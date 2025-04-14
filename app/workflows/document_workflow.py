from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.ocr_agent import process_with_ocr_agent
from app.agents.analysis_agent import process_with_analysis_agent
from app.agents.extraction_agent import process_with_extraction_agent
from app.core.logging import get_agent_logger, get_request_logger
from app.database.mongo import mongo_manager


class DocumentProcessingState(TypedDict):
    """State definition for document processing workflow"""
    
    # Input data
    document_id: str
    file_content: bytes
    filename: str
    language: str
    user_id: Optional[str]
    session_id: Optional[str]
    
    # Processing metadata
    status: str  # pending, processing, completed, failed, needs_review
    current_agent: str
    processing_start_time: datetime
    processing_end_time: Optional[datetime]
    
    # Agent outputs
    ocr_result: Optional[Dict[str, Any]]
    extracted_text: Optional[str]
    analysis_result: Optional[Dict[str, Any]]
    extraction_result: Optional[Dict[str, Any]]
    
    # Quality assessments
    ocr_quality: Optional[Dict[str, Any]]
    analysis_quality: Optional[Dict[str, Any]]
    extraction_quality: Optional[Dict[str, Any]]
    
    # Error handling
    error_message: Optional[str]
    retry_count: int
    needs_human_review: bool
    human_review_reason: Optional[str]
    
    # Performance metrics
    total_processing_time: float
    ocr_processing_time: float
    analysis_processing_time: float
    extraction_processing_time: float
    total_tokens_used: int


class DocumentWorkflow:
    """LangGraph workflow for document processing"""
    
    def __init__(self):
        self.workflow_name = "Document_Processing_Workflow"
        self.logger = get_agent_logger(self.workflow_name)
        self.request_logger = get_request_logger()
        self.graph = None
        self._build_workflow()
    
    def _build_workflow(self) -> None:
        """Build the LangGraph workflow"""
        
        # Create the state graph
        workflow = StateGraph(DocumentProcessingState)
        
        # Add nodes (agents)
        workflow.add_node("ocr_agent", self._ocr_agent_node)
        workflow.add_node("analysis_agent", self._analysis_agent_node)
        workflow.add_node("extraction_agent", self._extraction_agent_node)
        workflow.add_node("quality_check", self._quality_check_node)
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("finalize_results", self._finalize_results_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define the workflow entry point
        workflow.set_entry_point("ocr_agent")
        
        # Define edges from OCR agent
        workflow.add_conditional_edges(
            "ocr_agent",
            self._after_ocr_agent,
            {
                "proceed_to_analysis": "analysis_agent",
                "needs_human_review": "human_review",
                "error": "handle_error"
            }
        )
        
        # Define edges from analysis agent
        workflow.add_conditional_edges(
            "analysis_agent",
            self._after_analysis_agent,
            {
                "proceed_to_extraction": "extraction_agent",
                "needs_human_review": "human_review",
                "error": "handle_error"
            }
        )
        
        # Define edges from extraction agent
        workflow.add_conditional_edges(
            "extraction_agent",
            self._after_extraction_agent,
            {
                "proceed_to_quality_check": "quality_check",
                "needs_human_review": "human_review",
                "error": "handle_error"
            }
        )
        
        # Define edges from quality check
        workflow.add_conditional_edges(
            "quality_check",
            self._after_quality_check,
            {
                "complete": "finalize_results",
                "needs_human_review": "human_review",
                "error": "handle_error"
            }
        )
        
        # Define edges from human review
        workflow.add_edge("human_review", "finalize_results")
        
        # Define edges from error handling
        workflow.add_conditional_edges(
            "handle_error",
            self._after_error_handling,
            {
                "retry": "ocr_agent",
                "fail": END
            }
        )
        
        # Define edges from finalize
        workflow.add_edge("finalize_results", END)
        
        # Compile the graph
        # For now, we'll use in-memory checkpoints. In production, use SQLite or other persistent storage.
        memory = SqliteSaver.from_conn_string(":memory:")
        self.graph = workflow.compile(checkpointer=memory)
        
        self.logger.info("Document processing workflow built successfully")
    
    async def _ocr_agent_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """OCR agent node"""
        try:
            self.logger.agent_start(
                document_id=state['document_id'],
                extra={'agent': 'ocr_agent'}
            )
            
            # Prepare input for OCR agent
            document_data = {
                'file_content': state['file_content'],
                'filename': state['filename'],
                'language': state.get('language', 'eng')
            }
            
            # Process with OCR agent
            ocr_result = await process_with_ocr_agent(document_data, state)
            
            # Update state
            state['ocr_result'] = ocr_result
            state['extracted_text'] = ocr_result.get('extracted_text')
            state['ocr_processing_time'] = ocr_result.get('ocr_processing_time', 0)
            state['current_agent'] = 'ocr_agent'
            
            # Store workflow state in MongoDB
            if mongo_manager.db:
                await mongo_manager.store_agent_state(
                    document_id=state['document_id'],
                    agent_name='workflow_ocr',
                    state_data={
                        'input': document_data,
                        'output': ocr_result,
                        'state_snapshot': state
                    }
                )
            
            self.logger.agent_complete(
                document_id=state['document_id'],
                result={'ocr_completed': True}
            )
            
            return state
            
        except Exception as e:
            self.logger.agent_error(
                document_id=state['document_id'],
                error=e,
                extra={'agent': 'ocr_agent'}
            )
            state['error_message'] = f"OCR agent failed: {str(e)}"
            state['status'] = 'error'
            return state
    
    async def _analysis_agent_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Analysis agent node"""
        try:
            self.logger.agent_start(
                document_id=state['document_id'],
                extra={'agent': 'analysis_agent'}
            )
            
            # Check if we have extracted text
            if not state.get('extracted_text'):
                raise ValueError("No extracted text available for analysis")
            
            # Process with analysis agent
            analysis_result = await process_with_analysis_agent(state)
            
            # Update state
            state['analysis_result'] = analysis_result
            state['analysis_processing_time'] = analysis_result.get('analysis_processing_time', 0)
            state['total_tokens_used'] = analysis_result.get('analysis_tokens_used', 0)
            state['current_agent'] = 'analysis_agent'
            
            # Store workflow state in MongoDB
            if mongo_manager.db:
                await mongo_manager.store_agent_state(
                    document_id=state['document_id'],
                    agent_name='workflow_analysis',
                    state_data={
                        'input': {'extracted_text_length': len(state['extracted_text'])},
                        'output': analysis_result,
                        'state_snapshot': state
                    }
                )
            
            self.logger.agent_complete(
                document_id=state['document_id'],
                result={'analysis_completed': True}
            )
            
            return state
            
        except Exception as e:
            self.logger.agent_error(
                document_id=state['document_id'],
                error=e,
                extra={'agent': 'analysis_agent'}
            )
            state['error_message'] = f"Analysis agent failed: {str(e)}"
            state['status'] = 'error'
            return state
    
    async def _extraction_agent_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Extraction agent node"""
        try:
            self.logger.agent_start(
                document_id=state['document_id'],
                extra={'agent': 'extraction_agent'}
            )
            
            # Check if we have extracted text
            if not state.get('extracted_text'):
                raise ValueError("No extracted text available for field extraction")
            
            # Process with extraction agent
            extraction_result = await process_with_extraction_agent(state)
            
            # Update state
            state['extraction_result'] = extraction_result
            state['extraction_processing_time'] = extraction_result.get('extraction_processing_time', 0)
            state['total_tokens_used'] += extraction_result.get('extraction_tokens_used', 0)
            state['current_agent'] = 'extraction_agent'
            
            # Store workflow state in MongoDB
            if mongo_manager.db:
                await mongo_manager.store_agent_state(
                    document_id=state['document_id'],
                    agent_name='workflow_extraction',
                    state_data={
                        'input': {'extracted_text_length': len(state['extracted_text'])},
                        'output': extraction_result,
                        'state_snapshot': state
                    }
                )
            
            self.logger.agent_complete(
                document_id=state['document_id'],
                result={'extraction_completed': True}
            )
            
            return state
            
        except Exception as e:
            self.logger.agent_error(
                document_id=state['document_id'],
                error=e,
                extra={'agent': 'extraction_agent'}
            )
            state['error_message'] = f"Extraction agent failed: {str(e)}"
            state['status'] = 'error'
            return state
    
    async def _quality_check_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Quality check node"""
        try:
            self.logger.agent_start(
                document_id=state['document_id'],
                extra={'agent': 'quality_check'}
            )
            
            # Perform comprehensive quality assessment
            quality_assessment = await self._perform_quality_assessment(state)
            
            # Update state with quality assessment
            state['ocr_quality'] = quality_assessment.get('ocr_quality')
            state['analysis_quality'] = quality_assessment.get('analysis_quality')
            state['extraction_quality'] = quality_assessment.get('extraction_quality')
            state['needs_human_review'] = quality_assessment.get('needs_human_review', False)
            state['human_review_reason'] = quality_assessment.get('human_review_reason')
            state['current_agent'] = 'quality_check'
            
            self.logger.agent_complete(
                document_id=state['document_id'],
                result={'quality_check_completed': True}
            )
            
            return state
            
        except Exception as e:
            self.logger.agent_error(
                document_id=state['document_id'],
                error=e,
                extra={'agent': 'quality_check'}
            )
            state['error_message'] = f"Quality check failed: {str(e)}"
            state['status'] = 'error'
            return state
    
    async def _human_review_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Human review node (placeholder for human-in-the-loop)"""
        try:
            self.logger.agent_start(
                document_id=state['document_id'],
                extra={'agent': 'human_review'}
            )
            
            # In a real implementation, this would:
            # 1. Create a human review task
            # 2. Wait for human input
            # 3. Update state based on human review
            
            # For now, we'll just mark it as reviewed and continue
            state['needs_human_review'] = False
            state['human_review_reason'] = None
            state['current_agent'] = 'human_review'
            
            # Log that human review would be triggered here
            self.logger.info(
                f"Human review would be triggered for document: {state['document_id']}. "
                f"Reason: {state.get('human_review_reason', 'unknown')}"
            )
            
            self.logger.agent_complete(
                document_id=state['document_id'],
                result={'human_review_placeholder': True}
            )
            
            return state
            
        except Exception as e:
            self.logger.agent_error(
                document_id=state['document_id'],
                error=e,
                extra={'agent': 'human_review'}
            )
            state['error_message'] = f"Human review failed: {str(e)}"
            state['status'] = 'error'
            return state
    
    async def _finalize_results_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Finalize results node"""
        try:
            self.logger.agent_start(
                document_id=state['document_id'],
                extra={'agent': 'finalize_results'}
            )
            
            # Calculate total processing time
            state['total_processing_time'] = (
                state.get('ocr_processing_time', 0) +
                state.get('analysis_processing_time', 0) +
                state.get('extraction_processing_time', 0)
            )
            
            # Update status
            state['status'] = 'completed'
            state['processing_end_time'] = datetime.now()
            state['current_agent'] = 'finalize_results'
            
            # Store final state in MongoDB
            if mongo_manager.db:
                await mongo_manager.store_agent_state(
                    document_id=state['document_id'],
                    agent_name='workflow_final',
                    state_data={
                        'final_state': state,
                        'processing_summary': {
                            'total_time': state['total_processing_time'],
                            'tokens_used': state.get('total_tokens_used', 0),
                            'status': state['status']
                        }
                    }
                )
            
            self.logger.agent_complete(
                document_id=state['document_id'],
                result={'workflow_completed': True}
            )
            
            return state
            
        except Exception as e:
            self.logger.agent_error(
                document_id=state['document_id'],
                error=e,
                extra={'agent': 'finalize_results'}
            )
            state['error_message'] = f"Finalize results failed: {str(e)}"
            state['status'] = 'error'
            return state
    
    async def _handle_error_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Error handling node"""
        try:
            self.logger.agent_start(
                document_id=state['document_id'],
                extra={'agent': 'handle_error'}
            )
            
            # Increment retry count
            state['retry_count'] = state.get('retry_count', 0) + 1
            
            # Check if we should retry (max 3 retries)
            if state['retry_count'] <= 3:
                state['status'] = 'retrying'
                state['error_message'] = None
                self.logger.info(f"Retrying document processing (attempt {state['retry_count']})")
            else:
                state['status'] = 'failed'
                self.logger.error(f"Document processing failed after {state['retry_count']} attempts")
            
            state['current_agent'] = 'handle_error'
            
            self.logger.agent_complete(
                document_id=state['document_id'],
                result={'error_handled': True, 'retry_count': state['retry_count']}
            )
            
            return state
            
        except Exception as e:
            self.logger.agent_error(
                document_id=state['document_id'],
                error=e,
                extra={'agent': 'handle_error'}
            )
            state['error_message'] = f"Error handling failed: {str(e)}"
            state['status'] = 'failed'
            return state
    
    def _after_ocr_agent(self, state: DocumentProcessingState) -> str:
        """Determine next step after OCR agent"""
        if state.get('status') == 'error':
            return "error"
        
        # Check OCR quality
        ocr_confidence = state.get('ocr_result', {}).get('confidence_score', 0)
        if ocr_confidence < 0.5:  # Low confidence threshold
            state['needs_human_review'] = True
            state['human_review_reason'] = f"Low OCR confidence: {ocr_confidence:.2f}"
            return "needs_human_review"
        
        return "proceed_to_analysis"
    
    def _after_analysis_agent(self, state: DocumentProcessingState) -> str:
        """Determine next step after analysis agent"""
        if state.get('status') == 'error':
            return "error"
        
        # Check analysis quality
        analysis_result = state.get('analysis_result', {})
        if not analysis_result.get('brief_summary') or not analysis_result.get('detailed_summary'):
            state['needs_human_review'] = True
            state['human_review_reason'] = "Incomplete analysis results"
            return "needs_human_review"
        
        return "proceed_to_extraction"
    
    def _after_extraction_agent(self, state: DocumentProcessingState) -> str:
        """Determine next step after extraction agent"""
        if state.get('status') == 'error':
            return "error"
        
        # Check extraction quality
        extraction_result = state.get('extraction_result', {})
        field_extraction = extraction_result.get('field_extraction', {})
        
        if not field_extraction.get('document_type') or field_extraction.get('document_type') == 'unknown':
            state['needs_human_review'] = True
            state['human_review_reason'] = "Unable to determine document type"
            return "needs_human_review"
        
        if len(field_extraction.get('extracted_fields', {})) == 0:
            state['needs_human_review'] = True
            state['human_review_reason'] = "No fields extracted"
            return "needs_human_review"
        
        return "proceed_to_quality_check"
    
    def _after_quality_check(self, state: DocumentProcessingState) -> str:
        """Determine next step after quality check"""
        if state.get('status') == 'error':
            return "error"
        
        if state.get('needs_human_review', False):
            return "needs_human_review"
        
        return "complete"
    
    def _after_error_handling(self, state: DocumentProcessingState) -> str:
        """Determine next step after error handling"""
        if state.get('status') == 'retrying':
            return "retry"
        else:
            return "fail"
    
    async def _perform_quality_assessment(self, state: DocumentProcessingState) -> Dict[str, Any]:
        """Perform comprehensive quality assessment"""
        quality_assessment = {
            'ocr_quality': {},
            'analysis_quality': {},
            'extraction_quality': {},
            'needs_human_review': False,
            'human_review_reason': None
        }
        
        # OCR quality assessment
        ocr_result = state.get('ocr_result', {})
        ocr_confidence = ocr_result.get('confidence_score', 0)
        extracted_text = state.get('extracted_text', '')
        
        quality_assessment['ocr_quality'] = {
            'confidence_score': ocr_confidence,
            'text_length': len(extracted_text),
            'word_count': len(extracted_text.split()),
            'quality_rating': 'excellent' if ocr_confidence >= 0.8 else 'good' if ocr_confidence >= 0.6 else 'fair'
        }
        
        # Analysis quality assessment
        analysis_result = state.get('analysis_result', {})
        brief_summary = analysis_result.get('brief_summary', '')
        detailed_summary = analysis_result.get('detailed_summary', '')
        
        quality_assessment['analysis_quality'] = {
            'brief_summary_length': len(brief_summary),
            'detailed_summary_length': len(detailed_summary),
            'sentiment_identified': bool(analysis_result.get('sentiment_analysis', {}).get('sentiment')),
            'quality_rating': 'excellent' if len(detailed_summary) > 200 else 'good' if len(detailed_summary) > 100 else 'fair'
        }
        
        # Extraction quality assessment
        extraction_result = state.get('extraction_result', {})
        field_extraction = extraction_result.get('field_extraction', {})
        
        quality_assessment['extraction_quality'] = {
            'document_type': field_extraction.get('document_type'),
            'document_type_confidence': field_extraction.get('document_type_confidence', 0),
            'fields_extracted': len(field_extraction.get('extracted_fields', {})),
            'quality_rating': 'excellent' if field_extraction.get('document_type_confidence', 0) >= 0.8 else 'good' if field_extraction.get('document_type_confidence', 0) >= 0.6 else 'fair'
        }
        
        # Determine if human review is needed
        issues = []
        
        if quality_assessment['ocr_quality']['quality_rating'] == 'fair':
            issues.append("Poor OCR quality")
        
        if quality_assessment['analysis_quality']['quality_rating'] == 'fair':
            issues.append("Poor analysis quality")
        
        if quality_assessment['extraction_quality']['quality_rating'] == 'fair':
            issues.append("Poor extraction quality")
        
        if issues:
            quality_assessment['needs_human_review'] = True
            quality_assessment['human_review_reason'] = "; ".join(issues)
        
        return quality_assessment
    
    async def process_document(self, file_content: bytes, filename: str, user_id: str = None, session_id: str = None, language: str = 'eng') -> Dict[str, Any]:
        """Main method to process a document through the workflow"""
        try:
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Initialize state
            initial_state: DocumentProcessingState = {
                'document_id': document_id,
                'file_content': file_content,
                'filename': filename,
                'language': language,
                'user_id': user_id,
                'session_id': session_id,
                'status': 'processing',
                'current_agent': 'start',
                'processing_start_time': datetime.now(),
                'processing_end_time': None,
                'ocr_result': None,
                'extracted_text': None,
                'analysis_result': None,
                'extraction_result': None,
                'ocr_quality': None,
                'analysis_quality': None,
                'extraction_quality': None,
                'error_message': None,
                'retry_count': 0,
                'needs_human_review': False,
                'human_review_reason': None,
                'total_processing_time': 0,
                'ocr_processing_time': 0,
                'analysis_processing_time': 0,
                'extraction_processing_time': 0,
                'total_tokens_used': 0
            }
            
            self.logger.info(f"Starting document processing workflow for: {filename} (ID: {document_id})")
            
            # Execute the workflow
            final_state = await self.graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": document_id}}
            )
            
            # Return the final result
            return {
                'document_id': document_id,
                'status': final_state['status'],
                'results': {
                    'ocr': final_state.get('ocr_result'),
                    'analysis': final_state.get('analysis_result'),
                    'extraction': final_state.get('extraction_result')
                },
                'quality_assessment': {
                    'ocr': final_state.get('ocr_quality'),
                    'analysis': final_state.get('analysis_quality'),
                    'extraction': final_state.get('extraction_quality')
                },
                'processing_metrics': {
                    'total_time': final_state.get('total_processing_time', 0),
                    'ocr_time': final_state.get('ocr_processing_time', 0),
                    'analysis_time': final_state.get('analysis_processing_time', 0),
                    'extraction_time': final_state.get('extraction_processing_time', 0),
                    'tokens_used': final_state.get('total_tokens_used', 0)
                },
                'needs_human_review': final_state.get('needs_human_review', False),
                'human_review_reason': final_state.get('human_review_reason'),
                'error_message': final_state.get('error_message')
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise


# Global workflow instance
document_workflow = DocumentWorkflow()

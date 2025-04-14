from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from app.services.llm_service import llm_service
from app.database.postgres import Document, DocumentSummary
from app.database.mongo import mongo_manager
from app.core.logging import get_agent_logger, get_request_logger


class AnalysisAgent:
    """Analysis & Summarization Agent - Handles document analysis and summarization"""
    
    def __init__(self):
        self.agent_name = "Analysis_Summarization_Agent"
        self.logger = get_agent_logger(self.agent_name)
        self.request_logger = get_request_logger()
    
    async def process_document(self, extracted_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process extracted text through analysis and summarization"""
        try:
            document_id = state.get('document_id', 'unknown')
            self.logger.agent_start(
                document_id=document_id,
                extra={'text_length': len(extracted_text)}
            )
            
            if not extracted_text or len(extracted_text.strip()) == 0:
                raise ValueError("No text available for analysis")
            
            # Generate comprehensive analysis using LLM service
            analysis_result = await llm_service.generate_comprehensive_analysis(
                text=extracted_text,
                document_id=document_id
            )
            
            # Prepare result for state update
            result = {
                'brief_summary': analysis_result['brief_summary'],
                'detailed_summary': analysis_result['detailed_summary'],
                'sentiment_analysis': analysis_result['sentiment_analysis'],
                'analysis_processing_time': analysis_result['processing_metrics']['total_processing_time'],
                'analysis_tokens_used': analysis_result['processing_metrics']['total_tokens'],
                'llm_models_used': analysis_result['llm_models_used']
            }
            
            # Store agent state in MongoDB
            if mongo_manager.db:
                await mongo_manager.store_agent_state(
                    document_id=document_id,
                    agent_name=self.agent_name,
                    state_data={
                        'input_metadata': {
                            'text_length': len(extracted_text),
                            'word_count': len(extracted_text.split())
                        },
                        'output_result': result,
                        'processing_metrics': {
                            'processing_time': analysis_result['processing_metrics']['total_processing_time'],
                            'tokens_used': analysis_result['processing_metrics']['total_tokens']
                        }
                    }
                )
            
            self.logger.agent_complete(
                document_id=document_id,
                result={
                    'analysis_completed': True,
                    'processing_time': analysis_result['processing_metrics']['total_processing_time'],
                    'tokens_used': analysis_result['processing_metrics']['total_tokens']
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.agent_error(
                document_id=state.get('document_id', 'unknown'),
                error=e,
                extra={'text_length': len(extracted_text)}
            )
            raise
    
    async def save_analysis_result_to_database(self, session, document_id: str, analysis_data: Dict[str, Any]) -> str:
        """Save analysis result to PostgreSQL database"""
        try:
            # Create document summary record
            summary = DocumentSummary(
                id=str(uuid.uuid4()),
                document_id=document_id,
                brief_summary=analysis_data.get('brief_summary'),
                detailed_summary=analysis_data.get('detailed_summary'),
                key_topics=analysis_data.get('sentiment_analysis', {}).get('key_topics', []),
                entities=analysis_data.get('sentiment_analysis', {}).get('entities', {}),
                sentiment=analysis_data.get('sentiment_analysis', {}).get('sentiment'),
                tone=analysis_data.get('sentiment_analysis', {}).get('tone', []),
                llm_model=analysis_data.get('llm_models_used', {}).get('analysis_model'),
                llm_provider=analysis_data.get('llm_models_used', {}).get('analysis_provider'),
                tokens_used=analysis_data.get('analysis_tokens_used', 0),
                processing_time=analysis_data.get('analysis_processing_time')
            )
            
            session.add(summary)
            await session.commit()
            
            self.logger.info(f"Saved analysis result to database for document: {document_id}")
            
            return summary.id
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis result to database: {str(e)}")
            raise
    
    def validate_analysis_quality(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis quality and determine if human review is needed"""
        try:
            brief_summary = analysis_data.get('brief_summary', '')
            detailed_summary = analysis_data.get('detailed_summary', '')
            sentiment = analysis_data.get('sentiment_analysis', {}).get('sentiment', 'unknown')
            
            # Quality assessment
            quality_metrics = {
                'brief_summary_length': len(brief_summary),
                'detailed_summary_length': len(detailed_summary),
                'sentiment_identified': sentiment != 'unknown',
                'needs_human_review': False,
                'quality_issues': []
            }
            
            # Check for quality issues
            if len(brief_summary.strip()) == 0:
                quality_metrics['needs_human_review'] = True
                quality_metrics['quality_issues'].append("Brief summary is empty")
            
            if len(detailed_summary.strip()) == 0:
                quality_metrics['needs_human_review'] = True
                quality_metrics['quality_issues'].append("Detailed summary is empty")
            
            if len(brief_summary) < 50:  # Very short summary
                quality_metrics['quality_issues'].append("Brief summary appears to be very short")
            
            if sentiment == 'unknown':
                quality_metrics['quality_issues'].append("Sentiment could not be determined")
            
            # Quality rating based on issues
            if len(quality_metrics['quality_issues']) == 0:
                quality_metrics['quality_rating'] = 'excellent'
            elif len(quality_metrics['quality_issues']) <= 2 and not quality_metrics['needs_human_review']:
                quality_metrics['quality_rating'] = 'good'
            elif quality_metrics['needs_human_review']:
                quality_metrics['quality_rating'] = 'needs_review'
            else:
                quality_metrics['quality_rating'] = 'fair'
            
            self.logger.info(
                f"Analysis quality assessment: {quality_metrics['quality_rating']} "
                f"(issues: {len(quality_metrics['quality_issues'])})"
            )
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Analysis quality validation failed: {str(e)}")
            return {
                'needs_human_review': True,
                'quality_issues': [f"Validation error: {str(e)}"],
                'quality_rating': 'unknown'
            }
    
    async def generate_targeted_summary(self, text: str, document_id: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Generate targeted summary focusing on specific areas"""
        try:
            self.logger.agent_start(
                document_id=document_id,
                extra={'focus_areas': focus_areas, 'analysis_type': 'targeted'}
            )
            
            model = llm_service.get_preferred_model()
            if not model:
                raise ValueError("No LLM model available")
            
            system_prompt = f"""You are a professional document analyst. Create a targeted summary focusing specifically on: {', '.join(focus_areas)}.

Structure your response to address each focus area clearly and provide relevant details from the document."""
            
            # This would be implemented similarly to the LLM service methods
            # For now, we'll use the existing comprehensive analysis
            result = await llm_service.generate_comprehensive_analysis(text, document_id)
            
            self.logger.agent_complete(
                document_id=document_id,
                result={'targeted_analysis_completed': True}
            )
            
            return {
                'targeted_summary': result['detailed_summary'],
                'focus_areas': focus_areas
            }
            
        except Exception as e:
            self.logger.agent_error(
                document_id=document_id,
                error=e,
                extra={'focus_areas': focus_areas}
            )
            raise
    
    async def extract_key_insights(self, text: str, document_id: str) -> Dict[str, Any]:
        """Extract key insights and actionable information from document"""
        try:
            self.logger.agent_start(
                document_id=document_id,
                extra={'analysis_type': 'key_insights'}
            )
            
            model = llm_service.get_preferred_model()
            if not model:
                raise ValueError("No LLM model available")
            
            # This would be implemented with specific prompts for insight extraction
            # For now, we'll use the existing analysis
            result = await llm_service.generate_comprehensive_analysis(text, document_id)
            
            # Extract insights from the analysis
            insights = {
                'main_topics': result['sentiment_analysis'].get('key_topics', []),
                'sentiment': result['sentiment_analysis'].get('sentiment'),
                'document_type': result['sentiment_analysis'].get('document_type', 'unknown'),
                'key_entities': result['sentiment_analysis'].get('entities', {})
            }
            
            self.logger.agent_complete(
                document_id=document_id,
                result={'insights_extracted': True}
            )
            
            return insights
            
        except Exception as e:
            self.logger.agent_error(
                document_id=document_id,
                error=e,
                extra={'analysis_type': 'key_insights'}
            )
            raise


# Global analysis agent instance
analysis_agent = AnalysisAgent()


async def process_with_analysis_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper function for LangGraph workflow"""
    extracted_text = state.get('extracted_text', '')
    return await analysis_agent.process_document(extracted_text, state)

from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from app.services.ocr_service import ocr_service
from app.database.postgres import Document, OCRResult
from app.database.mongo import mongo_manager
from app.core.logging import get_agent_logger, get_request_logger


class OCRAgent:
    """OCR Extraction Agent - Handles text extraction from documents"""
    
    def __init__(self):
        self.agent_name = "OCR_Extraction_Agent"
        self.logger = get_agent_logger(self.agent_name)
        self.request_logger = get_request_logger()
    
    async def process_document(self, document_data: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Process document through OCR extraction"""
        try:
            self.logger.agent_start(
                document_id=state.get('document_id', 'unknown'),
                extra={'filename': document_data.get('filename')}
            )
            
            # Extract file content and metadata
            file_content = document_data.get('file_content')
            filename = document_data.get('filename')
            language = document_data.get('language', 'eng')
            
            if not file_content or not filename:
                raise ValueError("File content and filename are required")
            
            # Process document with OCR service
            ocr_result = await ocr_service.process_document(
                file_content=file_content,
                filename=filename,
                language=language
            )
            
            # Store raw OCR results in MongoDB
            if mongo_manager.db:
                await mongo_manager.store_ocr_raw_result(
                    document_id=state['document_id'],
                    ocr_data={
                        'ocr_engine': ocr_result['ocr_engine'],
                        'raw_results': ocr_result,
                        'page_data': ocr_result.get('page_results', []),
                        'confidence_scores': {
                            'overall': ocr_result['confidence_score'],
                            'per_page': [page.get('confidence', 0) for page in ocr_result.get('page_results', [])]
                        },
                        'processing_metadata': {
                            'processing_time': ocr_result['processing_time'],
                            'total_pages': ocr_result['total_pages'],
                            'language': language
                        }
                    }
                )
            
            # Prepare result for state update
            result = {
                'ocr_result': ocr_result,
                'extracted_text': ocr_result['extracted_text'],
                'confidence_score': ocr_result['confidence_score'],
                'total_pages': ocr_result['total_pages'],
                'ocr_processing_time': ocr_result['processing_time'],
                'ocr_engine_used': ocr_result['ocr_engine']
            }
            
            # Store agent state in MongoDB
            if mongo_manager.db:
                await mongo_manager.store_agent_state(
                    document_id=state['document_id'],
                    agent_name=self.agent_name,
                    state_data={
                        'input_metadata': {
                            'filename': filename,
                            'file_size': len(file_content),
                            'language': language
                        },
                        'output_result': result,
                        'processing_metrics': {
                            'processing_time': ocr_result['processing_time'],
                            'confidence_score': ocr_result['confidence_score']
                        }
                    }
                )
            
            self.logger.agent_complete(
                document_id=state['document_id'],
                result={
                    'pages_processed': ocr_result['total_pages'],
                    'confidence_score': ocr_result['confidence_score'],
                    'processing_time': ocr_result['processing_time']
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.agent_error(
                document_id=state.get('document_id', 'unknown'),
                error=e,
                extra={
                    'filename': document_data.get('filename'),
                    'file_size': len(document_data.get('file_content', b''))
                }
            )
            raise
    
    async def save_ocr_result_to_database(self, session, document_id: str, ocr_data: Dict[str, Any]) -> str:
        """Save OCR result to PostgreSQL database"""
        try:
            # Create OCR result record
            ocr_result = OCRResult(
                id=str(uuid.uuid4()),
                document_id=document_id,
                ocr_engine=ocr_data.get('ocr_engine_used', 'tesseract'),
                confidence_score=ocr_data.get('confidence_score'),
                total_pages=ocr_data.get('total_pages', 1),
                language='eng',  # Default language
                extracted_text=ocr_data.get('extracted_text', ''),
                page_metadata={
                    'page_results': ocr_data.get('ocr_result', {}).get('page_results', []),
                    'overall_confidence': ocr_data.get('confidence_score')
                },
                processing_time=ocr_data.get('ocr_processing_time')
            )
            
            session.add(ocr_result)
            await session.commit()
            
            self.logger.info(f"Saved OCR result to database for document: {document_id}")
            
            return ocr_result.id
            
        except Exception as e:
            self.logger.error(f"Failed to save OCR result to database: {str(e)}")
            raise
    
    def validate_ocr_quality(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate OCR quality and determine if human review is needed"""
        try:
            confidence_score = ocr_data.get('confidence_score', 0)
            extracted_text = ocr_data.get('extracted_text', '')
            
            # Quality assessment
            quality_metrics = {
                'confidence_score': confidence_score,
                'text_length': len(extracted_text),
                'word_count': len(extracted_text.split()),
                'needs_human_review': False,
                'quality_issues': []
            }
            
            # Check for quality issues
            if confidence_score < settings.ocr.ocr_confidence_threshold:
                quality_metrics['needs_human_review'] = True
                quality_metrics['quality_issues'].append(
                    f"Low confidence score: {confidence_score:.2f} (threshold: {settings.ocr.ocr_confidence_threshold})"
                )
            
            if len(extracted_text.strip()) == 0:
                quality_metrics['needs_human_review'] = True
                quality_metrics['quality_issues'].append("No text extracted")
            
            if len(extracted_text.split()) < 10:  # Very short document
                quality_metrics['quality_issues'].append("Document appears to be very short")
            
            # Quality rating
            if confidence_score >= 0.9:
                quality_metrics['quality_rating'] = 'excellent'
            elif confidence_score >= 0.7:
                quality_metrics['quality_rating'] = 'good'
            elif confidence_score >= 0.5:
                quality_metrics['quality_rating'] = 'fair'
            else:
                quality_metrics['quality_rating'] = 'poor'
            
            self.logger.info(
                f"OCR quality assessment for document: {quality_metrics['quality_rating']} "
                f"(confidence: {confidence_score:.2f})"
            )
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"OCR quality validation failed: {str(e)}")
            return {
                'confidence_score': 0,
                'needs_human_review': True,
                'quality_issues': [f"Validation error: {str(e)}"],
                'quality_rating': 'unknown'
            }
    
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported OCR languages"""
        return ocr_service.get_supported_languages()
    
    async def get_ocr_engine_info(self) -> Dict[str, Any]:
        """Get information about the OCR engine"""
        return ocr_service.get_ocr_engine_info()


# Global OCR agent instance
ocr_agent = OCRAgent()


async def process_with_ocr_agent(document_data: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper function for LangGraph workflow"""
    return await ocr_agent.process_document(document_data, state)

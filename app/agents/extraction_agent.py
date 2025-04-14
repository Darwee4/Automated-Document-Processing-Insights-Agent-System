from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import json

from app.services.llm_service import llm_service
from app.database.postgres import Document, StructuredData
from app.database.mongo import mongo_manager
from app.core.logging import get_agent_logger, get_request_logger
from app.core.config import settings


class ExtractionAgent:
    """Structured Field Detection Agent - Handles structured data extraction and document classification"""
    
    def __init__(self):
        self.agent_name = "Structured_Field_Detection_Agent"
        self.logger = get_agent_logger(self.agent_name)
        self.request_logger = get_request_logger()
    
    async def process_document(self, extracted_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process extracted text through structured field extraction"""
        try:
            document_id = state.get('document_id', 'unknown')
            self.logger.agent_start(
                document_id=document_id,
                extra={'text_length': len(extracted_text)}
            )
            
            if not extracted_text or len(extracted_text.strip()) == 0:
                raise ValueError("No text available for field extraction")
            
            # Extract structured fields using LLM service
            extraction_result = await llm_service.extract_structured_fields(
                text=extracted_text,
                document_id=document_id
            )
            
            # Prepare result for state update
            result = {
                'field_extraction': extraction_result['extraction'],
                'extraction_processing_time': extraction_result['processing_time'],
                'extraction_tokens_used': extraction_result['tokens_used'],
                'llm_model_used': extraction_result['llm_model'],
                'llm_provider_used': extraction_result['llm_provider']
            }
            
            # Validate extraction quality
            quality_metrics = self.validate_extraction_quality(result['field_extraction'])
            result['quality_metrics'] = quality_metrics
            
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
                            'processing_time': extraction_result['processing_time'],
                            'tokens_used': extraction_result['tokens_used']
                        },
                        'quality_metrics': quality_metrics
                    }
                )
            
            self.logger.agent_complete(
                document_id=document_id,
                result={
                    'extraction_completed': True,
                    'document_type': result['field_extraction'].get('document_type', 'unknown'),
                    'fields_extracted': len(result['field_extraction'].get('extracted_fields', {})),
                    'processing_time': extraction_result['processing_time'],
                    'tokens_used': extraction_result['tokens_used']
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
    
    async def save_extraction_result_to_database(self, session, document_id: str, extraction_data: Dict[str, Any]) -> str:
        """Save extraction result to PostgreSQL database"""
        try:
            field_extraction = extraction_data.get('field_extraction', {})
            
            # Create structured data record
            structured_data = StructuredData(
                id=str(uuid.uuid4()),
                document_id=document_id,
                document_type=field_extraction.get('document_type'),
                document_type_confidence=field_extraction.get('document_type_confidence'),
                extracted_fields=field_extraction.get('extracted_fields', {}),
                tables=field_extraction.get('tables', []),
                validation_results=field_extraction.get('validation_notes', []),
                extraction_model=extraction_data.get('llm_model_used'),
                confidence_threshold=0.7,  # Default threshold
                fields_with_low_confidence=self._identify_low_confidence_fields(
                    field_extraction.get('extracted_fields', {})
                ),
                processing_time=extraction_data.get('extraction_processing_time')
            )
            
            session.add(structured_data)
            await session.commit()
            
            self.logger.info(f"Saved extraction result to database for document: {document_id}")
            
            return structured_data.id
            
        except Exception as e:
            self.logger.error(f"Failed to save extraction result to database: {str(e)}")
            raise
    
    def _identify_low_confidence_fields(self, extracted_fields: Dict[str, Any]) -> List[str]:
        """Identify fields with low confidence scores"""
        low_confidence_fields = []
        
        for field_name, field_data in extracted_fields.items():
            confidence = field_data.get('confidence', 0)
            if confidence < 0.7:  # Confidence threshold
                low_confidence_fields.append(field_name)
        
        return low_confidence_fields
    
    def validate_extraction_quality(self, extraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extraction quality and determine if human review is needed"""
        try:
            document_type = extraction_data.get('document_type', 'unknown')
            extracted_fields = extraction_data.get('extracted_fields', {})
            document_type_confidence = extraction_data.get('document_type_confidence', 0)
            
            # Quality assessment
            quality_metrics = {
                'document_type': document_type,
                'document_type_confidence': document_type_confidence,
                'fields_extracted': len(extracted_fields),
                'low_confidence_fields': self._identify_low_confidence_fields(extracted_fields),
                'needs_human_review': False,
                'quality_issues': []
            }
            
            # Check for quality issues
            if document_type_confidence < 0.7:
                quality_metrics['needs_human_review'] = True
                quality_metrics['quality_issues'].append(
                    f"Low document type confidence: {document_type_confidence:.2f}"
                )
            
            if document_type == 'unknown':
                quality_metrics['needs_human_review'] = True
                quality_metrics['quality_issues'].append("Document type could not be determined")
            
            if len(extracted_fields) == 0:
                quality_metrics['needs_human_review'] = True
                quality_metrics['quality_issues'].append("No fields extracted")
            
            if len(quality_metrics['low_confidence_fields']) > 0:
                quality_metrics['quality_issues'].append(
                    f"Low confidence fields: {', '.join(quality_metrics['low_confidence_fields'])}"
                )
            
            # Quality rating
            if (document_type_confidence >= 0.8 and 
                len(extracted_fields) > 0 and 
                len(quality_metrics['low_confidence_fields']) == 0):
                quality_metrics['quality_rating'] = 'excellent'
            elif (document_type_confidence >= 0.7 and 
                  len(extracted_fields) > 0 and 
                  len(quality_metrics['low_confidence_fields']) <= 2):
                quality_metrics['quality_rating'] = 'good'
            elif quality_metrics['needs_human_review']:
                quality_metrics['quality_rating'] = 'needs_review'
            else:
                quality_metrics['quality_rating'] = 'fair'
            
            self.logger.info(
                f"Extraction quality assessment: {quality_metrics['quality_rating']} "
                f"(document_type: {document_type}, confidence: {document_type_confidence:.2f})"
            )
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Extraction quality validation failed: {str(e)}")
            return {
                'needs_human_review': True,
                'quality_issues': [f"Validation error: {str(e)}"],
                'quality_rating': 'unknown'
            }
    
    async def extract_specific_field_types(self, text: str, document_id: str, field_types: List[str]) -> Dict[str, Any]:
        """Extract specific types of fields from document"""
        try:
            self.logger.agent_start(
                document_id=document_id,
                extra={'field_types': field_types, 'extraction_type': 'targeted'}
            )
            
            model = llm_service.get_preferred_model()
            if not model:
                raise ValueError("No LLM model available")
            
            # Create targeted prompt for specific field types
            field_descriptions = {
                'dates': 'dates (issue date, due date, effective date, etc.)',
                'amounts': 'monetary amounts (total, subtotal, tax, etc.)',
                'names': 'names (sender, recipient, parties involved)',
                'addresses': 'addresses (street, city, state, zip code)',
                'contact_info': 'contact information (phone numbers, email addresses)',
                'document_ids': 'document identifiers (invoice numbers, contract IDs, etc.)'
            }
            
            requested_fields = [field_descriptions.get(ft, ft) for ft in field_types]
            
            system_prompt = f"""You are a professional document processor. Extract only the following specific field types from the document:
{', '.join(requested_fields)}

Return your extraction in JSON format with the following structure:
{{
    "extracted_fields": {{
        "field_name": {{
            "value": "extracted_value",
            "confidence": 0.9,
            "type": "field_type"
        }}
    }},
    "field_summary": {{
        "total_fields_extracted": 0,
        "fields_by_type": {{}}
    }}
}}"""
            
            # This would be implemented similarly to the LLM service methods
            # For now, we'll use the existing field extraction
            result = await llm_service.extract_structured_fields(text, document_id)
            
            # Filter fields to only include requested types
            filtered_fields = {}
            for field_name, field_data in result['extraction'].get('extracted_fields', {}).items():
                field_type = self._infer_field_type(field_name, field_data.get('value', ''))
                if field_type in field_types:
                    filtered_fields[field_name] = field_data
            
            self.logger.agent_complete(
                document_id=document_id,
                result={'targeted_extraction_completed': True}
            )
            
            return {
                'targeted_extraction': filtered_fields,
                'field_types': field_types
            }
            
        except Exception as e:
            self.logger.agent_error(
                document_id=document_id,
                error=e,
                extra={'field_types': field_types}
            )
            raise
    
    def _infer_field_type(self, field_name: str, field_value: str) -> str:
        """Infer field type based on field name and value"""
        field_name_lower = field_name.lower()
        field_value_lower = str(field_value).lower()
        
        # Date detection
        if any(keyword in field_name_lower for keyword in ['date', 'time', 'deadline']):
            return 'dates'
        
        # Amount detection
        if any(keyword in field_name_lower for keyword in ['amount', 'total', 'price', 'cost', 'fee']):
            return 'amounts'
        
        # Name detection
        if any(keyword in field_name_lower for keyword in ['name', 'person', 'contact', 'sender', 'recipient']):
            return 'names'
        
        # Address detection
        if any(keyword in field_name_lower for keyword in ['address', 'location', 'street', 'city', 'zip']):
            return 'addresses'
        
        # Contact info detection
        if any(keyword in field_name_lower for keyword in ['phone', 'email', 'contact']):
            return 'contact_info'
        
        # Document ID detection
        if any(keyword in field_name_lower for keyword in ['id', 'number', 'reference', 'invoice']):
            return 'document_ids'
        
        return 'other'
    
    async def validate_extracted_fields(self, extracted_fields: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """Validate extracted fields against document type expectations"""
        try:
            validation_results = {
                'valid_fields': [],
                'invalid_fields': [],
                'missing_fields': [],
                'validation_score': 0.0
            }
            
            # Define expected fields by document type
            expected_fields = self._get_expected_fields_by_type(document_type)
            
            # Check extracted fields
            for field_name, field_data in extracted_fields.items():
                if field_name in expected_fields:
                    validation_results['valid_fields'].append(field_name)
                else:
                    validation_results['invalid_fields'].append(field_name)
            
            # Check missing fields
            for expected_field in expected_fields:
                if expected_field not in extracted_fields:
                    validation_results['missing_fields'].append(expected_field)
            
            # Calculate validation score
            total_expected = len(expected_fields)
            if total_expected > 0:
                validation_results['validation_score'] = len(validation_results['valid_fields']) / total_expected
            
            self.logger.info(
                f"Field validation completed for {document_type}: "
                f"score={validation_results['validation_score']:.2f}"
            )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Field validation failed: {str(e)}")
            return {
                'valid_fields': [],
                'invalid_fields': [],
                'missing_fields': [],
                'validation_score': 0.0,
                'error': str(e)
            }
    
    def _get_expected_fields_by_type(self, document_type: str) -> List[str]:
        """Get expected fields for different document types"""
        field_templates = {
            'invoice': ['invoice_number', 'invoice_date', 'due_date', 'total_amount', 'vendor_name', 'customer_name'],
            'contract': ['contract_number', 'effective_date', 'expiration_date', 'parties', 'terms'],
            'report': ['report_title', 'author', 'date', 'summary', 'recommendations'],
            'form': ['form_name', 'submission_date', 'applicant_name', 'approval_status'],
            'email': ['sender', 'recipient', 'subject', 'date', 'body'],
            'letter': ['sender', 'recipient', 'date', 'subject', 'salutation', 'closing'],
            'memo': ['to', 'from', 'date', 'subject', 'body'],
            'proposal': ['proposal_title', 'client_name', 'submission_date', 'total_cost', 'timeline'],
            'resume': ['name', 'contact_info', 'education', 'experience', 'skills'],
            'academic_paper': ['title', 'authors', 'abstract', 'keywords', 'references']
        }
        
        return field_templates.get(document_type, [])


# Global extraction agent instance
extraction_agent = ExtractionAgent()


async def process_with_extraction_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper function for LangGraph workflow"""
    extracted_text = state.get('extracted_text', '')
    return await extraction_agent.process_document(extracted_text, state)

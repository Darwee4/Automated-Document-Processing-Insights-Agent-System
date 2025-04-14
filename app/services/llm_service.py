from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.logging import get_agent_logger, get_cost_logger


class LLMService:
    """LLM service for text analysis, summarization, and extraction"""
    
    def __init__(self):
        self.logger = get_agent_logger("LLM_Service")
        self.cost_logger = get_cost_logger()
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize LLM models based on configuration"""
        try:
            # Initialize OpenAI model if API key is available
            if settings.llm.openai_api_key:
                self.models['openai'] = ChatOpenAI(
                    model=settings.llm.openai_model,
                    temperature=settings.llm.analysis_temperature,
                    openai_api_key=settings.llm.openai_api_key
                )
                self.logger.info(f"Initialized OpenAI model: {settings.llm.openai_model}")
            
            # Initialize Anthropic model if API key is available
            if settings.llm.anthropic_api_key:
                self.models['anthropic'] = ChatAnthropic(
                    model=settings.llm.anthropic_model,
                    temperature=settings.llm.analysis_temperature,
                    anthropic_api_key=settings.llm.anthropic_api_key
                )
                self.logger.info(f"Initialized Anthropic model: {settings.llm.anthropic_model}")
            
            if not self.models:
                self.logger.warning("No LLM models initialized. Please set API keys in configuration.")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM models: {str(e)}")
            raise
    
    def get_preferred_model(self) -> Optional[Any]:
        """Get the preferred LLM model based on configuration"""
        # Prefer Anthropic if available, otherwise OpenAI
        if 'anthropic' in self.models:
            return self.models['anthropic']
        elif 'openai' in self.models:
            return self.models['openai']
        else:
            return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def generate_summary(self, text: str, document_id: str, summary_type: str = "brief") -> Dict[str, Any]:
        """Generate document summary using LLM"""
        try:
            start_time = datetime.now()
            model = self.get_preferred_model()
            
            if not model:
                raise ValueError("No LLM model available. Please check API key configuration.")
            
            # Define system prompts based on summary type
            if summary_type == "brief":
                system_prompt = """You are a professional document analyst. Create a concise 2-3 sentence summary that captures the main points and purpose of the document. Focus on the key information that would be most useful for someone quickly understanding the document's content."""
            else:  # detailed
                system_prompt = """You are a professional document analyst. Create a comprehensive summary that covers:
1. Main purpose and key objectives
2. Important findings or conclusions
3. Key recommendations or actions
4. Critical data points or statistics
5. Overall significance or implications

Provide a thorough analysis that would help someone understand the document in depth."""
            
            # Prepare messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Please analyze and summarize the following document:\n\n{text}")
            ]
            
            # Generate response with token tracking
            with get_openai_callback() as cb:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: model.invoke(messages)
                )
                
                # Extract content and track usage
                summary_content = response.content
                tokens_used = cb.total_tokens
                cost = cb.total_cost
                
                # Determine provider for cost tracking
                provider = "openai" if isinstance(model, ChatOpenAI) else "anthropic"
                model_name = settings.llm.openai_model if provider == "openai" else settings.llm.anthropic_model
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'summary': summary_content,
                'summary_type': summary_type,
                'tokens_used': tokens_used,
                'processing_time': processing_time,
                'llm_provider': provider,
                'llm_model': model_name
            }
            
            # Track cost
            self.cost_logger.track_llm_usage(
                provider=provider,
                model=model_name,
                tokens_used=tokens_used,
                cost=cost,
                document_id=document_id
            )
            
            self.logger.info(f"Generated {summary_type} summary for document: {document_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Summary generation failed for document {document_id}: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def analyze_sentiment_and_topics(self, text: str, document_id: str) -> Dict[str, Any]:
        """Analyze sentiment, tone, and extract key topics"""
        try:
            start_time = datetime.now()
            model = self.get_preferred_model()
            
            if not model:
                raise ValueError("No LLM model available. Please check API key configuration.")
            
            system_prompt = """You are a professional document analyst. Analyze the provided text and extract:
1. Overall sentiment (positive, negative, neutral)
2. Tone analysis (formal, informal, technical, persuasive, etc.)
3. Key topics and themes (list the 5-10 most important topics)
4. Named entities (people, organizations, locations, dates, etc.)
5. Document type classification

Return your analysis in JSON format with the following structure:
{
    "sentiment": "positive/negative/neutral",
    "tone": ["list", "of", "tone", "descriptors"],
    "key_topics": ["topic1", "topic2", "topic3"],
    "entities": {
        "people": [],
        "organizations": [],
        "locations": [],
        "dates": [],
        "other": []
    },
    "document_type": "invoice/contract/report/email/etc."
}"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Please analyze the following document:\n\n{text}")
            ]
            
            with get_openai_callback() as cb:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: model.invoke(messages)
                )
                
                analysis_content = response.content
                tokens_used = cb.total_tokens
                cost = cb.total_cost
                
                provider = "openai" if isinstance(model, ChatOpenAI) else "anthropic"
                model_name = settings.llm.openai_model if provider == "openai" else settings.llm.anthropic_model
            
            # Parse JSON response
            try:
                analysis_data = json.loads(analysis_content)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a basic structure
                analysis_data = {
                    "sentiment": "neutral",
                    "tone": ["informational"],
                    "key_topics": [],
                    "entities": {
                        "people": [],
                        "organizations": [],
                        "locations": [],
                        "dates": [],
                        "other": []
                    },
                    "document_type": "unknown"
                }
                self.logger.warning(f"Failed to parse JSON analysis for document {document_id}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'analysis': analysis_data,
                'tokens_used': tokens_used,
                'processing_time': processing_time,
                'llm_provider': provider,
                'llm_model': model_name
            }
            
            # Track cost
            self.cost_logger.track_llm_usage(
                provider=provider,
                model=model_name,
                tokens_used=tokens_used,
                cost=cost,
                document_id=document_id
            )
            
            self.logger.info(f"Completed analysis for document: {document_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed for document {document_id}: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def extract_structured_fields(self, text: str, document_id: str) -> Dict[str, Any]:
        """Extract structured fields and classify document type"""
        try:
            start_time = datetime.now()
            model = self.get_preferred_model()
            
            if not model:
                raise ValueError("No LLM model available. Please check API key configuration.")
            
            system_prompt = """You are a professional document processor. Extract structured information from the document and classify its type.

For document classification, choose from: invoice, contract, report, form, email, letter, memo, proposal, resume, academic_paper, legal_document, financial_statement, medical_record, other.

For field extraction, look for:
- Dates (issue date, due date, effective date, etc.)
- Amounts (total, subtotal, tax, etc.)
- Names (sender, recipient, parties involved)
- Addresses
- Contact information (phone, email)
- Document-specific fields (invoice number, contract terms, report findings, etc.)

Return your analysis in JSON format with the following structure:
{
    "document_type": "classified_type",
    "document_type_confidence": 0.95,
    "extracted_fields": {
        "field_name": {
            "value": "extracted_value",
            "confidence": 0.9,
            "source": "text_snippet"
        }
    },
    "tables": [],
    "validation_notes": []
}"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Please extract structured information from this document:\n\n{text}")
            ]
            
            with get_openai_callback() as cb:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: model.invoke(messages)
                )
                
                extraction_content = response.content
                tokens_used = cb.total_tokens
                cost = cb.total_cost
                
                provider = "openai" if isinstance(model, ChatOpenAI) else "anthropic"
                model_name = settings.llm.openai_model if provider == "openai" else settings.llm.anthropic_model
            
            # Parse JSON response
            try:
                extraction_data = json.loads(extraction_content)
            except json.JSONDecodeError:
                extraction_data = {
                    "document_type": "unknown",
                    "document_type_confidence": 0.0,
                    "extracted_fields": {},
                    "tables": [],
                    "validation_notes": ["Failed to parse extraction results"]
                }
                self.logger.warning(f"Failed to parse JSON extraction for document {document_id}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'extraction': extraction_data,
                'tokens_used': tokens_used,
                'processing_time': processing_time,
                'llm_provider': provider,
                'llm_model': model_name
            }
            
            # Track cost
            self.cost_logger.track_llm_usage(
                provider=provider,
                model=model_name,
                tokens_used=tokens_used,
                cost=cost,
                document_id=document_id
            )
            
            self.logger.info(f"Completed field extraction for document: {document_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Field extraction failed for document {document_id}: {str(e)}")
            raise
    
    async def generate_comprehensive_analysis(self, text: str, document_id: str) -> Dict[str, Any]:
        """Generate comprehensive analysis including summary, sentiment, and field extraction"""
        try:
            self.logger.agent_start(document_id=document_id, extra={"analysis_type": "comprehensive"})
            
            # Execute all analysis tasks concurrently
            tasks = [
                self.generate_summary(text, document_id, "brief"),
                self.generate_summary(text, document_id, "detailed"),
                self.analyze_sentiment_and_topics(text, document_id),
                self.extract_structured_fields(text, document_id)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Analysis task {i} failed: {str(result)}")
                    raise result
            
            brief_summary, detailed_summary, sentiment_analysis, field_extraction = results
            
            # Combine results
            comprehensive_result = {
                'brief_summary': brief_summary['summary'],
                'detailed_summary': detailed_summary['summary'],
                'sentiment_analysis': sentiment_analysis['analysis'],
                'field_extraction': field_extraction['extraction'],
                'processing_metrics': {
                    'total_tokens': (
                        brief_summary['tokens_used'] + 
                        detailed_summary['tokens_used'] + 
                        sentiment_analysis['tokens_used'] + 
                        field_extraction['tokens_used']
                    ),
                    'total_processing_time': (
                        brief_summary['processing_time'] + 
                        detailed_summary['processing_time'] + 
                        sentiment_analysis['processing_time'] + 
                        field_extraction['processing_time']
                    )
                },
                'llm_models_used': {
                    'brief_summary': brief_summary['llm_model'],
                    'detailed_summary': detailed_summary['llm_model'],
                    'sentiment_analysis': sentiment_analysis['llm_model'],
                    'field_extraction': field_extraction['llm_model']
                }
            }
            
            self.logger.agent_complete(
                document_id=document_id,
                result={
                    'analysis_completed': True,
                    'total_tokens': comprehensive_result['processing_metrics']['total_tokens'],
                    'total_processing_time': comprehensive_result['processing_metrics']['total_processing_time']
                }
            )
            
            return comprehensive_result
            
        except Exception as e:
            self.logger.agent_error(
                document_id=document_id,
                error=e,
                extra={'analysis_type': 'comprehensive'}
            )
            raise
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available LLM models"""
        available_models = {}
        
        if 'openai' in self.models:
            available_models['openai'] = {
                'model': settings.llm.openai_model,
                'temperature': settings.llm.analysis_temperature,
                'status': 'available'
            }
        
        if 'anthropic' in self.models:
            available_models['anthropic'] = {
                'model': settings.llm.anthropic_model,
                'temperature': settings.llm.analysis_temperature,
                'status': 'available'
            }
        
        return {
            'available_models': available_models,
            'preferred_model': 'anthropic' if 'anthropic' in self.models else 'openai' if 'openai' in self.models else 'none'
        }


# Global LLM service instance
llm_service = LLMService()

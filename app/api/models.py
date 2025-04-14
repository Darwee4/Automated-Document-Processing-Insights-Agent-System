from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status enum"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    
    filename: str = Field(..., description="Original filename")
    language: str = Field(default="eng", description="Language for OCR processing")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Filename cannot be empty")
        return v
    
    @validator('language')
    def validate_language(cls, v):
        # Basic language validation - can be extended with supported languages
        if not v or len(v.strip()) == 0:
            return "eng"
        return v.lower()


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    
    document_id: str = Field(..., description="Unique document identifier")
    status: DocumentStatus = Field(..., description="Initial processing status")
    message: str = Field(..., description="Status message")
    uploaded_at: datetime = Field(..., description="Upload timestamp")


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status"""
    
    document_id: str = Field(..., description="Unique document identifier")
    status: DocumentStatus = Field(..., description="Current processing status")
    current_agent: Optional[str] = Field(None, description="Current processing agent")
    progress_percentage: float = Field(..., description="Progress percentage (0-100)")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    needs_human_review: bool = Field(False, description="Whether human review is needed")
    human_review_reason: Optional[str] = Field(None, description="Reason for human review")


class OCRResultResponse(BaseModel):
    """Response model for OCR results"""
    
    extracted_text: str = Field(..., description="Extracted text from document")
    confidence_score: float = Field(..., description="Overall OCR confidence score")
    total_pages: int = Field(..., description="Total number of pages processed")
    processing_time: float = Field(..., description="OCR processing time in seconds")
    ocr_engine: str = Field(..., description="OCR engine used")
    language: str = Field(..., description="Language used for OCR")
    
    class Config:
        schema_extra = {
            "example": {
                "extracted_text": "This is sample extracted text from the document...",
                "confidence_score": 0.85,
                "total_pages": 1,
                "processing_time": 2.5,
                "ocr_engine": "tesseract",
                "language": "eng"
            }
        }


class AnalysisResultResponse(BaseModel):
    """Response model for analysis results"""
    
    brief_summary: str = Field(..., description="Brief 2-3 sentence summary")
    detailed_summary: str = Field(..., description="Detailed comprehensive summary")
    key_topics: List[str] = Field(..., description="Key topics identified")
    entities: Dict[str, List[str]] = Field(..., description="Named entities by category")
    sentiment: str = Field(..., description="Overall sentiment")
    tone: List[str] = Field(..., description="Tone analysis")
    processing_time: float = Field(..., description="Analysis processing time in seconds")
    tokens_used: int = Field(..., description="LLM tokens used")
    llm_model: str = Field(..., description="LLM model used")
    
    class Config:
        schema_extra = {
            "example": {
                "brief_summary": "This document discusses quarterly financial results...",
                "detailed_summary": "The quarterly financial report shows strong growth...",
                "key_topics": ["financial results", "revenue growth", "market analysis"],
                "entities": {
                    "people": ["John Doe", "Jane Smith"],
                    "organizations": ["Company Inc", "Market Research Corp"],
                    "locations": ["New York", "United States"]
                },
                "sentiment": "positive",
                "tone": ["formal", "informative"],
                "processing_time": 15.2,
                "tokens_used": 2450,
                "llm_model": "gpt-4"
            }
        }


class ExtractedField(BaseModel):
    """Model for individual extracted field"""
    
    value: str = Field(..., description="Extracted field value")
    confidence: float = Field(..., description="Extraction confidence (0-1)")
    source: Optional[str] = Field(None, description="Source text snippet")
    
    class Config:
        schema_extra = {
            "example": {
                "value": "2024-01-15",
                "confidence": 0.95,
                "source": "Invoice Date: 2024-01-15"
            }
        }


class ExtractionResultResponse(BaseModel):
    """Response model for field extraction results"""
    
    document_type: str = Field(..., description="Detected document type")
    document_type_confidence: float = Field(..., description="Document type confidence")
    extracted_fields: Dict[str, ExtractedField] = Field(..., description="Extracted key-value pairs")
    tables: List[Dict[str, Any]] = Field(default=[], description="Extracted table data")
    validation_notes: List[str] = Field(default=[], description="Validation and quality notes")
    processing_time: float = Field(..., description="Extraction processing time in seconds")
    fields_with_low_confidence: List[str] = Field(default=[], description="Fields with low confidence")
    
    class Config:
        schema_extra = {
            "example": {
                "document_type": "invoice",
                "document_type_confidence": 0.92,
                "extracted_fields": {
                    "invoice_number": {
                        "value": "INV-2024-001",
                        "confidence": 0.98,
                        "source": "Invoice #: INV-2024-001"
                    },
                    "invoice_date": {
                        "value": "2024-01-15",
                        "confidence": 0.95,
                        "source": "Date: 2024-01-15"
                    }
                },
                "tables": [],
                "validation_notes": ["All required fields extracted with high confidence"],
                "processing_time": 8.7,
                "fields_with_low_confidence": []
            }
        }


class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment"""
    
    ocr_quality: Dict[str, Any] = Field(..., description="OCR quality metrics")
    analysis_quality: Dict[str, Any] = Field(..., description="Analysis quality metrics")
    extraction_quality: Dict[str, Any] = Field(..., description="Extraction quality metrics")
    overall_quality_rating: str = Field(..., description="Overall quality rating")
    needs_human_review: bool = Field(..., description="Whether human review is recommended")
    quality_issues: List[str] = Field(default=[], description="List of quality issues")
    
    class Config:
        schema_extra = {
            "example": {
                "ocr_quality": {
                    "confidence_score": 0.85,
                    "text_length": 1250,
                    "word_count": 250,
                    "quality_rating": "good"
                },
                "analysis_quality": {
                    "brief_summary_length": 120,
                    "detailed_summary_length": 450,
                    "sentiment_identified": True,
                    "quality_rating": "excellent"
                },
                "extraction_quality": {
                    "document_type": "invoice",
                    "document_type_confidence": 0.92,
                    "fields_extracted": 8,
                    "quality_rating": "good"
                },
                "overall_quality_rating": "good",
                "needs_human_review": False,
                "quality_issues": []
            }
        }


class ProcessingMetricsResponse(BaseModel):
    """Response model for processing metrics"""
    
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    ocr_processing_time: float = Field(..., description="OCR processing time")
    analysis_processing_time: float = Field(..., description="Analysis processing time")
    extraction_processing_time: float = Field(..., description="Extraction processing time")
    total_tokens_used: int = Field(..., description="Total LLM tokens used")
    cost_estimate: float = Field(..., description="Estimated processing cost")
    success_rate: float = Field(..., description="Processing success rate")
    
    class Config:
        schema_extra = {
            "example": {
                "total_processing_time": 26.4,
                "ocr_processing_time": 3.2,
                "analysis_processing_time": 15.2,
                "extraction_processing_time": 8.0,
                "total_tokens_used": 3200,
                "cost_estimate": 0.064,
                "success_rate": 1.0
            }
        }


class DocumentResultsResponse(BaseModel):
    """Complete document processing results"""
    
    document_id: str = Field(..., description="Unique document identifier")
    status: DocumentStatus = Field(..., description="Processing status")
    filename: str = Field(..., description="Original filename")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    processing_completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    
    # Processing results
    ocr_results: Optional[OCRResultResponse] = Field(None, description="OCR extraction results")
    analysis_results: Optional[AnalysisResultResponse] = Field(None, description="Analysis results")
    extraction_results: Optional[ExtractionResultResponse] = Field(None, description="Field extraction results")
    
    # Quality and metrics
    quality_assessment: Optional[QualityAssessmentResponse] = Field(None, description="Quality assessment")
    processing_metrics: Optional[ProcessingMetricsResponse] = Field(None, description="Processing metrics")
    
    # Additional metadata
    needs_human_review: bool = Field(False, description="Whether human review is needed")
    human_review_reason: Optional[str] = Field(None, description="Reason for human review")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class DocumentListResponse(BaseModel):
    """Response model for document listing"""
    
    documents: List[DocumentResultsResponse] = Field(..., description="List of processed documents")
    total_count: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request identifier for debugging")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid file format",
                "detail": {"allowed_formats": ["pdf", "png", "jpg", "jpeg", "tiff"]},
                "request_id": "req_123456"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current timestamp")
    database_status: str = Field(..., description="Database connection status")
    mongo_status: str = Field(..., description="MongoDB connection status")
    llm_status: str = Field(..., description="LLM service status")
    uptime: float = Field(..., description="Service uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "database_status": "connected",
                "mongo_status": "connected",
                "llm_status": "available",
                "uptime": 86400.5
            }
        }


class CostMetricsResponse(BaseModel):
    """Cost tracking metrics response"""
    
    total_cost: float = Field(..., description="Total processing cost")
    total_documents: int = Field(..., description="Total documents processed")
    average_cost_per_document: float = Field(..., description="Average cost per document")
    cost_by_service: Dict[str, float] = Field(..., description="Cost breakdown by service")
    tokens_used: int = Field(..., description="Total tokens used")
    cost_trend: List[Dict[str, Any]] = Field(..., description="Cost trend over time")
    
    class Config:
        schema_extra = {
            "example": {
                "total_cost": 12.50,
                "total_documents": 25,
                "average_cost_per_document": 0.50,
                "cost_by_service": {
                    "ocr": 0.0,
                    "llm": 12.50
                },
                "tokens_used": 62500,
                "cost_trend": [
                    {"date": "2024-01-01", "cost": 2.50, "documents": 5},
                    {"date": "2024-01-02", "cost": 5.00, "documents": 10}
                ]
            }
        }


class BatchProcessingRequest(BaseModel):
    """Request model for batch processing"""
    
    documents: List[DocumentUploadRequest] = Field(..., description="List of documents to process")
    parallel_processing: bool = Field(default=True, description="Whether to process in parallel")
    priority: str = Field(default="normal", description="Processing priority")
    
    @validator('priority')
    def validate_priority(cls, v):
        allowed_priorities = ["low", "normal", "high", "urgent"]
        if v not in allowed_priorities:
            raise ValueError(f"Priority must be one of: {', '.join(allowed_priorities)}")
        return v


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing"""
    
    batch_id: str = Field(..., description="Batch processing identifier")
    total_documents: int = Field(..., description="Total documents in batch")
    accepted_documents: int = Field(..., description="Number of accepted documents")
    rejected_documents: int = Field(..., description="Number of rejected documents")
    document_ids: List[str] = Field(..., description="List of accepted document IDs")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_123456",
                "total_documents": 10,
                "accepted_documents": 8,
                "rejected_documents": 2,
                "document_ids": ["doc1", "doc2", "doc3"],
                "estimated_completion_time": "2024-01-15T11:30:00Z"
            }
        }

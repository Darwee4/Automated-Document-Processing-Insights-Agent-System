from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, JSON, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from datetime import datetime
import uuid
from typing import Optional, Dict, Any, List

from app.core.config import settings
from app.core.logging import get_request_logger


# Database base class
Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class Document(Base):
    """Main document table for storing document metadata"""
    
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    file_type = Column(String(50), nullable=False)  # pdf, png, jpg, etc.
    mime_type = Column(String(100), nullable=False)
    
    # Processing status
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    
    # User and session information
    user_id = Column(String(100), nullable=True)  # Optional user identification
    session_id = Column(String(100), nullable=True)  # Optional session tracking
    
    # Relationships
    ocr_results = relationship("OCRResult", back_populates="document", cascade="all, delete-orphan")
    summaries = relationship("DocumentSummary", back_populates="document", cascade="all, delete-orphan")
    structured_data = relationship("StructuredData", back_populates="document", cascade="all, delete-orphan")
    processing_metrics = relationship("ProcessingMetrics", back_populates="document", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary"""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "file_type": self.file_type,
            "mime_type": self.mime_type,
            "status": self.status,
            "error_message": self.error_message,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "processing_started_at": self.processing_started_at.isoformat() if self.processing_started_at else None,
            "processing_completed_at": self.processing_completed_at.isoformat() if self.processing_completed_at else None,
            "user_id": self.user_id,
            "session_id": self.session_id
        }


class OCRResult(Base):
    """OCR extraction results"""
    
    __tablename__ = "ocr_results"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    
    # OCR metadata
    ocr_engine = Column(String(50), nullable=False)  # tesseract, aws_textract, google_vision
    confidence_score = Column(Float, nullable=True)
    total_pages = Column(Integer, nullable=False)
    language = Column(String(10), default="eng")
    
    # Extracted content
    extracted_text = Column(Text, nullable=False)
    page_metadata = Column(JSON, nullable=True)  # Page-wise confidence and text
    
    # Processing metadata
    processing_time = Column(Float, nullable=True)  # Time in seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="ocr_results")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert OCR result to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "ocr_engine": self.ocr_engine,
            "confidence_score": self.confidence_score,
            "total_pages": self.total_pages,
            "language": self.language,
            "extracted_text": self.extracted_text,
            "page_metadata": self.page_metadata,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class DocumentSummary(Base):
    """Document analysis and summarization results"""
    
    __tablename__ = "document_summaries"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    
    # Summary types
    brief_summary = Column(Text, nullable=True)  # 2-3 sentences
    detailed_summary = Column(Text, nullable=True)  # More comprehensive
    
    # Analysis results
    key_topics = Column(JSON, nullable=True)  # List of key topics
    entities = Column(JSON, nullable=True)  # Named entities found
    sentiment = Column(String(20), nullable=True)  # positive, negative, neutral
    tone = Column(JSON, nullable=True)  # Tone analysis results
    
    # LLM metadata
    llm_model = Column(String(100), nullable=True)
    llm_provider = Column(String(50), nullable=True)  # openai, anthropic, etc.
    tokens_used = Column(Integer, nullable=True)
    
    # Processing metadata
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="summaries")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "brief_summary": self.brief_summary,
            "detailed_summary": self.detailed_summary,
            "key_topics": self.key_topics,
            "entities": self.entities,
            "sentiment": self.sentiment,
            "tone": self.tone,
            "llm_model": self.llm_model,
            "llm_provider": self.llm_provider,
            "tokens_used": self.tokens_used,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class StructuredData(Base):
    """Structured field extraction results"""
    
    __tablename__ = "structured_data"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    
    # Document classification
    document_type = Column(String(100), nullable=True)  # invoice, contract, report, etc.
    document_type_confidence = Column(Float, nullable=True)
    
    # Extracted fields
    extracted_fields = Column(JSON, nullable=True)  # Key-value pairs
    tables = Column(JSON, nullable=True)  # Table data
    validation_results = Column(JSON, nullable=True)  # Field validation results
    
    # Extraction metadata
    extraction_model = Column(String(100), nullable=True)
    confidence_threshold = Column(Float, nullable=True)
    fields_with_low_confidence = Column(JSON, nullable=True)
    
    # Processing metadata
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="structured_data")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert structured data to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "document_type": self.document_type,
            "document_type_confidence": self.document_type_confidence,
            "extracted_fields": self.extracted_fields,
            "tables": self.tables,
            "validation_results": self.validation_results,
            "extraction_model": self.extraction_model,
            "confidence_threshold": self.confidence_threshold,
            "fields_with_low_confidence": self.fields_with_low_confidence,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class ProcessingMetrics(Base):
    """Processing metrics and cost tracking"""
    
    __tablename__ = "processing_metrics"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    
    # Cost tracking
    total_cost = Column(Float, default=0.0)
    ocr_cost = Column(Float, default=0.0)
    llm_cost = Column(Float, default=0.0)
    
    # Token usage
    total_tokens = Column(Integer, default=0)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    
    # Processing times
    total_processing_time = Column(Float, default=0.0)
    ocr_processing_time = Column(Float, default=0.0)
    analysis_processing_time = Column(Float, default=0.0)
    extraction_processing_time = Column(Float, default=0.0)
    
    # Performance metrics
    success_rate = Column(Float, default=1.0)  # 0.0 to 1.0
    retry_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="processing_metrics")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "total_cost": self.total_cost,
            "ocr_cost": self.ocr_cost,
            "llm_cost": self.llm_cost,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_processing_time": self.total_processing_time,
            "ocr_processing_time": self.ocr_processing_time,
            "analysis_processing_time": self.analysis_processing_time,
            "extraction_processing_time": self.extraction_processing_time,
            "success_rate": self.success_rate,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DatabaseManager:
    """Database manager for PostgreSQL operations"""
    
    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        self.logger = get_request_logger()
    
    def init_db(self) -> None:
        """Initialize the database connection"""
        try:
            # Create async engine for FastAPI
            self.async_engine = create_async_engine(
                settings.database.postgres_url,
                echo=settings.server.debug,
                pool_pre_ping=True,
                pool_recycle=300
            )
            
            # Create sync engine for migrations and admin tasks
            sync_url = settings.database.postgres_url.replace("+asyncpg", "")
            self.engine = create_engine(
                sync_url,
                echo=settings.server.debug,
                pool_pre_ping=True,
                pool_recycle=300
            )
            
            # Create session factories
            self.async_session_factory = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self.session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False
            )
            
            self.logger.info("Database connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    async def get_async_session(self) -> AsyncSession:
        """Get an async database session"""
        if not self.async_session_factory:
            self.init_db()
        
        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Database session error: {str(e)}")
                raise
            finally:
                await session.close()
    
    def create_tables(self) -> None:
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create database tables: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.async_engine.connect() as conn:
                result = await conn.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            self.logger.error(f"Database health check failed: {str(e)}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_session():
    """Dependency for FastAPI to get database session"""
    async for session in db_manager.get_async_session():
        yield session

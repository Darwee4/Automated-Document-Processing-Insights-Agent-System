from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
from datetime import datetime
import asyncio

from app.api.models import (
    DocumentUploadResponse, ProcessingStatusResponse, DocumentResultsResponse,
    DocumentListResponse, ErrorResponse, HealthCheckResponse, CostMetricsResponse,
    BatchProcessingRequest, BatchProcessingResponse, DocumentUploadRequest
)
from app.workflows.document_workflow import document_workflow
from app.database.postgres import Document, db_manager, get_db_session
from app.database.mongo import mongo_manager
from app.core.logging import get_request_logger, setup_logging
from app.core.config import settings
from app.services.ocr_service import ocr_service
from app.services.llm_service import llm_service


# Setup logging
setup_logging()
logger = get_request_logger()

# Create API router
router = APIRouter(prefix="/api/v1", tags=["documents"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connections
        db_health = await db_manager.health_check()
        mongo_health = await mongo_manager.health_check() if mongo_manager.db else True
        
        # Check LLM service
        llm_models = llm_service.get_available_models()
        llm_health = "available" if llm_models['preferred_model'] != 'none' else "unavailable"
        
        # Calculate uptime (placeholder - would need to track startup time)
        uptime = 0.0  # In a real implementation, track from startup
        
        return HealthCheckResponse(
            status="healthy" if db_health and mongo_health else "degraded",
            version="1.0.0",
            timestamp=datetime.utcnow(),
            database_status="connected" if db_health else "disconnected",
            mongo_status="connected" if mongo_health else "disconnected",
            llm_status=llm_health,
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            version="1.0.0",
            timestamp=datetime.utcnow(),
            database_status="unknown",
            mongo_status="unknown",
            llm_status="unknown",
            uptime=0.0
        )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Query("eng", description="OCR language"),
    user_id: Optional[str] = Query(None, description="User identifier"),
    session_id: Optional[str] = Query(None, description="Session identifier")
):
    """Upload and process a document"""
    request_id = logger.generate_request_id()
    logger.set_request_id(request_id)
    
    try:
        logger.info(f"Processing document upload: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Validate file
        is_valid, validation_message = ocr_service.validate_file(file_content, file.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_message)
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Create document record in database
        async for session in db_manager.get_async_session():
            document = Document(
                id=document_id,
                filename=f"{document_id}_{file.filename}",
                original_filename=file.filename,
                file_size=len(file_content),
                file_type=file.filename.split('.')[-1].lower(),
                mime_type=file.content_type or "application/octet-stream",
                status="pending",
                user_id=user_id,
                session_id=session_id
            )
            
            session.add(document)
            await session.commit()
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            document_id,
            file_content,
            file.filename,
            language,
            user_id,
            session_id,
            request_id
        )
        
        logger.info(f"Document upload accepted: {document_id}")
        
        return DocumentUploadResponse(
            document_id=document_id,
            status="pending",
            message="Document accepted for processing",
            uploaded_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document: {str(e)}"
        )


async def process_document_background(
    document_id: str,
    file_content: bytes,
    filename: str,
    language: str,
    user_id: Optional[str],
    session_id: Optional[str],
    request_id: str
):
    """Background task to process document"""
    logger.set_request_id(request_id)
    
    try:
        # Update document status to processing
        async for session in db_manager.get_async_session():
            document = await session.get(Document, document_id)
            if document:
                document.status = "processing"
                document.processing_started_at = datetime.utcnow()
                await session.commit()
        
        logger.info(f"Starting background processing for document: {document_id}")
        
        # Process document through workflow
        result = await document_workflow.process_document(
            file_content=file_content,
            filename=filename,
            user_id=user_id,
            session_id=session_id,
            language=language
        )
        
        # Update document status based on workflow result
        async for session in db_manager.get_async_session():
            document = await session.get(Document, document_id)
            if document:
                document.status = result['status']
                if result['status'] == 'completed':
                    document.processing_completed_at = datetime.utcnow()
                elif result['status'] == 'failed':
                    document.error_message = result.get('error_message', 'Unknown error')
                await session.commit()
        
        logger.info(f"Background processing completed for document: {document_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for document {document_id}: {str(e)}")
        
        # Update document status to failed
        async for session in db_manager.get_async_session():
            document = await session.get(Document, document_id)
            if document:
                document.status = "failed"
                document.error_message = str(e)
                await session.commit()


@router.get("/documents/{document_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(document_id: str):
    """Get processing status for a document"""
    request_id = logger.generate_request_id()
    logger.set_request_id(request_id)
    
    try:
        async for session in db_manager.get_async_session():
            document = await session.get(Document, document_id)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Calculate progress percentage (simplified)
            progress = 0.0
            if document.status == "completed":
                progress = 100.0
            elif document.status == "processing":
                progress = 50.0  # Simplified - in real implementation, track actual progress
            elif document.status == "pending":
                progress = 0.0
            
            return ProcessingStatusResponse(
                document_id=document_id,
                status=document.status,
                current_agent=None,  # Would need to track this in workflow state
                progress_percentage=progress,
                estimated_completion_time=None,  # Could estimate based on processing time
                error_message=document.error_message,
                needs_human_review=False,  # Would need to check workflow state
                human_review_reason=None
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get processing status for {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get processing status")


@router.get("/documents/{document_id}/results", response_model=DocumentResultsResponse)
async def get_processing_results(document_id: str):
    """Get complete processing results for a document"""
    request_id = logger.generate_request_id()
    logger.set_request_id(request_id)
    
    try:
        async for session in db_manager.get_async_session():
            document = await session.get(Document, document_id)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            if document.status != "completed":
                raise HTTPException(
                    status_code=400, 
                    detail=f"Document processing not complete. Current status: {document.status}"
                )
            
            # In a real implementation, you would fetch the actual results from database
            # For now, we'll return a placeholder response
            return DocumentResultsResponse(
                document_id=document_id,
                status=document.status,
                filename=document.original_filename,
                uploaded_at=document.uploaded_at,
                processing_completed_at=document.processing_completed_at,
                ocr_results=None,  # Would fetch from OCRResult table
                analysis_results=None,  # Would fetch from DocumentSummary table
                extraction_results=None,  # Would fetch from StructuredData table
                quality_assessment=None,
                processing_metrics=None,
                needs_human_review=False,
                human_review_reason=None,
                error_message=document.error_message
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get results for {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get processing results")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    status: Optional[str] = Query(None, description="Filter by status"),
    user_id: Optional[str] = Query(None, description="Filter by user ID")
):
    """List processed documents with pagination"""
    request_id = logger.generate_request_id()
    logger.set_request_id(request_id)
    
    try:
        async for session in db_manager.get_async_session():
            # Build query
            query = session.query(Document)
            
            if status:
                query = query.filter(Document.status == status)
            if user_id:
                query = query.filter(Document.user_id == user_id)
            
            # Get total count
            total_count = await session.scalar(
                query.with_entities(func.count(Document.id))
            )
            
            # Apply pagination
            offset = (page - 1) * page_size
            documents = await session.execute(
                query.order_by(Document.uploaded_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            documents = documents.scalars().all()
            
            # Convert to response models
            document_responses = []
            for doc in documents:
                document_responses.append(
                    DocumentResultsResponse(
                        document_id=doc.id,
                        status=doc.status,
                        filename=doc.original_filename,
                        uploaded_at=doc.uploaded_at,
                        processing_completed_at=doc.processing_completed_at,
                        error_message=doc.error_message,
                        needs_human_review=False
                    )
                )
            
            return DocumentListResponse(
                documents=document_responses,
                total_count=total_count,
                page=page,
                page_size=page_size,
                has_next=(page * page_size) < total_count
            )
            
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all associated data"""
    request_id = logger.generate_request_id()
    logger.set_request_id(request_id)
    
    try:
        async for session in db_manager.get_async_session():
            # Get document
            document = await session.get(Document, document_id)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Delete document and related data (cascade should handle related records)
            await session.delete(document)
            await session.commit()
            
            # Clean up MongoDB data
            if mongo_manager.db:
                await mongo_manager.cleanup_document_data(document_id)
            
            logger.info(f"Deleted document: {document_id}")
            
            return JSONResponse(
                status_code=200,
                content={"message": f"Document {document_id} deleted successfully"}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.get("/metrics/costs", response_model=CostMetricsResponse)
async def get_cost_metrics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get cost tracking metrics"""
    request_id = logger.generate_request_id()
    logger.set_request_id(request_id)
    
    try:
        # In a real implementation, you would aggregate cost data from database
        # For now, return placeholder data
        
        # Calculate costs from database (simplified)
        async for session in db_manager.get_async_session():
            total_documents = await session.scalar(
                session.query(func.count(Document.id))
            )
        
        # Placeholder cost calculation
        total_cost = 0.0
        average_cost = 0.0
        if total_documents > 0:
            # Estimate cost based on average tokens per document
            average_cost = 0.05  # $0.05 per document estimate
            total_cost = total_documents * average_cost
        
        return CostMetricsResponse(
            total_cost=total_cost,
            total_documents=total_documents,
            average_cost_per_document=average_cost,
            cost_by_service={
                "ocr": 0.0,
                "llm": total_cost
            },
            tokens_used=total_documents * 1000,  # Estimate
            cost_trend=[]  # Would need time-series data
        )
        
    except Exception as e:
        logger.error(f"Failed to get cost metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get cost metrics")


@router.post("/documents/batch", response_model=BatchProcessingResponse)
async def batch_process_documents(
    background_tasks: BackgroundTasks,
    batch_request: BatchProcessingRequest
):
    """Process multiple documents in batch"""
    request_id = logger.generate_request_id()
    logger.set_request_id(request_id)
    
    try:
        # This is a simplified implementation
        # In a real system, you would use a proper job queue
        
        accepted_documents = []
        rejected_documents = []
        
        # For now, we'll accept all documents (in reality, validate each)
        for doc_request in batch_request.documents:
            # Generate document ID for each
            document_id = str(uuid.uuid4())
            accepted_documents.append(document_id)
            
            # In a real implementation, you would:
            # 1. Validate each document request
            # 2. Store batch metadata
            # 3. Queue each document for processing
        
        batch_id = f"batch_{str(uuid.uuid4())[:8]}"
        
        logger.info(f"Batch processing started: {batch_id} with {len(accepted_documents)} documents")
        
        return BatchProcessingResponse(
            batch_id=batch_id,
            total_documents=len(batch_request.documents),
            accepted_documents=len(accepted_documents),
            rejected_documents=len(rejected_documents),
            document_ids=accepted_documents,
            estimated_completion_time=None  # Could estimate based on document count
        )
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process batch")


@router.get("/system/info")
async def get_system_info():
    """Get system information and configuration"""
    request_id = logger.generate_request_id()
    logger.set_request_id(request_id)
    
    try:
        # Get OCR engine info
        ocr_info = ocr_service.get_ocr_engine_info()
        
        # Get LLM models info
        llm_info = llm_service.get_available_models()
        
        # Get database stats
        db_stats = {}
        mongo_stats = {}
        
        if mongo_manager.db:
            mongo_stats = await mongo_manager.get_collection_stats()
        
        return {
            "system": {
                "version": "1.0.0",
                "environment": settings.server.app_env,
                "debug_mode": settings.server.debug
            },
            "ocr": ocr_info,
            "llm": llm_info,
            "databases": {
                "postgres": db_stats,
                "mongodb": mongo_stats
            },
            "limits": {
                "max_file_size_mb": settings.security.max_file_size_mb,
                "allowed_extensions": settings.security.allowed_extensions,
                "rate_limits": {
                    "per_minute": settings.security.rate_limit_per_minute,
                    "per_hour": settings.security.rate_limit_per_hour
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system information")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import time
from datetime import datetime

from app.api.routes import router as api_router
from app.database.postgres import db_manager
from app.database.mongo import mongo_manager
from app.core.logging import setup_logging, get_request_logger
from app.core.config import settings


# Setup logging
setup_logging()
logger = get_request_logger()

# Track startup time
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup and shutdown events"""
    # Startup
    logger.info("Starting Document Processing API...")
    
    try:
        # Initialize database connections
        db_manager.init_db()
        db_manager.create_tables()
        logger.info("PostgreSQL database initialized")
        
        # Initialize MongoDB
        await mongo_manager.init_db()
        logger.info("MongoDB initialized")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Processing API...")
    
    # Close database connections
    if db_manager.engine:
        db_manager.engine.dispose()
        logger.info("PostgreSQL connections closed")
    
    if mongo_manager.client:
        mongo_manager.client.close()
        logger.info("MongoDB connections closed")
    
    logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="Automated Document Processing & Insights Agent System",
    description="A multi-agent workflow system for automated document processing, analysis, and insights extraction using LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan
)


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.server.debug else [
        "http://localhost:3000",
        "https://yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.server.debug else [
        "localhost",
        "yourdomain.com",
        "*.yourdomain.com"
    ]
)


# Include API routes
app.include_router(api_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    uptime = time.time() - startup_time
    
    return {
        "message": "Automated Document Processing & Insights Agent System",
        "version": "1.0.0",
        "status": "operational",
        "uptime_seconds": uptime,
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs",
        "health_check": "/api/v1/health"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "message": exc.detail,
                "request_id": logger._request_id
            }
        )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An internal server error occurred",
            "request_id": logger._request_id
        }
    )


# Health check endpoint (additional to the one in routes)
@app.get("/health")
async def health():
    """Simple health check endpoint"""
    try:
        # Check database connectivity
        db_healthy = await db_manager.health_check()
        mongo_healthy = await mongo_manager.health_check() if mongo_manager.db else True
        
        status = "healthy" if db_healthy and mongo_healthy else "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected" if db_healthy else "disconnected",
            "mongodb": "connected" if mongo_healthy else "disconnected",
            "uptime_seconds": time.time() - startup_time
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "uptime_seconds": time.time() - startup_time
        }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.server.host}:{settings.server.port}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug,
        workers=settings.server.workers if not settings.server.debug else 1,
        log_level=settings.server.log_level.lower()
    )

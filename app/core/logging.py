import logging
import sys
from loguru import logger
from typing import Dict, Any
import json
from datetime import datetime
import uuid

from .config import settings


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward Loguru"""

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """Setup logging configuration"""
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=settings.server.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True,
    )
    
    # Add file handler for production
    if settings.server.is_production:
        logger.add(
            "logs/app_{time}.log",
            level="INFO",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )
        
        # Error logs separately
        logger.add(
            "logs/error_{time}.log",
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set loguru as handler for uvicorn and fastapi
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False


class RequestLogger:
    """Request-specific logging with request IDs"""
    
    def __init__(self):
        self._request_id = None
    
    def set_request_id(self, request_id: str) -> None:
        """Set the current request ID for logging context"""
        self._request_id = request_id
    
    def generate_request_id(self) -> str:
        """Generate a new request ID"""
        self._request_id = str(uuid.uuid4())
        return self._request_id
    
    def _format_message(self, message: str, extra: Dict[str, Any] = None) -> str:
        """Format log message with request context"""
        base_msg = f"[RequestID: {self._request_id}] {message}"
        if extra:
            base_msg += f" | Extra: {json.dumps(extra)}"
        return base_msg
    
    def debug(self, message: str, extra: Dict[str, Any] = None) -> None:
        """Log debug message"""
        logger.debug(self._format_message(message, extra))
    
    def info(self, message: str, extra: Dict[str, Any] = None) -> None:
        """Log info message"""
        logger.info(self._format_message(message, extra))
    
    def warning(self, message: str, extra: Dict[str, Any] = None) -> None:
        """Log warning message"""
        logger.warning(self._format_message(message, extra))
    
    def error(self, message: str, extra: Dict[str, Any] = None, exc_info: bool = True) -> None:
        """Log error message"""
        logger.error(self._format_message(message, extra), exc_info=exc_info)
    
    def critical(self, message: str, extra: Dict[str, Any] = None) -> None:
        """Log critical message"""
        logger.critical(self._format_message(message, extra))


class AgentLogger:
    """Specialized logger for agent activities"""
    
    def __init__(self, agent_name: str, request_logger: RequestLogger):
        self.agent_name = agent_name
        self.request_logger = request_logger
    
    def agent_start(self, document_id: str, extra: Dict[str, Any] = None) -> None:
        """Log agent start"""
        message = f"Agent '{self.agent_name}' started processing document: {document_id}"
        self.request_logger.info(message, extra)
    
    def agent_complete(self, document_id: str, result: Dict[str, Any] = None) -> None:
        """Log agent completion"""
        message = f"Agent '{self.agent_name}' completed processing document: {document_id}"
        extra = {"result_summary": result} if result else None
        self.request_logger.info(message, extra)
    
    def agent_error(self, document_id: str, error: Exception, extra: Dict[str, Any] = None) -> None:
        """Log agent error"""
        message = f"Agent '{self.agent_name}' failed processing document: {document_id}"
        error_extra = {"error_type": type(error).__name__, "error_message": str(error)}
        if extra:
            error_extra.update(extra)
        self.request_logger.error(message, error_extra)
    
    def agent_transition(self, from_state: str, to_state: str, document_id: str, reason: str = None) -> None:
        """Log state transition"""
        message = f"Agent '{self.agent_name}' transitioned from '{from_state}' to '{to_state}' for document: {document_id}"
        extra = {"reason": reason} if reason else None
        self.request_logger.info(message, extra)


class CostLogger:
    """Logger for tracking costs and metrics"""
    
    def __init__(self, request_logger: RequestLogger):
        self.request_logger = request_logger
    
    def track_llm_usage(self, provider: str, model: str, tokens_used: int, cost: float, document_id: str) -> None:
        """Track LLM usage and costs"""
        message = f"LLM usage tracked for document: {document_id}"
        extra = {
            "provider": provider,
            "model": model,
            "tokens_used": tokens_used,
            "cost": cost,
            "document_id": document_id
        }
        self.request_logger.info(message, extra)
    
    def track_ocr_usage(self, pages_processed: int, cost: float, document_id: str) -> None:
        """Track OCR usage and costs"""
        message = f"OCR usage tracked for document: {document_id}"
        extra = {
            "pages_processed": pages_processed,
            "cost": cost,
            "document_id": document_id
        }
        self.request_logger.info(message, extra)
    
    def track_processing_time(self, agent_name: str, processing_time: float, document_id: str) -> None:
        """Track processing time for agents"""
        message = f"Processing time tracked for agent: {agent_name}"
        extra = {
            "agent_name": agent_name,
            "processing_time_seconds": processing_time,
            "document_id": document_id
        }
        self.request_logger.info(message, extra)


# Global logger instances
request_logger = RequestLogger()
cost_logger = CostLogger(request_logger)


def get_agent_logger(agent_name: str) -> AgentLogger:
    """Get an agent-specific logger"""
    return AgentLogger(agent_name, request_logger)


def get_request_logger() -> RequestLogger:
    """Get the request logger"""
    return request_logger


def get_cost_logger() -> CostLogger:
    """Get the cost logger"""
    return cost_logger

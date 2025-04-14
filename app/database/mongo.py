from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

from app.core.config import settings
from app.core.logging import get_request_logger


class MongoDBManager:
    """MongoDB manager for unstructured data storage"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.logger = get_request_logger()
    
    async def init_db(self) -> None:
        """Initialize MongoDB connection"""
        try:
            self.client = AsyncIOMotorClient(settings.database.mongodb_url)
            self.db = self.client[settings.database.mongodb_db]
            
            # Create indexes for better performance
            await self._create_indexes()
            
            self.logger.info("MongoDB connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MongoDB: {str(e)}")
            raise
    
    async def _create_indexes(self) -> None:
        """Create necessary indexes for collections"""
        try:
            # OCR raw results indexes
            await self.db.ocr_raw_results.create_index([("document_id", ASCENDING)])
            await self.db.ocr_raw_results.create_index([("created_at", DESCENDING)])
            
            # Agent states indexes
            await self.db.agent_states.create_index([("document_id", ASCENDING)])
            await self.db.agent_states.create_index([("agent_name", ASCENDING)])
            await self.db.agent_states.create_index([("created_at", DESCENDING)])
            
            # Processing logs indexes
            await self.db.processing_logs.create_index([("document_id", ASCENDING)])
            await self.db.processing_logs.create_index([("level", ASCENDING)])
            await self.db.processing_logs.create_index([("timestamp", DESCENDING)])
            
            # Cost tracking indexes
            await self.db.cost_tracking.create_index([("document_id", ASCENDING)])
            await self.db.cost_tracking.create_index([("service", ASCENDING)])
            await self.db.cost_tracking.create_index([("timestamp", DESCENDING)])
            
            self.logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create MongoDB indexes: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check MongoDB health"""
        try:
            # The ismaster command is cheap and does not require auth
            await self.client.admin.command('ismaster')
            return True
        except Exception as e:
            self.logger.error(f"MongoDB health check failed: {str(e)}")
            return False
    
    async def store_ocr_raw_result(self, document_id: str, ocr_data: Dict[str, Any]) -> str:
        """Store raw OCR results with confidence scores and page-level data"""
        try:
            ocr_record = {
                "_id": str(uuid.uuid4()),
                "document_id": document_id,
                "ocr_engine": ocr_data.get("ocr_engine", "tesseract"),
                "raw_results": ocr_data.get("raw_results", {}),
                "page_data": ocr_data.get("page_data", []),
                "confidence_scores": ocr_data.get("confidence_scores", {}),
                "processing_metadata": ocr_data.get("processing_metadata", {}),
                "created_at": datetime.utcnow()
            }
            
            result = await self.db.ocr_raw_results.insert_one(ocr_record)
            self.logger.info(f"Stored raw OCR result for document: {document_id}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Failed to store OCR raw result: {str(e)}")
            raise
    
    async def get_ocr_raw_results(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get raw OCR results for a document"""
        try:
            result = await self.db.ocr_raw_results.find_one(
                {"document_id": document_id},
                sort=[("created_at", DESCENDING)]
            )
            
            if result:
                # Convert ObjectId to string for JSON serialization
                result["_id"] = str(result["_id"])
                return result
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get OCR raw results: {str(e)}")
            raise
    
    async def store_agent_state(self, document_id: str, agent_name: str, state_data: Dict[str, Any]) -> str:
        """Store intermediate agent state"""
        try:
            state_record = {
                "_id": str(uuid.uuid4()),
                "document_id": document_id,
                "agent_name": agent_name,
                "state": state_data,
                "created_at": datetime.utcnow()
            }
            
            result = await self.db.agent_states.insert_one(state_record)
            self.logger.info(f"Stored agent state for {agent_name} and document: {document_id}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Failed to store agent state: {str(e)}")
            raise
    
    async def get_agent_states(self, document_id: str, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get agent states for a document, optionally filtered by agent name"""
        try:
            query = {"document_id": document_id}
            if agent_name:
                query["agent_name"] = agent_name
            
            cursor = self.db.agent_states.find(query).sort("created_at", ASCENDING)
            results = await cursor.to_list(length=None)
            
            # Convert ObjectIds to strings
            for result in results:
                result["_id"] = str(result["_id"])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get agent states: {str(e)}")
            raise
    
    async def store_processing_log(self, document_id: str, level: str, message: str, extra_data: Dict[str, Any] = None) -> str:
        """Store detailed processing logs"""
        try:
            log_record = {
                "_id": str(uuid.uuid4()),
                "document_id": document_id,
                "level": level,
                "message": message,
                "extra_data": extra_data or {},
                "timestamp": datetime.utcnow()
            }
            
            result = await self.db.processing_logs.insert_one(log_record)
            
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Failed to store processing log: {str(e)}")
            raise
    
    async def get_processing_logs(self, document_id: str, level: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get processing logs for a document"""
        try:
            query = {"document_id": document_id}
            if level:
                query["level"] = level
            
            cursor = self.db.processing_logs.find(query).sort("timestamp", DESCENDING).limit(limit)
            results = await cursor.to_list(length=limit)
            
            # Convert ObjectIds to strings
            for result in results:
                result["_id"] = str(result["_id"])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get processing logs: {str(e)}")
            raise
    
    async def track_cost(self, document_id: str, service: str, cost_data: Dict[str, Any]) -> str:
        """Track costs for different services"""
        try:
            cost_record = {
                "_id": str(uuid.uuid4()),
                "document_id": document_id,
                "service": service,
                "cost_data": cost_data,
                "timestamp": datetime.utcnow()
            }
            
            result = await self.db.cost_tracking.insert_one(cost_record)
            self.logger.info(f"Tracked cost for {service} and document: {document_id}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Failed to track cost: {str(e)}")
            raise
    
    async def get_cost_tracking(self, document_id: str, service: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cost tracking data for a document"""
        try:
            query = {"document_id": document_id}
            if service:
                query["service"] = service
            
            cursor = self.db.cost_tracking.find(query).sort("timestamp", ASCENDING)
            results = await cursor.to_list(length=None)
            
            # Convert ObjectIds to strings
            for result in results:
                result["_id"] = str(result["_id"])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get cost tracking: {str(e)}")
            raise
    
    async def store_unstructured_extraction(self, document_id: str, extraction_data: Dict[str, Any]) -> str:
        """Store unstructured extraction results"""
        try:
            extraction_record = {
                "_id": str(uuid.uuid4()),
                "document_id": document_id,
                "extraction_data": extraction_data,
                "created_at": datetime.utcnow()
            }
            
            result = await self.db.unstructured_extractions.insert_one(extraction_record)
            self.logger.info(f"Stored unstructured extraction for document: {document_id}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Failed to store unstructured extraction: {str(e)}")
            raise
    
    async def get_unstructured_extractions(self, document_id: str) -> List[Dict[str, Any]]:
        """Get unstructured extraction results for a document"""
        try:
            cursor = self.db.unstructured_extractions.find(
                {"document_id": document_id}
            ).sort("created_at", DESCENDING)
            
            results = await cursor.to_list(length=None)
            
            # Convert ObjectIds to strings
            for result in results:
                result["_id"] = str(result["_id"])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get unstructured extractions: {str(e)}")
            raise
    
    async def cleanup_document_data(self, document_id: str) -> bool:
        """Clean up all MongoDB data for a document"""
        try:
            # Delete all related data for the document
            collections_to_clean = [
                "ocr_raw_results",
                "agent_states", 
                "processing_logs",
                "cost_tracking",
                "unstructured_extractions"
            ]
            
            for collection_name in collections_to_clean:
                await self.db[collection_name].delete_many({"document_id": document_id})
            
            self.logger.info(f"Cleaned up MongoDB data for document: {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup document data: {str(e)}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        try:
            stats = {}
            collections = await self.db.list_collection_names()
            
            for collection_name in collections:
                count = await self.db[collection_name].count_documents({})
                stats[collection_name] = {
                    "document_count": count,
                    "size_bytes": await self.db.command("collstats", collection_name).get("size", 0)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            raise


# Global MongoDB manager instance
mongo_manager = MongoDBManager()


async def get_mongo_db():
    """Dependency for FastAPI to get MongoDB instance"""
    if not mongo_manager.db:
        await mongo_manager.init_db()
    return mongo_manager.db

from pydantic_settings import BaseSettings
from typing import Optional, List
from pydantic import Field, validator
import os


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="document_processing", env="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="password", env="POSTGRES_PASSWORD")
    
    mongodb_url: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    mongodb_db: str = Field(default="document_processing", env="MONGODB_DB")
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


class LLMSettings(BaseSettings):
    """LLM API configuration settings"""
    
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Model configurations
    openai_model: str = Field(default="gpt-4")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229")
    
    # Temperature settings
    summary_temperature: float = Field(default=0.1)
    analysis_temperature: float = Field(default=0.3)
    extraction_temperature: float = Field(default=0.1)


class OCRSettings(BaseSettings):
    """OCR configuration settings"""
    
    tesseract_cmd: str = Field(default="/usr/bin/tesseract", env="TESSERACT_CMD")
    use_cloud_ocr: bool = Field(default=False, env="USE_CLOUD_OCR")
    
    # Cloud OCR settings
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    
    # OCR quality settings
    ocr_confidence_threshold: float = Field(default=0.7)
    preprocessing_enabled: bool = Field(default=True)


class SecuritySettings(BaseSettings):
    """Security and authentication settings"""
    
    secret_key: str = Field(default="your_secret_key_here", env="SECRET_KEY")
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    # File upload security
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    allowed_extensions: List[str] = Field(default=["pdf", "png", "jpg", "jpeg", "tiff"], env="ALLOWED_EXTENSIONS")
    
    @validator("allowed_extensions", pre=True)
    def parse_allowed_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return v


class ServerSettings(BaseSettings):
    """Server configuration settings"""
    
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # Environment
    app_env: str = Field(default="development", env="APP_ENV")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings"""
    
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    enable_metrics: bool = Field(default=True)
    enable_health_checks: bool = Field(default=True)
    
    # Cost tracking
    track_costs: bool = Field(default=True)
    cost_reporting_interval: int = Field(default=3600)  # 1 hour in seconds


class Settings(BaseSettings):
    """Main application settings"""
    
    database: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()
    ocr: OCRSettings = OCRSettings()
    security: SecuritySettings = SecuritySettings()
    server: ServerSettings = ServerSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings"""
    return settings

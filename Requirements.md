# Project Prompt: Multi-Agent Document Processing & Insights System

Create a production-ready multi-agent workflow system using LangGraph for automated document processing. This system should be enterprise-grade with proper error handling, observability, and scalability.

## Core Requirements

### 1. Multi-Agent Architecture (LangGraph)
Build a LangGraph-based workflow with the following specialized agents:

**Agent 1: OCR Extraction Agent**
- Accept PDF files and images (PNG, JPG, JPEG, TIFF)
- Extract text using OCR (use Tesseract or cloud OCR services like AWS Textract/Google Vision)
- Handle multi-page PDFs
- Return structured text with page numbers and confidence scores
- Include preprocessing for image quality enhancement

**Agent 2: Analysis & Summarization Agent**
- Take extracted text as input
- Generate intelligent summaries using LLM (Claude/GPT-4)
- Identify key topics, entities, and themes
- Create both brief (2-3 sentences) and detailed summaries
- Extract sentiment and tone analysis

**Agent 3: Structured Field Detection Agent**
- Detect and extract structured data fields (dates, amounts, names, addresses, etc.)
- Identify document type (invoice, contract, form, report, etc.)
- Extract key-value pairs
- Handle tables and structured sections
- Validate extracted fields with confidence thresholds

### 2. LangGraph Workflow Design
- Define clear state management between agents
- Implement conditional edges based on document type or quality
- Add human-in-the-loop checkpoints for low-confidence results
- Create a graph visualization of the workflow
- Support parallel processing where appropriate

### 3. Database Storage
Implement dual database support:

**PostgreSQL (Primary)**
- Store document metadata (filename, upload time, user, status)
- Store extracted text and summaries
- Store structured fields in JSONB columns
- Create proper indexes for search performance
- Include full-text search capabilities

**MongoDB (Optional/Alternative)**
- Store raw OCR results with confidence scores
- Store intermediate agent states
- Handle unstructured extraction results
- Support flexible schema for various document types

### 4. FastAPI Backend
Create RESTful endpoints:

```
POST /api/v1/documents/upload
- Accept file upload (PDF/image)
- Return job ID for async processing

GET /api/v1/documents/{job_id}/status
- Check processing status
- Return progress percentage

GET /api/v1/documents/{job_id}/results
- Retrieve complete processing results
- Include OCR text, summary, and structured fields

GET /api/v1/documents/
- List all processed documents
- Support pagination and filtering

DELETE /api/v1/documents/{job_id}
- Remove document and all associated data
```

Additional features:
- Request validation with Pydantic models
- API versioning
- Rate limiting
- Authentication/API keys
- CORS configuration
- OpenAPI documentation

### 5. Production Features

**Logging**
- Structured logging with loguru or Python logging
- Log levels: DEBUG, INFO, WARNING, ERROR
- Log agent transitions and decisions
- Include request IDs for tracing
- Store logs in files and/or centralized logging (e.g., ELK stack)

**Retry Logic**
- Exponential backoff for API calls
- Retry failed OCR operations
- Handle transient database errors
- Circuit breaker pattern for external services
- Maximum retry attempts configuration

**Guardrails**
- Input validation (file size limits, allowed formats)
- Content safety checks (PII detection, sensitive data masking)
- Output validation (minimum confidence thresholds)
- Timeout limits for long-running operations
- Resource usage limits (memory, CPU)

**Cost Tracking**
- Track LLM API token usage per request
- Monitor OCR service costs
- Calculate cost per document
- Store cost metrics in database
- Create cost reporting endpoint (`GET /api/v1/metrics/costs`)

**Error Handling**
- Graceful degradation when agents fail
- Detailed error messages with remediation steps
- Store error logs with stack traces
- Implement dead letter queue for failed jobs

### 6. Infrastructure & DevOps

**Docker Configuration**
- Multi-stage Dockerfile for optimized images
- Docker Compose for local development
- Separate services for API, workers, databases
- Environment variable management

**Configuration Management**
- Use Pydantic Settings for configuration
- Support .env files
- Separate configs for dev/staging/prod
- Secret management best practices

**Testing**
- Unit tests for individual agents
- Integration tests for workflow
- API endpoint tests
- Mock external services
- Achieve >80% code coverage

**Monitoring**
- Health check endpoints
- Prometheus metrics export
- Processing time tracking
- Success/failure rates
- Queue depth monitoring

### 7. Code Structure

```
project/
├── app/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── ocr_agent.py
│   │   ├── analysis_agent.py
│   │   └── extraction_agent.py
│   ├── workflows/
│   │   ├── __init__.py
│   │   └── document_workflow.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── models.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── postgres.py
│   │   └── mongo.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ocr_service.py
│   │   └── storage_service.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── metrics.py
│   └── main.py
├── tests/
├── docker/
├── scripts/
├── requirements.txt
├── docker-compose.yml
├── README.md
└── .env.example
```

### 8. Documentation Requirements

**README.md**
- Project overview and architecture diagram
- Prerequisites and dependencies
- Installation instructions (local and Docker)
- Configuration guide
- API usage examples with curl commands
- Workflow diagram

**API Documentation**
- Auto-generated with FastAPI's OpenAPI
- Example requests and responses
- Error codes and meanings

**Code Documentation**
- Docstrings for all functions and classes
- Type hints throughout
- Architecture decision records (ADRs)

### 9. Performance Requirements
- Process a 10-page PDF in under 60 seconds
- Support concurrent processing of multiple documents
- Handle files up to 50MB
- API response time <200ms (excluding async jobs)
- Database query optimization

### 10. Security Considerations
- Sanitize file uploads
- Validate file types (not just extensions)
- Implement file size limits
- Use parameterized database queries
- Store sensitive configs in secrets manager
- Add rate limiting and DDoS protection
- HTTPS only in production

## Technology Stack

**Required:**
- Python 3.10+
- LangGraph (latest version)
- LangChain
- FastAPI
- PostgreSQL
- SQLAlchemy or asyncpg
- Pydantic
- Tesseract OCR or cloud OCR API
- Anthropic/OpenAI API

**Optional:**
- MongoDB with Motor (async driver)
- Redis (for caching and job queue)
- Celery (for background tasks)
- Prometheus + Grafana (monitoring)

## Deliverables

1. Complete, working codebase with all features
2. Docker Compose setup for one-command deployment
3. Comprehensive README with setup instructions
4. API documentation
5. Test suite with >80% coverage
6. Example documents for testing
7. Environment configuration templates
8. Database migration scripts
9. Workflow visualization diagram

## Success Criteria

- All agents work correctly in the LangGraph workflow
- API endpoints respond correctly with proper error handling
- Documents are successfully processed and stored
- Logging captures all important events
- Cost tracking accurately measures expenses
- System handles failures gracefully with retries
- Docker setup works on first try
- Documentation is clear and complete

## Bonus Features (If Time Permits)

- Web UI for document upload and viewing results
- Webhook support for job completion notifications
- Batch processing endpoint
- Document comparison feature
- Export results to JSON/CSV/Excel
- Admin dashboard for system monitoring
- Multi-language OCR support
- Document classification training interface

---

**Note**: Focus on code quality, proper abstractions, and production-readiness. This should be a reference implementation that demonstrates best practices for multi-agent systems.
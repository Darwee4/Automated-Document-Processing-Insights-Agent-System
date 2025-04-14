# Automated Document Processing & Insights Agent System

A production-ready multi-agent workflow system using LangGraph for automated document processing, analysis, and insights extraction. This enterprise-grade system provides intelligent document processing with proper error handling, observability, and scalability.

## 🚀 Features

### Multi-Agent Architecture
- **OCR Extraction Agent**: Extracts text from PDFs and images using Tesseract OCR
- **Analysis & Summarization Agent**: Generates intelligent summaries and analyzes sentiment/topics
- **Structured Field Detection Agent**: Extracts structured data and classifies document types

### Core Capabilities
- 📄 Support for PDF, PNG, JPG, JPEG, TIFF formats
- 🔍 Multi-page PDF processing
- 🤖 AI-powered analysis using Claude/GPT-4
- 📊 Structured field extraction (dates, amounts, names, addresses)
- 📈 Quality assessment and human-in-the-loop review
- 💾 Dual database storage (PostgreSQL + MongoDB)
- 📡 RESTful API with async processing
- 🐳 Docker containerization
- 📊 Monitoring and metrics (Prometheus + Grafana)

## 🏗️ Architecture

### System Components
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI API   │    │  LangGraph       │    │   PostgreSQL    │
│                 │◄──►│  Workflow        │◄──►│   (Primary)     │
│   /api/v1/*     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Background    │    │   Agent          │    │   MongoDB       │
│   Tasks         │    │   Orchestration  │    │   (Secondary)   │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Agent Workflow
```
Document Upload → OCR Extraction → Quality Check → Analysis → Field Extraction → Final Results
        │              │               │            │             │
        └─ Human Review (if needed) ──┘            └─ Human Review (if needed)
```

## 🛠️ Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Tesseract OCR
- LLM API keys (OpenAI or Anthropic)

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/your-org/document-processing-system.git
cd document-processing-system
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run with Docker (recommended)**
```bash
docker-compose up -d
```

5. **Or run locally**
```bash
python -m app.main
```

The API will be available at `http://localhost:8000`

### Configuration

Create a `.env` file with the following variables:

```env
# Server Configuration
APP_ENV=development
DEBUG=true
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=document_processing
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017/document_processing

# LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_MODEL=gpt-4
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# OCR Configuration
TESSERACT_CMD=/usr/bin/tesseract
USE_CLOUD_OCR=false
OCR_CONFIDENCE_THRESHOLD=0.7

# Security
MAX_FILE_SIZE_MB=50
ALLOWED_EXTENSIONS=pdf,png,jpg,jpeg,tiff
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

## 📚 API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload and process document |
| `GET` | `/api/v1/documents/{id}/status` | Check processing status |
| `GET` | `/api/v1/documents/{id}/results` | Get processing results |
| `GET` | `/api/v1/documents` | List processed documents |
| `DELETE` | `/api/v1/documents/{id}` | Delete document and data |
| `GET` | `/api/v1/health` | System health check |
| `GET` | `/api/v1/metrics/costs` | Cost tracking metrics |

### Example Usage

**Upload a document:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "language=eng"
```

**Check processing status:**
```bash
curl "http://localhost:8000/api/v1/documents/{document_id}/status"
```

**Get results:**
```bash
curl "http://localhost:8000/api/v1/documents/{document_id}/results"
```

## 🗄️ Database Schema

### PostgreSQL (Primary Storage)
- `documents` - Document metadata and status
- `ocr_results` - OCR extraction results
- `document_summaries` - Analysis and summarization results
- `structured_data` - Extracted fields and classification

### MongoDB (Secondary Storage)
- `ocr_raw_results` - Raw OCR data with confidence scores
- `agent_states` - Intermediate workflow states
- `processing_metrics` - Performance and cost metrics

## 🔧 Development

### Project Structure
```
project/
├── app/
│   ├── agents/              # Specialized agents
│   │   ├── ocr_agent.py
│   │   ├── analysis_agent.py
│   │   └── extraction_agent.py
│   ├── workflows/           # LangGraph workflows
│   │   └── document_workflow.py
│   ├── api/                 # FastAPI endpoints
│   │   ├── routes.py
│   │   └── models.py
│   ├── database/           # Database connections
│   │   ├── postgres.py
│   │   └── mongo.py
│   ├── services/           # Core services
│   │   ├── ocr_service.py
│   │   └── llm_service.py
│   ├── core/               # Configuration and utilities
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── metrics.py
│   └── main.py            # Application entry point
├── tests/                 # Test suite
├── docker/               # Docker configuration
├── scripts/              # Deployment scripts
├── requirements.txt
├── docker-compose.yml
└── README.md
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_ocr_service.py
```

### Adding New Agents

1. Create agent class in `app/agents/`
2. Implement the `process_document` method
3. Add agent to workflow in `app/workflows/document_workflow.py`
4. Add corresponding API models in `app/api/models.py`

## 🚀 Deployment

### Production with Docker
```bash
# Build and run
docker-compose -f docker-compose.prod.yml up -d

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale worker=4
```

### Kubernetes (Optional)
```bash
kubectl apply -f k8s/
```

### Environment Variables for Production
```env
APP_ENV=production
DEBUG=false
WORKERS=4
LOG_LEVEL=warning

# Use external databases in production
POSTGRES_HOST=your-production-db-host
MONGODB_URL=your-production-mongo-url

# Enable monitoring
PROMETHEUS_MULTIPROC_DIR=/tmp
```

## 📊 Monitoring & Observability

### Health Checks
- Application: `GET /health`
- Database connectivity
- LLM service availability
- System resources

### Metrics
- Processing time per document
- OCR confidence scores
- LLM token usage and costs
- Success/failure rates
- Queue depth and throughput

### Logging
- Structured JSON logging
- Request tracing with correlation IDs
- Agent-specific logging
- Error tracking with stack traces

## 🔒 Security

### Features
- File type validation (MIME + extension)
- File size limits
- Rate limiting
- CORS configuration
- Input sanitization
- PII detection and masking
- API key authentication

### Best Practices
- Use HTTPS in production
- Store secrets in environment variables
- Regular security updates
- Database connection pooling
- Request validation with Pydantic

## 📈 Performance

### Benchmarks
- Process 10-page PDF in under 60 seconds
- Support concurrent processing of multiple documents
- Handle files up to 50MB
- API response time <200ms (excluding async jobs)

### Optimization Tips
- Enable parallel processing for batch operations
- Use cloud OCR services for high-volume processing
- Implement caching for frequently accessed documents
- Monitor and optimize database queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Write tests for new features
- Follow PEP 8 style guide
- Use type hints
- Update documentation
- Add API examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is a production-ready reference implementation. Ensure proper security measures and monitoring when deploying to production environments.

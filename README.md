# CV RAG System

A production-ready Retrieval-Augmented Generation system that intelligently answers questions about my professional experience. Using semantic search and large language models.

## Motivation

I built this to demonstrate my LLM engineering skills while solving a practical problem: making my CV information instantly searchable. Instead of asking people to read through lengthy documents, they can ask natural questions and get accurate answers drawn directly from my experience.

The system showcases the ML engineering work I do professionally: building RAG pipelines, implementing vector databases, integrating LLM APIs, and deploying production systems. It's designed to be scalable, maintainable, and solve real problems.

## System Architecture

### Core Components

**Source Code (`src/`)**
- `text_processor.py`: Handles document chunking with token-aware splitting (512 tokens, 50 token overlap) and context assembly within LLM token budgets
- `hf_client.py`: Manages HuggingFace API integration for embeddings (sentence-transformers/all-MiniLM-L6-v2) and text generation (Mistral-7B-Instruct)
- `pgvector_client.py`: PostgreSQL client with pgvector extension for vector similarity search using cosine distance

**Backend (`main.py`)**
FastAPI application serving both REST API and web interface. Orchestrates the RAG pipeline: embedding generation, vector search, context assembly, and LLM answer generation. Includes health checks, error handling, and comprehensive logging.

**Frontend (`templates/` + `static/`)**
Responsive web interface built with Pico CSS framework. Users submit questions via a simple form, results display with answer text, source attribution, retrieval metrics (chunks used, similarity scores), and execution time.

**Configuration (`config/`)**
Centralized configuration management with environment variable support for API keys, model selection, and RAG parameters (chunk size, top-k retrieval, similarity thresholds).

### Query Flow

When a user submits a question through the frontend, the system executes a four-stage pipeline:

1. **Embedding Generation**: The question is sent to HuggingFace's embedding API, which transforms it into a 384-dimensional semantic vector representing its meaning.

2. **Vector Search**: The query embedding is compared against all stored document chunk embeddings in PostgreSQL using cosine similarity. The database returns the top-k most semantically similar chunks (default: 5) that exceed the similarity threshold (0.1), leveraging IVFFlat indexing for efficient approximate nearest neighbor search.

3. **Context Assembly**: Retrieved chunks are assembled into a context string within the LLM's token budget (2000 tokens reserved for context). The text processor includes metadata headers with source files and similarity scores, then truncates if necessary to fit the budget while prioritizing higher-relevance chunks.

4. **Answer Generation**: The assembled context and original question are formatted into a prompt and sent to the Mistral-7B-Instruct model via HuggingFace's Inference API. The LLM generates a natural language answer grounded in the provided context, instructed to only use information from retrieved chunks.

The frontend displays the complete response including the answer, source documents, number of chunks used, and total execution time (typically 3-6 seconds).

## Documentation

- [Backend Setup](BACKEND_SETUP.md) - Detailed deployment instructions
- [Requirements](requirements.txt) - Python dependencies

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- HuggingFace API token ([get one here](https://huggingface.co/settings/tokens))
- PostgreSQL database with pgvector extension (or use bundled Docker setup)

### With Bundled Database

```bash
# Create environment file
cat > .env << EOF
HF_TOKEN=your_huggingface_token
EOF

# Start application
docker compose -f docker-compose.external-db.yml up --build

# Access application
open http://localhost:8000
```

### With External PostgreSQL

```bash
# Create environment file with database URL
cat > .env << EOF
HF_TOKEN=your_huggingface_token
DATABASE_URL=postgresql://user:password@host:5432/database
EOF

# Start application
docker compose -f docker-compose.external-db.yml up --build
```

## Technical Features

- **FastAPI**: Async REST API with automatic OpenAPI documentation
- **HuggingFace Integration**: Remote Inference API for embeddings and LLM generation
- **Vector Search**: PostgreSQL with pgvector extension for semantic similarity
- **Responsive UI**: Modern Pico CSS framework with loading states
- **Docker Deployment**: Multi-container setup with health checks
- **Production Ready**: Logging, error handling, configuration management

## Project Structure

```
cv_rag/
├── main.py                      # FastAPI application & RAG pipeline
├── config/
│   └── config.py               # Environment-based configuration
├── src/
│   ├── text_processor.py       # Chunking & context assembly
│   ├── hf_client.py            # HuggingFace API client
│   └── pgvector_client.py      # PostgreSQL vector operations
├── templates/
│   └── index.html              # Web interface
├── static/
│   ├── css/custom.css          # Styling
│   └── js/app.js               # Frontend logic
├── documents/                   # Source CV documents
├── docker-compose.external-db.yml
└── Dockerfile
```

## Docker Commands

```bash
# Start services (foreground with logs)
docker compose -f docker-compose.external-db.yml up --build

# Start in background
docker compose -f docker-compose.external-db.yml up --build -d

# Stop services
docker compose -f docker-compose.external-db.yml down

# View logs
docker compose -f docker-compose.external-db.yml logs -f app

# Restart app only
docker compose -f docker-compose.external-db.yml restart app

# Access database
docker compose -f docker-compose.external-db.yml exec postgres psql -U raguser -d ragdb
```

## Running Without Docker

```bash
# Install dependencies
pip install -r requirements-backend.txt

# Set environment variables
export HF_TOKEN=your_token
export DATABASE_URL=postgresql://user:password@host:5432/database

# Run application
python main.py
```

## Health Check

```bash
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","database":"connected"}
```

## Data Ingestion

After starting the services, load your documents into the database:

```bash
# Run ingestion script (creates embeddings and stores chunks)
docker compose -f docker-compose.external-db.yml exec app python ingest_documents.py
```

## Configuration Options

Set these in your `.env` file to customize behavior:

```bash
# Required
HF_TOKEN=your_token
DATABASE_URL=postgresql://user:pass@host:5432/db

# Optional (defaults shown)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_CHUNKS=5
SIMILARITY_THRESHOLD=0.1
```

## Technology Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **ML/AI**: HuggingFace Inference API, tiktoken
- **Database**: PostgreSQL 16 with pgvector extension
- **Deployment**: Docker, Docker Compose
- **Frontend**: Pico CSS, Vanilla JavaScript

# CV RAG System

Retrieval-Augmented Generation system for answering questions about your CV using FastAPI, HuggingFace, and pgvector.

## Documentation

- [Quick Start Guide](QUICKSTART.md) - Get started in 2 minutes
- [Backend Setup](BACKEND_SETUP.md) - Detailed setup instructions
- [Requirements](requirements.txt) - Python dependencies

## Quick Start

```bash
# Create environment file
echo "HF_TOKEN=your_token" > .env
echo "DATABASE_URL=postgresql://user:password@host:5432/db" >> .env

# Start application
docker-compose up --build

# Access application
open http://localhost:8000
```

For external PostgreSQL database:

```bash
docker-compose -f docker-compose.external-db.yml up --build
```

## Features

- FastAPI REST API with web interface
- HuggingFace Inference API integration
- PostgreSQL with pgvector for semantic search
- Bootstrap responsive frontend
- Docker containerization
- Health check endpoints

## Project Structure

```
cv_rag/
├── main.py              # FastAPI application
├── config/
│   └── config.py       # Configuration
├── src/
│   ├── hf_client.py
│   ├── pgvector_client.py
│   └── text_processor.py
├── templates/
│   └── index.html
├── static/
│   └── css/
└── docker-compose.yml
```

## Docker Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Rebuild
docker-compose up --build

# View logs
docker-compose logs -f app

# Access database
docker-compose exec db psql -U rag_user -d rag_db
```
---

# Quick Start

## With Bundled Database

```bash
# Create .env
echo "HF_TOKEN=your_token" > .env

# Start
docker-compose up --build

# Access
open http://localhost:8000
```

## With External Database

```bash
# Create .env
cat > .env << EOF
HF_TOKEN=your_token
DATABASE_URL=postgresql://user:password@host:5432/database
EOF

# Start
docker-compose -f docker-compose.external-db.yml up --build

# Access
open http://localhost:8000
```

## Without Docker

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
export DATABASE_URL=postgresql://user:password@host:5432/database
python main.py
```

## Verify

```bash
curl http://localhost:8000/health
```



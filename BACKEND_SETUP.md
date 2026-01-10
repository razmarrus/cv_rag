# Backend Setup Instructions

## Prerequisites

- Docker and Docker Compose installed
- PostgreSQL database with pgvector extension and populated data
- HuggingFace API token

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
HF_TOKEN=your_huggingface_api_token
DATABASE_URL=postgresql://user:password@host:port/database
```

Replace the following:
- `your_huggingface_api_token`: Your HuggingFace API token
- `user`: Database username
- `password`: Database password
- `host`: Database host (use `db` for Docker Compose, or external host)
- `port`: Database port (default: 5432)
- `database`: Database name

### 2. Optional Configuration

Additional environment variables can be set in `.env`:

```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_CHUNKS=5
SIMILARITY_THRESHOLD=0.7
```

## Running with Docker Compose

### Using Bundled PostgreSQL

If you want to use the bundled PostgreSQL container:

```bash
docker-compose up --build
```

This starts both the database and application containers.

### Using External PostgreSQL

If you have an existing PostgreSQL server with data, use the external database compose file:

```bash
docker-compose -f docker-compose.external-db.yml up --build
```

Ensure your `.env` file contains the correct `DATABASE_URL` pointing to your external database.

## Running Without Docker

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Export environment variables or create `.env` file as described above.

### 3. Run Application

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Verification

### Health Check

Test the application is running:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "database": "connected"
}
```

### Frontend Access

Open browser and navigate to:

```
http://localhost:8000
```

## Troubleshooting

### Database Connection Failed

Verify database credentials and network connectivity:

```bash
psql "postgresql://user:password@host:port/database"
```

Ensure pgvector extension is installed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Application Fails to Start

Check logs:

```bash
docker-compose logs app
```

Or without Docker:

```bash
python main.py
```

### Port Already in Use

Change the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "8080:8000"  # Use port 8080 instead
```

Or when running directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

## Stopping the Application

With Docker Compose:

```bash
docker-compose down
```

Without Docker, use Ctrl+C to stop the process.

## Production Deployment

For production deployments:

1. Remove volume mounts in `docker-compose.yml`
2. Set appropriate resource limits
3. Use production WSGI server configuration
4. Enable HTTPS with reverse proxy
5. Configure log aggregation
6. Set up monitoring and alerting

Refer to deployment documentation for platform-specific instructions.


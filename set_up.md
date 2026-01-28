# Setup and Common Commands

## Docker Service Management

### Start Docker Service
```bash
sudo systemctl start docker.service
```

## Container Management


### Check All Containers (including stopped)
```bash
docker ps -a
```

### Start Individual Containers
```bash
sudo docker start cv_rag_app
sudo docker start rag_pgvector
```

## Application Deployment

### Build and Start with Docker Compose
```bash
docker-compose -f docker-compose.external-db.yml up --build
```

### Build and Start in Background
```bash
docker-compose -f docker-compose.external-db.yml up --build -d
```

### Restart Application Only
```bash
docker-compose -f docker-compose.external-db.yml restart app
```

### Stop All Services
```bash
docker-compose -f docker-compose.external-db.yml down
```

`docker rm -f container_name` - remove docker container

## Virtual Environment

### Activate Virtual Environment
```bash
source /home/margot/Code/LLM/cv_rag/rag_cv_env/bin/activate
```

## Quick Start Workflow

1. Start Docker service:
   ```bash
   sudo systemctl start docker.service
   ```

2. Build and start application:
   ```bash
   docker-compose -f docker-compose.external-db.yml up --build
   ```

3. Access application:
   ```
   http://localhost:8000
   ```

## Troubleshooting

### View Application Logs
```bash
docker logs cv_rag_app -f
```

### View Database Logs
```bash
docker logs rag_pgvector -f
```

### Check Container Status
```bash
docker ps
```

Compose
`docker-compose -f docker-compose.external-db.yml up --build -d`

With logs
`docker-compose -f docker-compose.external-db.yml up --build`

Restart
`docker-compose -f docker-compose.external-db.yml restart app`

## Pi

`ssh razmarrus@192.168.0.180`
`sudo shutdown -h now` or `sudo poweroff`

ingest data
`docker compose exec app python ingest_documents.py`

Clean up db
`docker compose exec postgres psql -U raguser -d ragdb -c "TRUNCATE TABLE documents;"`

# Health check
curl http://192.168.0.180:8000/health

# Ask a question via API
curl -X POST http://192.168.0.180:8000/ask \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=What is your experience with Python?"

restart conatiner
`docker compose restart app`
`docker compose up -d --build app`
  
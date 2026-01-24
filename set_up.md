# Setup and Common Commands

## Docker Service Management

### Start Docker Service
```bash
sudo systemctl start docker.service
```

## Container Management

### Check Running Containers
```bash
docker ps
```

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

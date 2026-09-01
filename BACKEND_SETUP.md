# Backend setup

Run the FastAPI RAG app with Docker Compose (app + pgvector). Same path on laptop or Pi.

## Prerequisites

- Docker Compose
- Hugging Face token with **Inference Providers** access
- `.env` in the repo root (never commit it)

## `.env`

All of these are required. Missing one fails container start (`:?` in compose).

```bash
HF_TOKEN=hf_...
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=openai/gpt-oss-20b
HF_PROVIDER=groq
USE_LOCAL_EMBEDDINGS=true
```

| Var | Notes |
| --- | --- |
| `HF_TOKEN` | HF token used for Inference Providers |
| `EMBEDDING_MODEL` | Local encoder when `USE_LOCAL_EMBEDDINGS=true` |
| `LLM_MODEL` | Must be **conversational** on the chosen provider |
| `HF_PROVIDER` | Provider slug (`groq`, …). Do not use `auto` |
| `USE_LOCAL_EMBEDDINGS` | `true` on Pi/prod. Do not mix true/false against the same DB |

The app container **ignores** `DATABASE_URL` from `.env`. Compose sets:

`postgresql://raguser:ragpass@postgres:5432/ragdb`

Host tools (notebook / `psql`) use `127.0.0.1:5433` → same DB.

RAG knobs (chunk size, top-k, thresholds) live in `config/config.py`, not env.

## Start

```bash
docker compose up -d --build
docker compose exec app python ingest_documents.py
curl http://localhost:8000/health
# open http://localhost:8000
```

After changing `.env`:

```bash
docker compose up -d --force-recreate app
```

`docker compose restart app` does **not** reload env.

## Without Docker

```bash
# Prefer requirements-backend.txt. Install torch from the PyTorch CPU index
# (see Dockerfile) — plain PyPI pulls CUDA wheels.
export HF_TOKEN=... EMBEDDING_MODEL=... LLM_MODEL=... HF_PROVIDER=... USE_LOCAL_EMBEDDINGS=...
export DATABASE_URL=postgresql://raguser:ragpass@127.0.0.1:5433/ragdb
python main.py
```

## Checks

```bash
curl http://localhost:8000/health
# {"status":"healthy","database":"connected"}

docker compose logs -f app
docker compose exec postgres psql -U raguser -d ragdb
```

## Common failures

| Symptom | Cause |
| --- | --- |
| App exits on start | Missing `.env` var |
| `chat_completion` HTTP 400 | Model not conversational on `HF_PROVIDER` |
| HTTP 402 | HF Inference Providers quota |
| Dim mismatch at startup | Table width ≠ encoder — truncate + re-ingest |
| Weird retrieval after toggling local/remote embed | Different vector spaces — re-ingest |

```bash
docker compose exec postgres psql -U raguser -d ragdb -c "TRUNCATE TABLE documents;"
docker compose exec app python ingest_documents.py
```

See also: root `README.md`, `documents/SYSTEM_INFRASTRUCTURE.md`, `documents/set_up.md`.

# CV RAG System

A Retrieval-Augmented Generation app that answers questions about my professional experience from indexed CV documents. 

**Live portfolio:** [https://who-is-margot.duckdns.org/](https://who-is-margot.duckdns.org/)

## Motivation

I thought a chatbot on top of my CV would be cool. I like building things. It runs at home on my own Raspberry Pi 5 — that part is cool too.

Under the hood it is still a real RAG pipeline: search my documents first, then an LLM answers only from what it found. Not a bot that invents a bio from the open web.

## Current stack

| Piece | What runs |
|---|---|
| Encoder | `BAAI/bge-small-en-v1.5` locally via sentence-transformers (384-dim) |
| LLM | `openai/gpt-oss-20b` via Hugging Face Inference Providers |
| Provider | `groq` (pinned; do not use `auto`) |
| Vector store | PostgreSQL 16 + pgvector, cosine similarity, IVFFlat index |
| App | FastAPI + Jinja2, one uvicorn worker |
| Deploy | Docker Compose, app `mem_limit: 1g`, postgres `256m` |

Model names and the provider slug live in `.env`. There are no code-side fallbacks: a missing variable fails at startup.

The LLM must be mapped for the **conversational** task on the chosen provider. `text-generation`-only models fail `chat_completion` with HTTP 400. Local and remote encoders must not be mixed: they produce different vector spaces even at the same width. Ingestion and queries both take `USE_LOCAL_EMBEDDINGS` from the same config.

## Architecture

**`src/hf_client.py`** — One encoder path (local *or* remote, never both). Reads embedding width from the loaded model. LLM calls go to `InferenceClient(provider=...)`. Empty completions raise instead of returning a blank 200.

**`src/pgvector_client.py`** — Connection pool, schema, cosine search. After `CREATE TABLE IF NOT EXISTS`, `_verify_embedding_dim` compares the encoder width to the stored `VECTOR(n)` column and refuses to start on mismatch.

**`src/text_processor.py`** — Token-aware chunking and context assembly under the LLM token budget.

**`main.py`** — FastAPI. Pipeline: embed → search (threshold 0.5, then 0.1 if empty) → assemble context → generate. Off-topic questions deflect; personal chunks use a warmer prompt. Daily query cap is 25.

**`templates/` + `static/`** — Single-page UI with preset question pills (including Off Script) and a privacy page.

**`config/config.py`** — Required: `HF_TOKEN`, `DATABASE_URL`, `EMBEDDING_MODEL`, `LLM_MODEL`, `HF_PROVIDER`. RAG constants (chunk size, top-k, thresholds) are in code, not env.

## Query flow

1. **Embed** the question locally with bge-small (384-dim).
2. **Search** pgvector for the top 4 chunks above similarity 0.5. If none, retry at 0.1. If still none, generate a deflect answer with no document context.
3. **Assemble** retrieved chunks into a prompt, capped at 2500 context tokens.
4. **Generate** via Inference Providers (`chat_completion` on the pinned provider). Typical latency is well under a second on Groq once the encoder is warm.

Chunking used at ingest: 430 tokens, 25 overlap. Changing the embedding model requires dropping `documents` and re-ingesting; width is checked at startup, vector-space drift is not.

## Quick start

Prerequisites: Docker Compose, a Hugging Face token with **Make calls to Inference Providers**.

```bash
cat > .env << EOF
HF_TOKEN=hf_...
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=openai/gpt-oss-20b
HF_PROVIDER=groq
USE_LOCAL_EMBEDDINGS=true
EOF

docker compose up -d --build
docker compose exec app python ingest_documents.py
# open http://localhost:8000
```

The app container ignores `DATABASE_URL` from `.env` and uses `postgresql://raguser:ragpass@postgres:5432/ragdb` on the compose network. Host tools (notebooks, `psql`) reach the same database at `127.0.0.1:5433` because the host already occupies 5432.

`.env` is read only at container create. After changing it:

```bash
docker compose up -d --force-recreate app
```

`docker compose restart app` will not pick up new values.

## Configuration

Set in `.env` (all required except `DATABASE_URL` for the container):

```bash
HF_TOKEN=...
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=openai/gpt-oss-20b
HF_PROVIDER=groq
USE_LOCAL_EMBEDDINGS=true
```

`HF_PROVIDER` is a provider slug (`groq`, `featherless-ai`, `together`, `novita`, …), not a model id. Pin a provider that actually serves `LLM_MODEL` for `conversational`. Switching models without re-ingest only works if the new encoder has the same width *and* the same vector space — in practice, re-ingest.

In-code RAG settings (`config/config.py`):

| Setting | Value |
|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 430 / 25 |
| `TOP_K_CHUNKS` | 4 |
| `SIMILARITY_THRESHOLD` | 0.5 (relaxed 0.1) |
| `MAX_CONTEXT_TOKENS` | 2500 |
| `MAX_NEW_TOKENS` | 350 (500 for personal) |
| `DAILY_QUERY_LIMIT` | 25 |

## Docker

```bash
docker compose up -d --build
docker compose logs -f app
docker compose exec postgres psql -U raguser -d ragdb
docker compose down
```

Ingest must run inside the app container so it uses local embeddings and the compose database:

```bash
docker compose exec app python ingest_documents.py
```

Without Docker, install `requirements-backend.txt` (CPU torch is installed from the PyTorch index in the Dockerfile; from PyPI it pulls CUDA wheels). Export the same env vars, point `DATABASE_URL` at a pgvector instance, then `python main.py`.

## Health check

```bash
curl http://localhost:8000/health
# {"status":"healthy","database":"connected"}
```

## Project layout

```
cv_rag/
├── main.py                 # FastAPI + RAG orchestration
├── ingest_documents.py     # ETL into pgvector
├── config/config.py
├── src/
│   ├── hf_client.py        # local encoder + Inference Providers LLM
│   ├── pgvector_client.py  # search, schema, dim check
│   └── text_processor.py
├── templates/              # index + privacy
├── static/
├── documents/              # source CV text
├── docker-compose.yml
├── Dockerfile
└── requirements-backend.txt
```

## Notes on hardware and billing

Local bge-small plus torch needs about 520 MB resident; the app container is capped at 1g for that reason. A larger encoder (bge-large, 1024-dim) does not fit this budget and would also invalidate the existing 384-dim table.

LLM calls are billed against Hugging Face Inference Providers credits when routed with an HF token. Credits apply to eligible providers; a 402 is quota, not an application bug. A custom provider API key in Hugging Face settings bills the provider instead and leaves the code unchanged.


# System infrastructure

Live site: `https://who-is-margot.duckdns.org/`

Pi at home runs the app + Postgres. A small VPS only terminates HTTPS and proxies over WireGuard. LLM calls leave the Pi through Hugging Face → Groq.

---

## Topology

| Piece | Role |
| --- | --- |
| DuckDNS | Name → public VPS |
| Caddy (VPS) | TLS `:443`, reverse proxy |
| WireGuard | VPS ↔ Pi private net (`10.0.0.1` ↔ `10.0.0.2`) |
| `cv_rag_app` on Pi | FastAPI `:8000`, local bge-small, outbound LLM |
| `rag_pgvector` on Pi | Postgres 16 + pgvector, Compose-only |
| Hugging Face router | Auth + route `chat_completion` |
| Groq | Runs `openai/gpt-oss-20b` (pinned via `HF_PROVIDER=groq`) |

```
Browser → HTTPS → Caddy (VPS)
       → WireGuard → http://10.0.0.2:8000 (Pi app)
       → local embed + Postgres on Compose network
       → HTTPS router.huggingface.co → Groq
       ← answer back the same path
```

Caddy is the only public face. Postgres is not on the internet (`127.0.0.1:5433` on the Pi for local tools only).

---

## Pi containers (`docker-compose.yml`)

**App** — `mem_limit: 1g`. Env from `.env` (`HF_*`, `USE_LOCAL_EMBEDDINGS`). DB URL forced to `postgresql://raguser:ragpass@postgres:5432/ragdb`. Cache volume for encoder weights.

**Postgres** — `mem_limit: 256m`. Internal `:5432`. Host map `127.0.0.1:5433:5432`.

`.env` change → `docker compose up -d --force-recreate app` (restart is not enough).

---

## Data path

1. Embed question locally (`BAAI/bge-small-en-v1.5`, 384-dim).
2. Cosine search in pgvector (top-k 4; threshold 0.5 then 0.1).
3. Build prompt from chunks (or deflect).
4. `InferenceClient(..., provider=HF_PROVIDER).chat_completion(...)`.

Ingest with the same embed mode as queries:

```bash
docker compose exec app python ingest_documents.py
```

Switching local ↔ remote embeddings or encoder model ⇒ truncate `documents` and re-ingest. Width is checked at startup; vector-space drift is not.

---

## Hugging Face

| Call | How |
| --- | --- |
| Embed (prod) | Local sentence-transformers — no HF HTTP after cache warm |
| Embed (dev flag) | `provider="hf-inference"` feature-extraction — separate vector space |
| LLM | `provider=HF_PROVIDER` (e.g. `groq`) — always remote |

Model must be **conversational** on that provider. HTTP 402 = quota.

---

## Ports

| Port | Where | Who |
| --- | --- | --- |
| `:443` | VPS Caddy | browsers |
| `:8000` | Pi app | Caddy via WireGuard |
| `:5432` | Compose `postgres` | app container |
| `:5433` | Pi localhost | notebooks / `psql` on that host |

---

## Eraser diagrams

Paste into [Eraser](https://www.eraser.io).

### Deploy

```eraser
title CV RAG — deploy
direction right

Browser [icon: monitor] > Caddy [icon: server]: HTTPS :443
Caddy > App [icon: server]: WireGuard → 10.0.0.2:8000
App > DB [icon: database]: pgvector cosine search
App > HF [icon: cloud]: chat_completion + HF_TOKEN
HF > Groq [icon: cloud]: openai/gpt-oss-20b
Groq > HF > App > Caddy > Browser: answer
```

### One ask

```eraser
title One ask
Browser > Caddy: 1. HTTPS ask
Caddy > App: 2. proxy
App > Encoder [icon: cpu]: 3. local embed
App > DB [icon: database]: 4. vector search
DB > App: 5. chunks
App > HF [icon: cloud]: 6. chat_completion
HF > Groq: 7. run
Groq > HF > App > Caddy > Browser: 8–11. answer
```

---



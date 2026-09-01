# Ops cheat sheet

Day-to-day commands for this repo. Run from the project root unless noted.

Compose DB (inside containers): user `raguser`, db `ragdb`, host `postgres`.  
From the host: `127.0.0.1:5433`.

---

## Daily query quota

Quota = rows in `query_logs` for your IP with `DATE(created_at) = CURRENT_DATE`  
Limit: `DAILY_QUERY_LIMIT` (default **25**) in `config/config.py` / env.

Behind Caddy the IP is often `X-Forwarded-For`. Locally it is often `127.0.0.1` or a Docker bridge IP.

### See today's usage

```bash
docker compose exec postgres psql -U raguser -d ragdb -c "
SELECT user_ip, COUNT(*) AS n
FROM query_logs
WHERE DATE(created_at) = CURRENT_DATE
GROUP BY user_ip
ORDER BY n DESC;
"
```

### Reset **your** quota for today (one IP)

Use `CURRENT_DATE` for today, or an ISO date string `'YYYY-MM-DD'` — not `01-09-2026` (Postgres treats that as integer math).

```bash
# replace IP
docker compose exec postgres psql -U raguser -d ragdb -c "
DELETE FROM query_logs
WHERE user_ip = '127.0.0.1'
  AND DATE(created_at) = CURRENT_DATE;
"
```

Specific calendar day (example: 1 Sep 2026):

```bash
docker compose exec postgres psql -U raguser -d ragdb -c "
DELETE FROM query_logs
WHERE user_ip = '127.0.0.1'
  AND DATE(created_at) = DATE '2026-09-01';
"
```

Or drop the IP filter to clear **all** rows for that day:

```bash
docker compose exec postgres psql -U raguser -d ragdb -c "
DELETE FROM query_logs
WHERE DATE(created_at) = DATE '2026-09-01';
"
```

### Reset **everyone's** quota for today

```bash
docker compose exec postgres psql -U raguser -d ragdb -c "
DELETE FROM query_logs
WHERE DATE(created_at) = CURRENT_DATE;
"
```

### Wipe one IP entirely (all days — GDPR-style)

```bash
docker compose exec postgres psql -U raguser -d ragdb -c "
DELETE FROM query_logs WHERE user_ip = '127.0.0.1';
"
```

Reload the page after delete; remaining count is recomputed on `/` and `/ask`.

---

## Docker Compose

```bash
docker compose up -d --build
docker compose up -d --force-recreate app   # after .env change
docker compose restart app                 # code only; does NOT reload .env
docker compose logs -f app
docker compose logs -f postgres
docker compose ps
docker compose down
```

---

## Ingest / documents table

```bash
docker compose exec app python ingest_documents.py

docker compose exec postgres psql -U raguser -d ragdb -c "TRUNCATE TABLE documents;"
docker compose exec app python ingest_documents.py   # re-ingest after truncate
```

Ingest only picks up `documents/*.txt` (not `docks_backup/`, not `.md`).

---

## Postgres shell

```bash
docker compose exec postgres psql -U raguser -d ragdb

# host tools / notebooks
psql "postgresql://raguser:ragpass@127.0.0.1:5433/ragdb"
```

Useful SQL:

```sql
SELECT COUNT(*) FROM documents;
SELECT COUNT(*) FROM query_logs;
SELECT user_ip, created_at, LEFT(question, 60)
FROM query_logs
ORDER BY created_at DESC
LIMIT 20;
```

---

## Health / ask

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=What is your experience with Python?"
```

Public (through Caddy): `https://YOUR_DOMAIN/health`

---

## `.env` (required)

```bash
HF_TOKEN=hf_...
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=openai/gpt-oss-20b
HF_PROVIDER=groq
USE_LOCAL_EMBEDDINGS=true
```

Optional: `DAILY_QUERY_LIMIT=25`

App container ignores host `DATABASE_URL`; compose sets `postgres:5432`.

---

## WireGuard (Pi ↔ VPS)

```bash
sudo systemctl enable --now wg-quick@wg0
sudo wg show
sudo wg-quick down wg0 && sudo wg-quick up wg0
```

---

## Notebook env

```bash
source rag_cv_env/bin/activate
# pins: requirements-backend.txt (skip Pi torch pin on CUDA laptop)
# guide (if present): documents/docks_backup/NOTEBOOK_ENV_UPDATE.md
```

---

## Related

- `README.md` — stack overview  
- `documents/vps_setup.md` — Pi + VPS + Caddy  
- `documents/SYSTEM_INFRASTRUCTURE.md` — if present; else README topology notes

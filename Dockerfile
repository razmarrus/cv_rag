FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files and buffer stdout/stderr
# Prevents Python from writing .pyc files (cached compiled Python code

ENV PYTHONDONTWRITEBYTECODE=1

# Makes Python output appear immediately in logs
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements-backend.txt .

# Optional offline install:
# - put pre-downloaded wheels into ./wheels (see README below)
COPY wheels/ /wheels/

# Install Python dependencies with BuildKit cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    if ls /wheels/*.whl >/dev/null 2>&1; then \
        python -m pip install --find-links=/wheels -r requirements-backend.txt || \
        python -m pip install -r requirements-backend.txt; \
    else \
        python -m pip install -r requirements-backend.txt; \
    fi

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/css static/js templates

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5).read()" || exit 1

# Run the application (optimized for Raspberry Pi - single worker to conserve memory)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
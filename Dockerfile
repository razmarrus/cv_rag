FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
# Prevents Python from writing .pyc files (cached compiled Python code

ENV PYTHONDONTWRITEBYTECODE=1 

# Makes Python output appear immediately in logs
ENV PYTHONUNBUFFERED=1

# libpq-dev - Header files and libraries for PostgreSQL
# gcc - GNU Compiler Collection (GCC)
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/css static/js templates

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
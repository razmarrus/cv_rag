import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Database. Embedding width is not configured here: it is read off the loaded
    # encoder and checked against the table, so the two cannot drift apart.

    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Model Configuration. Values come from .env only - no fallbacks here, so a
    # missing variable fails at startup instead of silently using a stale model.
    USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    # LLM_MODEL must be exposed for the "conversational" task by HF_PROVIDER;
    # text-generation-only models fail chat_completion with HTTP 400.
    LLM_MODEL = os.getenv("LLM_MODEL")
    # Inference Providers routing (huggingface_hub >= 1.0). Pin a provider rather
    # than using "auto", which can reroute onto one that maps LLM_MODEL differently.
    HF_PROVIDER = os.getenv("HF_PROVIDER")
    
    # CHUNK_SIZE = 512
    # CHUNK_OVERLAP = 50
    # MAX_CONTEXT_TOKENS = 2000
    # TOP_K_CHUNKS = 5
    # SIMILARITY_THRESHOLD = 0.1

    # RAG Settings
    # CHUNK_SIZE = 750
    CHUNK_SIZE = 430
    CHUNK_OVERLAP = 25


    SIMILARITY_THRESHOLD = 0.5
    RELAXED_SIMILARITY_THRESHOLD = 0.1
    TOP_K_CHUNKS = 4             
    MAX_CONTEXT_TOKENS = 2500    
    TEMPERATURE = 0.7
    OFF_TOPIC_TEMPERATURE = float(os.getenv("OFF_TOPIC_TEMPERATURE", "0.9"))
    PERSONAL_TEMPERATURE = float(os.getenv("PERSONAL_TEMPERATURE", "0.8"))
    MAX_NEW_TOKENS = 350
    MAX_PERSONAL_NEW_TOKENS = int(os.getenv("MAX_PERSONAL_NEW_TOKENS", "500"))

    RATE_LIMIT = "10/hour"

    # Rate Limiting
    DAILY_QUERY_LIMIT = int(os.getenv("DAILY_QUERY_LIMIT", "25"))

    # Data Retention
    QUERY_LOG_RETENTION_DAYS = int(os.getenv("QUERY_LOG_RETENTION_DAYS", "90"))

    # Network Configuration
    WIREGUARD_IP = os.getenv("WIREGUARD_IP")

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        required = [
            "HF_TOKEN",
            "DATABASE_URL",
            "EMBEDDING_MODEL",
            "LLM_MODEL",
            "HF_PROVIDER",
        ]
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(
                f"Missing required config (set in .env): {', '.join(missing)}"
            )
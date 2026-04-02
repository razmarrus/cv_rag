import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    EMBEDDING_DIM = 384
    
    # Model Configuration 
    USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    # LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    
    # CHUNK_SIZE = 512
    # CHUNK_OVERLAP = 50
    # MAX_CONTEXT_TOKENS = 2000
    # TOP_K_CHUNKS = 5
    # SIMILARITY_THRESHOLD = 0.1

    # RAG Settings
    # CHUNK_SIZE = 750
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 25


    SIMILARITY_THRESHOLD = 0.5
    RELAXED_SIMILARITY_THRESHOLD = 0.1
    TOP_K_CHUNKS = 4             
    MAX_CONTEXT_TOKENS = 2500    
    TEMPERATURE = 0.2
    MAX_NEW_TOKENS = 350  # Shorter answers (2-3 sentences)

    RATE_LIMIT = "10/hour"

    # Rate Limiting
    DAILY_QUERY_LIMIT = int(os.getenv("DAILY_QUERY_LIMIT", "12"))

    # Data Retention
    QUERY_LOG_RETENTION_DAYS = int(os.getenv("QUERY_LOG_RETENTION_DAYS", "90"))

    # Network Configuration
    WIREGUARD_IP = os.getenv("WIREGUARD_IP")

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        required = ["HF_TOKEN", "DATABASE_URL"]
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")
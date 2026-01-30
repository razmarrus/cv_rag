import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys
    HF_TOKEN = os.getenv("HF_TOKEN")

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", 
        "BAAI/bge-small-en-v1.5"
        if USE_LOCAL_EMBEDDINGS else "sentence-transformers/all-MiniLM-L6-v2"
    )
    LLM_MODEL = os.getenv(
        "LLM_MODEL", 
        "mistralai/Mistral-7B-Instruct-v0.2"
    )
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    EMBEDDING_DIM = 384
    
    # CHUNK_SIZE = 512
    # CHUNK_OVERLAP = 50
    # MAX_CONTEXT_TOKENS = 2000
    # TOP_K_CHUNKS = 5
    # SIMILARITY_THRESHOLD = 0.1

    # RAG Settings
    CHUNK_SIZE = 750
    CHUNK_OVERLAP = 125


    SIMILARITY_THRESHOLD = 0.5

    # more context
    # TOP_K_CHUNKS = 6
    # MAX_CONTEXT_TOKENS = 3500

    # Optimized 
    TOP_K_CHUNKS = 4             
    MAX_CONTEXT_TOKENS = 2500    


    TEMPERATURE = 0.2
    MAX_NEW_TOKENS = 500

    RATE_LIMIT = "10/hour"
    
    # Local embeddings (faster, uses ~200MB more RAM)
    USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        required = ["HF_TOKEN", "DATABASE_URL"]
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")
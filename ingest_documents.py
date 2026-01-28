#!/usr/bin/env python3
"""
Data Ingestion Script for RAG System.

This script implements an ETL pipeline to process documents from the
documents/ folder and load them into PostgreSQL with vector embeddings.
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple

from config.config import Config
from src.text_processor import TextProcessor
from src.hf_client import HuggingFaceClient
from src.pgvector_client import PgVectorClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_components() -> Tuple[TextProcessor, PgVectorClient,
                                     HuggingFaceClient]:
    """
    Initialize RAG system components.

    Returns:
        Tuple containing TextProcessor, PgVectorClient, and HuggingFaceClient
        instances.

    Raises:
        ValueError: If required configuration is missing.
    """
    logger.info("Initializing RAG components...")

    text_processor = TextProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        max_context_tokens=Config.MAX_CONTEXT_TOKENS
    )

    db_client = PgVectorClient(
        connection_string=Config.DATABASE_URL,
        embedding_dim=Config.EMBEDDING_DIM
    )

    hf_client = HuggingFaceClient(
        hf_token=Config.HF_TOKEN,
        embedding_model=Config.EMBEDDING_MODEL,
        llm_model=Config.LLM_MODEL
    )

    logger.info("Components initialized successfully")
    return text_processor, db_client, hf_client


def extract_documents(text_processor: TextProcessor,
                      docs_dir: Path = Path("documents")) -> List[Dict]:
    """
    Extract and chunk documents from directory.

    Args:
        text_processor: TextProcessor instance for chunking.
        docs_dir: Path to documents directory.

    Returns:
        List of chunk dictionaries with content and metadata.

    Raises:
        FileNotFoundError: If documents directory does not exist.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    all_chunks = []

    for file_path in sorted(docs_dir.glob("*.txt")):
        logger.info(f"Processing {file_path.name}...")
        try:
            chunks = text_processor.chunk_file(str(file_path))
            all_chunks.extend(chunks)
            logger.info(f"{len(chunks)} chunks created")
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    logger.info(f"Total chunks extracted: {len(all_chunks)}")
    return all_chunks


def generate_embeddings(hf_client: HuggingFaceClient,
                       chunks: List[Dict]) -> List[Dict]:
    """
    Generate embeddings for document chunks.

    Args:
        hf_client: HuggingFaceClient instance for embedding generation.
        chunks: List of chunk dictionaries.

    Returns:
        List of chunks with embeddings added.
    """
    logger.info("Generating embeddings...")

    texts = [chunk['content'] for chunk in chunks]
    embeddings = hf_client.get_embeddings(texts)

    for i, chunk in enumerate(chunks):
        embedding = embeddings[i]
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        chunk['embedding'] = embedding

    logger.info(f"Generated {len(embeddings)} embeddings")
    return chunks


def load_to_database(db_client: PgVectorClient, chunks: List[Dict]) -> None:
    """
    Load chunks with embeddings into database.

    Args:
        db_client: PgVectorClient instance for database operations.
        chunks: List of chunk dictionaries with embeddings.
    """
    logger.info("Inserting into database...")
    db_client.insert_chunks(chunks)
    logger.info(f"Inserted {len(chunks)} chunks into database")


def ingest_documents() -> None:
    """
    Main ETL pipeline for document ingestion.

    Orchestrates the complete ingestion process:
    1. Validate configuration
    2. Initialize components
    3. Extract and chunk documents
    4. Generate embeddings
    5. Load into database
    """
    logger.info("Starting document ingestion pipeline")

    # Validate configuration
    Config.validate()

    # Initialize components
    text_processor, db_client, hf_client = initialize_components()

    try:
        # Extract: Read and chunk documents
        all_chunks = extract_documents(text_processor)

        if not all_chunks:
            logger.warning("No documents found or processed")
            return

        # Transform: Generate embeddings
        all_chunks = generate_embeddings(hf_client, all_chunks)

        # Load: Insert into database
        load_to_database(db_client, all_chunks)

        logger.info("Document ingestion pipeline completed successfully")

    finally:
        # Cleanup: Close database connection
        db_client.close()
        logger.info("Database connection closed")


if __name__ == "__main__":
    ingest_documents()

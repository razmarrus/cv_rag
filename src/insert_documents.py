"""
Document insertion ETL script for RAG pipeline.

This script processes documents, generates embeddings, and inserts them
into a PostgreSQL vector database.
"""

import logging
import os
from typing import List, Dict

from dotenv import load_dotenv

from src.text_processor import TextProcessor
from src.pgvector_client import PgVectorClient
from src.hf_client import HuggingFaceClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_configuration():
    """
    Load environment variables and configuration constants.
    
    Returns:
        dict: Configuration dictionary with all necessary parameters.
    """
    load_dotenv()
    
    config = {
        'pg_conn_string': os.getenv("PG_CONNECTION_STRING"),
        'hf_token': os.getenv("HF_TOKEN"),
        'file_paths': [
            "documents/contact_info_chunk_750.txt",
            "documents/hr_qa_chunk_750.txt",
            "documents/projects_chunk_750.txt",
            "documents/motivation_chunk_750.txt",
        ],
        'chunk_size': 750,
        'chunk_overlap': 125,
        'max_context_tokens': 3500,
        'embedding_dim': 384,
        'embedding_model': "BAAI/bge-small-en-v1.5",
        'llm_model': "mistralai/Mistral-7B-Instruct-v0.2",
    }
    
    logger.info("Configuration loaded successfully")
    return config


def initialize_components(config):
    """
    Initialize TextProcessor, PgVectorClient, and HuggingFaceClient.
    
    Args:
        config (dict): Configuration dictionary.
    
    Returns:
        tuple: (text_processor, db_client, hf_client)
    """
    text_processor = TextProcessor(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        max_context_tokens=config['max_context_tokens']
    )
    
    db_client = PgVectorClient(
        connection_string=config['pg_conn_string'],
        embedding_dim=config['embedding_dim']
    )
    
    hf_client = HuggingFaceClient(
        hf_token=config['hf_token'],
        embedding_model=config['embedding_model'],
        llm_model=config['llm_model'],
    )
    
    logger.info("Components initialized successfully")
    return text_processor, db_client, hf_client


def extract_chunks(text_processor, file_paths):
    """
    Extract and chunk documents from file paths.
    
    Args:
        text_processor (TextProcessor): TextProcessor instance.
        file_paths (list): List of file paths to process.
    
    Returns:
        list: List of chunk dictionaries with content and metadata.
    """
    all_chunks = []
    
    for file_path in file_paths:
        chunks = text_processor.chunk_file(file_path)
        all_chunks.extend(chunks)
        logger.info(f"Processed {file_path}: {len(chunks)} chunks")
    
    logger.info(f"Total chunks from all files: {len(all_chunks)}")
    return all_chunks


def transform_chunks_with_embeddings(hf_client, chunks, batch_size=32):
    """
    Generate embeddings for chunks with batching and add them to chunk dictionaries.
    
    Args:
        hf_client (HuggingFaceClient): HuggingFaceClient instance.
        chunks (list): List of chunk dictionaries.
        batch_size (int): Number of chunks per API call.
    
    Returns:
        list: Chunks with embeddings added.
    """
    logger.info(f"Generating embeddings for {len(chunks)} chunks in batches of {batch_size}...")
    
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk['content'] for chunk in batch]
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(texts)} chunks)")
        batch_embeddings = hf_client.get_embeddings(texts)
        all_embeddings.extend(batch_embeddings)
    
    for i, chunk in enumerate(chunks):
        embedding = all_embeddings[i]
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        chunk['embedding'] = embedding
    
    logger.info(f"Generated {len(all_embeddings)} embeddings successfully")
    return chunks


def load_chunks_to_database(db_client, chunks):
    """
    Insert chunks with embeddings into the database.
    
    Args:
        db_client (PgVectorClient): PgVectorClient instance.
        chunks (list): List of chunk dictionaries with embeddings.
    """
    db_client.insert_chunks(chunks)
    logger.info(f"Inserted {len(chunks)} chunks into database")


def main():
    """
    Main ETL pipeline for document insertion.
    
    Steps:
        1. Load configuration
        2. Initialize components
        3. Extract: Read and chunk documents
        4. Transform: Generate embeddings
        5. Load: Insert into database
    """
    logger.info("Starting document insertion ETL pipeline")
    
    config = load_configuration()
    
    text_processor, db_client, hf_client = initialize_components(config)
    
    all_chunks = extract_chunks(text_processor, config['file_paths'])
    
    all_chunks = transform_chunks_with_embeddings(hf_client, all_chunks)
    
    load_chunks_to_database(db_client, all_chunks)
    
    logger.info("Document insertion ETL pipeline completed successfully")


if __name__ == "__main__":
    main()

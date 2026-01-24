"""
Class PgVectorClient:: Database CRUD, vector search, schema management
"""
import json
import logging
from typing import List, Dict, Optional, Any

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)


class PgVectorClient:
    """PostgreSQL database with pgvector for embeddings."""
    
    def __init__(self, connection_string: str, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        
        try:
            self.conn = psycopg2.connect(connection_string)
            self.conn.autocommit = True
            logger.info("Database connected")
        except Exception as e:
            raise RuntimeError(f"Database connection failed: {e}") from e
        
        self._create_extension()
        self._create_table()
    
    
    def _create_extension(self):
        """Enable pgvector extension."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("pgvector extension enabled")
        except Exception as e:
            logger.error(f"Failed to create extension: {e}")
            raise
    

    def _create_table(self):
        """Create documents table with vector index."""
        sql = f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR({self.embedding_dim}),
            source TEXT,
            chunk_id INTEGER,
            start_token INTEGER,
            end_token INTEGER,
            token_count INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_embedding 
        ON documents USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        
        CREATE INDEX IF NOT EXISTS idx_source_chunk
        ON documents(source, chunk_id);
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql)
            logger.info("Database schema created")
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise

    
    def insert_chunks(self, chunks: List[Dict]):
        """Insert chunks with embeddings (matches schema: source, chunk_id)."""
        with self.conn.cursor() as cur:
            for chunk in chunks:
                embedding = chunk['embedding']
                
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                embedding_str = f"[{','.join(map(str, embedding))}]"
                
                cur.execute("""
                    INSERT INTO documents 
                    (content, embedding, source, chunk_id, start_token, end_token, token_count)
                    VALUES (%s, %s::vector, %s, %s, %s, %s, %s)
                """, (
                    chunk['content'],
                    embedding_str,
                    chunk.get('source', 'unknown'),
                    chunk.get('chunk_id', 0),
                    chunk.get('start_token', 0),
                    chunk.get('end_token', 0),
                    chunk.get('token_count', 0)
                ))
        
        logger.info(f"Inserted {len(chunks)} chunks")


    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> list[dict]:
        """
        Perform a vector similarity search using cosine distance.

        Args:
            query_embedding: The embedding to search for (list of floats).
            k: Number of results to return.
            similarity_threshold: Minimum similarity score (0 to 1) to include in results.

        Returns:
            List of dictionaries, each containing document details and similarity score.
        """
        query_embedding_str = f"[{','.join(map(str, query_embedding))}]"

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        id,
                        content,
                        source,
                        chunk_id,
                        start_token,
                        end_token,
                        token_count,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM documents
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (query_embedding_str, query_embedding_str, similarity_threshold, k))

                results = cur.fetchall()
                logger.info(f"Found {len(results)} results above threshold {similarity_threshold}")
                
                # If no results, check top matches without threshold
                if not results:
                    logger.info("Checking top 3 matches without threshold...")
                    cur.execute("""
                        SELECT
                            source,
                            chunk_id,
                            1 - (embedding <=> %s::vector) AS similarity
                        FROM documents
                        ORDER BY similarity DESC
                        LIMIT 3
                    """, (query_embedding_str,))
                    top_matches = cur.fetchall()
                    for i, match in enumerate(top_matches):
                        logger.info(f"Top {i+1}: source={match[0]}, chunk={match[1]}, similarity={match[2]:.3f}")

                return [
                    {
                        "id": row[0],
                        "content": row[1],
                        "source": row[2],
                        "chunk_id": row[3],
                        "start_token": row[4],
                        "end_token": row[5],
                        "token_count": row[6],
                        "similarity": float(row[7]),
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    # def count_documents(self) -> int:
    #     """Get total number of chunks in database."""
    #     with self.conn.cursor() as cur:
    #         cur.execute("SELECT COUNT(*) FROM documents")
    #         return cur.fetchone()[0]
    
    # def get_sources(self) -> List[str]:
    #     """Get list of unique sources in database."""
    #     with self.conn.cursor() as cur:
    #         cur.execute("SELECT DISTINCT source FROM documents ORDER BY source")
    #         return [row[0] for row in cur.fetchall()]
    
    # def delete_by_source(self, source: str) -> int:
    #     """Delete all chunks from a specific source."""
    #     try:
    #         with self.conn.cursor() as cur:
    #             cur.execute("DELETE FROM documents WHERE source = %s", (source,))
    #             deleted = cur.rowcount
    #         logger.info(f"Deleted {deleted} chunks from {source}")
    #         return deleted
    #     except Exception as e:
    #         raise RuntimeError(f"Delete failed: {e}") from e
    
    # def clear_all(self):
    #     """Delete all documents from database."""
    #     try:
    #         with self.conn.cursor() as cur:
    #             cur.execute("TRUNCATE TABLE documents")
    #         logger.warning("All documents cleared from database")
    #     except Exception as e:
    #         raise RuntimeError(f"Clear failed: {e}") from e
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")


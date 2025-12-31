"""
Class PgVectorClient:: Database CRUD, vector search, schema management
"""

import logging
from typing import List, Dict, Optional

import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)


class PgVectorClient:
    """PostgreSQL database with pgvector for embeddings."""
    
    def __init__(self, connection_string: str, embedding_dim: int = 1024):
        """
        Initialize database connection.
        
        Args:
            connection_string: PostgreSQL connection string
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        
        try:
            self.conn = psycopg2.connect(connection_string)
            self.cursor = self.conn.cursor()
            logger.info("Database connected")
        except Exception as e:
            raise RuntimeError(f"Database connection failed: {e}") from e
        
        self._create_extension()
        self._create_table()
    

    def _create_extension(self):
        """Enable pgvector extension. Enable extension to work with vector data."""
        try:
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.conn.commit()
            logger.info("pgvector extension enabled")
        except Exception as e:
            self.conn.rollback()
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
            # changes exist only in the transaction buffer and are rolled back when the connection closes
            self.cursor.execute(sql) # Change executed but not saved
            
            # Writes all pending changes to disk
            self.conn.commit()
            logger.info("Database schema created")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create table: {e}")
            raise
            def insert_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        Batch insert chunks with embeddings.
        
        Args:
            chunks: List of chunk dicts (from TextProcessor)
            embeddings: List of embedding vectors (from HuggingFaceClient)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch")
        
        sql = """
        INSERT INTO documents 
        (content, embedding, source, chunk_id, start_token, end_token, token_count)
        VALUES %s
        """
        
        data = [
            (
                chunk["content"],
                embedding,
                chunk.get("source", "unknown"),
                chunk.get("chunk_id", 0),
                chunk.get("start_token", 0),
                chunk.get("end_token", 0),
                chunk.get("token_count", 0)
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        
        try:
            execute_values(self.cursor, sql, data)
            self.conn.commit()
            logger.info(f"Inserted {len(data)} chunks")
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Insert failed: {e}") from e
            self,
        query_embedding: List[float],
        k: int = 5,
        similarity_threshold: float = 0.0

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Vector similarity search.
        
        Args:
            query_embedding: Query vector (from HuggingFaceClient)
            k: Number of results to return
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of chunk dicts with similarity scores
        """
        sql = """
        SELECT 
            content,
            source,
            chunk_id,
            start_token,
            end_token,
            token_count,
            1 - (embedding <=> %s) as similarity
        FROM documents
        WHERE 1 - (embedding <=> %s) >= %s
        ORDER BY embedding <=> %s
        LIMIT %s
        """
        
        try:
            self.cursor.execute(
                sql,
                (query_embedding, query_embedding, similarity_threshold, query_embedding, k)
            )
            
            results = [
                {
                    "content": row[0],
                    "source": row[1],
                    "chunk_id": row[2],
                    "start_token": row[3],
                    "end_token": row[4],
                    "token_count": row[5],
                    "similarity": float(row[6])
                }
                for row in self.cursor.fetchall()
            ]
            
            logger.info(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    


    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Vector similarity search.

        Vector Similarity Mechanics:
        - <=> operator: pgvector cosine distance operator
        - cosine_distance range: [0, 2]
            - 0 = identical vectors (angle 0°)
            - 1 = orthogonal vectors (angle 90°)
            - 2 = opposite vectors (angle 180°)
        
        - similarity = 1 - cosine_distance
        - similarity range: [-1, 1]
            - 1.0 = identical (same direction)
            - 0.0 = orthogonal (perpendicular)
            - -1.0 = opposite (opposing direction)
        
        - WHERE clause: 1 - (embedding <=> %s) >= %s
            - Filters documents by minimum similarity threshold
            - Only returns chunks meeting quality standards
            - Example: threshold=0.7 keeps similarity >= 0.7

            
        Args:
            query_embedding: Query vector (from HuggingFaceClient)
            k: Number of results to return
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of chunk dicts with similarity scores
        """

        # (embedding <=> %s) - compare similarity to threshold.
        # similarity = 1 - cosine_distance
        # Range: [-1, 1]
        # # 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite

        sql = """
        SELECT 
            content,
            source,
            chunk_id,
            start_token,
            end_token,
            token_count,
            1 - (embedding <=> %s) as similarity
        FROM documents
        WHERE 1 - (embedding <=> %s) >= %s
        ORDER BY embedding <=> %s
        LIMIT %s
        """
        
        try:
            self.cursor.execute(
                sql,
                (query_embedding, query_embedding, similarity_threshold, query_embedding, k)
            )
            
            results = [
                {
                    "content": row[0],
                    "source": row[1],
                    "chunk_id": row[2],
                    "start_token": row[3],
                    "end_token": row[4],
                    "token_count": row[5],
                    "similarity": float(row[6])
                }
                for row in self.cursor.fetchall()
            ]
            
            logger.info(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    

    # def count_documents(self) -> int:
    #     """Get total number of chunks in database."""
    #     self.cursor.execute("SELECT COUNT(*) FROM documents")
    #     return self.cursor.fetchone()[0]
    

    # def get_sources(self) -> List[str]:
    #     """Get list of unique sources in database."""
    #     self.cursor.execute("SELECT DISTINCT source FROM documents ORDER BY source")
    #     return [row[0] for row in self.cursor.fetchall()]
    

    # def delete_by_source(self, source: str) -> int:
    #     """Delete all chunks from a specific source."""
    #     sql = "DELETE FROM documents WHERE source = %s"
        
    #     try:
    #         self.cursor.execute(sql, (source,))
    #         deleted = self.cursor.rowcount
    #         self.conn.commit()
    #         logger.info(f"Deleted {deleted} chunks from {source}")
    #         return deleted
    #     except Exception as e:
    #         self.conn.rollback()
    #         raise RuntimeError(f"Delete failed: {e}") from e
    

    # def clear_all(self):
    #     """Delete all documents from database."""
    #     try:
    #         self.cursor.execute("TRUNCATE TABLE documents")
    #         self.conn.commit()
    #         logger.warning("All documents cleared from database")
    #     except Exception as e:
    #         self.conn.rollback()
    #         raise RuntimeError(f"Clear failed: {e}") from e
    
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")

"""
Class PgVectorClient:: Database CRUD, vector search, schema management with connection pooling
"""
import json
import logging
from contextlib import contextmanager
from typing import List, Dict, Optional, Any

import numpy as np
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)


class PgVectorClient:
    """PostgreSQL database with pgvector for embeddings and connection pooling."""
    
    def __init__(
        self, 
        connection_string: str, 
        embedding_dim: int,
        min_connections: int = 2,
        max_connections: int = 10
    ):
        """
        Initialize database client with connection pooling.
        
        Args:
            connection_string: PostgreSQL connection string
            embedding_dim: Width of the encoder's vectors, from HuggingFaceClient
            min_connections: Minimum connections to keep in pool
            max_connections: Maximum connections allowed in pool
        """
        self.embedding_dim = embedding_dim
        self.connection_string = connection_string
        
        try:
            # Create threaded connection pool
            self.pool = pool.ThreadedConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                dsn=connection_string
            )
            logger.info(f"Database pool created: {min_connections}-{max_connections} connections")
        except Exception as e:
            raise RuntimeError(f"Database pool creation failed: {e}") from e
        
        self._create_extension()
        self._create_table()
        self._verify_embedding_dim()
    
    @contextmanager
    def get_connection(self):
        """
        Context manager to get connection from pool.
        
        Yields:
            Database connection
            
        Example:
            with client.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM documents")
        """
        conn = self.pool.getconn()
        try:
            conn.autocommit = True
            yield conn
        finally:
            self.pool.putconn(conn)
    
    
    def _create_extension(self):
        """Enable pgvector extension."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
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

        CREATE TABLE IF NOT EXISTS query_logs (
            id SERIAL PRIMARY KEY,
            user_ip VARCHAR(45) NOT NULL,
            question TEXT NOT NULL,
            answer TEXT,
            execution_time FLOAT,
            num_chunks INTEGER,
            sources TEXT[],
            created_at TIMESTAMP DEFAULT NOW(),
            status VARCHAR(20) DEFAULT 'success'
        );

        CREATE INDEX IF NOT EXISTS idx_user_ip_date 
        ON query_logs(user_ip, DATE(created_at));
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
            logger.info("Database schema created")
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise

    def _verify_embedding_dim(self):
        """Fail startup when the encoder's width disagrees with the stored column."""
        # CREATE TABLE IF NOT EXISTS silently keeps the old width, so a change of
        # EMBEDDING_MODEL would otherwise surface as a query-time pgvector error.
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT atttypmod
                    FROM pg_attribute
                    WHERE attrelid = 'documents'::regclass
                      AND attname = 'embedding'
                      AND NOT attisdropped
                """)
                row = cur.fetchone()

        if row is None:
            raise RuntimeError("Table 'documents' has no 'embedding' column")

        table_dim = row[0]
        if table_dim != self.embedding_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: encoder produces {self.embedding_dim}, "
                f"table 'documents' stores VECTOR({table_dim}). The embedding model "
                f"changed since ingestion. Drop the table and re-ingest, or restore "
                f"the previous EMBEDDING_MODEL."
            )

        logger.info(f"Embedding dimension verified: {table_dim}")

    
    def insert_chunks(self, chunks: List[Dict]):
        """Insert chunks with embeddings (matches schema: source, chunk_id)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
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
            with self.get_connection() as conn:
                with conn.cursor() as cur:
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
            # Never return [] here: the caller reads an empty result as "nothing
            # relevant found" and deflects, turning a database fault into a
            # plausible-looking answer.
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Vector search failed: {e}") from e

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
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
        logger.info("Database connection pool closed")
    
    @property
    def conn(self):
        """
        Deprecated property for backward compatibility.
        Returns a connection from the pool (should be used with caution).
        """
        logger.warning("Direct conn access is deprecated, use get_connection() context manager")
        return self.pool.getconn()

    
    def log_query(
        self, 
        user_ip: str, 
        question: str, 
        answer: str,
        execution_time: float,
        num_chunks: int,
        sources: List[str],
        status: str = "success"
    ):
        """Log query to database."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO query_logs 
                    (user_ip, question, answer, execution_time, num_chunks, sources, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (user_ip, question, answer, execution_time, num_chunks, sources, status))


    def get_daily_query_count(self, user_ip: str) -> int:
        """Get number of queries from IP today."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM query_logs 
                    WHERE user_ip = %s 
                    AND DATE(created_at) = CURRENT_DATE
                """, (user_ip,))
                return cur.fetchone()[0]


    def cleanup_old_logs(self, retention_days: int = 90):
        """Delete query logs older than retention period."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM query_logs 
                    WHERE created_at < NOW() - INTERVAL '%s days'
                """, (retention_days,))
                deleted = cur.rowcount
        logger.info(f"Deleted {deleted} old query logs")
        return deleted


    def get_user_data(self, user_ip: str) -> List[Dict]:
        """Retrieve all data for a specific IP (GDPR access request)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, question, answer, execution_time, num_chunks, 
                        sources, created_at, status
                    FROM query_logs 
                    WHERE user_ip = %s
                    ORDER BY created_at DESC
                """, (user_ip,))
                
                results = cur.fetchall()
                return [
                    {
                        "id": row[0],
                        "question": row[1],
                        "answer": row[2],
                        "execution_time": row[3],
                        "num_chunks": row[4],
                        "sources": row[5],
                        "created_at": row[6].isoformat(),
                        "status": row[7]
                    }
                    for row in results
                ]

    def delete_user_data(self, user_ip: str) -> int:
        """Delete all data for a specific IP (GDPR erasure request)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM query_logs 
                    WHERE user_ip = %s
                """, (user_ip,))
                deleted = cur.rowcount
        logger.info(f"Deleted {deleted} records for IP {user_ip} (GDPR erasure)")
        return deleted
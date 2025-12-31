"""
Class TextProcessor: chunking, tokenization, context assembly.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text chunking and context assembly for RAG systems."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_context_tokens: int = 2000,
        model_name: str = "gpt-3.5-turbo"
    ):
        """Initialize text processor with chunking parameters.

        Args:
            chunk_size: Target tokens per chunk
            chunk_overlap: Overlapping tokens between chunks
            max_context_tokens: Maximum tokens for assembled context
            model_name: Model name for tokenizer selection
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_context_tokens = max_context_tokens

        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.warning(
                f"Model {model_name} not found, using cl100k_base"
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using model tokenizer."""
        return len(self.tokenizer.encode(text)) if text else 0

    def chunk_text(
        self,
        text: str,
        source: str = "unknown"
    ) -> List[Dict]:
        """Split text into overlapping chunks with metadata.

        Args:
            text: Input text to chunk
            source: Source identifier for provenance tracking

        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text or not text.strip():
            return []

        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            chunks.append({
                "content": self.tokenizer.decode(chunk_tokens),
                "chunk_id": chunk_id,
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": end,
                "source": source
            })

            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1

        return chunks

    def chunk_file(self, file_path: str) -> List[Dict]:
        """Read and chunk a single text file.

        Args:
            file_path: Path to text file

        Returns:
            List of chunks from file

        Raises:
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = self.chunk_text(text, source=path.name)
        logger.info(f"Chunked {path.name}: {len(chunks)} chunks")

        return chunks


    def assemble_context(
        self,
        chunks: List[Dict],
        question: str,
        system_prompt_tokens: int = 150,
        answer_reserve_tokens: int = 500
    ) -> str:
        """Assemble chunks into context within token budget.

        Args:
            chunks: Retrieved chunks with similarity scores
            question: User question for token budget calculation
            system_prompt_tokens: Reserved tokens for system prompt
            answer_reserve_tokens: Reserved tokens for model answer

        Returns:
            Formatted context string within token budget
        """
        if not chunks:
            return ""

        # Calculate available token budget
        question_tokens = self.count_tokens(question)
        reserved = (
            question_tokens + system_prompt_tokens + answer_reserve_tokens
        )
        available = max(self.max_context_tokens - reserved, 500)

        # Assemble chunks within budget
        parts = []
        used_tokens = 0

        for chunk in chunks:
            chunk_tokens = chunk.get("token_count", 0)

            if used_tokens + chunk_tokens > available:
                if not parts:
                    # Always include first chunk, truncated if necessary
                    remaining = available - used_tokens
                    content = self._truncate(chunk["content"], remaining)
                    parts.append(self._format_chunk(chunk, content))
                break

            parts.append(self._format_chunk(chunk, chunk["content"]))
            used_tokens += chunk_tokens

        context = "\n\n---\n\n".join(parts)
        logger.info(
            f"Context: {used_tokens}/{available} tokens, "
            f"{len(parts)}/{len(chunks)} chunks"
        )

        return context
    

    def _format_chunk(self, chunk: Dict, content: str) -> str:
        """Format chunk with metadata header."""
        source = chunk.get("source", "unknown")
        chunk_id = chunk.get("chunk_id", 0)
        similarity = chunk.get("similarity", 0)

        header = f"[{source} | Chunk {chunk_id}"
        if similarity > 0:
            header += f" | Similarity {similarity:.3f}"
        header += "]"

        return f"{header}\n{content}"


    def _truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to token limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens]) + "..."

    
    # def chunk_directory(self, directory: str, pattern: str = "*.txt") -> List[Dict]:
    #     """Chunk all files in directory matching pattern."""
    #     path = Path(directory)
        
    #     if not path.exists():
    #         raise FileNotFoundError(f"Directory not found: {directory}")
        
    #     all_chunks = []
    #     files = list(path.glob(pattern))
        
    #     for file_path in files:
    #         try:
    #             chunks = self.chunk_file(str(file_path))
    #             all_chunks.extend(chunks)
    #         except Exception as e:
    #             logger.error(f"Failed to chunk {file_path.name}: {e}")
        
    #     logger.info(f"Total: {len(files)} files, {len(all_chunks)} chunks")
    #     return all_chunks
    


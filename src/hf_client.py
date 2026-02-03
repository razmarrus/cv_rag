"""
HuggingFaceClient: Embeddings generation, LLM inference
"""
import gc
from contextlib import contextmanager
from typing import List, Optional
from huggingface_hub import InferenceClient
import logging

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Client for Hugging Face Inference API."""
    
    def __init__(
        self,
        hf_token: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_local_embeddings: bool = False
    ):
        """
        Initialize Hugging Face client.
        
        Args:
            hf_token: Hugging Face API token
            embedding_model: Model for embeddings
            llm_model: Model for text generation
            use_local_embeddings: Use local sentence-transformers model instead of API
        """
        self.hf_token = hf_token
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.use_local_embeddings = use_local_embeddings
        self.local_embedding_model = None
        
        # Load local embedding model if requested
        if use_local_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.local_embedding_model = SentenceTransformer(embedding_model)
            except (ImportError, Exception) as e:
                logger.warning(f"Local embeddings unavailable ({type(e).__name__}), using remote API")
                self.use_local_embeddings = False
        
        # Initialize remote clients
        # Note: Modern huggingface-hub versions auto-route to correct endpoint
        self.embedding_client = InferenceClient(
            model=embedding_model, 
            token=hf_token
        )
        self.llm_client = InferenceClient(
            model=llm_model, 
            token=hf_token
        )
        
        embedding_mode = "LOCAL" if self.use_local_embeddings else "REMOTE"
        logger.info(f"Embeddings: {embedding_mode} | LLM: {llm_model}")


    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using local model or remote API.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors (as Python lists)
        """
        # Try local model first if enabled
        if self.use_local_embeddings and self.local_embedding_model is not None:
            try:
                logger.debug(f"Generating {len(texts)} embeddings using LOCAL model")
                embeddings = self.local_embedding_model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Convert numpy arrays to Python lists
                if hasattr(embeddings, 'tolist'):
                    embeddings = embeddings.tolist()
                
                logger.info(f"Generated {len(embeddings)} embeddings (LOCAL)")
                return embeddings
                
            except Exception as e:
                logger.warning(f"Local embedding generation failed: {e}")
                logger.info("Falling back to remote API")
        
        # Use remote API (either by default or as fallback)
        try:
            logger.debug(f"Generating {len(texts)} embeddings using REMOTE API")
            embeddings = self.embedding_client.feature_extraction(texts)
            
            # Convert numpy arrays to Python lists
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            elif isinstance(embeddings, list) and len(embeddings) > 0:
                if hasattr(embeddings[0], 'tolist'):
                    embeddings = [emb.tolist() for emb in embeddings]
            
            logger.info(f"Generated {len(embeddings)} embeddings (REMOTE)")
            return embeddings
            
        except Exception as e:
            logger.error(f"Remote embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    @contextmanager
    def embedding_context(self, text: str):
        """
        Context manager for embedding generation with automatic cleanup.
        
        Args:
            text: Single text to embed
            
        Yields:
            Single embedding vector (as Python list)
            
        Example:
            with hf_client.embedding_context(question) as query_embedding:
                chunks = db.search(query_embedding)
        """
        embedding_list = None
        query_embedding = None
        
        try:
            embedding_list = self.get_embeddings([text])
            query_embedding = embedding_list[0]
            
            if not isinstance(query_embedding, list):
                query_embedding = query_embedding.tolist()
            
            yield query_embedding
            
        finally:
            del query_embedding
            del embedding_list
            gc.collect()

    def build_prompt(self, question: str, context: str) -> str:
        """
        Build prompt for LLM.
        
        Args:
            question: User question
            context: Retrieved context
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""<s>[INST] You are a helpful assistant. Answer the question based on the provided context. Answer in human written style. Keep friendly and easy to read tone.

Context:
{context}

Question: {question}

Answer based only on the context provided. If the answer is not in the context, say so. [/INST]
"""
        return prompt

    def generate_answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 500,
        temperature: float = 0.2
    ) -> str:
        """
        Generate answer using LLM.
        
        Args:
            question: User question
            context: Context string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated answer text
        """
        prompt = self.build_prompt(question, context)
        
        try:
            # Use chat_completion for new router endpoint
            messages = [
            {
                "role": "user",
                "content": prompt
            }]

            response = self.llm_client.chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
                        
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise RuntimeError(f"Failed to generate answer: {e}") from e

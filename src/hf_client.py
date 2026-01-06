"""
HuggingFaceClient: Embeddings generation, LLM inference
"""
from typing import List
from huggingface_hub import InferenceClient
import logging

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Client for Hugging Face Inference API."""
    
    def __init__(
        self,
        hf_token: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    ):
        """
        Initialize Hugging Face client.
        
        Args:
            hf_token: Hugging Face API token
            embedding_model: Model for embeddings
            llm_model: Model for text generation
        """
        self.hf_token = hf_token
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        self.embedding_client = InferenceClient(
            model=embedding_model,
            token=hf_token
        )

        # llm_client=InferenceClient(
        #             model="mistralai/Mistral-7B-Instruct-v0.2",
        #             token=HF_TOKEN
        #         )

        
        self.llm_client = InferenceClient(
            model=llm_model,
            token=hf_token
        )
        
        logger.info(f"Initialized embedding model: {embedding_model}")
        logger.info(f"Initialized LLM: {llm_model}")


    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors (as Python lists)
        """

        try:
            embeddings = self.embedding_client.feature_extraction(texts)
            
            # Convert numpy arrays to Python lists
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            elif isinstance(embeddings, list) and len(embeddings) > 0:
                if hasattr(embeddings[0], 'tolist'):
                    embeddings = [emb.tolist() for emb in embeddings]
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e


    def build_prompt(self, question: str, context: str) -> str:
        """
        Build prompt for LLM.
        
        Args:
            question: User question
            context: Retrieved context
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""<s>[INST] You are a helpful assistant. Answer the question based on the provided context.

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
        max_new_tokens: int = 512,
        temperature: float = 0.7
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
            # response = self.llm_client.text_generation(
            #     prompt,
            #     max_new_tokens=max_new_tokens,
            #     temperature=temperature,
            #     do_sample=True,
            #     top_p=0.9,
            #     return_full_text=False
            # )
            messages = [
            {
                "role": "user",
                "content": prompt #f"Based on this context:\n\n{context_text}\n\nAnswer: {llm_question}?"
            }]

            response = self.llm_client.chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
                        
            answer = response.choices[0].message.content.strip() #strip()
            logger.info(f"Generated answer ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise RuntimeError(f"Failed to generate answer: {e}") from e



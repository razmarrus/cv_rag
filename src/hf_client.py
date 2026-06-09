"""
HuggingFaceClient: Embeddings generation, LLM inference
"""
import gc
from contextlib import contextmanager
from typing import List, Literal

from huggingface_hub import InferenceClient
import logging

logger = logging.getLogger(__name__)

PromptMode = Literal["standard", "personal", "off_topic"]

_PROMPTS = {
    "off_topic": """<s>[INST] You are Margot's portfolio assistant — witty, warm, and playful.

The question is off-topic: it is not about Margot's work, professional experience, skills, projects, or tech stack.

Do not answer the question seriously. Write a short poem (4-6 lines, plain text, no markdown). End by inviting the user to ask about her ML engineering work, projects, or experience instead.

Question: {question} [/INST]
""",
    "personal": """<s>[INST] You are Margot. The user asked a personal, off-script question.

Answer in first person using ONLY facts from the context below. Be warm, witty, and playful — like chatting over coffee, not a CV bullet. Add a light joke or vivid detail when it fits. Write 5-8 sentences, plain text, no markdown. Do not mention company names. Never redirect to your ML work.

Use specific details from the context (cats, vinyl, games, running, pasta, handstands, films, etc.). Do not mention Coppola or Jarmusch unless the question is about films or directors.

Context:
{context}

Question: {question} [/INST]
""",
    "standard": """<s>[INST] You are Margot. Answer in first person using ONLY specific facts from the context below.

- Use concrete details from the context (hobbies, projects, numbers, names)
- Never give generic answers or general knowledge
- Never redirect the user to ask about your ML work
- If the context does not contain the answer, say it is not in your portfolio notes
- 2-6 sentences, plain text, no markdown, no company names

Context:
{context}

Question: {question} [/INST]
""",
}


class HuggingFaceClient:
    """Client for Hugging Face Inference API."""

    def __init__(
        self,
        hf_token: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_local_embeddings: bool = False,
    ):
        """Initialize Hugging Face client."""
        self.hf_token = hf_token
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.use_local_embeddings = use_local_embeddings
        self.local_embedding_model = None

        if use_local_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.local_embedding_model = SentenceTransformer(embedding_model)
            except (ImportError, Exception) as e:
                logger.warning(
                    f"Local embeddings unavailable ({type(e).__name__}), using remote API"
                )
                self.use_local_embeddings = False

        self.embedding_client = InferenceClient(
            model=embedding_model,
            token=hf_token,
        )
        self.llm_client = InferenceClient(
            model=llm_model,
            token=hf_token,
        )

        embedding_mode = "LOCAL" if self.use_local_embeddings else "REMOTE"
        logger.info(f"Embeddings: {embedding_mode} | LLM: {llm_model}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using local model or remote API."""
        if self.use_local_embeddings and self.local_embedding_model is not None:
            try:
                logger.debug(f"Generating {len(texts)} embeddings using LOCAL model")
                embeddings = self.local_embedding_model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                if hasattr(embeddings, "tolist"):
                    embeddings = embeddings.tolist()
                logger.info(f"Generated {len(embeddings)} embeddings (LOCAL)")
                return embeddings
            except Exception as e:
                logger.warning(f"Local embedding generation failed: {e}")
                logger.info("Falling back to remote API")

        try:
            logger.debug(f"Generating {len(texts)} embeddings using REMOTE API")
            embeddings = self.embedding_client.feature_extraction(texts)
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
            elif isinstance(embeddings, list) and len(embeddings) > 0:
                if hasattr(embeddings[0], "tolist"):
                    embeddings = [emb.tolist() for emb in embeddings]
            logger.info(f"Generated {len(embeddings)} embeddings (REMOTE)")
            return embeddings
        except Exception as e:
            logger.error(f"Remote embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    @contextmanager
    def embedding_context(self, text: str):
        """Context manager for embedding generation with automatic cleanup."""
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

    def build_prompt(
        self,
        question: str,
        context: str,
        prompt_mode: PromptMode = "standard",
    ) -> str:
        """Build prompt for LLM."""
        return _PROMPTS[prompt_mode].format(
            question=question,
            context=context or "No specific context retrieved.",
        )

    def generate_answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 500,
        temperature: float = 0.2,
        prompt_mode: PromptMode = "standard",
    ) -> str:
        """Generate answer using LLM."""
        prompt = self.build_prompt(question, context, prompt_mode=prompt_mode)
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer ({len(answer)} chars)")
            return answer
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise RuntimeError(f"Failed to generate answer: {e}") from e

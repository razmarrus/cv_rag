"""
HuggingFaceClient: Embeddings generation, LLM inference
"""
import gc
from contextlib import contextmanager
from typing import List, Literal

from huggingface_hub import InferenceClient
import logging

logger = logging.getLogger(__name__)

PromptMode = Literal["standard", "personal", "deflect", "deflect_poetry"]


def _deflect_rule(include_contact: bool) -> str:
    """Fallback rule when retrieved context does not answer the question."""
    if include_contact:
        return (
            "If the context is empty or does not contain the answer, do NOT answer the question "
            "or use outside knowledge. Either say warmly their question is fun and they can email "
            "Margot at margo.razumeyeva@gmail.com, OR one short joke on the theme, then suggest "
            "finding Margot on LinkedIn (say LinkedIn only — never paste a URL). "
            "2-4 sentences, plain text."
        )
    return (
        "If the context is empty or does not contain the answer, do NOT answer the question "
        "or use outside knowledge. Say warmly that is not in your portfolio notes, OR one short "
        "playful line on the theme. Do not suggest contacting Margot, email, or LinkedIn. "
        "2-4 sentences, plain text."
    )


def _build_deflect_prompt(question: str, poetry: bool, include_contact: bool) -> str:
    """Prompt for questions with no matching portfolio documents."""
    if poetry:
        if include_contact:
            closing = (
                "Always finish with one plain sentence inviting them to email Margot at "
                "margo.razumeyeva@gmail.com or find her on LinkedIn (say LinkedIn only — never paste a URL)."
            )
        else:
            closing = (
                "Finish with one plain sentence that you do not have that in Margot's portfolio notes. "
                "Do not suggest contacting Margot."
            )
        return f"""<s>[INST] You are Margot's portfolio assistant — calm, warm, lightly poetic.

The question is not in Margot's portfolio documents — unrelated to her work, experience, or what is documented about her.

Do NOT answer the question. Do not use general knowledge.

Write a short calm poem (4-6 lines, plain text, no markdown) inspired by the question's theme only — playful, not a real answer. {closing}

Question: {question} [/INST]
"""

    if include_contact:
        body = (
            "Either: say warmly their question is fun and they can ask Margot at margo.razumeyeva@gmail.com\n"
            "OR: one short playful joke on the question's theme, then suggest finding Margot on LinkedIn "
            "(say LinkedIn only — never paste a URL)"
        )
    else:
        body = (
            "Say warmly that you do not have that in Margot's portfolio notes, "
            "OR one short playful joke on the question's theme. "
            "Do not suggest contacting Margot, email, or LinkedIn."
        )

    return f"""<s>[INST] You are Margot's portfolio assistant.

The question is not in Margot's portfolio documents — unrelated to her work, experience, or what is documented about her.

Do NOT answer the question. Do not use general knowledge.

{body}

Plain text, 2-4 sentences.

Question: {question} [/INST]
"""


_STANDARD_TEMPLATE = """<s>[INST] You are Margot. Answer in first person using ONLY specific facts from the context below.

- Use concrete details from the context (hobbies, projects, numbers, names)
- Never give generic answers or general knowledge
- {deflect_rule}
- 2-6 sentences when answering from context, plain text, no markdown, no company names

Context:
{context}

Question: {question} [/INST]
"""

_PERSONAL_TEMPLATE = """<s>[INST] You are Margot. The user asked a personal, off-script question.

Answer in first person using the context below — but lightly. Pick only one or two details that fit the question. Do not pile on facts or list everything you know. Stay calm and unhurried.

Often add a short poetic touch: a line of verse, a metaphor, or a gentle mini-poem mixed with plain sentences. Vary the style; not every answer needs poetry, but use it regularly.

About 5-7 sentences total when answering from context, plain text, no markdown. Warm, a little witty, never breathless. Do not mention company names.

{deflect_rule}

Do not mention Coppola or Jarmusch unless the question is about films or directors.

Context:
{context}

Question: {question} [/INST]
"""

_CONTEXT_TEMPLATES = {
    "standard": _STANDARD_TEMPLATE,
    "personal": _PERSONAL_TEMPLATE,
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
        is_preset: bool = False,
    ) -> str:
        """Build prompt for LLM."""
        include_contact = not is_preset
        if prompt_mode == "deflect":
            return _build_deflect_prompt(question, poetry=False, include_contact=include_contact)
        if prompt_mode == "deflect_poetry":
            return _build_deflect_prompt(question, poetry=True, include_contact=include_contact)

        template = _CONTEXT_TEMPLATES[prompt_mode]
        return template.format(
            question=question,
            context=context or "No specific context retrieved.",
            deflect_rule=_deflect_rule(include_contact),
        )

    def generate_answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 500,
        temperature: float = 0.2,
        prompt_mode: PromptMode = "standard",
        is_preset: bool = False,
    ) -> str:
        """Generate answer using LLM."""
        prompt = self.build_prompt(
            question,
            context,
            prompt_mode=prompt_mode,
            is_preset=is_preset,
        )
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

"""
HuggingFaceClient: Embeddings generation, LLM inference
"""
import gc
from contextlib import contextmanager
from typing import List, Literal

from huggingface_hub import InferenceClient
import logging

logger = logging.getLogger(__name__)

PromptMode = Literal["standard", "personal", "deflect"]


def _deflect_rule(include_contact: bool, off_script: bool = False) -> str:
    """Fallback rule when retrieved context does not answer the question."""
    if off_script:
        return (
            "If the context does not contain the answer, write a short calm poem "
            "(4-6 lines) inspired by the theme only — playful, not a real answer. "
            "No disclaimer, no contact info."
        )
    if include_contact:
        return (
            "If the context is empty or does not contain the answer, do NOT answer the question "
            "or use outside knowledge. Write a short calm poem (4-6 lines, plain text, no markdown) "
            "inspired by the theme only — playful, not a real answer. "
            "Finish with one plain sentence inviting them to email Margot at margo.razumeyeva@gmail.com "
            "or find her on LinkedIn (say LinkedIn only — never paste a URL)."
        )
    return (
        "If the context is empty or does not contain the answer, do NOT answer the question "
        "or use outside knowledge. Write a short calm poem (4-6 lines, plain text, no markdown) "
        "inspired by the theme only — playful, not a real answer. "
        "Finish with one plain sentence that you do not have that in Margot's portfolio notes. "
        "Do not suggest contacting Margot, email, or LinkedIn."
    )


def _build_deflect_prompt(question: str, include_contact: bool, off_script: bool = False) -> str:
    """Prompt for questions with no matching portfolio documents."""
    if off_script:
        closing = "No disclaimer or contact information — just the poem."
    elif include_contact:
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

Answer in first person using the context below — but lightly. Pick only one detail that fits. Do not list hobbies or pile on facts.

Always answer as a short calm poem: 4-6 lines, plain text, no markdown. Warm and a little witty. Weave in one fact from the context. Do not mention company names.

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


def _describe_error(exc: Exception) -> str:
    """Error description including HTTP status when the exception carries one."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    label = type(exc).__name__ if status is None else f"{type(exc).__name__} HTTP {status}"
    return f"{label}: {exc}"


class HuggingFaceClient:
    """Client for Hugging Face Inference Providers."""

    def __init__(
        self,
        hf_token: str,
        embedding_model: str,
        llm_model: str,
        use_local_embeddings: bool,
        provider: str,
    ):
        """Initialize Hugging Face client."""
        self.hf_token = hf_token
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.use_local_embeddings = use_local_embeddings
        self.provider = provider
        self.local_embedding_model = None
        self.embedding_client = None

        # One encoder, chosen by config. Never both: local and remote pooling
        # differ, so switching at runtime would mix incompatible vector spaces.
        if use_local_embeddings:
            from sentence_transformers import SentenceTransformer
            self.local_embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.local_embedding_model.get_embedding_dimension()
        else:
            # feature-extraction is served by hf-inference, not the chat providers.
            self.embedding_client = InferenceClient(
                model=embedding_model,
                api_key=hf_token,
                provider="hf-inference",
            )
            # The remote model exposes no metadata endpoint for width, so pay for
            # one probe call rather than trusting a configured number.
            self.embedding_dim = len(self.get_embeddings(["dimension probe"])[0])

        self.llm_client = InferenceClient(
            model=llm_model,
            api_key=hf_token,
            provider=provider,
        )

        embedding_mode = "LOCAL" if use_local_embeddings else "REMOTE"
        logger.info(
            f"Embeddings: {embedding_mode} ({embedding_model}, dim={self.embedding_dim}) "
            f"| LLM: {llm_model} | provider: {provider}"
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the configured encoder."""
        try:
            if self.use_local_embeddings:
                embeddings = self.local_embedding_model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            else:
                embeddings = self.embedding_client.feature_extraction(texts)

            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
            elif isinstance(embeddings, list) and embeddings:
                if hasattr(embeddings[0], "tolist"):
                    embeddings = [emb.tolist() for emb in embeddings]

            mode = "LOCAL" if self.use_local_embeddings else "REMOTE"
            logger.info(f"Generated {len(embeddings)} embeddings ({mode})")
            return embeddings
        except Exception as e:
            detail = _describe_error(e)
            logger.error(f"Embedding generation failed: {detail}")
            raise RuntimeError(f"Failed to generate embeddings: {detail}") from e

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
        include_contact: bool = True,
        off_script: bool = False,
    ) -> str:
        """Build prompt for LLM."""
        if prompt_mode == "deflect":
            return _build_deflect_prompt(
                question, include_contact=include_contact, off_script=off_script
            )

        template = _CONTEXT_TEMPLATES[prompt_mode]
        return template.format(
            question=question,
            context=context or "No specific context retrieved.",
            deflect_rule=_deflect_rule(include_contact, off_script=off_script),
        )

    def _is_reasoning_model(self) -> bool:
        """True for models that spend tokens on internal reasoning (e.g. gpt-oss)."""
        return "gpt-oss" in self.llm_model.lower()

    def _chat_completion(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        reasoning_effort: str | None = None,
    ):
        """Call chat_completion, passing reasoning_effort when supported."""
        kwargs: dict = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        return self.llm_client.chat_completion(**kwargs)

    def generate_answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 500,
        temperature: float = 0.2,
        prompt_mode: PromptMode = "standard",
        include_contact: bool = True,
        off_script: bool = False,
        reasoning_effort: str | None = None,
    ) -> str:
        """Generate answer using LLM."""
        from config.config import Config

        prompt = self.build_prompt(
            question,
            context,
            prompt_mode=prompt_mode,
            include_contact=include_contact,
            off_script=off_script,
        )
        if reasoning_effort is None and self._is_reasoning_model():
            reasoning_effort = Config.REASONING_EFFORT

        attempts = [
            (max_new_tokens, reasoning_effort),
            (max_new_tokens * 2, "none"),
        ]
        last_error: RuntimeError | None = None

        for attempt_idx, (tokens, effort) in enumerate(attempts):
            try:
                response = self._chat_completion(
                    prompt, tokens, temperature, reasoning_effort=effort
                )
            except Exception as e:
                detail = _describe_error(e)
                logger.error(
                    f"Answer generation failed (model={self.llm_model}, "
                    f"provider={self.provider}): {detail}"
                )
                raise RuntimeError(f"Failed to generate answer: {detail}") from e

            choice = response.choices[0]
            answer = (choice.message.content or "").strip()
            if answer:
                if attempt_idx:
                    logger.info(
                        f"Retry succeeded (tokens={tokens}, reasoning_effort={effort})"
                    )
                logger.info(f"Generated answer ({len(answer)} chars)")
                return answer

            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(response, "usage", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            last_error = RuntimeError(
                f"LLM returned an empty answer (finish_reason={finish_reason}, "
                f"completion_tokens={completion_tokens}/{tokens})"
            )
            if attempt_idx == 0:
                logger.warning(
                    f"Empty completion (model={self.llm_model}, "
                    f"finish_reason={finish_reason}, "
                    f"completion_tokens={completion_tokens}/{tokens}) — retrying"
                )

        logger.error(
            f"Empty completion after retry (model={self.llm_model}, "
            f"provider={self.provider})"
        )
        raise last_error

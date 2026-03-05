"""LLM client using Groq for chat/structured output and Gemini for embeddings.

Provides an LLMClient class with:
    - chat_completion() — chat completions via Groq (Llama 3.3 70B).
    - stream_chat_completion() — streaming chat via Groq returning async generator.
    - structured_output() — completions parsed into a Pydantic model via JSON mode.
    - embed_text() / embed_texts() — text embedding via Gemini gemini-embedding-001.
    - Automatic retries (tenacity) and usage logging.

Usage::

    from app.services.llm_client import LLMClient

    client = LLMClient()
    reply = await client.chat_completion([{"role": "user", "content": "Hello"}])
    report = await client.structured_output(messages, LCIReportSchema)
    vec = await client.embed_text("some campaign description")
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import TypeVar

from google import genai
from groq import AsyncGroq
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings, settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# Retry decorator shared by API-calling methods
_retry = retry(
    retry=retry_if_exception_type((Exception,)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class LLMClient:
    """Async client using Groq for chat and Gemini for embeddings.

    Args:
        cfg: Application settings (defaults to the global singleton).
    """

    def __init__(
        self,
        cfg: Settings | None = None,
    ) -> None:
        self._cfg = cfg or settings
        self._groq_api_key = self._cfg.groq_api_key
        self._gemini = genai.Client(api_key=self._cfg.google_api_key)
        self._default_model = self._cfg.llm_model
        self._embedding_model = self._cfg.embedding_model
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0

    def _get_groq(self) -> AsyncGroq:
        """Create a fresh AsyncGroq client per call for serverless compatibility."""
        return AsyncGroq(api_key=self._groq_api_key)

    # ── Chat completion (Groq) ─────────────────────────────────────────

    @_retry
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Send a chat completion request via Groq and return the text reply.

        Args:
            messages: OpenAI-format message list.
            model: Override the default model.
            temperature: Sampling temperature.

        Returns:
            The assistant's response text.
        """
        model = model or self._default_model

        logger.info(
            "chat_completion | model=%s | messages=%d | temp=%.1f",
            model,
            len(messages),
            temperature,
        )

        response = await self._get_groq().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        reply = response.choices[0].message.content or ""

        # Token accounting
        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens or 0
            self._total_output_tokens += response.usage.completion_tokens or 0
            logger.info(
                "chat_completion done | prompt_tokens=%d | completion_tokens=%d",
                response.usage.prompt_tokens or 0,
                response.usage.completion_tokens or 0,
            )

        return reply

    # ── Streaming chat completion (Groq) ───────────────────────────────

    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion via Groq, yielding content chunks.

        Args:
            messages: OpenAI-format message list.
            model: Override the default model.
            temperature: Sampling temperature.

        Yields:
            Content string chunks from the assistant's response.
        """
        model = model or self._default_model

        logger.info(
            "stream_chat_completion | model=%s | temp=%.1f",
            model,
            temperature,
        )

        stream = await self._get_groq().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    # ── Structured output (Groq with JSON mode) ───────────────────────

    @_retry
    async def structured_output(
        self,
        messages: list[dict[str, str]],
        response_schema: type[T],
        model: str | None = None,
        temperature: float = 0.3,
    ) -> T:
        """Send a chat completion and parse the response into a Pydantic model.

        Uses Groq's JSON mode with schema instruction in the system message.

        Args:
            messages: OpenAI-format message list.
            response_schema: Pydantic model class to parse into.
            model: Override the default model.
            temperature: Sampling temperature (lower for structured output).

        Returns:
            An instance of response_schema populated from the LLM response.
        """
        model = model or self._default_model
        schema_name = response_schema.__name__

        logger.info(
            "structured_output | model=%s | schema=%s",
            model,
            schema_name,
        )

        # Add JSON schema instruction to messages
        json_schema = response_schema.model_json_schema()
        schema_prompt = (
            f"\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"{json.dumps(json_schema, indent=2)}"
        )

        augmented = list(messages)
        if augmented and augmented[0].get("role") == "system":
            augmented[0] = {
                "role": "system",
                "content": augmented[0]["content"] + schema_prompt,
            }
        else:
            augmented.insert(0, {"role": "system", "content": schema_prompt})

        response = await self._get_groq().chat.completions.create(
            model=model,
            messages=augmented,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        # Token accounting
        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens or 0
            self._total_output_tokens += response.usage.completion_tokens or 0

        raw = response.choices[0].message.content or "{}"
        return response_schema.model_validate(json.loads(raw))

    # ── Embeddings (Gemini) ────────────────────────────────────────────

    @_retry
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string using the Gemini embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats (3072 dimensions).
        """
        return (await self.embed_texts([text]))[0]

    @_retry
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts via the Gemini embedding API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text (3072 dimensions each).
        """
        model = self._embedding_model

        logger.info(
            "embed_texts | model=%s | n=%d",
            model,
            len(texts),
        )

        result = await self._gemini.aio.models.embed_content(
            model=model,
            contents=texts,
        )

        logger.info("embed_texts done | n=%d", len(texts))

        return [e.values for e in result.embeddings]

    # ── Helpers ────────────────────────────────────────────────────────

    @property
    def cumulative_stats(self) -> dict:
        """Return cumulative token and cost stats for this client instance.

        Returns:
            Dict with total_input_tokens, total_output_tokens, and total_cost.
        """
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": round(self._total_cost, 4),
        }


# ── Module-level singleton ────────────────────────────────────────────

_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return a module-level singleton LLMClient.

    Returns:
        The shared LLMClient instance.
    """
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client

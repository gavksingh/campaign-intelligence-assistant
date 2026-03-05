"""Wrapper around the Google Gemini SDK with structured output, embeddings, and retry logic.

Provides an LLMClient class with:
    - chat_completion() — standard chat completions returning text.
    - stream_chat_completion() — streaming chat returning async generator of chunks.
    - structured_output() — completions parsed into a Pydantic model via
      Gemini's response_mime_type="application/json" with response_schema.
    - embed_text() / embed_texts() — text embedding via Gemini text-embedding-004.
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
from google.genai import types
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


def _messages_to_contents(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[types.Content]]:
    """Convert OpenAI-format messages to Gemini contents + system instruction.

    Args:
        messages: OpenAI-format message list with 'role' and 'content'.

    Returns:
        Tuple of (system_instruction, contents list).
    """
    system_instruction = None
    contents: list[types.Content] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            # Gemini uses system_instruction separately
            if system_instruction is None:
                system_instruction = content
            else:
                system_instruction += "\n\n" + content
        elif role == "assistant":
            contents.append(
                types.Content(role="model", parts=[types.Part(text=content)])
            )
        else:
            # "user" and any other role
            contents.append(
                types.Content(role="user", parts=[types.Part(text=content)])
            )

    return system_instruction, contents


class LLMClient:
    """Async Google Gemini client with structured output, embeddings, and observability.

    Args:
        cfg: Application settings (defaults to the global singleton).
        genai_client: Optional pre-configured genai.Client for testing.
    """

    def __init__(
        self,
        cfg: Settings | None = None,
        genai_client: genai.Client | None = None,
    ) -> None:
        self._cfg = cfg or settings
        self._client = genai_client or genai.Client(api_key=self._cfg.google_api_key)
        self._default_model = self._cfg.llm_model
        self._embedding_model = self._cfg.embedding_model
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0

    # ── Chat completion ───────────────────────────────────────────────

    @_retry
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Send a chat completion request and return the assistant's text reply.

        Args:
            messages: OpenAI-format message list.
            model: Override the default model.
            temperature: Sampling temperature.

        Returns:
            The assistant's response text.
        """
        model = model or self._default_model
        system_instruction, contents = _messages_to_contents(messages)

        logger.info(
            "chat_completion | model=%s | messages=%d | temp=%.1f",
            model,
            len(contents),
            temperature,
        )

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            ),
        )

        reply = response.text or ""

        # Token accounting
        if response.usage_metadata:
            um = response.usage_metadata
            self._total_input_tokens += um.prompt_token_count or 0
            self._total_output_tokens += um.candidates_token_count or 0
            logger.info(
                "chat_completion done | prompt_tokens=%d | completion_tokens=%d",
                um.prompt_token_count or 0,
                um.candidates_token_count or 0,
            )

        return reply

    # ── Streaming chat completion ────────────────────────────────────

    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion, yielding content chunks as they arrive.

        Args:
            messages: OpenAI-format message list.
            model: Override the default model.
            temperature: Sampling temperature.

        Yields:
            Content string chunks from the assistant's response.
        """
        model = model or self._default_model
        system_instruction, contents = _messages_to_contents(messages)

        logger.info(
            "stream_chat_completion | model=%s | temp=%.1f",
            model,
            temperature,
        )

        async for chunk in self._client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            ),
        ):
            if chunk.text:
                yield chunk.text

    # ── Structured output ─────────────────────────────────────────────

    @_retry
    async def structured_output(
        self,
        messages: list[dict[str, str]],
        response_schema: type[T],
        model: str | None = None,
        temperature: float = 0.3,
    ) -> T:
        """Send a chat completion and parse the response into a Pydantic model.

        Uses Gemini's response_mime_type="application/json" with response_schema
        for structured JSON output.

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
        system_instruction, contents = _messages_to_contents(messages)

        logger.info(
            "structured_output | model=%s | schema=%s",
            model,
            schema_name,
        )

        # Add instruction to output valid JSON matching the schema
        json_schema = response_schema.model_json_schema()
        schema_prompt = (
            f"\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"{json.dumps(json_schema, indent=2)}"
        )
        if system_instruction:
            system_instruction += schema_prompt
        else:
            system_instruction = schema_prompt

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )

        # Token accounting
        if response.usage_metadata:
            um = response.usage_metadata
            self._total_input_tokens += um.prompt_token_count or 0
            self._total_output_tokens += um.candidates_token_count or 0

        raw = response.text or "{}"
        return response_schema.model_validate(json.loads(raw))

    # ── Embeddings ────────────────────────────────────────────────────

    @_retry
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string using the configured embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats (768 dimensions).
        """
        return (await self.embed_texts([text]))[0]

    @_retry
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts via the Gemini embedding API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text (768 dimensions each).
        """
        model = self._embedding_model

        logger.info(
            "embed_texts | model=%s | n=%d",
            model,
            len(texts),
        )

        result = await self._client.aio.models.embed_content(
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

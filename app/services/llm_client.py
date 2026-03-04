"""Wrapper around the OpenAI SDK with structured output, embeddings, and retry logic.

Provides an LLMClient class with:
    - chat_completion() — standard chat completions returning text.
    - structured_output() — completions parsed into a Pydantic model via
      OpenAI's response_format json_schema, with function-calling fallback.
    - embed_text() / embed_texts() — text embedding via OpenAI or local fallback.
    - Automatic retries (tenacity), token counting, and cost logging.

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
from typing import TypeVar

import tiktoken
from openai import AsyncOpenAI
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

# ── Approximate pricing per 1K tokens (as of early 2025) ──────────────
_COST_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a single API call.

    Args:
        model: The model name used.
        input_tokens: Number of prompt tokens.
        output_tokens: Number of completion tokens.

    Returns:
        Estimated cost in USD.
    """
    rates = _COST_PER_1K.get(model, {"input": 0.005, "output": 0.015})
    return (input_tokens / 1000 * rates["input"]) + (
        output_tokens / 1000 * rates["output"]
    )


def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a string using tiktoken.

    Args:
        text: The text to tokenize.
        model: The model whose tokenizer to use.

    Returns:
        Token count.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _messages_token_count(messages: list[dict[str, str]], model: str) -> int:
    """Estimate token count for a list of chat messages.

    Args:
        messages: OpenAI-format message list.
        model: Model name for tokenizer selection.

    Returns:
        Approximate total token count.
    """
    total = 0
    for msg in messages:
        total += 4  # role + content overhead per message
        total += _count_tokens(msg.get("content", ""), model)
    total += 2  # priming tokens
    return total


# Retry decorator shared by API-calling methods
_retry = retry(
    retry=retry_if_exception_type((Exception,)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class LLMClient:
    """Async OpenAI client with structured output, embeddings, and observability.

    Args:
        settings: Application settings (defaults to the global singleton).
        openai_client: Optional pre-configured AsyncOpenAI instance for testing.
    """

    def __init__(
        self,
        cfg: Settings | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self._cfg = cfg or settings
        self._client = openai_client or AsyncOpenAI(api_key=self._cfg.openai_api_key)
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
        input_tokens = _messages_token_count(messages, model)

        logger.info(
            "chat_completion | model=%s | est_input_tokens=%d | temp=%.1f",
            model,
            input_tokens,
            temperature,
        )

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        choice = response.choices[0]
        reply = choice.message.content or ""

        # Token accounting from API response
        usage = response.usage
        if usage:
            cost = _estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
            self._total_input_tokens += usage.prompt_tokens
            self._total_output_tokens += usage.completion_tokens
            self._total_cost += cost
            logger.info(
                "chat_completion done | prompt_tokens=%d | completion_tokens=%d | "
                "cost=$%.4f | finish_reason=%s",
                usage.prompt_tokens,
                usage.completion_tokens,
                cost,
                choice.finish_reason,
            )

        return reply

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

        Uses OpenAI's native structured output (response_format with json_schema).
        Falls back to function-calling extraction if structured output fails.

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
        json_schema = response_schema.model_json_schema()

        logger.info(
            "structured_output | model=%s | schema=%s",
            model,
            schema_name,
        )

        try:
            return await self._structured_via_response_format(
                messages, response_schema, json_schema, schema_name, model, temperature
            )
        except Exception as exc:
            logger.warning(
                "response_format structured output failed (%s), "
                "falling back to function calling",
                exc,
            )
            return await self._structured_via_function_calling(
                messages, response_schema, json_schema, schema_name, model, temperature
            )

    async def _structured_via_response_format(
        self,
        messages: list[dict[str, str]],
        response_schema: type[T],
        json_schema: dict,
        schema_name: str,
        model: str,
        temperature: float,
    ) -> T:
        """Attempt structured output using response_format json_schema.

        Args:
            messages: Chat messages.
            response_schema: Target Pydantic model.
            json_schema: The JSON schema dict from the model.
            schema_name: Name for logging and the schema wrapper.
            model: OpenAI model to use.
            temperature: Sampling temperature.

        Returns:
            Parsed Pydantic model instance.
        """
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            },
        )

        self._log_usage(model, response.usage)
        raw = response.choices[0].message.content or "{}"
        return response_schema.model_validate_json(raw)

    async def _structured_via_function_calling(
        self,
        messages: list[dict[str, str]],
        response_schema: type[T],
        json_schema: dict,
        schema_name: str,
        model: str,
        temperature: float,
    ) -> T:
        """Fallback: extract structured data via function calling.

        Args:
            messages: Chat messages.
            response_schema: Target Pydantic model.
            json_schema: The JSON schema dict from the model.
            schema_name: Name for the function tool.
            model: OpenAI model to use.
            temperature: Sampling temperature.

        Returns:
            Parsed Pydantic model instance.
        """
        tool_def = {
            "type": "function",
            "function": {
                "name": f"extract_{schema_name.lower()}",
                "description": f"Extract structured {schema_name} data from the conversation.",
                "parameters": json_schema,
            },
        }

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=[tool_def],
            tool_choice={"type": "function", "function": {"name": tool_def["function"]["name"]}},
        )

        self._log_usage(model, response.usage)

        tool_call = response.choices[0].message.tool_calls[0]
        raw_args = tool_call.function.arguments
        return response_schema.model_validate(json.loads(raw_args))

    # ── Embeddings ────────────────────────────────────────────────────

    @_retry
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string using the configured embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        return (await self.embed_texts([text]))[0]

    @_retry
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        model = self._embedding_model

        total_tokens = sum(_count_tokens(t, model) for t in texts)
        logger.info(
            "embed_texts | model=%s | n=%d | est_tokens=%d",
            model,
            len(texts),
            total_tokens,
        )

        response = await self._client.embeddings.create(
            model=model,
            input=texts,
        )

        if response.usage:
            cost = _estimate_cost(model, response.usage.total_tokens, 0)
            self._total_input_tokens += response.usage.total_tokens
            self._total_cost += cost
            logger.info(
                "embed_texts done | tokens=%d | cost=$%.6f",
                response.usage.total_tokens,
                cost,
            )

        return [item.embedding for item in response.data]

    # ── Helpers ────────────────────────────────────────────────────────

    def _log_usage(self, model: str, usage) -> None:
        """Log token usage and cost from an API response.

        Args:
            model: The model that was called.
            usage: The usage object from the OpenAI response.
        """
        if not usage:
            return
        cost = _estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        self._total_input_tokens += usage.prompt_tokens
        self._total_output_tokens += usage.completion_tokens
        self._total_cost += cost
        logger.info(
            "usage | prompt_tokens=%d | completion_tokens=%d | cost=$%.4f",
            usage.prompt_tokens,
            usage.completion_tokens,
            cost,
        )

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

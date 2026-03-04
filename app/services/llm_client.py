"""Thin wrapper around the OpenAI SDK with structured output support.

Provides a single entry point for all LLM calls with:
- Automatic retries via tenacity
- Structured output parsing into Pydantic models
- Configurable model selection from app settings
"""

from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

T = TypeVar("T", bound=BaseModel)

_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    """Return a singleton async OpenAI client."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
async def chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Send a chat completion request and return the assistant's text reply.

    Args:
        messages: OpenAI-format message list.
        model: Override the default model from settings.
        temperature: Sampling temperature.

    Returns:
        The assistant's response text.
    """
    raise NotImplementedError("LLM chat completion not yet implemented.")


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
async def structured_output(
    messages: list[dict[str, str]],
    response_model: type[T],
    model: str | None = None,
) -> T:
    """Send a chat completion request and parse the response into a Pydantic model.

    Uses OpenAI's structured output / function-calling to guarantee the response
    conforms to the given schema.

    Args:
        messages: OpenAI-format message list.
        response_model: Pydantic model class to parse the response into.
        model: Override the default model from settings.

    Returns:
        An instance of response_model populated from the LLM response.
    """
    raise NotImplementedError("Structured LLM output not yet implemented.")

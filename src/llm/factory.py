"""
factory.py — Resolve an LLMProvider enum + model string into a concrete provider instance.
"""

from __future__ import annotations

from llm.base import BaseLLMProvider
from models.schema import LLMProvider


def get_llm_provider(provider: LLMProvider, model: str) -> BaseLLMProvider:
    """
    Factory that returns the appropriate provider instance.

    Parameters
    ----------
    provider : LLMProvider
        One of ANTHROPIC, HUGGINGFACE, or GEMINI.
    model : str
        The model identifier to pass to the provider SDK
        (e.g. "claude-sonnet-4-20250514", "mistralai/Mistral-7B-Instruct-v0.3",
        "gemini-2.0-flash").

    Returns
    -------
    BaseLLMProvider
        A ready-to-use provider instance.
    """
    if provider == LLMProvider.ANTHROPIC:
        from llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model)

    if provider == LLMProvider.HUGGINGFACE:
        from llm.providers.huggingface import HuggingFaceProvider

        return HuggingFaceProvider(model)

    if provider == LLMProvider.GEMINI:
        from llm.providers.gemini import GeminiProvider

        return GeminiProvider(model)

    raise ValueError(f"Unknown LLM provider: {provider!r}")

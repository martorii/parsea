"""
anthropic.py — Anthropic (Claude) LLM provider.

Uses the `anthropic` Python SDK. Requires ANTHROPIC_API_KEY in .env.
"""

from __future__ import annotations

import os

import anthropic

from llm.base import BaseLLMProvider
from utils import get_logger

log = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Extract document fields using Anthropic's Claude models."""

    ALLOWED_MODELS: list[str] = [
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-latest",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(self, model: str) -> None:
        super().__init__(model)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required. "
                "Set it in your .env file."
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    def _call_api(self, prompt: str) -> str:
        log.info("🤖 Calling Anthropic model=%s", self.model)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        # Extract text from the first content block
        response_text = message.content[0].text
        log.info(
            "🤖 Anthropic responded (%d chars, stop_reason=%s)",
            len(response_text),
            message.stop_reason,
        )
        return response_text

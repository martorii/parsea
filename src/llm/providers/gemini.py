"""
gemini.py — Google Gemini LLM provider.

Uses the `google-genai` SDK. Requires GOOGLE_API_KEY in .env.
"""

from __future__ import annotations

import os

from google import genai

from llm.base import BaseLLMProvider
from utils import get_logger

log = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Extract document fields using Google's Gemini models."""

    ALLOWED_MODELS: list[str] = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    def __init__(self, model: str) -> None:
        super().__init__(model)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required. "
                "Set it in your .env file."
            )
        self.client = genai.Client(api_key=api_key)

    def _call_api(self, prompt: str) -> str:
        log.info("🤖 Calling Gemini model=%s", self.model)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        response_text = response.text
        log.info("🤖 Gemini responded (%d chars)", len(response_text))
        return response_text

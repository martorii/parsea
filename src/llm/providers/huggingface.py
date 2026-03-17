"""
huggingface.py — Hugging Face Inference API LLM provider.

Uses the `huggingface_hub` InferenceClient. Requires HUGGINGFACEHUB_API_TOKEN in .env.
"""

from __future__ import annotations

import os

from huggingface_hub import InferenceClient

from llm.base import BaseLLMProvider
from utils import get_logger

log = get_logger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """Extract document fields using Hugging Face Inference API models."""

    ALLOWED_MODELS: list[str] = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2-9b-it",
    ]

    def __init__(self, model: str) -> None:
        super().__init__(model)
        api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not api_token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN environment variable is required. "
                "Set it in your .env file."
            )
        self.client = InferenceClient(
            model=self.model,
            token=api_token,
        )

    def _call_api(self, prompt: str) -> str:
        log.info("🤖 Calling Hugging Face model=%s", self.model)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        # Extract response from response object
        response = response.choices[0].message.content
        log.info("🤖 Hugging Face responded (%d chars)", len(response))
        return response

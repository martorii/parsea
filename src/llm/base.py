"""
base.py — Abstract base class for LLM extraction providers.

Each provider builds a prompt from document chunks and field definitions,
calls the respective LLM API, and returns a dict of raw extractions.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from models import ExtractionInstructions
from processing.chunker import Chunk
from utils import get_logger

log = get_logger(__name__)


class BaseLLMProvider(ABC):
    """Interface that every LLM provider must implement."""

    ALLOWED_MODELS: list[str] = []

    def __init__(self, model: str) -> None:
        self.model = model
        self._validate_model()

    def _validate_model(self) -> None:
        """Check if the chosen model is in the allowed models list."""
        if self.model not in self.ALLOWED_MODELS:
            raise ValueError(
                f"Model {self.model!r} is not supported by {self.__class__.__name__}. "
                f"Allowed models: {', '.join(self.ALLOWED_MODELS)}"
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def extract(
        self,
        chunks: list[Chunk],
        instructions: ExtractionInstructions,
    ) -> dict[str, dict]:
        """
        Run extraction: build prompt ➜ call LLM ➜ parse response.

        Returns a dict keyed by field key, each value being:
            {"raw_value": str | None, "confidence": float, "chunk_id": str, "page": int}
        """
        prompt = self._build_prompt(chunks, instructions)
        raw_response = self._call_api(prompt)

        if not raw_response:
            log.warning("Received empty response from LLM API.")
            return self._fallback_results(instructions)

        try:
            return self._parse_response(raw_response, instructions)
        except json.JSONDecodeError as exc:
            log.warning(
                "Initial JSON parsing failed: %s. Attempting to re-format...", exc
            )

            reformat_prompt = self._build_reformat_prompt(raw_response, exc)
            reformat_response = self._call_api(reformat_prompt)

            if not reformat_response:
                return self._fallback_results(instructions)

            try:
                return self._parse_response(reformat_response, instructions)
            except json.JSONDecodeError as final_exc:
                log.error(
                    "Re-formatting also failed to produce valid JSON: %s", final_exc
                )
                return self._fallback_results(instructions)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_prompt(
        self,
        chunks: list[Chunk],
        instructions: ExtractionInstructions,
    ) -> str:
        """Build a structured prompt from document chunks and field definitions."""
        field_specs = []
        for f in instructions.fields:
            spec = {
                "key": f.key,
                "name": f.name,
                "description": f.description,
                "type": f.type.value,
                "required": f.required,
            }
            if f.example:
                spec["example"] = f.example
            if f.categories:
                spec["categories"] = f.categories
            field_specs.append(spec)

        chunks_text = "\n\n".join(
            f"--- Chunk {c.id} (page {c.page}) ---\n{c.text}" for c in chunks
        )

        prompt = (
            f"You are a document extraction assistant.\n\n"
            f"DOCUMENT DESCRIPTION:\n{instructions.document_description}\n\n"
            f"FIELDS TO EXTRACT:\n{json.dumps(field_specs, indent=2, ensure_ascii=False)}\n\n"
            f"DOCUMENT CONTENT:\n{chunks_text}\n\n"
            f"INSTRUCTIONS:\n"
            f"Extract each field from the document content. "
            f"Return a JSON object where each key is a field key and the value is an object with:\n"
            f'  - "raw_value": the extracted text exactly as it appears (string or null)\n'
            f'  - "confidence": your confidence from 0.0 to 1.0\n'
            f'  - "chunk_id": the chunk ID where you found it\n'
            f'  - "page": the page number where you found it\n\n'
            f"Return ONLY valid JSON, no markdown fences or extra text."
        )
        return prompt

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Send the prompt to the LLM API and return the raw text response."""
        ...

    def _build_reformat_prompt(
        self, raw_response: str, exc: json.JSONDecodeError
    ) -> str:
        """Build a prompt asking the LLM to fix invalid JSON output."""
        return (
            f"You previously generated the following extraction output, which is not valid JSON:\n\n"
            f"```json\n{raw_response}\n```\n\n"
            f"The JSON parser encountered this error: {exc}\n\n"
            f"Please output exactly the same data but strictly formatted as valid JSON. "
            f"Do not include any other text except the JSON object."
        )

    def _fallback_results(
        self, instructions: ExtractionInstructions
    ) -> dict[str, dict]:
        """Return a dictionary of empty/unfound results for all fields."""
        return {
            f.key: {
                "raw_value": None,
                "confidence": 0.0,
                "chunk_id": "unknown",
                "page": 1,
            }
            for f in instructions.fields
        }

    def _parse_response(
        self,
        raw_response: str,
        instructions: ExtractionInstructions,
    ) -> dict[str, dict]:
        """Parse the JSON response from the LLM into the expected dict format."""
        text = (raw_response or "").strip()

        # Strip potential markdown code fences
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()

        # Let JSONDecodeError propagate to extract()
        parsed = json.loads(text)

        results: dict[str, dict] = {}
        for field_def in instructions.fields:
            entry = parsed.get(field_def.key)
            if entry and isinstance(entry, dict) and entry.get("raw_value") is not None:
                results[field_def.key] = {
                    "raw_value": entry.get("raw_value"),
                    "confidence": float(entry.get("confidence", 0.9)),
                    "chunk_id": str(entry.get("chunk_id", "unknown")),
                    "page": int(entry.get("page", 1)),
                }
            else:
                results[field_def.key] = {
                    "raw_value": None,
                    "confidence": 0.0,
                    "chunk_id": "unknown",
                    "page": 1,
                }

        return results

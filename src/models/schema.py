from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FieldType(str, Enum):
    TEXT = "text"
    CURRENCY = "currency"
    DATE = "date"
    CATEGORY = "category"
    NUMBER = "number"
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    PHONE = "phone"
    EMAIL = "email"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    GEMINI = "gemini"


class FieldDefinition(BaseModel):
    name: str = Field(..., description="Human-readable name, e.g. 'Invoice Total'")
    key: str = Field(..., description="snake_case output key, e.g. 'invoice_total'")
    description: str | None = Field(None, description="What this field represents")
    example: str | None = Field(
        None, description="An example value to guide extraction"
    )
    type: FieldType = Field(
        FieldType.TEXT, description="Output type for standardization"
    )
    currency_code: str = Field("USD", description="Currency code for currency types")

    # Type-specific options
    categories: list[str] | None = Field(
        None, description="Allowed values when type=category"
    )
    required: bool = Field(False, description="Emit a warning if not found")


class ExtractionInstructions(BaseModel):
    document_description: str = Field(
        ...,
        description="What kind of document this is — helps the LLM extract more accurately",
    )
    llm_provider: LLMProvider = Field(
        ..., description="Which LLM provider to use for extraction"
    )
    model: str = Field(
        ...,
        description="Model identifier, e.g. 'claude-sonnet-4-20250514' or 'gemini-2.0-flash'",
    )
    fields: list[FieldDefinition]


class ChunkReference(BaseModel):
    page: int
    chunk_id: str
    chunk_preview: str


class ExtractedField(BaseModel):
    key: str
    value: Any
    raw_value: str | None
    confidence: float = Field(..., ge=0, le=1)
    found: bool
    warning: str | None = None
    reference: ChunkReference | None = None


class ExtractionResponse(BaseModel):
    document_id: str
    total_pages: int
    total_chunks: int
    fields: list[ExtractedField]

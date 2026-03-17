from __future__ import annotations

from llm import get_llm_provider
from models import (
    ChunkReference,
    ExtractedField,
    ExtractionInstructions,
    ExtractionResponse,
)
from processing.chunker import Chunk
from processing.retriever import ChunkRetriever
from processing.standardize import standardize
from utils import get_logger

log = get_logger(__name__)


def extract_fields(
    document_id: str,
    chunks: list[Chunk],
    instructions: ExtractionInstructions,
) -> ExtractionResponse:
    """Run the full extraction pipeline: LLM → standardize → response model."""

    # Step A: Instantiate the right LLM provider and call it
    provider = get_llm_provider(instructions.llm_provider, instructions.model)
    raw_extractions = provider.extract(chunks, instructions)

    # Step B: Standardize each field and build ExtractedField objects
    extracted_fields: list[ExtractedField] = []

    # Instantiate retriever
    retriever = ChunkRetriever(chunks)

    for field_def in instructions.fields:
        extraction = raw_extractions.get(field_def.key, {})
        raw_value = extraction.get("raw_value")
        name = field_def.get("name") if isinstance(field_def, dict) else field_def.name

        # Standardize the raw value
        if raw_value is not None:
            std_value, warning = standardize(raw_value, field_def)
            found = True

            # Use BM25+CrossEncoder to find the best matching chunk and get a confidence score
            source_chunk, confidence = retriever.find_best_chunk(
                str(name) + ": " + str(raw_value)
            )

            # Fallback to the LLM's chunk ID and confidence if the retriever didn't find anything
            if not source_chunk:
                chunk_id = extraction.get("chunk_id", "unknown")
                source_chunk = next((c for c in chunks if c.id == chunk_id), None)
                # Ensure confidence is in [0, 1]. Some models might return negative values or logs.
                confidence = max(
                    0.0, min(1.0, float(extraction.get("confidence", 0.0)))
                )
        else:
            std_value, warning = standardize("", field_def)
            found = False
            confidence = 0.0
            source_chunk = None

        reference = None
        if source_chunk:
            reference = ChunkReference(
                page=source_chunk.page,
                chunk_id=source_chunk.id,
                chunk_preview=source_chunk.text[:120].replace("\n", " ") + "...",
            )

        extracted_fields.append(
            ExtractedField(
                key=getattr(field_def, "key", ""),
                value=std_value,
                raw_value=raw_value,
                confidence=confidence,  # Already in [0, 1]
                found=found,
                warning=warning,
                reference=reference,
            )
        )

    return ExtractionResponse(
        document_id=document_id,
        total_pages=max((c.page for c in chunks), default=0),
        total_chunks=len(chunks),
        fields=extracted_fields,
    )

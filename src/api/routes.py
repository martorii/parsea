import json

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from api.auth import require_auth
from models import ExtractionInstructions
from processing.chunker import chunk_pages
from processing.extractor import extract_fields
from processing.parser import parse_document
from utils import get_logger

log = get_logger(__name__)
router = APIRouter()


@router.post(
    "/extract",
    summary="Extract fields from a document",
    description="Upload a document and provide JSON extraction instructions.",
    dependencies=[Depends(require_auth)],
)
async def extract_document(
    file: UploadFile,
    instructions_json: str = Form(
        ...,
        description="JSON string matching the ExtractionInstructions schema.",
    ),
):
    log.info("Received request to process document: %s", file.filename)

    # 1. Parse instructions
    try:
        parsed_instructions_dict = json.loads(instructions_json)
        instructions = ExtractionInstructions(**parsed_instructions_dict)
    except json.JSONDecodeError as e:
        log.warning("Failed to parse instructions JSON: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON provided in 'instructions_json': {e}",
        )
    except ValidationError as e:
        log.warning("Instructions failed validation against the schema: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Instructions failed validation: {e.errors()}",
        )

    # 2. Read the file
    try:
        data = await file.read()
    except Exception as e:
        log.error("Failed to read uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read file: {e}",
        )

    # 3. Parse the document
    try:
        parsed_document = parse_document(data, filename=file.filename or "uploaded.pdf")
        log.info("Parsed %d pages from %s", len(parsed_document.pages), file.filename)
    except ValueError as e:
        log.warning("Unsupported file type: %s", e)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {e}",
        )
    except Exception as e:
        log.error("Error during document parsing: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing document: {e}",
        )

    # 4. Chunk
    try:
        chunks = chunk_pages(parsed_document.pages, chunk_size=600, overlap=120)
        log.info("Created %d chunks", len(chunks))
    except Exception as e:
        log.error("Error during chunking: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document text chunks: {e}",
        )

    # 5. Extract fields using the provided instructions
    provider_label = f"{instructions.llm_provider.value} / {instructions.model}"
    log.info("Extracting fields using %s...", provider_label)

    try:
        response = extract_fields(
            document_id=file.filename or "unknown",
            chunks=chunks,
            instructions=instructions,
        )
    except Exception as e:
        log.error("Error during extraction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {e}",
        )

    return response

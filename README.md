# Parsea Backend

Parsea is a powerful document extraction backend that uses Large Language Models (LLMs) and advanced retrieval techniques to transform unstructured documents into structured data.

## Features

- **Document Parsing**: Robust PDF parsing using `pdfplumber` and `pypdf`.
- **Advanced Chunking**: Context-aware chunking that preserves tables and spatial structure.
- **RAG-based Extraction**: Uses BM25 and Cross-Encoder re-ranking to provide relevant context to LLMs.
- **Multi-Cloud LLM Support**: Support for Anthropic (Claude), Google (Gemini), and Hugging Face (Llama, Mistral, etc.) models.
- **Data Standardization**: Automatic validation and formatting of extracted values (Dates, Currencies, Boolean, etc.).
- **FastAPI Interface**: Clean REST API for easy integration.
- **Docker Ready**: Production-optimized multi-stage Docker build.

## Project Structure

```text
├── src/
│   ├── api/            # FastAPI routes and authentication
│   ├── llm/            # LLM provider implementations
│   ├── models/         # Pydantic and Dataclass models
│   ├── processing/     # Core logic (parsing, chunking, retrieval, standardize)
│   ├── utils/          # Logging and shared helpers
│   ├── app.py          # FastAPI entry point
│   ├── client.py       # Example CLI client
├── docs/               # Put your sample documents here (ignored by git)
├── tests/              # Pytest suite
├── Dockerfile          # Multi-stage Docker build
├── docker-compose.yml  # Deployment orchestration
├── Makefile            # Quick commands for dev/deploy
└── pyproject.toml      # Dependency management (uv)
```

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- Docker (for containerized deployment)

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/martorii/parsea.git
   cd parsea/backend
   ```

2. **Environment Setup**:
   Copy `.env.example` to `.env` and fill in your API tokens:
   ```bash
   cp .env.example .env
   ```

3. **Install Dependencies**:
   ```bash
   uv sync
   ```

4. **Run the API**:
   ```bash
   uv run uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir src --reload
   ```

5. **Test with the Client**:
   ```bash
   uv run python src/client.py
   ```

## Docker Deployment

The easiest way to deploy Parsea is using the provided `Makefile`:

```bash
# Build the images
make build

# Start the services
make run

# Run health tests
make test

# View logs
make logs
```

## API Usage

The main endpoint is `POST /api/v1/extract`. It expects:
- `file`: The document as a file upload.
- `instructions_json`: A JSON string defining the fields to extract.

Example `instructions_json`:
```json
{
  "document_description": "Invoices",
  "llm_provider": "anthropic",
  "model": "claude-3-5-sonnet-latest",
  "fields": [
    {
      "name": "Invoice Number",
      "key": "invoice_id",
      "type": "text",
      "required": true
    },
    {
      "name": "Total Amount",
      "key": "total",
      "type": "currency",
      "currency_code": "EUR"
    }
  ]
}
```

## Quality and Standards

This project uses:
- **Ruff** for linting and formatting.
- **Mypy** (optional) for type checking.
- **Pytest** for unit and integration testing.

Pre-commit hooks are installed to ensure code quality before every commit.

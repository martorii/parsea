# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv 
RUN pip install uv

# Create a virtual environment and install dependencies using uv
COPY pyproject.toml uv.lock ./
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv -r pyproject.toml

# --- Stage 2: Runner ---
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m appuser

WORKDIR /app

# Copy the built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Add the venv to the PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code
COPY src /app/src
# COPY docs /app/docs

# Switch to the non-root user
USER appuser

# Expose port (metadata only)
EXPOSE 8000

# Start the application, using API_HOST and API_PORT from environment (or defaults)
CMD ["sh", "-c", "uvicorn app:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000} --app-dir src"]


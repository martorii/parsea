import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from utils import configure_logging, get_logger

# Ensure environment variables are loaded
load_dotenv()

# Configure logging early
configure_logging(level="DEBUG")
log = get_logger(__name__)

app = FastAPI(
    title="Parsea Document Extraction API",
    description="API for extracting structured fields from documents.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/", summary="Health Check")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "app": "parsea-api"}


if __name__ == "__main__":
    log.info("Starting Parsea API server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, app_dir="src")

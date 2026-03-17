import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from utils import get_logger

log = get_logger(__name__)

_VALID_TOKEN = os.environ.get("API_TOKEN", "dev-secret-token")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_auth(api_key: str = Security(api_key_header)) -> str:
    if not api_key:
        log.warning("Request rejected: missing X-API-Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )
    if api_key != _VALID_TOKEN:
        log.warning("Request rejected: invalid token (last 4: ...%s)", api_key[-4:])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    log.debug("Auth passed")
    return api_key

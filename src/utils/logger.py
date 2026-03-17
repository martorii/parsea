import logging
import sys

_ROOT_LOGGER_NAME = "parsea"
_configured = False


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Call once at application startup (done automatically in main.py).
    Subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Always log to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Optionally also write to a file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "anthropic", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(module_name: str) -> logging.Logger:
    """
    Return a logger named `parsea.<last_segment_of_module_name>`.
    Keeps log lines short while still being traceable to the source file.

    Example:
        get_logger("parsea.parsers") -> "parsea.parsers"
        get_logger(__name__)         -> "parsea.<filename>"
    """
    # Strip any leading package path so names stay short in the log output
    short_name = module_name.split(".")[-1]
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short_name}")

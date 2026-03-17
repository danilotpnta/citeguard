import sys
import logging
import contextvars
from pathlib import Path
from typing import Optional

# Context variable to store request ID (for tracking requests across files)
request_id_ctx = contextvars.ContextVar("request_id", default=None)
user_id_ctx = contextvars.ContextVar("user_id", default=None)


class ContextFilter(logging.Filter):
    """
    Add request_id and user_id to every log record
    This lets you trace a single request through the entire system
    """

    def filter(self, record):
        record.request_id = request_id_ctx.get() or "NO-REQUEST-ID"
        record.user_id = user_id_ctx.get() or "NO-USER"
        return True


class ColoredFormatter(logging.Formatter):
    """
    Add colors to console output for easier reading
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        return super().format(record)


def setup_logging(
    log_level: str = "INFO", log_file: Optional[Path] = None, enable_colors: bool = True
) -> None:
    """
    Configure logging for the entire application

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        enable_colors: Whether to use colored output in console
    """

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Format string
    format_string = (
        "%(asctime)s | %(levelname)-8s | "
        "[user:%(user_id)s] [req:%(request_id)s] | "
        "%(name)s:%(lineno)d | %(message)s"
    )

    # Use colored formatter for console
    if enable_colors:
        console_formatter = ColoredFormatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        console_formatter = logging.Formatter(
            format_string, datefmt="%Y-%m-%d %H:%M:%S"
        )

    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ContextFilter())
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(ContextFilter())
        root_logger.addHandler(file_handler)

    # Silence noisy libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("python_multipart").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module

    Usage:
        logger = get_logger(__name__)
        logger.info("Something happened")

    Args:
        name: Usually __name__ (the module name)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_request_context(request_id: str, user_id: Optional[str] = None) -> None:
    """
    Set request context for logging

    Call this at the start of each request to track it through the system

    Args:
        request_id: Unique ID for this request
        user_id: User ID if authenticated
    """
    request_id_ctx.set(request_id)
    if user_id:
        user_id_ctx.set(user_id)


def clear_request_context() -> None:
    """Clear request context after request completes"""
    request_id_ctx.set(None)
    user_id_ctx.set(None)

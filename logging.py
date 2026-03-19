"""Logging configuration for the churn prediction pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "churn_prediction",
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up and configure a logger instance.

    Args:
        name: Name of the logger.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a log file.
        format_string: Optional custom format string.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "churn_prediction") -> logging.Logger:
    """Get an existing logger or create a new one.

    Args:
        name: Name of the logger.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
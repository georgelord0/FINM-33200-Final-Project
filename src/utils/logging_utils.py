"""Logging configuration for the src pipeline.

Uses *loguru* as the sole logging backend.  Call :func:`setup_logger` once at
application startup (e.g. in a CLI entry-point or notebook preamble) and then
obtain module-specific loggers via :func:`get_logger`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

# Module-level flag so setup only runs once.
_configured: bool = False

# Default log directory (relative to project root).
_DEFAULT_LOG_DIR: str = "logs"

# Default format string.
_FORMAT: str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[module]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

_FILE_FORMAT: str = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{extra[module]}:{function}:{line} | "
    "{message}"
)


def setup_logger(
    *,
    level: str = "DEBUG",
    log_dir: str | Path | None = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    serialize_json: bool = False,
) -> Logger:
    """Configure the global *loguru* logger.

    Parameters
    ----------
    level:
        Minimum severity for the stderr sink.
    log_dir:
        Directory for rotating log files.  Defaults to ``logs/`` relative to
        the current working directory.
    rotation:
        When to rotate the log file (size or time string).
    retention:
        How long to keep old log files.
    serialize_json:
        If ``True`` the file sink emits structured JSON lines.

    Returns
    -------
    Logger
        The configured *loguru* logger instance.
    """
    global _configured  # noqa: PLW0603

    if _configured:
        return logger

    # Remove the default stderr handler so we can replace it.
    logger.remove()

    # -- stderr sink (coloured) ------------------------------------------------
    logger.add(
        sys.stderr,
        format=_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # -- file sink (rotating) --------------------------------------------------
    log_path = Path(log_dir) if log_dir is not None else Path(_DEFAULT_LOG_DIR)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path / "src.log",
        format=_FILE_FORMAT,
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        serialize=serialize_json,
        backtrace=True,
        diagnose=True,
        enqueue=True,  # thread-safe writes
    )

    _configured = True
    logger.bind(module="logging_utils").info(
        "Logger initialised (stderr={}, file={})", level, log_path
    )
    return logger


def get_logger(name: str) -> Logger:
    """Return a *loguru* logger bound to *name*.

    This makes it easy to identify which module produced a given log line
    without requiring callers to configure anything beyond a single import.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    Logger
        A bound *loguru* logger instance.
    """
    return logger.bind(module=name)

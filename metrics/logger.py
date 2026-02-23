from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union


def _resolve_level(level: Union[str, int]) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def setup_logger(
    name: str,
    log_dir: Union[Path, str],
    *,
    console_level: Union[str, int] = "INFO",
    file_level: Union[str, int] = "DEBUG",
    log_filename: str = "training.log",
) -> logging.Logger:
    """Create or reset a logger with unified console and rotating file handlers."""

    resolved_dir = Path(log_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False

    console_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_resolve_level(console_level))
    console_handler.setFormatter(console_fmt)

    file_path = resolved_dir / log_filename
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(_resolve_level(file_level))
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.setLevel(min(console_handler.level, file_handler.level))

    return logger


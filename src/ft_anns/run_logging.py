"""Bounded human-readable run logs alongside structured experiment artifacts."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time


_PACKAGE_LOGGER = "ft_anns"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 2


class _UtcFormatter(logging.Formatter):
    converter = time.gmtime


def configure_run_logging(
    path: str | Path,
    *,
    level: str = "INFO",
    append: bool = False,
) -> Path:
    """Configure one bounded package log without polluting JSON stdout."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not append:
        for candidate in (
            log_path,
            *(
                log_path.with_name(f"{log_path.name}.{index}")
                for index in range(1, _BACKUP_COUNT + 1)
            ),
        ):
            candidate.unlink(missing_ok=True)

    package_logger = logging.getLogger(_PACKAGE_LOGGER)
    for handler in package_logger.handlers:
        handler.close()
    package_logger.handlers.clear()
    package_logger.setLevel(level.upper())
    package_logger.propagate = False

    handler = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level.upper())
    handler.setFormatter(
        _UtcFormatter(
            fmt=(
                "%(asctime)s.%(msecs)03dZ | %(levelname)-8s | pid=%(process)d | "
                "%(name)s (%(filename)s:%(funcName)s:%(lineno)d) | %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    package_logger.addHandler(handler)
    package_logger.info(
        "run logging configured path=%s max_bytes=%d backups=%d append=%s",
        log_path,
        _MAX_BYTES,
        _BACKUP_COUNT,
        append,
    )
    return log_path

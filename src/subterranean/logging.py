"""Loguru-based logging setup.

The logger is the only acceptable piece of global state in the library. Default
level is INFO; pass ``verbose=True`` (wired to the CLI's ``--verbose`` flag) for DEBUG.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

__all__ = ["configure_logging", "logger"]

_DEFAULT_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan> - <level>{message}</level>"
)


def _safe_serialize(record: Record) -> bool:
    """Strip values that cannot be safely repr'd (e.g. circular references).

    Loguru eagerly formats ``extra`` bindings; a circular structure passed via
    ``logger.bind(...)`` would raise during emission. We replace any value whose
    ``repr`` fails with a placeholder so logging never takes down a real run.

    Returns ``True`` unconditionally so it can double as a loguru ``filter``.
    """
    extra = record.get("extra", {})
    for key, value in list(extra.items()):
        try:
            repr(value)
        except Exception:
            extra[key] = "<unrepresentable>"
    return True


def configure_logging(*, verbose: bool = False) -> None:
    """Configure the global logger.

    Args:
        verbose: When True, emit DEBUG-level records; otherwise INFO.

    Example:
        >>> configure_logging(verbose=True)
        >>> logger.debug("detailed trace")
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format=_DEFAULT_FORMAT,
        filter=_safe_serialize,
        backtrace=False,
        diagnose=False,
    )

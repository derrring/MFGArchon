"""Shared utilities for the MFGarchon workflow module.

Consolidates duplicate patterns across experiment_tracker, workflow_manager,
and parameter_sweep: status enums, serialization, and logging setup.

Issue #621: Consolidate duplicate patterns in workflow/ module.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger


class ExecutionStatus(Enum):
    """Unified execution status for workflows and experiments.

    Superset of the previously separate ExperimentStatus and WorkflowStatus enums.
    """

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


def serialize_value(value: Any, name: str = "") -> Any:
    """Serialize a value for JSON-compatible storage.

    Handles numpy arrays, objects with ``to_dict()``, JSON-native types,
    and falls back to ``str()`` for everything else.

    Args:
        value: The value to serialize.
        name: Optional name used for the numpy data_file key.

    Returns:
        A JSON-serializable representation of *value*.
    """
    if isinstance(value, np.ndarray):
        return {
            "type": "numpy_array",
            "shape": value.shape,
            "dtype": str(value.dtype),
            "data_file": f"result_{name}.npy" if name else "result.npy",
        }

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    if isinstance(value, (dict, list, str, int, float, bool)):
        return value

    return str(value)


def setup_workflow_logging(
    name: str,
    log_file: Path | None = None,
    *,
    console: bool = False,
) -> logging.Logger:
    """Configure a logger with an optional file handler and optional console handler.

    All five ``_setup_logging()`` methods across the workflow module follow the
    same pattern. This function consolidates that pattern.

    Args:
        name: Logger name passed to :func:`get_logger`.
        log_file: Path to the log file, or ``None`` for console only. `None` is what a
            caller passes before it owns a directory to write into -- attach the file
            later with :func:`attach_log_file`.
        console: If ``True``, also attach a :class:`~logging.StreamHandler`.

    Returns:
        Configured :class:`~logging.Logger`.
    """
    logger = get_logger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(_FORMATTER)
            logger.addHandler(console_handler)

    if log_file is not None:
        attach_log_file(logger, log_file)

    return logger


_FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def attach_log_file(logger: logging.Logger, log_file: Path) -> None:
    """Attach a file handler, once, to a logger that may already have one.

    `delay=True` matters: `logging.FileHandler` opens its file in `__init__` by default, so
    constructing the handler is itself a write. Deferring means the file appears when a record
    is actually emitted, which is the same moment the caller has committed to owning the
    directory. #1917.
    """
    resolved = os.path.abspath(str(log_file))
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved:
            return
    file_handler = logging.FileHandler(log_file, delay=True)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)


__all__ = [
    "ExecutionStatus",
    "attach_log_file",
    "serialize_value",
    "setup_workflow_logging",
]

"""Shared utilities for the MFGarchon workflow module.

Consolidates duplicate patterns across experiment_tracker, workflow_manager,
and parameter_sweep: status enums, serialization, and logging setup.

Issue #621: Consolidate duplicate patterns in workflow/ module.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from mfgarchon.utils.mfg_logging import MFGFormatter, get_logger


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

    `log_file` is optional so a caller that does not yet own a directory can still get a logger;
    `Workflow` and `WorkflowManager` construct theirs before any directory exists (#1917).

    The `if not logger.handlers:` guard is deliberate and is restored here after a review found
    that lifting it revived a path dead since #621 (`d424be1d`, 2026-02-06). `get_logger` always
    attaches a StreamHandler before returning, so for any caller that obtains its logger through
    `get_logger` this branch is False and no `FileHandler` is constructed. (~~for any caller~~
    [CORRECTED 2026-08-14] -- `tests/unit/test_workflow/test_common.py` patches `get_logger` to
    return a bare logger, so the guard is True there and a real FileHandler IS built. That is why
    that test is the second one to redden when the guard is lifted. The operative claim is about
    production callers; the universal was false.)

    An earlier revision of this change moved the file handler outside the guard; the result wrote
    `parameter_sweep.log` and `experiment.log` into the caller's working directory, and -- because
    `mfg_workflow_manager`, `mfg_experiment_tracker` and `mfg_parameter_sweep` are FIXED logger
    names -- appended a new handler per instance, so records from **37** distinct output
    directories landed in one file at this repository's root, 298,009 bytes of it. 37 counts the
    directories named by the `Saved complete results to` records dated 2026-08-14; a figure that
    moves with the extraction rule is not settled, so the rule is stated. (~~126~~ [CORRECTED
    2026-08-14] -- the whole file holds 128 across 2025-10-09..2026-08-14, of which 91 predate
    #621 disabling the logging, with zero overlap. Attributing that total to the 2026-08-14
    window overstated the mechanism ~3.5x. The window spans 01:08:43..01:22:18 with gaps, so
    whether it is one gate run or several is not established.)

    Reviving workflow file logging is a separate decision from stopping import-time writes, and
    it needs the shared-name loggers fixed first. Tracked separately; not done here.
    """
    logger = get_logger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # MFGFormatter owns this format; a hand-rolled copy here was the second of two rivals
        # (#2058). This branch is dead for production callers -- `get_logger` always attaches a
        # StreamHandler, so the guard above is False and neither handler is built, measured -- but
        # a dead rival string is still a rival, and it is what a future lift of that guard would
        # install.
        formatter = MFGFormatter()

        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger


__all__ = [
    "ExecutionStatus",
    "serialize_value",
    "setup_workflow_logging",
]

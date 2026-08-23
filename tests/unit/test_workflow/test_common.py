"""Tests for mfgarchon.workflow.common — shared workflow utilities.

Issue #621: Consolidate duplicate patterns in workflow/ module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from mfgarchon.workflow.common import (
    ExecutionStatus,
    serialize_value,
    setup_workflow_logging,
)

# ---------------------------------------------------------------------------
# ExecutionStatus tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecutionStatus:
    """ExecutionStatus enum covers all expected values."""

    def test_execution_status_values(self):
        """All six status values exist and have the expected string values."""
        assert ExecutionStatus.CREATED.value == "created"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.CANCELLED.value == "cancelled"
        assert ExecutionStatus.PAUSED.value == "paused"

    def test_execution_status_backward_compat(self):
        """ExperimentStatus and WorkflowStatus aliases resolve to ExecutionStatus."""
        from mfgarchon.workflow.experiment_tracker import ExperimentStatus
        from mfgarchon.workflow.workflow_manager import WorkflowStatus

        assert ExperimentStatus is ExecutionStatus
        assert WorkflowStatus is ExecutionStatus

        # Existing value checks work identically
        assert ExperimentStatus.COMPLETED is ExecutionStatus.COMPLETED
        assert WorkflowStatus.PAUSED is ExecutionStatus.PAUSED


# ---------------------------------------------------------------------------
# serialize_value tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSerializeValue:
    """serialize_value handles all expected types."""

    def test_numpy_array(self):
        """Numpy arrays produce a metadata dict."""
        arr = np.ones((3, 4), dtype=np.float64)
        result = serialize_value(arr, name="my_arr")

        assert result["type"] == "numpy_array"
        assert result["shape"] == (3, 4)
        assert result["dtype"] == "float64"
        assert result["data_file"] == "result_my_arr.npy"

    def test_passthrough_types(self):
        """Primitive JSON-native types are returned unchanged."""
        assert serialize_value({"a": 1}) == {"a": 1}
        assert serialize_value([1, 2, 3]) == [1, 2, 3]
        assert serialize_value("hello") == "hello"
        assert serialize_value(42) == 42
        assert serialize_value(3.14) == 3.14
        assert serialize_value(True) is True

    def test_custom_to_dict(self):
        """Objects with .to_dict() are serialized via that method."""

        class Custom:
            def to_dict(self):
                return {"custom": True}

        result = serialize_value(Custom())
        assert result == {"custom": True}

    def test_fallback_str(self):
        """Unknown types fall back to str()."""

        class Opaque:
            def __str__(self):
                return "opaque_repr"

        result = serialize_value(Opaque())
        assert result == "opaque_repr"


# ---------------------------------------------------------------------------
# setup_workflow_logging tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetupWorkflowLogging:
    """setup_workflow_logging configures handlers correctly.

    We patch ``get_logger`` to return a bare logger (no pre-existing handlers)
    so that the handler-attachment logic inside ``setup_workflow_logging``
    can be tested in isolation — independent of MFGLogger's own initialization.
    """

    def _bare_logger(self, name: str) -> logging.Logger:
        """Create a bare logger with no handlers for testing."""
        logger = logging.Logger(name, level=logging.DEBUG)
        logger.handlers.clear()
        return logger

    def test_file_handler_only(self, tmp_path: Path):
        """Without console=True, only a file handler is attached."""
        log_file = tmp_path / "test.log"
        bare = self._bare_logger("test_file_only")

        with patch("mfgarchon.workflow.common.get_logger", return_value=bare):
            logger = setup_workflow_logging("test_file_only", log_file)

        assert isinstance(logger, logging.Logger)
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        # No console handler should be attached
        stream_only = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_only) == 0

        # Clean up
        logger.handlers.clear()

    def test_with_console(self, tmp_path: Path):
        """With console=True, both file and stream handlers are attached."""
        log_file = tmp_path / "test_console.log"
        bare = self._bare_logger("test_with_console")

        with patch("mfgarchon.workflow.common.get_logger", return_value=bare):
            logger = setup_workflow_logging("test_with_console", log_file, console=True)

        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]

        # #2058: both handlers must carry the OWNED format, not a hand-rolled copy. Asserting
        # handler TYPES alone left this branch green under any formatter, which is how a second
        # format string lived here. Type, not string: the string belongs to MFGFormatter.
        from mfgarchon.utils.mfg_logging import MFGFormatter

        for handler in logger.handlers:
            assert isinstance(handler.formatter, MFGFormatter), (
                f"{type(handler).__name__} carries {type(handler.formatter).__name__}, not MFGFormatter"
            )

        assert len(file_handlers) >= 1
        assert len(stream_handlers) >= 1

        # Clean up
        logger.handlers.clear()

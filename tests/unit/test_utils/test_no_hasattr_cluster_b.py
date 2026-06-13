"""Pinning tests for Issue #1068 Cluster B (utils/ hasattr removal).

Cluster A (core/) was handled by PR #1307. Cluster B replaces the two
internal-duck-typing `hasattr()` sites in `utils/` with CLAUDE.md patterns:

- ``utils/exceptions.py`` ``validate_solver_state`` — getattr-with-default.
- ``utils/memory_management.py`` ``memory_monitored`` cleanup — getattr-with-default
  (the decorator wraps methods of arbitrary classes, so ``_temp_arrays`` is an
  optional attribute on a foreign instance; there is no __init__ to None-init).

The two array-likeness checks in ``cleanup_arrays`` (``hasattr(arr, "shape")`` /
``"size"`` / ``"__len__"``) are intentionally KEPT: they are external duck-typing
on array-like objects (numpy/torch/jax), allowed by CLAUDE.md's "External Library
Feature Detection". Narrowing them to ``isinstance(np.ndarray)`` would wrongly
exclude torch/jax tensors.
"""

from __future__ import annotations

import ast
import inspect

import pytest

import numpy as np

from mfgarchon.utils import exceptions, memory_management
from mfgarchon.utils.exceptions import SolutionNotAvailableError
from mfgarchon.utils.memory_management import memory_monitored


def _hasattr_call_count(func) -> int:
    """Count ``hasattr(...)`` call nodes in a function's source body."""
    source = inspect.getsource(func)
    tree = ast.parse(source.lstrip())
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hasattr"
    )


# ---------------------------------------------------------------------------
# Structural pins: the refactored sites must contain no hasattr() duck-typing.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_solver_state_has_no_hasattr():
    assert _hasattr_call_count(exceptions.validate_solver_state) == 0


@pytest.mark.unit
def test_memory_monitored_has_no_hasattr():
    # memory_monitored is a decorator factory; its source includes the inner
    # wrapper where the `_temp_arrays` check lived.
    assert _hasattr_call_count(memory_management.memory_monitored) == 0


# ---------------------------------------------------------------------------
# Behavioral pins: getattr-with-default preserves the original truthiness.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_solver_state_missing_attr_raises():
    """A solver lacking ``_solution_computed`` entirely must still raise.

    This exercises the getattr default-False branch that replaced
    ``not hasattr(...) or not ...``.
    """

    class BareSolver:
        pass

    with pytest.raises(SolutionNotAvailableError):
        exceptions.validate_solver_state(BareSolver(), "get_solution")


@pytest.mark.unit
def test_validate_solver_state_false_flag_raises():
    class NotComputed:
        def __init__(self):
            self._solution_computed = False

    with pytest.raises(SolutionNotAvailableError):
        exceptions.validate_solver_state(NotComputed(), "get_solution")


@pytest.mark.unit
def test_validate_solver_state_true_flag_passes():
    class Computed:
        def __init__(self):
            self._solution_computed = True

    assert exceptions.validate_solver_state(Computed(), "get_solution") is None


@pytest.mark.unit
def test_memory_monitored_cleanup_without_temp_arrays_does_not_raise():
    """A decorated class that never defines ``_temp_arrays`` must not raise.

    This is the foreign-instance path: the getattr default (None) is what keeps
    a bare attribute access from raising AttributeError here.
    """

    class NoTempArrays:
        @memory_monitored(max_memory_gb=100.0, cleanup_on_exit=True)
        def run(self):
            return "ok"

    assert NoTempArrays().run() == "ok"


@pytest.mark.unit
def test_memory_monitored_cleanup_with_temp_arrays_runs():
    """A decorated class that defines ``_temp_arrays`` still gets cleaned up."""

    class WithTempArrays:
        def __init__(self):
            self._temp_arrays: list[np.ndarray] = []

        @memory_monitored(max_memory_gb=100.0, cleanup_on_exit=True)
        def run(self):
            self._temp_arrays.append(np.ones((1500, 700)))
            return "done"

    assert WithTempArrays().run() == "done"

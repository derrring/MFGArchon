r"""The named stop outcomes must be reachable where the docs say they are (#1745).

Review found the whole path unexercised: `grep -rn "info\.extra" tests/` returned nothing, and
the full suite emitted zero "Newton stopped:" warnings, so `_stop` shipped with its payload
nested one level under `extra["extra"]` and nobody noticed. These tests are that coverage.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.utils.numerical.nonlinear_solvers import NewtonSolver


def _rootless(v):
    """x^2 + 1 has no real root, and a non-singular Jacobian away from the origin."""
    return v**2 + 1.0


def test_a_failed_line_search_names_itself_in_extra():
    """The changelog promises `SolverInfo.extra["reason"]`. It must be there, not nested.

    `SolverInfo.__init__` takes `**extra`, so passing a keyword literally named `extra` buries
    the payload at `info.extra["extra"]` while every document says `info.extra["reason"]`.
    """
    with pytest.warns(RuntimeWarning, match="Newton stopped"):
        _, info = NewtonSolver(max_iterations=40, tolerance=1e-10, sparse=False).solve(_rootless, np.array([1.3, 2.7]))
    assert not info.converged
    assert info.extra["reason"] == "line_search_failed"
    assert "extra" not in info.extra, "payload is nested one level; splat it instead"


def test_the_stop_path_does_not_shadow_jacobian_evals():
    """`newton_mfg_solver.py` reads `extra.get("jacobian_evals", 0)`.

    Nesting made that getter return its fallback on every stop path -- a pre-existing field
    silently reporting 0 where the true count was 20. The `.get()` default absorbed it, so
    there was no error to notice.
    """
    with pytest.warns(RuntimeWarning):
        _, info = NewtonSolver(max_iterations=40, tolerance=1e-10, sparse=False).solve(_rootless, np.array([1.3, 2.7]))
    assert info.extra.get("jacobian_evals", 0) > 0


def test_iterations_matches_the_residual_history_on_every_exit():
    """The three exits disagreed: converged returned `iteration + 1`, budget-exhausted returned
    `max_iterations`, and `_stop` returned the raw 0-based index -- so a stop that had performed
    a Jacobian and twenty residual evaluations reported "after 0 iterations"."""
    with pytest.warns(RuntimeWarning):
        _, stopped = NewtonSolver(max_iterations=40, tolerance=1e-10, sparse=False).solve(
            _rootless, np.array([1.3, 2.7])
        )
    assert stopped.iterations == len(stopped.residual_history)

    _, converged = NewtonSolver(max_iterations=40, tolerance=1e-10, sparse=False).solve(
        lambda v: v - 1.0, np.array([5.0, -3.0])
    )
    assert converged.converged
    assert converged.iterations == len(converged.residual_history)

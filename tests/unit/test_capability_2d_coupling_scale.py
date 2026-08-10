"""The 2-D smoke cell must run at a coupling strength comparable to its 1-D sibling's.

`coupling=lambda m: m` reads the density directly, and a probability density on a 2-D grid is
intrinsically peakier than the 1-D one carrying the same mass -- 1.818 against 9.549 on these
fixtures. Unscaled, the 2-D cell applied 5.3x the interaction and the matrix reported that as an
effect of dimension.

`capability_matrix._coupling_scale_2d()` derives the correction from the two fixtures, so asserting
that the constant equals its own derivation would be a tautology and is not what this file does.
What is not tautological is that the scale is **applied** -- delete it from the builder and the two
crowds see different `f(m)` again, which is the defect this exists to prevent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "capability_matrix.py"


def _load():
    spec = importlib.util.spec_from_file_location("capability_matrix_scale", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _crowd_f(problem) -> float:
    """`f(m)` at the densest point of the initial crowd, through the public Hamiltonian.

    Evaluated through `SeparableHamiltonian.__call__`, and taken as a difference at `p = 0` so
    nothing is assumed about the other terms: the control cost and any potential cancel, leaving
    `f(m_peak) - f(0)`.

    Two earlier versions of this helper reached for `hamiltonian.coupling` and then
    `hamiltonian.evaluate_hamiltonian`; neither exists on this class, so both assertions were red
    with `AttributeError` for a reason unrelated to what they test -- and the mutation that removes
    the scale entirely produced the identical red. A test that fails for the wrong reason
    discriminates nothing, and only running the mutation showed it.
    """
    peak = float(np.asarray(problem.m_initial, dtype=float).max())
    hamiltonian = problem.components.hamiltonian
    x = np.atleast_1d(0.5)
    p_zero = np.zeros(1)
    at_peak = float(np.asarray(hamiltonian(x, peak, p_zero)).ravel()[0])
    at_zero = float(np.asarray(hamiltonian(x, 0.0, p_zero)).ravel()[0])
    return at_peak - at_zero


def test_both_smoke_cells_apply_the_same_coupling_at_their_crowds():
    cm = _load()
    f_1d = _crowd_f(cm._smoke_problem())
    f_2d = _crowd_f(cm._smoke_problem_2d())
    assert f_1d == pytest.approx(f_2d, rel=1e-6), (
        f"the crowd sees f(m)={f_1d:.6f} in 1-D and {f_2d:.6f} in 2-D. The 2-D cell would report the "
        "difference in interaction strength as an effect of dimension, which is what it exists to "
        "measure. Check that `_smoke_problem_2d` still applies `_coupling_scale_2d()`."
    )


def test_the_scale_is_a_correction_and_not_a_no_op():
    """A scale of 1 would satisfy nothing above by accident, but it would satisfy everything if the
    two peaks ever coincided. Assert the correction is real on the fixtures as they stand, so a
    future edit that flattens one of them cannot make this file pass vacuously."""
    cm = _load()
    scale = cm._coupling_scale_2d()
    assert 0.05 < scale < 0.5, (
        f"the derived 2-D coupling scale is {scale:.6f}. Outside this range the two fixtures' crowds "
        "are no longer the several-fold apart that motivates the correction, and the comparability "
        "argument in `_smoke_problem_2d`'s docstring needs re-deriving rather than the bound relaxing."
    )

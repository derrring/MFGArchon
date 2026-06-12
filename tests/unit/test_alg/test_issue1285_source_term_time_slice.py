"""Issue #1285 (secondary bug): time-dependent source terms must see the time-t density slice.

``FixedPointIterator._compose_hjb_source`` / ``_compose_fp_source`` bind the current density
iterate ``M_old`` (full ``(Nt+1, Nx)`` array) into a ``(t, x)`` closure handed to the solver's
``source_term``. Before this fix the closure passed the WHOLE array to the problem-level
``source_term_hjb/fp(x, m, v, t)`` without slicing by ``t``, so a time-dependent source — e.g.
``lions_correction`` ``F[m]`` whose 2-D branch falls back to ``m[-1]`` — silently used the
terminal density slice for every backward time step. The sibling nonlocal branch already sliced
the value function via ``_get_time_slice`` (Issue #1259); the source branch did not.

These pins assert the source callback receives the time-``t`` slice (``m[k]``, 1-D), not the full
2-D array — failing against the pre-fix code and passing after.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_NX = 5
_NT = 4


class _Stub:
    """Minimal stand-in carrying only ``.problem`` — the only attribute the compose methods read."""

    def __init__(self, problem: MFGProblem) -> None:
        self.problem = problem


def _problem_with_source(captured: list, *, kind: str) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_NX], boundary_conditions=no_flux_bc(dimension=1))

    def spy(x, m, v, t):  # problem-level signature (x, m, v, t)
        captured.append((t, np.asarray(m).copy()))
        return np.zeros(np.asarray(x).shape[0])

    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    # source_term_hjb/fp are MFGProblem-level kwargs (mfg_problem.py:568), not MFGComponents fields.
    source_kw = {"source_term_hjb": spy} if kind == "hjb" else {"source_term_fp": spy}
    return MFGProblem(geometry=grid, T=0.4, Nt=_NT, sigma=0.3, components=comps, **source_kw)


def _row_indexed_field() -> np.ndarray:
    """(Nt+1, Nx) array whose row k is the constant k — so a slice reveals which time index was used."""
    return np.arange(_NT + 1, dtype=float)[:, None] * np.ones((_NT + 1, _NX))


def test_hjb_source_receives_time_slice_not_full_array():
    captured: list = []
    problem = _problem_with_source(captured, kind="hjb")
    M_full = _row_indexed_field()
    U_full = np.zeros((_NT + 1, _NX))
    closure = FixedPointIterator._compose_hjb_source(_Stub(problem), M_full, U_full)
    assert closure is not None
    x = np.linspace(0.0, 1.0, _NX)
    dt = problem.dt

    for k in (0, 2, _NT):  # t = 0, mid, terminal
        captured.clear()
        closure(k * dt, x)
        _, m = captured[-1]
        assert m.ndim == 1, f"source got the full {m.ndim}-D array, not the time-t slice (Issue #1285)"
        assert np.allclose(m, k), f"t={k * dt}: expected the k={k} density slice, got {m} (terminal-slice bug)"


def test_fp_source_receives_time_slice_not_full_array():
    captured: list = []
    problem = _problem_with_source(captured, kind="fp")
    M_full = _row_indexed_field()
    V_full = np.zeros((_NT + 1, _NX))
    closure = FixedPointIterator._compose_fp_source(_Stub(problem), M_full, V_full)
    assert closure is not None
    x = np.linspace(0.0, 1.0, _NX)
    dt = problem.dt

    for k in (0, 2, _NT):
        captured.clear()
        closure(k * dt, x)
        _, m = captured[-1]
        assert m.ndim == 1, f"FP source got the full {m.ndim}-D array, not the time-t slice (Issue #1285)"
        assert np.allclose(m, k), f"t={k * dt}: expected the k={k} density slice, got {m}"

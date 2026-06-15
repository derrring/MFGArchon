"""Issue #1361: single-source agreement for source/nonlocal/obstacle composition.

``_compose_hjb_source`` / ``_compose_fp_source`` were lifted out of
``FixedPointIterator`` into the shared ``coupling/source_composition.py`` so the
Picard coupler and the coupled-Newton ``MFGResidual`` path consume one copy of
the convention (the bug class behind #1259 and #1285 was a private second copy).

These pins assert:

1. The shared ``compose_hjb_source`` / ``compose_fp_source`` reproduce the prior
   ``FixedPointIterator`` closures byte-for-byte across a battery of inputs
   (a verbatim reference copy of the pre-#1361 logic is held below).
2. ``FixedPointIterator._compose_*`` now delegate to the shared helpers and
   therefore agree with them exactly.

If the shared helper ever drifts from the pinned convention, these fail.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.coupling.graph_coupling import _get_time_slice
from mfgarchon.alg.numerical.coupling.source_composition import (
    compose_fp_source,
    compose_hjb_source,
)
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_NX = 6  # Nx=6 intervals -> 7 grid points
_NT = 4


# ---------------------------------------------------------------------------
# Verbatim reference copy of the pre-#1361 FixedPointIterator closures.
# (Pinning target: the shared helper must reproduce this exactly.)
# ---------------------------------------------------------------------------


def _ref_compose_hjb_source(problem, m_current, u_current):
    has_nonlocal = problem.nonlocal_operator is not None
    has_source = problem.source_term_hjb is not None
    has_obstacle = problem.obstacle is not None
    if not (has_nonlocal or has_source or has_obstacle):
        return None

    def composed(t, x):
        terms = []
        if has_source:
            m_t = _get_time_slice(m_current, t, problem.dt)
            terms.append(problem.source_term_hjb(x, m_t, np.zeros_like(m_t), t))
        if has_obstacle:
            psi = problem.obstacle(x)
            eps = getattr(problem, "_penalty_eps", 1e6)
            terms.append((1.0 / eps) * np.maximum(0.0, psi.ravel()))
        if has_nonlocal:
            v_t = _get_time_slice(u_current, t, problem.dt)
            terms.append(problem.nonlocal_operator @ v_t)
        return sum(terms) if terms else np.zeros(x.shape[0])

    return composed


def _ref_compose_fp_source(problem, m_current, v_current):
    has_source = problem.source_term_fp is not None
    if not has_source:
        return None

    def composed(t, x):
        m_t = _get_time_slice(m_current, t, problem.dt)
        v_t = _get_time_slice(v_current, t, problem.dt)
        return problem.source_term_fp(x, m_t, v_t, t)

    return composed


# ---------------------------------------------------------------------------
# Problem factory
# ---------------------------------------------------------------------------


def _make_problem(**extra) -> MFGProblem:
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0)
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[_NX + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=0.4,
        Nt=_NT,
        sigma=0.3,
        components=comp,
        **extra,
    )


def _grid_size(problem) -> int:
    return int(np.prod(problem.geometry.get_grid_shape()))


def _row_indexed(gs: int) -> np.ndarray:
    """(Nt+1, gs) array whose row k is the constant k (reveals the slice index)."""
    return np.arange(_NT + 1, dtype=float)[:, None] * np.ones((_NT + 1, gs))


# Field configurations spanning every branch and their combinations.
def _field_configs(gs: int):
    src_hjb = lambda x, m, v, t: 0.7 * np.ones(x.shape[0]) + 0.1 * t + 0.01 * m  # noqa: E731
    src_fp = lambda x, m, v, t: 0.05 * np.ones(x.shape[0]) + 0.02 * v  # noqa: E731
    obstacle = lambda x: np.asarray(x) - 0.5  # noqa: E731
    nonlocal_op = 0.3 * np.eye(gs) + 0.05 * np.ones((gs, gs))
    return [
        ("source_hjb", dict(source_term_hjb=src_hjb)),
        ("source_fp", dict(source_term_fp=src_fp)),
        ("obstacle", dict(obstacle=obstacle)),
        ("nonlocal", dict(nonlocal_operator=nonlocal_op)),
        ("hjb+obstacle+nonlocal", dict(source_term_hjb=src_hjb, obstacle=obstacle, nonlocal_operator=nonlocal_op)),
        ("all", dict(source_term_hjb=src_hjb, source_term_fp=src_fp, obstacle=obstacle, nonlocal_operator=nonlocal_op)),
    ]


# ---------------------------------------------------------------------------
# Agreement: shared helper == verbatim reference, and == FixedPointIterator delegate
# ---------------------------------------------------------------------------


def test_hjb_composition_matches_reference_and_delegate():
    gs = _grid_size(_make_problem())
    M = _row_indexed(gs)
    U = 2.0 * _row_indexed(gs) + 0.5
    x = np.linspace(0.0, 1.0, gs)

    for name, kw in _field_configs(gs):
        problem = _make_problem(**kw)
        shared = compose_hjb_source(problem, M, U)
        ref = _ref_compose_hjb_source(problem, M, U)
        delegate = FixedPointIterator._compose_hjb_source(_StubIterator(problem), M, U)

        # Existence agreement: helper returns None iff reference returns None.
        assert (shared is None) == (ref is None), name
        assert (shared is None) == (delegate is None), name
        if shared is None:
            continue
        for k in range(_NT + 1):
            t = k * problem.dt
            a = shared(t, x)
            b = ref(t, x)
            c = delegate(t, x)
            np.testing.assert_array_equal(a, b, err_msg=f"{name} t={t}: shared != reference (HJB)")
            np.testing.assert_array_equal(a, c, err_msg=f"{name} t={t}: shared != FixedPointIterator delegate (HJB)")


def test_fp_composition_matches_reference_and_delegate():
    gs = _grid_size(_make_problem())
    M = _row_indexed(gs)
    V = 3.0 * _row_indexed(gs) - 1.0
    x = np.linspace(0.0, 1.0, gs)

    for name, kw in _field_configs(gs):
        problem = _make_problem(**kw)
        shared = compose_fp_source(problem, M, V)
        ref = _ref_compose_fp_source(problem, M, V)
        delegate = FixedPointIterator._compose_fp_source(_StubIterator(problem), M, V)

        assert (shared is None) == (ref is None), name
        assert (shared is None) == (delegate is None), name
        if shared is None:
            continue
        for k in range(_NT + 1):
            t = k * problem.dt
            np.testing.assert_array_equal(shared(t, x), ref(t, x), err_msg=f"{name} t={t}: shared != reference (FP)")
            np.testing.assert_array_equal(
                shared(t, x), delegate(t, x), err_msg=f"{name} t={t}: shared != delegate (FP)"
            )


def test_no_fields_returns_none():
    problem = _make_problem()
    M = _row_indexed(_grid_size(problem))
    assert compose_hjb_source(problem, M, M) is None
    assert compose_fp_source(problem, M, M) is None


class _StubIterator:
    """Minimal stand-in carrying only ``.problem`` (the only attribute compose reads)."""

    def __init__(self, problem) -> None:
        self.problem = problem

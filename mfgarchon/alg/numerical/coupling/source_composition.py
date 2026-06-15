"""Single-source composition of problem-level source/nonlocal/obstacle terms.

Issue #1361: lift ``_compose_hjb_source`` / ``_compose_fp_source`` out of
``FixedPointIterator`` so the Picard coupler (``FixedPointIterator``) and the
coupled-Newton path (``MFGResidual`` behind ``NewtonMFGSolver``) consume **one**
copy of the convention. A second private copy is the "parallel physics paths
with private convention copies" bug class that already produced two silent
divergences in the Picard copy alone:

- #1259 — ``nonlocal_operator`` ``J[v]`` computed but never applied.
- #1285 — time-dependent source used the terminal density slice ``m[-1]`` for
  every backward time step instead of the time-``t`` slice.

Both functions read the extended PDE fields off ``MFGProblem`` and return a
solver-level ``(t, x) -> array`` closure compatible with
``BaseHJBSolver.solve_hjb_system(source_term=...)`` /
``BaseFPSolver.solve_fp_system(source_term=...)``, or ``None`` when no relevant
field is active.

Conventions (mirrored verbatim from the prior ``FixedPointIterator`` copy):

- **Time-``t`` slicing** of the bound density / value iterates via
  :func:`graph_coupling._get_time_slice` (round-to-nearest index), so a
  time-dependent source sees ``m[k]`` / ``v[k]``, not the full ``(Nt+1, Nx)``
  array (Issue #1285).
- **Nonlocal term** applied as ``s += nonlocal_operator @ v_t`` with ``v_t`` the
  time-``t`` slice of the value function (Issue #1259), matching
  ``graph_mfg_solver``'s sign convention.
- **Obstacle** uses the approximate ``v = 0`` penalty ``(1/eps) * max(0, psi)``
  (``eps = problem._penalty_eps`` if set, else ``1e6``). Proper handling is the
  ``PenaltyHJBSolver`` wrapper (#924); both coupling paths use this same
  approximation so they **match** rather than silently diverge.
- The HJB source passes ``v = 0`` to ``source_term_hjb(x, m, v, t)`` (the
  problem-level source does not receive the value function in the HJB path),
  preserved from the prior copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .graph_coupling import _get_time_slice

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.core.mfg_problem import MFGProblem


def compose_hjb_source(
    problem: MFGProblem,
    m_current: NDArray,
    u_current: NDArray,
) -> Callable[[float, NDArray], NDArray] | None:
    """Compose problem-level HJB source terms into a solver-level callable.

    Reads ``source_term_hjb``, ``nonlocal_operator``, and ``obstacle`` from the
    problem, binds the spatial grid, current density ``m_current``, and current
    value function ``u_current``, and returns a ``(t, x) -> array`` closure.

    Args:
        problem: MFG problem definition carrying the extended PDE fields.
        m_current: Density iterate ``(Nt+1, Nx)`` bound for time-``t`` slicing.
        u_current: Value-function iterate ``(Nt+1, Nx)`` bound for the nonlocal
            term's time-``t`` slicing.

    Returns:
        Callable, or ``None`` if no HJB source / nonlocal / obstacle field is set.
    """
    has_nonlocal = problem.nonlocal_operator is not None
    has_source = problem.source_term_hjb is not None
    has_obstacle = problem.obstacle is not None

    if not (has_nonlocal or has_source or has_obstacle):
        return None

    def composed(t: float, x: NDArray) -> NDArray:
        terms: list[NDArray] = []
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


def compose_fp_source(
    problem: MFGProblem,
    m_current: NDArray,
    v_current: NDArray,
) -> Callable[[float, NDArray], NDArray] | None:
    """Compose problem-level FP source terms into a solver-level callable.

    Reads ``source_term_fp`` from the problem, binds the current density and
    value-function iterates, and returns a ``(t, x) -> array`` closure that
    evaluates the source at the time-``t`` slices of ``m`` and ``v``.

    Args:
        problem: MFG problem definition carrying the extended PDE fields.
        m_current: Density iterate ``(Nt+1, Nx)`` bound for time-``t`` slicing.
        v_current: Value-function iterate ``(Nt+1, Nx)`` bound for time-``t``
            slicing.

    Returns:
        Callable, or ``None`` if ``source_term_fp`` is not set.
    """
    has_source = problem.source_term_fp is not None

    if not has_source:
        return None

    def composed(t: float, x: NDArray) -> NDArray:
        m_t = _get_time_slice(m_current, t, problem.dt)
        v_t = _get_time_slice(v_current, t, problem.dt)
        return problem.source_term_fp(x, m_t, v_t, t)

    return composed

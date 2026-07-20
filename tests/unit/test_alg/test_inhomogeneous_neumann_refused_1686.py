"""FP solvers must refuse a Neumann value they do not honour (Issue #1686).

Every FP family declares `BCType.NEUMANN` in `_SUPPORTED_BC_TYPES` and then reads only the type:
`neumann_bc(value=g)` with `g != 0` is applied by the HJB side and silently discarded by the FP
side, so the coupled solve integrates a pair that is not adjoint and still reports
`converged=True`. Measured before this gate: `max|M(g=0) - M(g=-100)| = 0.0` on the FP side while
the HJB side moved by `1.6e+03`.

Declaring the type without honouring the value is the RFC #1574 class -- a declared surface
broader than the honoured code, silent in the gap. Until an inhomogeneous flux wall exists, the
library refuses the problem rather than solving a different one.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers import (
    FPFDMSolver,
    FPFVMSolver,
    FPGFDMSolver,
    FPParticleSolver,
    FPSLAdjointSolver,
    FPSLSolver,
)
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc, no_flux_bc

# All six families that declare BCType.NEUMANN. Enumerated from the package exports rather
# than hand-listed: a first attempt at this file guessed the module names and missed two of
# them (fp_particle's multi-line declaration and fp_semi_lagrangian_adjoint entirely).
FP_FAMILIES = [
    FPFDMSolver,
    FPFVMSolver,
    FPGFDMSolver,
    FPParticleSolver,
    FPSLSolver,
    FPSLAdjointSolver,
]


def _construct(solver_cls, problem):
    """Build a solver, supplying the extra arguments a family requires.

    FPGFDMSolver needs collocation points; every other family takes the problem alone. Handled
    here rather than by dropping GFDM from the list, since excluding a family would leave it
    unpinned -- and it declares BCType.NEUMANN like the rest.
    """
    if solver_cls is FPGFDMSolver:
        return solver_cls(problem, collocation_points=np.linspace(0.0, 1.0, 21).reshape(-1, 1))
    return solver_cls(problem)


def _problem(bc):
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=bc),
        Nt=10,
        T=1.0,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        ),
    )


@pytest.mark.parametrize("solver_cls", FP_FAMILIES)
def test_fp_refuses_a_neumann_value_it_would_discard(solver_cls):
    """A non-zero Neumann value must raise, naming the value and the issue."""
    with pytest.raises(NotImplementedError, match="honours only the homogeneous case"):
        _construct(solver_cls, _problem(neumann_bc(value=5.0, dimension=1)))


@pytest.mark.parametrize("solver_cls", FP_FAMILIES)
@pytest.mark.parametrize(
    ("bc_factory", "label"),
    [
        (lambda: neumann_bc(value=0.0, dimension=1), "neumann g=0"),
        (lambda: no_flux_bc(dimension=1), "no_flux"),
    ],
)
def test_fp_still_accepts_the_homogeneous_case(solver_cls, bc_factory, label):
    """`g = 0` is the whole of current usage and must remain accepted."""
    _construct(solver_cls, _problem(bc_factory()))


def test_hjb_still_accepts_a_neumann_value():
    """The HJB side honours `du/dn = g` and must not be caught by the FP-side refusal.

    This is what makes the flag per-solver rather than global: the same `BCType.NEUMANN` means
    `du/dn = g` on the HJB side and a prescribed flux `J.n = g` on the FP side.
    """
    HJBFDMSolver(_problem(neumann_bc(value=5.0, dimension=1)))


def test_the_flag_is_what_decides():
    """Pin the mechanism, not just the outcome -- flipping the flag flips the behaviour."""
    problem = _problem(neumann_bc(value=5.0, dimension=1))

    assert FPFDMSolver.honors_inhomogeneous_neumann is False
    assert HJBFDMSolver.honors_inhomogeneous_neumann is True

    class _Claims(FPFDMSolver):
        honors_inhomogeneous_neumann = True

    _Claims(problem)  # must not raise: the refusal is driven by the flag, not by the class


@pytest.mark.parametrize("solver_cls", FP_FAMILIES)
def test_a_time_dependent_neumann_value_is_also_refused(solver_cls):
    """`isinstance(value, (int, float))` alone lets a callable through.

    A first version of this gate guarded with that check, so
    `neumann_bc(value=lambda t: 5.0)` was accepted and discarded silently -- the behaviour the
    gate exists to stop, reintroduced by the guard itself. A callable cannot be shown identically
    zero here, so it is refused rather than assumed homogeneous, and the message says so.
    """
    with pytest.raises(NotImplementedError, match="time-dependent value cannot be checked"):
        _construct(solver_cls, _problem(neumann_bc(value=lambda t: 5.0, dimension=1)))


def test_hjb_still_accepts_a_time_dependent_neumann_value():
    """The HJB side honours `du/dn = g(t)`; the refusal must not reach it."""
    HJBFDMSolver(_problem(neumann_bc(value=lambda t: 5.0, dimension=1)))

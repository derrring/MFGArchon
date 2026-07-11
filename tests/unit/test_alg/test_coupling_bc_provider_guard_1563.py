"""Issue #1563 / RFC #1574: coupling loops that do not resolve dynamic BC providers must fail loud.

Only FixedPointIterator resolves a BCValueProvider (e.g. AdjointConsistentProvider) stored in a
BCSegment.value, via problem.using_resolved_bc() each Picard step. The other coupling loops
(FictitiousPlay, Block*, MFGResidual/Newton, MultiPopulation, RegimeSwitching) do not, so a provider
would otherwise reach the solver unresolved -- a deep GFDM ValueError, or a silent miss. They now
raise NotImplementedError up front via the single-source guard assert_bc_providers_resolvable.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import assert_bc_providers_resolvable
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    AdjointConsistentProvider,
    BCSegment,
    BCType,
    BoundaryConditions,
    no_flux_bc,
)


def _components():
    return MFGComponents(
        m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        u_terminal=lambda x: 0.0 * np.asarray(x, dtype=float),
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )


def _provider_problem(sigma=0.3):
    """A 1D problem whose BC carries an AdjointConsistentProvider (a dynamic BC provider)."""
    provider_bc = BoundaryConditions(
        segments=[
            BCSegment(
                name="left_ac",
                bc_type=BCType.ROBIN,
                alpha=0.0,
                beta=1.0,
                value=AdjointConsistentProvider(side="left", sigma=sigma),
                boundary="x_min",
            ),
            BCSegment(
                name="right_ac",
                bc_type=BCType.ROBIN,
                alpha=0.0,
                beta=1.0,
                value=AdjointConsistentProvider(side="right", sigma=sigma),
                boundary="x_max",
            ),
        ],
        dimension=1,
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=provider_bc)
    return MFGProblem(geometry=grid, T=0.1, Nt=2, sigma=sigma, components=_components())


def _plain_problem(sigma=0.3):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=grid, T=0.1, Nt=2, sigma=sigma, components=_components())


def test_guard_helper_discriminates():
    """The single-source guard raises for a provider-bearing problem, passes for a static BC."""
    assert _provider_problem().get_boundary_conditions().has_providers() is True
    with pytest.raises(NotImplementedError, match="1563"):
        assert_bc_providers_resolvable(_provider_problem(), "SomeLoop")
    # Static BC -> no providers -> no raise.
    assert_bc_providers_resolvable(_plain_problem(), "SomeLoop")


def test_fictitious_play_rejects_provider_bc():
    """The wiring: constructing a non-resolving loop with a provider-bearing problem raises up front
    (before the solvers are used), naming the loop. Solvers are built on a static problem since the
    guard fires on self.problem before touching them."""
    from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

    plain = _plain_problem()
    hjb, fp = HJBFDMSolver(plain), FPFDMSolver(plain)
    with pytest.raises(NotImplementedError, match="1563"):
        FictitiousPlayIterator(_provider_problem(), hjb_solver=hjb, fp_solver=fp)

    # A static-BC problem constructs fine (the guard is scoped to providers only).
    FictitiousPlayIterator(plain, hjb_solver=hjb, fp_solver=fp)

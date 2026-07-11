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


def test_single_problem_loops_reject_provider_bc():
    """BlockIterator and MFGResidual (the other two single-self.problem loops) also fail loud."""
    from mfgarchon.alg.numerical.coupling.block_iterators import BlockJacobiIterator
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

    plain = _plain_problem()
    hjb, fp = HJBFDMSolver(plain), FPFDMSolver(plain)
    for cls in (BlockJacobiIterator, MFGResidual):
        with pytest.raises(NotImplementedError, match="1563"):
            cls(_provider_problem(), hjb_solver=hjb, fp_solver=fp)


def test_regime_switching_guards_every_regime_not_just_the_first():
    """Load-bearing claim: RegimeSwitchingIterator guards EVERY regime's problem, not just
    problems[0] (which is the representative self.problem). A provider on the SECOND regime must
    raise, naming 'regime 1' -- a revert to guarding only problems[0] would let it through."""
    from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
    from mfgarchon.core.regime_switching import RegimeSwitchingConfig

    plain = _plain_problem()
    # regime 0 = static, regime 1 = provider-bearing. Solvers built on the static problem (the guard
    # fires on the problems list before any solver is used).
    problems = [plain, _provider_problem()]
    config = RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]]))
    hjbs = [HJBFDMSolver(plain), HJBFDMSolver(plain)]
    fps = [FPFDMSolver(plain), FPFDMSolver(plain)]
    with pytest.raises(NotImplementedError, match="regime 1"):
        RegimeSwitchingIterator(problems=problems, regime_config=config, hjb_solvers=hjbs, fp_solvers=fps)

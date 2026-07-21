"""RFC #1574 Phase 0 capability-honesty guards: fail loud where a declared/dispatched surface is
broader than the code that honors it.

- #1560: HJBSemiLagrangianSolver collapses a mixed per-axis BC (segments mapping to different
  geometric operations, e.g. no-flux + periodic) to the first segment's single op applied to all axes.
- #1564: HJBFDMSolver.build_linearized_operator (the strict-adjoint FP operator, #707) hardcodes
  no-flux at every boundary while HJBFDMSolver declares DIRICHLET supported for the normal solve.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry.boundary import BCSegment, BCType, dirichlet_bc, mixed_bc, no_flux_bc
from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid


def _components() -> MFGComponents:
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))
    return MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0)


def _problem(bc, bounds, npts) -> MFGProblem:
    grid = TensorProductGrid(bounds=bounds, num_points=npts, boundary_conditions=bc)
    return MFGProblem(geometry=grid, T=0.2, Nt=2, sigma=0.1, components=_components())


def test_sl_mixed_per_axis_bc_fails_loud_1560():
    """no-flux (x) + periodic (y) map to different geometric ops (reflect vs periodic); SL collapses
    them to one op on all axes, so construction must raise rather than silently pick the first."""
    mixed = mixed_bc(
        dimension=2,
        segments=[
            BCSegment(name="wx", boundary="x_min", bc_type=BCType.NO_FLUX),
            BCSegment(name="ex", boundary="x_max", bc_type=BCType.NO_FLUX),
            BCSegment(name="py0", boundary="y_min", bc_type=BCType.PERIODIC),
            BCSegment(name="py1", boundary="y_max", bc_type=BCType.PERIODIC),
        ],
    )
    with pytest.raises(NotImplementedError, match=r"mixed per-axis|1560"):
        HJBSemiLagrangianSolver(problem=_problem(mixed, [(0.0, 1.0), (0.0, 1.0)], [6, 6]))


def test_sl_uniform_bc_still_constructs_1560():
    """A single BC type across all axes must be unaffected by the mixed-BC guard."""
    solver = HJBSemiLagrangianSolver(problem=_problem(no_flux_bc(dimension=2), [(0.0, 1.0), (0.0, 1.0)], [6, 6]))
    assert solver is not None


def test_build_linearized_operator_fails_loud_on_dirichlet_1564():
    """The strict-adjoint FP operator hardcodes no-flux; a Dirichlet BC must raise, not be silently
    treated as mass-conserving no-flux."""
    solver = HJBFDMSolver(problem=_problem(dirichlet_bc(dimension=1), [(0.0, 1.0)], [11]))
    U, M = np.zeros(11), np.ones(11) / 11
    with pytest.raises(NotImplementedError, match=r"1564|no-flux"):
        solver.build_linearized_operator(U, M, time=0.0)


def test_build_linearized_operator_ok_on_no_flux_1564():
    """No-flux (the honored BC) must still build the operator."""
    solver = HJBFDMSolver(problem=_problem(no_flux_bc(dimension=1), [(0.0, 1.0)], [11]))
    U, M = np.zeros(11), np.ones(11) / 11
    A = solver.build_linearized_operator(U, M, time=0.0)
    assert A.shape == (11, 11)

"""The boundary-condition owner's refusals answer on every coupled route, whatever BC its solvers carry (#3005).

`IterationState.start`, which every coupled route calls per component at each solve, reads the problem's BC from its
one owner, `problem.get_boundary_conditions()` (#2424). When both solvers carry their own BC, no solver asks the owner, so this
read is the one that refuses two declarations that differ (#2530) and a FLUX on the shared BC (#2581). Up to e14eecd6
GraphMFGSolver never asked, and the six other routes asked through the provider check #2424 deleted.

The pin this fix owes (AGENTS.md § Review before merge): its sixteen refusal cases fail at 56aa9fba, where every route
solved them, and Graph's two at e14eecd6. The control, one declaration, solves on every route.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import (
    BlockGaussSeidelIterator,
    BlockJacobiIterator,
    FictitiousPlayIterator,
    FixedPointIterator,
)
from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc

_FLUX = BoundaryConditions(
    dimension=1,
    segments=[
        BCSegment(name="in", bc_type=BCType.FLUX, value=0.1, boundary="x_min"),
        BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_max"),
    ],
)
_CASES = {
    "one declaration": (no_flux_bc(dimension=1), None, None),
    "two declarations that differ": (no_flux_bc(dimension=1), dirichlet_bc(dimension=1, value=0.7), r"declares its"),
    "a FLUX on the shared BC": (_FLUX, None, r"a FLUX boundary condition"),
}
_ROUTES = [
    "FixedPoint",
    "FictitiousPlay",
    "Block GS",
    "Block Jacobi",
    "Newton",
    "Graph",
    "RegimeSwitching",
    "MultiPopulation",
]


def _problem(grid_bc, components_bc):
    kwargs = {
        "hamiltonian": SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        "m_initial": lambda x: 1.0 + 0.5 * np.cos(np.pi * np.asarray(x[0], dtype=float)),
        "u_terminal": lambda x: (np.asarray(x[0], dtype=float) - 0.3) ** 2,
    }
    if components_bc is not None:
        kwargs["boundary_conditions"] = components_bc
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=grid_bc)
    return MFGProblem(geometry=grid, components=MFGComponents(**kwargs), T=0.5, Nt=10, volatility=0.4)


def _solvers_with_their_own_bc(problem):
    """Both solvers carry their own BC, so neither asks the problem's owner."""
    points = np.asarray(problem.geometry.get_spatial_grid()).reshape(-1, 1)
    own = no_flux_bc(dimension=1)
    return HJBGFDMSolver(problem, collocation_points=points, boundary_conditions=own), FPFDMSolver(
        problem, boundary_conditions=own
    )


def _build_and_solve(route, grid_bc, components_bc):
    k = 2 if route in ("Graph", "RegimeSwitching") else 1
    problems = [_problem(grid_bc, components_bc) for _ in range(k)]
    pairs = [_solvers_with_their_own_bc(p) for p in problems]
    hjb, fp = [h for h, _ in pairs], [f for _, f in pairs]
    if route == "Graph":
        GraphMFGSolver(problems, AdjacencyCoupling(np.zeros((2, 2))), hjb, fp, max_iterations=2).solve()
    elif route == "RegimeSwitching":
        Q = np.zeros((2, 2))
        RegimeSwitchingIterator(problems, RegimeSwitchingConfig(transition_matrix=Q), hjb, fp, max_iterations=2).solve()
    elif route == "MultiPopulation":
        MultiPopulationIterator(MultiPopulationProblem(populations=problems, population_names=["a"]), hjb, fp).solve(
            max_iterations=2
        )
    else:
        cls = {
            "FixedPoint": FixedPointIterator,
            "FictitiousPlay": FictitiousPlayIterator,
            "Block GS": BlockGaussSeidelIterator,
            "Block Jacobi": BlockJacobiIterator,
            "Newton": NewtonMFGSolver,
        }[route]
        cls(problems[0], hjb[0], fp[0]).solve(max_iterations=2, verbose=False)


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("route", _ROUTES)
def test_the_owner_refuses_what_it_refuses_on_every_route(route, case):
    grid_bc, components_bc, refusal = _CASES[case]
    if refusal is None:
        _build_and_solve(route, grid_bc, components_bc)
    else:
        with pytest.raises(ValueError, match=refusal):
            _build_and_solve(route, grid_bc, components_bc)

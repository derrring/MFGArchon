"""A coupled route handed network problems refuses them by name (#2923).

A network problem's boundary conditions are None or a ``GraphBCConfig``. Up to e14eecd6, FictitiousPlay, both Block
iterators, Newton's residual, MultiPopulation and RegimeSwitching read them through ``assert_bc_providers_resolvable``
when they were built, and each died there with ``AttributeError: ... has no attribute 'has_providers'``. That check
went with the boundary-value provider concept (#2424). Each route now reaches its own refusal of the geometry: the
iterate owner's (``IterationState``), or, for two network populations, MultiPopulation's #2335 refusal, the answer
#2923 asked for.

A labelled defect pin (AGENTS.md, admission class 3): its fourteen AttributeError cases fail at e14eecd6.
FixedPointIterator and GraphMFGSolver, which refused the geometry by name before and after, are the controls.
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
from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkPolicyIterationHJBSolver
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.extensions.topology import NetworkMFGProblem
from mfgarchon.geometry.boundary.applicator_graph import GraphBCConfig, GraphBCType, NodeBC
from mfgarchon.geometry.graph.network_geometry import GridNetwork

_GEOMETRY = r"the problem's geometry must be a CartesianGrid"
_DOORS = r"population\(s\) \[0, 1\] sit on a grid of dimension \['network', 'network'\]"


def _network_problem(with_node_bc: bool) -> NetworkMFGProblem:
    bc = GraphBCConfig(node_bcs=[NodeBC(nodes=[0], bc_type=GraphBCType.ABSORBING, value=0.0)]) if with_node_bc else None
    net = GridNetwork(width=3, height=3, boundary_conditions=bc)
    net.create_network()
    return NetworkMFGProblem(geometry=net, T=0.5, Nt=10)


def _build_and_solve(route: str, with_node_bc: bool) -> None:
    if route == "Graph":
        problems = [_network_problem(with_node_bc) for _ in range(2)]
        hjb = [NetworkPolicyIterationHJBSolver(p) for p in problems]
        fp = [FPNetworkSolver(p) for p in problems]
        GraphMFGSolver(
            problems, AdjacencyCoupling(np.array([[0.0, 1.0], [1.0, 0.0]])), hjb, fp, max_iterations=1
        ).solve()
        return
    if route in ("MultiPopulation, two", "MultiPopulation, one", "RegimeSwitching"):
        k = 1 if route == "MultiPopulation, one" else 2
        problems = [_network_problem(with_node_bc) for _ in range(k)]
        hjb = [NetworkPolicyIterationHJBSolver(p) for p in problems]
        fp = [FPNetworkSolver(p) for p in problems]
        if route == "RegimeSwitching":
            Q = np.array([[-0.5, 0.5], [0.5, -0.5]])
            RegimeSwitchingIterator(
                problems, RegimeSwitchingConfig(transition_matrix=Q), hjb, fp, max_iterations=1
            ).solve()
            return
        names = [f"p{i}" for i in range(k)]
        MultiPopulationIterator(MultiPopulationProblem(populations=problems, population_names=names), hjb, fp).solve(
            max_iterations=1
        )
        return
    problem = _network_problem(with_node_bc)
    cls = {
        "FixedPoint": FixedPointIterator,
        "FictitiousPlay": FictitiousPlayIterator,
        "Block Gauss-Seidel": BlockGaussSeidelIterator,
        "Block Jacobi": BlockJacobiIterator,
        "Newton": NewtonMFGSolver,
    }[route]
    cls(problem, NetworkPolicyIterationHJBSolver(problem), FPNetworkSolver(problem)).solve(
        max_iterations=1, verbose=False
    )


@pytest.mark.parametrize("with_node_bc", [True, False], ids=["node BC", "no BC"])
@pytest.mark.parametrize(
    ("route", "refusal"),
    [
        ("FixedPoint", _GEOMETRY),
        ("FictitiousPlay", _GEOMETRY),
        ("Block Gauss-Seidel", _GEOMETRY),
        ("Block Jacobi", _GEOMETRY),
        ("Newton", _GEOMETRY),
        ("MultiPopulation, two", _DOORS),
        ("MultiPopulation, one", _GEOMETRY),
        ("RegimeSwitching", _GEOMETRY),
        ("Graph", _GEOMETRY),
    ],
)
def test_a_network_problem_is_refused_by_name(route, refusal, with_node_bc):
    with pytest.raises((ValueError, NotImplementedError), match=refusal):
        _build_and_solve(route, with_node_bc)

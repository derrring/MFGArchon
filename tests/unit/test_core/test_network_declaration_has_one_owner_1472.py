"""#1472: a network MFG is declared in one place, `MFGProblem`'s model/domain/conditions form, and its graph has one
holder, the network geometry (#1472 plan v1, #1472 comment 7224953131, approved in 7224969065; stage 1's plan v2,
7226702955, approved in 7226717273). Stage 1 of two.

Stage 1 holds:
- **The solvers read the graph from the geometry.** By AST over the network solvers: no read of a graph attribute
  (``GRAPH``) on the problem they hold.
- **No attribute outside the geometry holds the graph.** By AST over ``mfgarchon/`` outside ``geometry/graph/``: no
  assignment stores on an attribute a value read from a graph (``GRAPH_READS``), except those ``HOLDERS_ALLOWED`` names
  with why: a solver's copy taken at construction, which the geometry's graph generation guards; a member stage 2
  removes; or a name the census matches that is not the network's graph. An allowance that covers nothing fails.
- **A rebuilt graph reaches the Hamiltonian and the problem, and is refused at a solver built before it,** through the
  generation the geometry advances on each build.
- **A network solver refuses by name, at construction, a problem that lacks what it reads** (#2045).

What the scans cannot follow: a read through ``getattr``, a graph object stored in a container rather than on an
attribute (the solvers' neighbour lists), and a value held under a local name more than one assignment back.

Stage 2 removes the remaining members, each a strict xfail or an allowance here naming it: `NetworkMFGProblem` and
`NetworkMFGComponents`, the problem's two sources for its initial density and terminal value, which disagree,
``MFGProblem(network=...)``, and the node-BC readers.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

import numpy as np

import mfgarchon

PACKAGE = Path(mfgarchon.__file__).resolve().parent
SOLVERS = sorted(str(p.relative_to(PACKAGE)) for p in (PACKAGE / "alg/numerical/network_solvers").glob("*.py"))
GRAPH = {
    "num_nodes",
    "network_data",
    "get_adjacency_matrix",
    "get_laplacian_matrix",
    "get_node_neighbors",
    "adjacency_matrix",
    "laplacian_matrix",
    "incidence_matrix",
}
GRAPH_READS = GRAPH | {
    "get_graph_laplacian",
    "get_sparse_laplacian",
    "get_neighbors",
    "get_edge_weight",
    "number_of_nodes",
    "nodes",
}
HOLDERS = {"problem", "network_problem"}
STAGE_2 = "#1472 stage 2: MFGProblem takes the model/domain/conditions form on a network, and the two classes go"
GUARDED = "a solver's copy taken at construction, which refuse_a_rebuilt_graph guards at each solve"
NOT_THE_GRAPH = "a name the scan matches that holds no network graph"
# (path, the enclosing class and function, the attribute) -> why it may hold what it reads
HOLDERS_ALLOWED = {
    ("alg/numerical/network_solvers/fp_network.py", "FPNetworkSolver.__init__", "self.num_nodes"): GUARDED,
    ("alg/numerical/network_solvers/fp_network.py", "FPNetworkSolver.__init__", "self.adjacency_matrix"): GUARDED,
    ("alg/numerical/network_solvers/fp_network.py", "FPNetworkSolver.__init__", "self.laplacian_matrix"): GUARDED,
    (
        "alg/numerical/network_solvers/fp_network.py",
        "FPNetworkSolver._initialize_network_operators",
        "self.dt_stable",
    ): GUARDED,
    ("alg/numerical/network_solvers/hjb_network.py", "NetworkHJBSolver.__init__", "self.num_nodes"): GUARDED,
    ("alg/numerical/network_solvers/hjb_network.py", "NetworkHJBSolver.__init__", "self.adjacency_matrix"): GUARDED,
    ("alg/numerical/network_solvers/hjb_network.py", "NetworkHJBSolver.__init__", "self.laplacian_matrix"): GUARDED,
    ("core/mfg_problem.py", "MFGProblem._init_network", "self.num_nodes"): STAGE_2 + " (MFGProblem(network=...))",
    ("core/mfg_problem.py", "MFGProblem._init_network", "self.adjacency_matrix"): STAGE_2
    + " (MFGProblem(network=...))",
    ("core/mfg_problem.py", "MFGProblem._init_network", "self.spatial_shape"): STAGE_2 + " (MFGProblem(network=...))",
    ("core/mfg_problem.py", "MFGProblem._init_network", "self.geometry"): STAGE_2 + " (MFGProblem(network=...))",
    ("core/mfg_problem.py", "MFGProblem._init_network", "self.num_spatial_points"): STAGE_2
    + " (MFGProblem(network=...))",
    ("geometry/boundary/protocols.py", "BaseGraphApplicator.__init__", "self._num_nodes"): STAGE_2
    + " (the node-BC readers)",
    (
        "alg/numerical/network_solvers/fp_network.py",
        "FPNetworkSolver.solve_fp_system",
        "self._current_rates",
    ): NOT_THE_GRAPH,
    (
        "alg/numerical/network_solvers/fp_network.py",
        "FPNetworkSolver.solve_fp_system",
        "self._reported_mass_drift",
    ): NOT_THE_GRAPH,
    (
        "alg/numerical/meshless_galerkin/discretization.py",
        "MeshlessGalerkinDiscretization.__init__",
        "self._nodes",
    ): NOT_THE_GRAPH,
    (
        "alg/numerical/meshless_galerkin/scni_discretization.py",
        "MeshlessSCNIDiscretization.__init__",
        "self._nodes",
    ): NOT_THE_GRAPH,
    (
        "alg/numerical/pde_solvers/implicit_heat.py",
        "ImplicitHeatSolver._build_system_matrices",
        "self.L",
    ): NOT_THE_GRAPH,
    (
        "operators/integro_diff/graphon_coupling.py",
        "GraphonCouplingOperator._setup_quadrature",
        "self._quad_y",
    ): NOT_THE_GRAPH,
    (
        "operators/interaction/energy_functionals.py",
        "QuadraticInteractionEnergy.__init__",
        "self._nodes",
    ): NOT_THE_GRAPH,
}


def graph_reads_on_the_problem(source: str, path: str) -> list[str]:
    """Each ``problem.<graph attribute>`` or ``self.network_problem.<graph attribute>`` in ``source``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in GRAPH:
            holder = node.value
            name = getattr(holder, "id", getattr(holder, "attr", ""))
            if name in HOLDERS:
                found.append(f"{path}:{node.lineno} reads {ast.unparse(node)}")
    return found


def _reads_the_graph(value: ast.AST, local: set[str]) -> bool:
    for node in ast.walk(value):
        name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)
        if name in GRAPH_READS or (isinstance(node, ast.Name) and node.id in local):
            return True
    return False


def graph_holders(source: str, path: str) -> list[tuple[str, str, str, int]]:
    """Each assignment in ``source`` that stores on an attribute a value read from a graph: (path, the enclosing class and
    function, the attribute, the line). A name bound to such a value, or a parameter named ``network_data``, counts as
    one when it is stored."""
    found = []
    tree = ast.parse(source)
    for owner in [tree, *(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))]:
        for fn in (n for n in owner.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            local = {a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)} & {"network_data"}
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and _reads_the_graph(node.value, local):
                            local.add(target.id)
            for node in ast.walk(fn):
                pairs = [(t, node.value) for t in node.targets] if isinstance(node, ast.Assign) else []
                if isinstance(node, ast.AnnAssign) and node.value is not None:
                    pairs.append((node.target, node.value))
                for target, value in pairs:
                    if isinstance(target, ast.Attribute) and _reads_the_graph(value, local):
                        where = f"{getattr(owner, 'name', '<module>')}.{fn.name}"
                        found.append((path, where, ast.unparse(target), node.lineno))
    return found


def _package_sources() -> dict[str, str]:
    return {
        str(p.relative_to(PACKAGE)): p.read_text()
        for p in sorted(PACKAGE.rglob("*.py"))
        if not str(p.relative_to(PACKAGE)).startswith("geometry/graph/")
    }


@pytest.mark.parametrize(
    "planted",
    [
        "n = problem.num_nodes",
        "A = self.network_problem.get_adjacency_matrix()",
        "w = self.network_problem.network_data.get_edge_weight(i, j)",
    ],
)
def test_the_scan_flags_a_planted_read(planted):
    assert len(graph_reads_on_the_problem(planted, "a/solver.py")) == 1


def test_the_scan_passes_a_read_of_the_geometry():
    clean = "n = problem.geometry.num_nodes\nA = self.network_problem.geometry.network_data.adjacency_matrix\n"
    assert graph_reads_on_the_problem(clean, "a/solver.py") == []


def test_the_network_solvers_read_the_graph_from_the_geometry():
    found = [read for path in SOLVERS for read in graph_reads_on_the_problem((PACKAGE / path).read_text(), path)]
    assert found == [], "\n".join(found)


@pytest.mark.parametrize(
    "planted",
    [
        "class H:\n    def __init__(self, network_data):\n        self.network_data = network_data",
        "class P:\n    def __init__(self, geometry):\n        self.graph = geometry.network_data",
        "class S:\n    def __init__(self, geometry):\n        A = geometry.network_data.adjacency_matrix\n        self.A = A",
    ],
)
def test_the_holder_scan_flags_a_planted_holder(planted):
    assert len(graph_holders(planted, "a/holder.py")) == 1


def test_no_attribute_outside_the_geometry_holds_the_graph():
    found = [h for path, source in _package_sources().items() for h in graph_holders(source, path)]
    left = [h for h in found if h[:3] not in HOLDERS_ALLOWED]
    unused = sorted(set(HOLDERS_ALLOWED) - {h[:3] for h in found})
    assert left == [], "\n".join(f"{p}:{line} {where} | {target}" for p, where, target, line in left)
    assert unused == [], "\n".join(f"{p} {where} {target}: holds nothing" for p, where, target in unused)


def _network_problem():
    pytest.importorskip("igraph")
    from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import RandomNetwork

    np.random.seed(1)
    network = RandomNetwork(8, connection_prob=0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = NetworkMFGProblem(geometry=network, T=0.3, Nt=4, components=NetworkMFGComponents())
    return network, problem


def test_a_rebuilt_graph_reaches_the_hamiltonian_and_the_problem():
    """The Hamiltonian and the problem hold no graph of their own: after the geometry rebuilds it, both read the new one."""
    network, problem = _network_problem()
    old = network.network_data
    np.random.seed(2)
    network.create_network()
    assert network.network_data is not old
    assert problem.hamiltonian_class.network_data is network.network_data
    assert problem.network_data is network.network_data


@pytest.mark.parametrize("solver", ["FPNetworkSolver", "NetworkHJBSolver", "NetworkPolicyIterationHJBSolver"])
def test_a_solver_built_before_the_graph_was_rebuilt_refuses_to_solve(solver):
    """Mutant: the geometry's generation not advanced on a rebuild, so the solver solves on the old graph's operators
    while the Hamiltonian reads the new graph."""
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
    from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver, NetworkPolicyIterationHJBSolver

    network, problem = _network_problem()
    n, nt = problem.num_nodes, problem.Nt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = {
            "FPNetworkSolver": lambda: FPNetworkSolver(problem, diffusion_coefficient=0.05),
            "NetworkHJBSolver": lambda: NetworkHJBSolver(problem),
            "NetworkPolicyIterationHJBSolver": lambda: NetworkPolicyIterationHJBSolver(problem),
        }[solver]()
    np.random.seed(2)
    network.create_network()
    if solver == "FPNetworkSolver":

        def solve():
            return built.solve_fp_system(np.full(n, 1.0 / n), np.zeros((nt + 1, n)), show_progress=False)
    else:

        def solve():
            return built.solve_hjb_system(
                M_density=np.full((nt + 1, n), 1.0 / n), U_terminal=np.zeros(n), volatility=0.2
            )

    with pytest.raises(RuntimeError, match="graph was rebuilt after this"):
        solve()


@pytest.mark.parametrize("solver", ["FPNetworkSolver", "NetworkHJBSolver", "NetworkPolicyIterationHJBSolver"])
def test_a_network_solver_refuses_a_problem_it_cannot_read_at_construction(solver):
    """#2045's refusal, for a problem built otherwise than as a NetworkMFGProblem on a network geometry: an MFGProblem
    with a network Hamiltonian bound to the same geometry. Mutant: the refusal at construction removed, so the solve
    fails later on a missing attribute."""
    pytest.importorskip("igraph")
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
    from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver, NetworkPolicyIterationHJBSolver
    from mfgarchon.extensions.topology import NetworkHamiltonian
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    network = GridNetwork(width=3, height=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            model=Model(hamiltonian=NetworkHamiltonian().bind(network), volatility=0.2),
            domain=network,
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0 / 6, T=0.3),
            Nt=4,
        )
    cls = {
        "FPNetworkSolver": FPNetworkSolver,
        "NetworkHJBSolver": NetworkHJBSolver,
        "NetworkPolicyIterationHJBSolver": NetworkPolicyIterationHJBSolver,
    }[solver]
    with pytest.raises(NotImplementedError, match="from its problem, which this MFGProblem on a GridNetwork lacks"):
        cls(problem)


@pytest.mark.xfail(strict=True, reason=STAGE_2)
def test_no_class_declares_a_network_beside_the_model_form():
    sources = {str(p.relative_to(PACKAGE)): p.read_text() for p in sorted(PACKAGE.rglob("*.py"))}
    found = [
        f"{path}:{node.lineno} {node.name}"
        for path, source in sources.items()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef)
        and "Network" in node.name
        and any(getattr(b, "id", "") in ("MFGProblem", "MFGComponents") for b in node.bases)
    ]
    assert found == [], "\n".join(found)


@pytest.mark.xfail(strict=True, reason=STAGE_2)
def test_a_network_problems_readers_of_its_conditions_agree():
    pytest.importorskip("igraph")
    from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    given = [0.4, 0.3, 0.2, 0.1]
    components = NetworkMFGComponents(
        initial_node_density_func=lambda node: given[node], terminal_node_value_func=lambda node: float(node)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = NetworkMFGProblem(geometry=GridNetwork(width=4, height=1), T=0.5, Nt=5, components=components)
    np.testing.assert_array_equal(np.ravel(problem.get_m_initial()), problem.get_initial_density())
    np.testing.assert_array_equal(np.ravel(problem.get_u_terminal()), problem.terminal_value(None))

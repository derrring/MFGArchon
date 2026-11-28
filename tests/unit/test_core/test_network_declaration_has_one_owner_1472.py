"""#1472: a network MFG is declared in one place, `MFGProblem`'s model/domain/conditions form, and its graph has one
holder, the network geometry (#1472 plan v1, #1472 comment 7224953131, approved in 7224969065; stage 1's plan v2,
7226702955, approved in 7226717273; stage 2's plan, 7232724466, approved in 7232886239). Stages 1 and 2a of three.

Stage 1 holds:
- **The solvers read the graph from the geometry.** By AST over the network solvers: no read of a graph attribute
  (``GRAPH``) on the problem they hold.
- **No attribute outside the geometry holds the graph.** By AST over ``mfgarchon/`` outside ``geometry/graph/``: no
  assignment stores on an attribute a value read from a graph (``GRAPH_READS``), except those ``HOLDERS_ALLOWED`` names
  with why: a solver's copy taken at construction, which the geometry's graph generation guards; a member stage 2
  removes; or a name the census matches that is not the network's graph. An allowance that covers nothing fails.
- **A Hamiltonian reads a graph only through the network geometry it is bound to,** and refuses by name to read one
  unbound or to bind a geometry that holds no graph or whose graph is not built.
- **A rebuilt graph reaches the Hamiltonian and the problem, and is refused at a solver built before it,** through the
  generation the geometry advances on each build.
- **A network solver refuses by name, at construction, a problem that lacks what it reads** (#2045), and nothing it
  does not read.

What the scans cannot follow: a read through ``getattr``, a graph object stored in a container rather than on an
attribute (the solvers' neighbour lists), and a value held under a local name more than one assignment back.

Stage 2a holds, each by a pin below: a problem on a network binds the `NetworkHamiltonian` it is given and refuses any
other; its `Conditions` are the one source of its initial density and terminal value, read at each node with the node as
x; the network solvers read the node BCs from the geometry where they apply them.

Stage 2b removes the remaining members, each a strict xfail or an allowance here naming it: `NetworkMFGProblem`,
`NetworkMFGComponents` and ``MFGProblem(network=...)``.
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
STAGE_2B = "#1472 stage 2b: NetworkMFGProblem, NetworkMFGComponents and MFGProblem(network=...) are deleted"
READ_WHERE_USED = "an applicator built from the geometry where its node BCs are read (#1472 stage 2a)"
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
    ("core/mfg_problem.py", "MFGProblem._init_network", "self.num_nodes"): STAGE_2B,
    ("core/mfg_problem.py", "MFGProblem._init_network", "self.adjacency_matrix"): STAGE_2B,
    ("core/mfg_problem.py", "MFGProblem._init_network", "self.spatial_shape"): STAGE_2B,
    ("core/mfg_problem.py", "MFGProblem._init_network", "self.geometry"): STAGE_2B,
    ("core/mfg_problem.py", "MFGProblem._init_network", "self.num_spatial_points"): STAGE_2B,
    ("geometry/boundary/protocols.py", "BaseGraphApplicator.__init__", "self._num_nodes"): READ_WHERE_USED,
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


def test_a_hamiltonian_reads_a_graph_only_through_a_network_geometry_it_is_bound_to():
    """An unbound `NetworkHamiltonian` refuses by name at its first read of the graph, and ``bind`` refuses by name a
    geometry that holds no graph, or whose graph is not built. Mutants: each refusal removed, so the read fails on
    ``None`` or the bind succeeds."""
    pytest.importorskip("igraph")
    from mfgarchon.extensions.topology import NetworkHamiltonian
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    with pytest.raises(RuntimeError, match=r"bound to no network geometry"):
        NetworkHamiltonian().source_term(t=0.0, x=0, m=np.zeros(3))
    with pytest.raises(TypeError, match=r"bind takes a network geometry.*TensorProductGrid"):
        NetworkHamiltonian().bind(
            TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[5], boundary_conditions=no_flux_bc(dimension=1))
        )
    unbuilt = GridNetwork(width=3, height=2)
    unbuilt.network_data = None
    with pytest.raises(ValueError, match=r"whose graph is built, and this GridNetwork holds none"):
        NetworkHamiltonian().bind(unbuilt)


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


def _model_form(network, hamiltonian=None, m_initial=None, u_terminal=None, cls=None):
    """A problem in the model form on ``network``: a NetworkHamiltonian, the uniform density and a zero terminal value
    where none is given."""
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.extensions.topology import NetworkHamiltonian

    n = network.num_nodes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (cls or MFGProblem)(
            model=Model(hamiltonian=NetworkHamiltonian() if hamiltonian is None else hamiltonian, volatility=0.0),
            domain=network,
            conditions=Conditions(
                m_initial=m_initial or (lambda x: 1.0 / n), u_terminal=u_terminal or (lambda x: 0.0), T=0.3
            ),
            Nt=4,
        )


def _solver(name):
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
    from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver, NetworkPolicyIterationHJBSolver

    return {
        "FPNetworkSolver": FPNetworkSolver,
        "NetworkHJBSolver": NetworkHJBSolver,
        "NetworkPolicyIterationHJBSolver": NetworkPolicyIterationHJBSolver,
    }[name]


SOLVER_NAMES = ["FPNetworkSolver", "NetworkHJBSolver", "NetworkPolicyIterationHJBSolver"]


@pytest.mark.parametrize("solver", SOLVER_NAMES)
def test_a_network_solver_refuses_a_problem_it_cannot_read_at_construction(solver):
    """#2045's refusal, for a problem built otherwise than as the library builds one on a network geometry: here, one
    whose Hamiltonian cannot be read. Mutant: the refusal at construction removed, so the solve fails later on a missing
    attribute."""
    pytest.importorskip("igraph")
    from mfgarchon import MFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    class WithoutAHamiltonian(MFGProblem):
        @property
        def hamiltonian_class(self):
            raise AttributeError("hamiltonian_class")

    problem = _model_form(GridNetwork(width=3, height=2), cls=WithoutAHamiltonian)
    with pytest.raises(NotImplementedError, match=r"reads .*hamiltonian_class.* which this WithoutAHamiltonian on a"):
        _solver(solver)(problem)


def test_each_hjb_solver_refuses_only_what_it_reads():
    """`NetworkHJBSolver` never reads its Hamiltonian's ``has_custom_hamiltonian`` and builds without it; policy
    iteration reads it, and refuses by name a problem whose Hamiltonian lacks it. Mutants: the base solver's reads widened
    to policy iteration's (review 2's MF4), or policy iteration's narrowed to the base solver's."""
    pytest.importorskip("igraph")
    from mfgarchon.extensions.topology import NetworkHamiltonian
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    class WithoutTheFlag(NetworkHamiltonian):
        @property
        def has_custom_hamiltonian(self):
            raise AttributeError("has_custom_hamiltonian")

    problem = _model_form(GridNetwork(width=3, height=2), hamiltonian=WithoutTheFlag())
    _solver("NetworkHJBSolver")(problem)
    with pytest.raises(
        NotImplementedError, match=r"NetworkPolicyIterationHJBSolver reads hamiltonian_class\.has_custom"
    ):
        _solver("NetworkPolicyIterationHJBSolver")(problem)


def test_a_problem_on_a_network_refuses_a_hamiltonian_that_is_not_a_networks():
    """Stage 2's Q3 (7232886239): the network solvers read a `NetworkHamiltonian`'s methods, so a problem on a network
    refuses any other Hamiltonian by name when it is built. Mutant: the refusal removed."""
    pytest.importorskip("igraph")
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    with pytest.raises(TypeError, match=r"on a GridNetwork takes a NetworkHamiltonian.*given SeparableHamiltonian"):
        _model_form(GridNetwork(width=3, height=2), hamiltonian=SeparableHamiltonian(QuadraticControlCost(1.0)))


@pytest.mark.parametrize("solver", SOLVER_NAMES)
def test_the_model_form_on_a_network_binds_its_hamiltonian_and_solves_as_the_network_problem(solver):
    """The problem binds the `NetworkHamiltonian` it is given to its domain and leaves the public field as given; with
    the same Hamiltonian and conditions, each network solver's solution equals a `NetworkMFGProblem`'s. Mutant: the bind
    removed, so a solve reads an unbound Hamiltonian."""
    pytest.importorskip("igraph")
    from mfgarchon.extensions.topology import NetworkHamiltonian, NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    given = [0.3, 0.25, 0.2, 0.15, 0.1, 0.0]
    hamiltonian = NetworkHamiltonian(node_potential_func=lambda t, node: 0.2 * node)
    network = GridNetwork(width=3, height=2)
    model_form = _model_form(
        network, hamiltonian=hamiltonian, m_initial=lambda x: given[x], u_terminal=lambda x: 0.1 * x
    )
    assert model_form.components.hamiltonian is hamiltonian
    assert model_form.hamiltonian_class.geometry is network
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = NetworkMFGProblem(
            geometry=GridNetwork(width=3, height=2),
            T=0.3,
            Nt=4,
            components=NetworkMFGComponents(
                node_potential_func=lambda t, node: 0.2 * node,
                initial_node_density_func=lambda node: given[node],
                terminal_node_value_func=lambda node: 0.1 * node,
            ),
        )
    n, nt = network.num_nodes, 4
    U = np.stack([np.linspace(0.0, 1.0, n) * (1 + 0.1 * k) for k in range(nt + 1)])
    solutions = []
    for problem in (model_form, reference):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if solver == "FPNetworkSolver":
                built = _solver(solver)(problem, diffusion_coefficient=0.05)
                solutions.append(built.solve_fp_system(problem.get_m_initial(), U, show_progress=False))
            else:
                solutions.append(
                    _solver(solver)(problem).solve_hjb_system(
                        M_density=np.full((nt + 1, n), 1.0 / n), U_terminal=problem.get_u_terminal(), volatility=0.2
                    )
                )
    np.testing.assert_array_equal(solutions[0], solutions[1])


def test_a_conditions_callable_on_a_network_takes_each_node_as_an_int():
    """The user's ruling (Q1, 7232886239): a `Conditions` callable on a network is read once at each node, with x the
    node as an int, through the network geometry's ``node_x``, by the check that reads it first and by the problem.
    Mutant: the per-node read removed, so the callable is handed the geometry's points."""
    pytest.importorskip("igraph")
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    seen = []

    def m_initial(x):
        seen.append(x)
        return [0.4, 0.3, 0.2, 0.1][x]

    problem = _model_form(GridNetwork(width=4, height=1), m_initial=m_initial)
    assert {type(x) for x in seen} == {int}, seen
    assert seen[-4:] == [0, 1, 2, 3]
    np.testing.assert_array_equal(problem.get_m_initial(), [0.4, 0.3, 0.2, 0.1])


def test_a_network_problems_readers_of_its_conditions_agree():
    """Retired from a strict xfail by stage 2a: `get_initial_density` and `terminal_value` read the problem's own
    conditions, which the node fields are now. Mutant: the builder hands the problem the uniform density whatever the
    field."""
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
    np.testing.assert_array_equal(problem.get_m_initial(), given)
    np.testing.assert_array_equal(np.ravel(problem.get_m_initial()), problem.get_initial_density())
    np.testing.assert_array_equal(np.ravel(problem.get_u_terminal()), problem.terminal_value(None))


@pytest.mark.parametrize("route", ["FPNetworkSolver.solve", "FPFDMSolver.solve", "FPFVMSolver, no initial density"])
def test_an_fp_solve_handed_no_initial_density_starts_from_the_problems(route):
    """`BaseFPSolver.solve`, and `FPFVMSolver` handed no initial density, read `get_m_initial`, the one source (#1472
    stage 2a). They read `get_initial_density`, which only `NetworkMFGProblem` defines, so on any other problem they
    raised `AttributeError`. Mutants: either reads `get_initial_density` again."""
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    if route == "FPNetworkSolver.solve":
        pytest.importorskip("igraph")
        from mfgarchon.geometry.graph.network_geometry import GridNetwork

        problem = _model_form(GridNetwork(width=4, height=1), m_initial=lambda x: [0.4, 0.3, 0.2, 0.1][x])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            M = _solver("FPNetworkSolver")(problem, diffusion_coefficient=0.05).solve()
        np.testing.assert_array_equal(M[0], [0.4, 0.3, 0.2, 0.1])
        return
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            model=Model(hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)), volatility=0.2),
            domain=grid,
            conditions=Conditions(m_initial=lambda x: 1.0 + x[0], u_terminal=lambda x: 0.0, T=0.3),
            Nt=4,
        )
        if route == "FPFDMSolver.solve":
            M = FPFDMSolver(problem).solve()
        else:
            M = FPFVMSolver(problem).solve_fp_system(None, potential_field=np.zeros((5, 11)), show_progress=False)
    np.testing.assert_array_equal(M[0], problem.get_m_initial())


@pytest.mark.parametrize("solver", SOLVER_NAMES)
def test_a_node_bc_the_geometry_is_given_after_a_solver_is_built_is_the_one_read(solver):
    """P1 (review 1 of #3014): the geometry is the node BCs' one holder, read where they are applied (#1472 stage 2a).
    A BC set after the solver is built is read: FP applies an ABSORBING node, policy iteration pins a DIRICHLET node, and
    `NetworkHJBSolver`, which applies none, refuses at the solve. Mutants: each reader reads the BC at construction."""
    pytest.importorskip("igraph")
    from mfgarchon.geometry.boundary.applicator_graph import GraphBCConfig, GraphBCType, NodeBC
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    network = GridNetwork(width=3, height=3)
    problem = _model_form(network)
    n, nt = network.num_nodes, problem.Nt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = _solver(solver)(problem, scheme="explicit") if solver == "FPNetworkSolver" else _solver(solver)(problem)
    kind = GraphBCType.ABSORBING if solver == "FPNetworkSolver" else GraphBCType.DIRICHLET
    network.set_boundary_conditions(GraphBCConfig(node_bcs=[NodeBC(nodes=[0], bc_type=kind, value=1.0)]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if solver == "FPNetworkSolver":
            M = built.solve_fp_system(M_initial=np.full(n, 1.0 / n), potential_field=np.zeros((nt + 1, n)))
            np.testing.assert_array_equal(M[1:, 0], 0.0)
        elif solver == "NetworkPolicyIterationHJBSolver":
            U = built.solve_hjb_system(M_density=np.full((nt + 1, n), 1.0 / n), U_terminal=np.zeros(n), volatility=0.2)
            np.testing.assert_array_equal(U[:-1, 0], 1.0)
        else:
            with pytest.raises(NotImplementedError, match="does not support node boundary conditions"):
                built.solve_hjb_system(M_density=np.full((nt + 1, n), 1.0 / n), U_terminal=np.zeros(n), volatility=0.2)


@pytest.mark.xfail(strict=True, reason=STAGE_2B)
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

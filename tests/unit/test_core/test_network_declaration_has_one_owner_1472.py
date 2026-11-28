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
import re
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


def _model_form(network, hamiltonian=None, m_initial=None, u_terminal=None, cls=None, volatility=0.0, **problem_kw):
    """A problem in the model form on ``network``: a NetworkHamiltonian, the uniform density and a zero terminal value
    where none is given, and no volatility unless one is."""
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.extensions.topology import NetworkHamiltonian

    n = network.num_nodes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (cls or MFGProblem)(
            model=Model(
                hamiltonian=NetworkHamiltonian() if hamiltonian is None else hamiltonian, volatility=volatility
            ),
            domain=network,
            conditions=Conditions(
                m_initial=m_initial or (lambda x: 1.0 / n), u_terminal=u_terminal or (lambda x: 0.0), T=0.3
            ),
            Nt=4,
            **problem_kw,
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
        network, hamiltonian=hamiltonian, m_initial=lambda x: given[x], u_terminal=lambda x: 0.1 * x, volatility=0.3
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
    # The model form's FP reads its D from the problem's volatility; the NetworkMFGProblem, which declares none, is
    # handed the same D
    diffusion = _solver("FPNetworkSolver")(model_form).diffusion_coefficient
    for problem in (model_form, reference):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if solver == "FPNetworkSolver":
                kw = {} if problem is model_form else {"diffusion_coefficient": diffusion}
                built = _solver(solver)(problem, **kw)
                solutions.append(built.solve_fp_system(problem.get_m_initial(), U, show_progress=False))
            else:
                solutions.append(
                    _solver(solver)(problem).solve_hjb_system(
                        M_density=np.full((nt + 1, n), 1.0 / n), U_terminal=problem.get_u_terminal(), volatility=0.2
                    )
                )
    np.testing.assert_array_equal(solutions[0], solutions[1])


def test_a_conditions_callable_on_a_network_takes_each_node_as_an_int():
    """The user's ruling (Q1, 7232886239): a `Conditions` callable on a network is read at each node, with x the
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
            M = _solver("FPNetworkSolver")(problem).solve()
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
    A solver solves, the geometry is given a BC, and the solver solves again: FP applies an ABSORBING node, policy
    iteration pins a DIRICHLET node, and `NetworkHJBSolver`, which applies none, refuses the second solve. Mutants: each
    reader reads the BC when it is built, or at its first solve."""
    pytest.importorskip("igraph")
    from mfgarchon.geometry.boundary.applicator_graph import GraphBCConfig, GraphBCType, NodeBC
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    network = GridNetwork(width=3, height=3)
    problem = _model_form(network)
    n, nt = network.num_nodes, problem.Nt
    M_flat = np.full((nt + 1, n), 1.0 / n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = _solver(solver)(problem, scheme="explicit") if solver == "FPNetworkSolver" else _solver(solver)(problem)

        def solve():
            if solver == "FPNetworkSolver":
                return built.solve_fp_system(M_initial=np.full(n, 1.0 / n), potential_field=np.zeros((nt + 1, n)))
            return built.solve_hjb_system(M_density=M_flat, U_terminal=np.zeros(n), volatility=0.2)

        before = solve()
        kind = GraphBCType.ABSORBING if solver == "FPNetworkSolver" else GraphBCType.DIRICHLET
        network.set_boundary_conditions(GraphBCConfig(node_bcs=[NodeBC(nodes=[0], bc_type=kind, value=1.0)]))
        if solver == "FPNetworkSolver":
            assert np.all(before[1:, 0] > 0.0)
            np.testing.assert_array_equal(solve()[1:, 0], 0.0)
        elif solver == "NetworkPolicyIterationHJBSolver":
            assert np.all(before[:-1, 0] != 1.0)
            np.testing.assert_array_equal(solve()[:-1, 0], 1.0)
        else:
            with pytest.raises(NotImplementedError, match="does not support node boundary conditions"):
                solve()


@pytest.mark.parametrize(
    ("solver", "given", "named"),
    [
        ("FPNetworkSolver", {"source_term_fp": lambda t, x, v, m: 1.0}, "source_term_fp="),
        ("NetworkHJBSolver", {"source_term_hjb": lambda t, x, v, m: 1.0}, "source_term_hjb="),
        ("NetworkPolicyIterationHJBSolver", {"source_term_hjb": lambda t, x, v, m: 1.0}, "source_term_hjb="),
        ("NetworkHJBSolver", {"nonlocal_operator": object()}, "nonlocal_operator="),
        ("NetworkPolicyIterationHJBSolver", {"nonlocal_operator": object()}, "nonlocal_operator="),
        *(
            (
                name,
                {"hamiltonian": {"hamiltonian_dm_func": lambda t, node, neighbors, p, m: 0.0}},
                "hamiltonian_dm_func=",
            )
            for name in SOLVER_NAMES
        ),
        *((name, {"hamiltonian": {"population_index": 1}}, "population_index=1") for name in SOLVER_NAMES),
    ],
)
def test_a_network_solver_refuses_at_construction_an_input_it_does_not_read(solver, given, named):
    """Items 1-5 of stage 2a's plan v2 (7236116911): an input the problem was given and the solver would drop is refused
    by name when the solver is built, not dropped from the solve. Mutants: each solver's refusal removed, or one input
    left off it."""
    pytest.importorskip("igraph")
    from mfgarchon.extensions.topology import NetworkHamiltonian
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    kw = dict(given)
    hamiltonian = NetworkHamiltonian(**kw.pop("hamiltonian")) if "hamiltonian" in kw else None
    problem = _model_form(GridNetwork(width=3, height=2), hamiltonian=hamiltonian, **kw)
    with pytest.raises(NotImplementedError, match=rf"{solver} does not read .*{re.escape(named)}"):
        _solver(solver)(problem)


def test_the_network_fp_reads_its_diffusion_from_the_owner_the_hjb_solvers_read():
    """Q2 of plan v2's approval (7236127464): `FPNetworkSolver` resolves its D once, through `network_diffusion`: its
    coefficient when given, refused where it disagrees with the volatility the problem declares; else the problem's
    volatility. A `Model`'s volatility defaults to 0.1, so an explicit coefficient with a default `Model` is refused
    unless the volatility is set to match. A `NetworkMFGProblem` declares none and keeps D = 0.1. Mutants: FP ignores
    the problem's volatility; the disagreement is not refused; a NetworkMFGProblem reads its 0.0."""
    pytest.importorskip("igraph")
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.extensions.topology import NetworkHamiltonian, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork
    from mfgarchon.utils.numerical.volatility import diffusion_from_volatility

    FP = _solver("FPNetworkSolver")
    assert FP(_model_form(GridNetwork(width=3, height=2), volatility=0.3)).diffusion_coefficient == float(
        diffusion_from_volatility(0.3)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = MFGProblem(
            model=Model(hamiltonian=NetworkHamiltonian()),
            domain=GridNetwork(width=3, height=2),
            conditions=Conditions(m_initial=lambda x: 1.0 / 6, u_terminal=lambda x: 0.0, T=0.3),
            Nt=4,
        )
    assert FP(default).diffusion_coefficient == float(diffusion_from_volatility(0.1))
    with pytest.raises(ValueError, match=r"diffusion_coefficient=0\.05\) disagrees with the problem's volatility"):
        FP(default, diffusion_coefficient=0.05)
    assert FP(default, diffusion_coefficient=0.005).diffusion_coefficient == 0.005
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        network_problem = NetworkMFGProblem(geometry=GridNetwork(width=3, height=2), T=0.3, Nt=4)
    assert FP(network_problem).diffusion_coefficient == 0.1
    assert FP(network_problem, diffusion_coefficient=0.05).diffusion_coefficient == 0.05


def test_a_components_bc_on_a_network_is_refused_when_the_problem_is_built():
    """Item 6: a network's node BCs are its geometry's (#1471), and no network solver reads `components.boundary_conditions`,
    so the problem refuses it when it is built. Mutant: the owner's refusal not called at construction."""
    pytest.importorskip("igraph")
    from mfgarchon import MFGProblem
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.extensions.topology import NetworkHamiltonian
    from mfgarchon.geometry.boundary.applicator_graph import GraphBCConfig, GraphBCType, NodeBC
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    bc = GraphBCConfig(node_bcs=[NodeBC(nodes=[0], bc_type=GraphBCType.DIRICHLET, value=1.0)])
    with pytest.raises(ValueError, match=r"components\.boundary_conditions is not read on a network"):
        MFGProblem(
            geometry=GridNetwork(width=3, height=2),
            components=MFGComponents(
                hamiltonian=NetworkHamiltonian(),
                m_initial=lambda x: 1.0 / 6,
                u_terminal=lambda x: 0.0,
                boundary_conditions=bc,
            ),
            T=0.3,
            Nt=4,
        )


@pytest.mark.parametrize(
    "method",
    [
        "hamiltonian",
        "hamiltonian_dm",
        "lagrangian",
        "node_potential",
        "density_coupling",
        "get_initial_density",
        "get_terminal_value",
        "terminal_value",
        "get_initial_m",
        "get_final_u",
        "apply_boundary_conditions",
    ],
)
def test_a_network_problem_subclass_overriding_a_method_no_solver_reads_is_refused(method):
    """Item 8 (ruling B2 of review 1 of #3015): no network solver reads these through a `NetworkMFGProblem` subclass, or,
    for the two terminal ones, a subclass's would be a second terminal source; a subclass overriding one is refused at
    class creation. Mutant: one method left off the list."""
    from mfgarchon.extensions.topology import NetworkMFGProblem

    with pytest.raises(TypeError, match=rf"overrides NetworkMFGProblem's {method}, which no network solver reads"):
        type("Overriding", (NetworkMFGProblem,), {method: lambda self, *args, **kwargs: None})


@pytest.mark.parametrize(
    ("m_initial", "error", "message"),
    [
        (lambda x: "0.25", (TypeError, ValueError), r"numeric value, got str|a number at each node"),
        (
            lambda x: 0.25 if x != 2 else "0.25",
            TypeError,
            r"must return a number at each node, and returned str at node 2",
        ),
        (lambda x: np.array([0.25, 0.25]), ValueError, r"one number at each node, and returned shape \(2,\) at node 0"),
        (lambda x: 1.0 / x, ValueError, r"raised exception at node 0: float division by zero"),
    ],
    ids=["a str", "a str past node 0", "two entries", "raising at node 0"],
)
def test_a_network_condition_meets_the_condition_value_contract(m_initial, error, message):
    """B3 and B4 of review 1 of #3015: a network read of a `Conditions` callable meets the one condition-value contract,
    as the grid route does: a str is refused, at a node the check reads or at one it does not, and so is more than one
    number at a node; the error of a callable raising at a node names the node and says x is an int, not a coordinate
    tuple. Mutants: the network read's value check or its size check removed; its error advises a tuple."""
    pytest.importorskip("igraph")
    from mfgarchon.geometry.graph.network_geometry import GridNetwork
    from mfgarchon.utils.validation import ValidationError

    with pytest.raises((ValidationError, *(error if isinstance(error, tuple) else (error,))), match=message) as caught:
        _model_form(GridNetwork(width=4, height=1), m_initial=m_initial)
    assert "coordinate tuple" not in str(caught.value)


def test_a_network_condition_returning_a_one_entry_array_is_read_as_its_entry():
    """B3: a one-entry array per node is read as its entry, as the grid route reads one. Mutant: the network read refuses
    an array."""
    pytest.importorskip("igraph")
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    problem = _model_form(GridNetwork(width=4, height=1), m_initial=lambda x: np.array([[0.4, 0.3, 0.2, 0.1][x]]))
    np.testing.assert_array_equal(problem.get_m_initial(), [0.4, 0.3, 0.2, 0.1])


def test_a_network_condition_raising_past_node_0_names_the_node():
    """SF4 of review 1 of #3015: a condition raising at a node the check did not read, here node 2, keeps its type and is
    named at that node. Mutant: the note removed."""
    pytest.importorskip("igraph")
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    with pytest.raises(IndexError) as caught:
        _model_form(GridNetwork(width=4, height=1), m_initial=lambda x: [0.5, 0.5][x])
    assert any("read at node 2; on a network, x is the node, an int" in note for note in caught.value.__notes__)


def test_a_network_problems_messages_and_info_name_its_own_inputs():
    """SF2, SF3 and SF8 of review 1 of #3015: `get_problem_info` reads the problem's conditions source; a refusal of the
    initial density names the input it came from; and a network problem given no Hamiltonian is told so. Mutants:
    get_problem_info reads the components; the message names MFGComponents; the refusal says NoneType."""
    pytest.importorskip("igraph")
    from mfgarchon import MFGProblem
    from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        given = NetworkMFGProblem(
            geometry=GridNetwork(width=4, height=1),
            T=0.3,
            Nt=4,
            components=NetworkMFGComponents(initial_node_density_func=lambda node: [0.4, 0.3, 0.2, 0.1][node]),
        )
    assert given.get_problem_info()["has_custom_initial"] is True
    with pytest.raises(ValueError, match=r"Check NetworkMFGComponents\(initial_node_density_func=\.\.\.\)"):
        NetworkMFGProblem(
            geometry=GridNetwork(width=4, height=1),
            T=0.3,
            Nt=4,
            components=NetworkMFGComponents(initial_node_density_func=lambda node: -1.0),
        )
    with pytest.raises(ValueError, match=r"Check Conditions\(m_initial=\.\.\.\)"):
        _model_form(GridNetwork(width=4, height=1), m_initial=lambda x: -1.0)
    with pytest.raises(TypeError, match=r"takes a NetworkHamiltonian, .* and was given none"):
        MFGProblem(
            geometry=GridNetwork(width=4, height=1),
            components=NetworkMFGComponents(m_initial=lambda x: 0.25, u_terminal=lambda x: 0.0),
            T=0.3,
            Nt=4,
        )


def test_a_read_at_a_node_refuses_an_x_that_is_not_an_int():
    """SF9 of review 1 of #3015: `BoundCallable.at_a_node` refuses an x that is not a node, a door defect `node_x` never
    makes. Mutant: the check removed."""
    from mfgarchon.types.callable_protocols import CONDITION_SLOTS, _remembered

    bound = _remembered(lambda x: x, CONDITION_SLOTS, "m_initial")
    assert bound.at_a_node(x=2) == 2
    with pytest.raises(TypeError, match=r"x as float, and on a network x is a node, an int"):
        bound.at_a_node(x=2.0)


@pytest.mark.xfail(strict=True, reason=STAGE_2B)
@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("mfgarchon.extensions.topology", "create_grid_mfg_problem"),
        ("mfgarchon.extensions.topology", "create_random_mfg_problem"),
        ("mfgarchon.extensions.topology", "create_scale_free_mfg_problem"),
        ("mfgarchon.alg.numerical.coupling.network_mfg_solver", "create_network_mfg_solver"),
        ("mfgarchon.alg.numerical.coupling.network_mfg_solver", "create_simple_network_solver"),
    ],
)
def test_no_factory_declares_a_network_beside_the_model_form(module, name):
    """M4 of review 1 of #3015: the five factories that build a NetworkMFGProblem, or a solver typed on one, are remaining
    members of the class line, removed by 2b."""
    import importlib

    assert name not in vars(importlib.import_module(module))


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

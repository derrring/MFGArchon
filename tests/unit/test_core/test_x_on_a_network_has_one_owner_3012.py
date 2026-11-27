"""#3012: the form of the node the library hands a network Hamiltonian as ``x`` has one owner, the network geometry's
``node_x`` (#1472 plan v1, #1472 comment 7224953131, approved in 7224969065, item 1; stage 1's plan v2, 7226702955,
approved in 7226717273).

By AST over the network solvers and ``extensions/topology.py``, an *x position* is a value a call hands as ``x=``, or
hands as ``node=`` to one of the network problem's relays that pass the node on as ``x`` (``NODE_RELAYS``). Each is the
geometry's ``node_x(...)``, a parameter of the function the call is in (a node that function was handed), or a name bound
only to ``_node(x, ...)``, `NetworkHamiltonian`'s read of the node it was handed. What the scan cannot follow: a node
handed by position, a name bound otherwise than by a plain assignment, a relay other than ``NODE_RELAYS``, and a call
outside these files.

At runtime, a recording `NetworkHamiltonian` subclass takes the type of each ``x`` the three network solvers hand it.
And `NetworkHamiltonian`'s read of a node takes what ``node_x`` takes, through ``node_x``: an ``int``, a numpy integer, or a
one-entry array of integer dtype, one of the graph's nodes. It refuses by name every other form main read, each of which
main read as a node (the cases below).
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

import numpy as np

import mfgarchon
from mfgarchon.extensions.topology import NetworkHamiltonian, NetworkMFGComponents, NetworkMFGProblem

PACKAGE = Path(mfgarchon.__file__).resolve().parent
FILES = [
    *sorted(str(p.relative_to(PACKAGE)) for p in (PACKAGE / "alg/numerical/network_solvers").glob("*.py")),
    "extensions/topology.py",
]
# The network problem's methods that take the node as ``node=`` and hand it on to its Hamiltonian as ``x``
NODE_RELAYS = {"hamiltonian", "hamiltonian_dm", "node_potential", "density_coupling"}


def _functions(tree: ast.AST):
    def walk(node, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child, enclosing
                yield from walk(child, [child, *enclosing])
            else:
                yield from walk(child, enclosing)

    yield from walk(tree, [])


def _own_calls(fn: ast.AST):
    for child in ast.iter_child_nodes(fn):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _own_calls(child)


def _callee(call: ast.Call) -> str:
    return getattr(call.func, "attr", getattr(call.func, "id", ""))


def _the_owners(value: ast.AST, fn: ast.AST, enclosing: list[ast.AST]) -> bool:
    if isinstance(value, ast.Call) and _callee(value) == "node_x":
        return True
    if not isinstance(value, ast.Name):
        return False
    bound = [
        n.value
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == value.id for t in n.targets)
    ]
    if bound:
        return all(isinstance(v, ast.Call) and _callee(v) == "_node" for v in bound)
    return any(
        value.id in [a.arg for a in (*f.args.posonlyargs, *f.args.args, *f.args.kwonlyargs)] for f in [fn, *enclosing]
    )


def node_doors(source: str, path: str) -> list[str]:
    """Each x position in ``source`` that is not the geometry's node, a node the function was handed, or one `_node`
    read."""
    found = []
    for fn, enclosing in _functions(ast.parse(source)):
        for call in _own_calls(fn):
            for keyword in call.keywords:
                position = keyword.arg == "x" or (keyword.arg == "node" and _callee(call) in NODE_RELAYS)
                if position and not _the_owners(keyword.value, fn, enclosing):
                    found.append(
                        f"{path}:{call.lineno} in {fn.name} hands {keyword.arg} as {ast.unparse(keyword.value)}"
                    )
    return found


@pytest.mark.parametrize(
    "planted",
    [
        "def rates(self, i):\n    return H.optimal_control(x=np.array([i]), m=m, p=u, t=t)",
        "def sources(self):\n    return [H.source_term(x=i, m=m, t=t) for i in range(n)]",
        "def rate(self, i):\n    return H.optimal_control(x=int(i), m=m, p=u, t=t)",
        "def value(self, x):\n    node = int(np.asarray(x).flat[0])\n    return self.source_term(x=node, m=m, t=t)",
        # The relay review 1 found (B3): the node handed as node= to the problem, which hands it on as x
        "def batch(self, i):\n    return self.network_problem.hamiltonian(t=t, node=np.array([i]), neighbors=n, p=u, m=m)",
        "def batch(self):\n    return [self.network_problem.hamiltonian(t=t, node=i, p=u, m=m) for i in range(n)]",
    ],
)
def test_the_scan_flags_a_planted_door(planted):
    assert len(node_doors(planted, "a/door.py")) == 1


def test_the_scan_passes_the_owners_node_and_a_node_handed_on():
    clean = (
        "def rate(self, i):\n    return H.optimal_control(x=self.network.node_x(i), m=m, p=u, t=t)\n\n\n"
        "def batch(self, i):\n    return self.network_problem.hamiltonian(t=t, node=self.network.node_x(i), p=u, m=m)\n\n\n"
        "def relay(self, t, node, m):\n    return H.source_term(x=node, m=m, t=t)\n\n\n"
        "def read(self, t, x, m):\n    node = self._node(x, 'dm')\n    return self.coupling_value(x=node, m=m, t=t)\n\n\n"
        "def condition(self):\n    return [density(t=0.0, node=i) for i in range(n)]\n"
    )
    assert node_doors(clean, "a/door.py") == []


def test_only_the_geometry_decides_the_form_of_a_node():
    found = [door for path in FILES for door in node_doors((PACKAGE / path).read_text(), path)]
    assert found == [], "\n".join(found)


def _network():
    pytest.importorskip("igraph")
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    network = GridNetwork(width=3, height=2)
    network.create_network()
    return network


def _recording_problem():
    class Recording(NetworkHamiltonian):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.forms = set()

        def __call__(self, t, x, p, m):
            self.forms.add(type(x))
            return super().__call__(t=t, x=x, p=p, m=m)

        def source_term(self, t, x, m):
            self.forms.add(type(x))
            return super().source_term(t=t, x=x, m=m)

        def optimal_control(self, t, x, p, m):
            self.forms.add(type(x))
            return super().optimal_control(t=t, x=x, p=p, m=m)

    network = _network()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = NetworkMFGProblem(geometry=network, T=0.2, Nt=4, components=NetworkMFGComponents())
    H = Recording().bind(network)
    problem.components._hamiltonian_class = H  # the backing field of MFGProblem.hamiltonian_class
    assert problem.hamiltonian_class is H
    return problem, H


@pytest.mark.parametrize("solver", ["FPNetworkSolver", "NetworkHJBSolver", "NetworkPolicyIterationHJBSolver"])
def test_each_network_solver_hands_its_hamiltonian_the_node_as_an_int(solver):
    """Mutant: a door hands ``np.array([i])`` again, as main did at FP's and policy iteration's rates."""
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
    from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver, NetworkPolicyIterationHJBSolver

    problem, H = _recording_problem()
    n, nt = problem.num_nodes, problem.Nt
    U = np.stack([np.linspace(0.0, 1.0, n) * (1 + 0.1 * k) for k in range(nt + 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if solver == "FPNetworkSolver":
            FPNetworkSolver(problem, scheme="explicit").solve_fp_system(np.full(n, 1.0 / n), U, show_progress=False)
        else:
            hjb = {
                "NetworkHJBSolver": NetworkHJBSolver,
                "NetworkPolicyIterationHJBSolver": NetworkPolicyIterationHJBSolver,
            }
            hjb[solver](problem).solve_hjb_system(M_density=np.full((nt + 1, n), 1.0 / n), U_terminal=np.zeros(n))
    assert H.forms == {int}, f"{solver}: {sorted(t.__name__ for t in H.forms)}"


@pytest.mark.parametrize(
    "x", [2, np.int64(2), np.array([2]), np.array([[2]])], ids=["int", "numpy int", "one-entry array", "1x1 array"]
)
def test_a_node_the_contract_names_is_read_as_that_node(x):
    """Mutant: the read refuses a one-entry array, a form main read and the contract names."""
    assert NetworkHamiltonian().bind(_network())._node(x, "__call__") == 2


@pytest.mark.parametrize(
    ("x", "message"),
    [
        (np.array([1, 2]), "an array of 2 entries"),
        (np.array([], dtype=int), "an array of 0 entries"),
        (np.array([2.0]), "of dtype float64"),
        (2.5, "was handed 2.5"),
        (2.0, "was handed 2.0"),
        (np.float64(2.0), "of type float64"),
        (True, "of type bool"),
        (np.bool_(True), "of type bool"),
        ("2", "was handed '2'"),
        (2 + 0j, "of type complex"),
        ([2], "of type list"),
        ((2,), "of type tuple"),
        (-1, "-1 is not one of them"),
        (6, "6 is not one of them"),
    ],
    ids=[
        "two entries",
        "no entry",
        "float array",
        "not an integer",
        "integral float",
        "numpy float",
        "bool",
        "numpy bool",
        "string",
        "complex",
        "list",
        "tuple",
        "negative",
        "off the graph",
    ],
)
def test_a_node_the_contract_does_not_name_is_refused_by_name(x, message):
    """Main read each of these as a node, through ``int(np.asarray(x).flat[0])``. Mutants: the array's size check, its
    dtype check, ``node_x``'s type check, and its range check, each removed."""
    with pytest.raises((TypeError, ValueError), match=message):
        NetworkHamiltonian().bind(_network())._node(x, "optimal_control")


def test_the_geometry_refuses_an_index_off_the_graph():
    """S2 of #3014's review 1: ``node_x`` refuses an index that is not one of the graph's nodes. Mutant: its range check
    removed."""
    network = _network()
    assert network.node_x(5) == 5
    for index in (-1, 6):
        with pytest.raises(ValueError, match=f"{index} is not one of them"):
            network.node_x(index)

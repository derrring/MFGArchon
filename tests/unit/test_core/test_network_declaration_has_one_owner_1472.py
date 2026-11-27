"""#1472: a network MFG is declared in one place, `MFGProblem`'s model/domain/conditions form, with the graph on the network
geometry (#1472 plan v1, #1472 comment 7224953131, approved in 7224969065). Stage 1 of two.

Stage 1 holds: the network solvers read the graph from the geometry, not from the problem. By AST over the network
solvers: no read of a graph attribute (``GRAPH``) on the problem they hold. What the scan cannot follow: a read through
``getattr``, or through a name the problem was copied to.

Stage 2 removes the remaining members, each a strict xfail here naming it: the classes that declare a network beside
that form (`NetworkMFGProblem`, `NetworkMFGComponents`), and the problem's two sources for its initial density and
terminal value, which disagree.
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
HOLDERS = {"problem", "network_problem"}
STAGE_2 = "#1472 stage 2: MFGProblem takes the model/domain/conditions form on a network, and the two classes go"


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

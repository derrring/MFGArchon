"""A graph node's value is a callable of (t, node), time first, bound once and read at its time (#2296, #2512 row N2).

Before, `NodeBC.get_value` called a value as `value(node, t)`, so one written `(t, node)`, the order #2378 phase 5 set
for every other user callable, was read node-first; a CUSTOM rule shared the field with a different contract; and the
readers took `t=0` when given no time. Each row is an oracle: the declared value at its time, or a refusal by name.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary.applicator_graph import GraphApplicator, GraphBCConfig, GraphBCType, NodeBC


def _g(t, node):
    return t + 10.0 * node


def test_a_node_value_is_read_at_its_time_through_a_network_solve():
    """O1 with g(t, node) = t + 10 node at a DIRICHLET node: the policy-iteration HJB pins U[n] at t_n for n < Nt. Read
    node-first, as `get_value` called it, the pin was 4 + 10 t_n at node 4."""
    from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkPolicyIterationHJBSolver
    from mfgarchon.extensions.topology import NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    network = GridNetwork(
        width=3,
        height=3,
        boundary_conditions=GraphBCConfig(node_bcs=[NodeBC(nodes=[4], bc_type=GraphBCType.DIRICHLET, value=_g)]),
    )
    network.create_network()
    problem = NetworkMFGProblem(geometry=network, T=0.5, Nt=4)
    n = problem.num_nodes
    U = NetworkPolicyIterationHJBSolver(problem).solve_hjb_system(M_density=np.ones((5, n)) / n, U_terminal=np.zeros(n))
    times = np.linspace(0.0, 0.5, 5)
    np.testing.assert_allclose(U[:-1, 4], _g(times[:-1], 4), rtol=0, atol=1e-12)


def test_each_reader_reads_a_value_and_a_rule_at_their_time():
    """`apply`, `apply_direct` and `get_bc_value` read a value at the time they are given, and CUSTOM reads its rule,
    a field of its own, with the field: g(t, node) at node 2 and t = 0.5 is 20.5."""
    applicator = GraphApplicator(4).set_dirichlet_nodes([2], value=_g, name="d")
    assert applicator.apply(np.zeros(4), t=0.5)[2] == 20.5
    assert applicator.apply_direct(np.zeros(4), [2], _g, t=0.5)[2] == 20.5
    assert applicator.get_bc_value("d", 2, 0.5) == 20.5
    custom = GraphApplicator(4).set_custom_bc([1], lambda t, node, field: field[node] + t)
    np.testing.assert_array_equal(custom.apply(np.full(4, 3.0), t=0.5), [3.0, 3.5, 3.0, 3.0])


class _NoSignature:
    """A callable whose signature cannot be read, written in the old order: main called it (node, t) and read it right;
    bound by position, as time first, it would read t as the node (#2296)."""

    __signature__ = "unreadable"

    def __call__(self, *args):
        node, t = args
        return 100.0 * t + node


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (
            lambda: NodeBC(nodes=[0], value=lambda node, t: 0.0),
            r"takes \(node, t\), which is out of order.*\(t, node\)",
        ),
        (lambda: NodeBC(nodes=[0], value=lambda a, b: 0.0), r"cannot be matched to \(t, node\)"),
        (lambda: NodeBC(nodes=[0], value=_NoSignature()), "not a plain def or lambda"),
        (lambda: NodeBC(nodes=[0], bc_type=GraphBCType.CUSTOM, value=1.0), "takes its rule as rule="),
        (lambda: NodeBC(nodes=[0], bc_type=GraphBCType.CUSTOM, rule=lambda field, node, t: 0.0), "out of order"),
        (lambda: NodeBC(nodes=[0], rule=lambda t, node, field: 0.0), "rule= is a CUSTOM condition's"),
    ],
    ids=["old order", "names no slot", "no signature", "CUSTOM scalar", "CUSTOM old order", "rule off CUSTOM"],
)
def test_a_value_or_rule_it_cannot_read_by_its_names_is_refused_where_it_is_built(build, match):
    with pytest.raises(TypeError, match=match):
        build()


def test_a_value_that_reads_t_needs_a_time_and_a_row_of_its_own():
    """Given no time, the readers read a callable at t = 0 in silence; on a 2-D (Nt, N) field, at the one t given, into
    every row, and so did a CUSTOM rule; a value returning an array was written down its node's column. Each is refused
    by name. A constant reads as before; a value that reads no t, which main called with two arguments, is read with
    none, into every row."""
    applicator = GraphApplicator(4).set_dirichlet_nodes([2], value=_g)
    for read in (
        lambda: applicator.apply(np.zeros(4)),
        lambda: applicator.apply_hjb(np.zeros(4)),
        lambda: applicator.apply_direct(np.zeros(4), [2], _g),
    ):
        with pytest.raises(TypeError, match="no time was given"):
            read()
    column = GraphApplicator(4).set_dirichlet_nodes([2], value=lambda node: np.array([1.0, 2.0, 3.0]))
    for read, match in (
        (lambda: applicator.apply(np.zeros((3, 4)), t=0.5), "2-D"),
        (lambda: applicator.apply_direct(np.zeros((3, 4)), [2], _g, t=0.5), "2-D"),
        (lambda: GraphApplicator(4).set_custom_bc([2], lambda t, node, field: t).apply(np.zeros((3, 4)), t=0.5), "2-D"),
        # A value that reads no t is the one callable left to reach a 2-D field. Main called this one with two arguments
        # and raised; one written `lambda node, t:` it wrote down the node's column, which the order refusal closes now.
        (lambda: column.apply(np.zeros((3, 4))), "not one real number"),
    ):
        with pytest.raises(TypeError, match=match):
            read()
    timeless = GraphApplicator(4).set_dirichlet_nodes([2], value=lambda node: 10.0 * node)
    np.testing.assert_array_equal(timeless.apply(np.zeros((3, 4)))[:, 2], [20.0, 20.0, 20.0])

"""A graph node's constant is read by the reader its return is read by, `types._read_constant` (#2861, #2512 row B2).

Its form stays the graph's own, a real number a float holds, so what a grid constant takes beyond that stays refused by
name at the doors that take one value. A NaN or an infinity, which the graph wrote into the field, is refused at each
door, as the grid refuses it (#2593) and as a graph callable's return already was. A finite value reads as it did, but for
the sign of a zero: the reader's zero test reads ``-0.0`` as ``0.0``.
"""

import math
from fractions import Fraction

import pytest

import numpy as np

from mfgarchon.geometry.boundary.applicator_graph import GraphApplicator, GraphBCType, NodeBC, create_graph_applicator
from mfgarchon.geometry.boundary.types import _read_constant, boundary_value

_WRITTEN_BEFORE = [math.nan, math.inf, -math.inf, np.float64(np.inf), np.float32(np.nan)]


def _maze():
    from mfgarchon.geometry.graph.maze_generator import MazeGeometry

    return MazeGeometry(3, 3, seed=0)


def _at_a_node(value):
    return GraphApplicator(4).add_node_bc(NodeBC(nodes=[2], value=value)).apply(np.zeros(4))[2]


_DOORS = {
    "NodeBC": _at_a_node,
    "set_dirichlet_nodes": lambda v: GraphApplicator(4).set_dirichlet_nodes([2], value=v).apply(np.zeros(4))[2],
    "set_source_nodes": lambda v: GraphApplicator(4).set_source_nodes([2], value=v).apply_fp(np.zeros(4))[2],
    "apply_direct, a value": lambda v: GraphApplicator(4).apply_direct(np.zeros(4), [2], v)[2],
    "apply_direct, a list": lambda v: GraphApplicator(4).apply_direct(np.zeros(4), [2], [v])[2],
    "apply_direct, a tuple": lambda v: GraphApplicator(4).apply_direct(np.zeros(4), [2], (v,))[2],
    "apply_direct, an array": lambda v: GraphApplicator(4).apply_direct(np.zeros(4), [2], np.array([v]))[2],
    "create_graph_applicator, boundary_value": lambda v: create_graph_applicator(
        4, boundary_nodes=[2], boundary_value=v
    ).apply(np.zeros(4))[2],
    "create_graph_applicator, source_value": lambda v: create_graph_applicator(
        4, source_nodes=[2], source_value=v
    ).apply_fp(np.zeros(4))[2],
}


@pytest.mark.parametrize("value", _WRITTEN_BEFORE, ids=repr)
@pytest.mark.parametrize("door", list(_DOORS))
def test_a_non_finite_constant_is_refused_at_each_door(door, value):
    with pytest.raises(NotImplementedError, match=r"non-finite value .*the graph applicator .*#2593"):
        _DOORS[door](value)


@pytest.mark.parametrize("value", [math.inf, math.nan], ids=repr)
def test_a_non_finite_boundary_value_is_refused_by_the_geometry_factory(value):
    with pytest.raises(NotImplementedError, match=r"from_network_geometry: boundary_value has a non-finite value"):
        GraphApplicator.from_network_geometry(_maze(), "degree", boundary_value=value)


@pytest.mark.parametrize("door", list(_DOORS))
@pytest.mark.parametrize("value", [2.5, 0, -3, np.float32(0.5)], ids=repr)
def test_a_finite_constant_reads_at_each_door_as_the_grid_reads_it(door, value):
    assert _DOORS[door](value) == _read_constant(value, kind="a grid constant", where="a grid")


def test_a_fraction_is_read_where_it_is_given():
    """An array of one is an object array, which `values` refuses by its dtype."""
    assert _at_a_node(Fraction(1, 4)) == 0.25
    assert GraphApplicator(4).apply_direct(np.zeros(4), [2], [Fraction(1, 4)])[2] == 0.25


@pytest.mark.parametrize(
    "value", [None, np.ma.masked, np.array([5.0]), np.array(5.0), "0"], ids=["None", "masked", "[5.0]", "5.0", "'0'"]
)
def test_a_form_only_the_grid_takes_stays_refused_on_a_graph_constant(value):
    """Where the two differ at a door that takes one value, the graph refuses by name, and no number reads differently.
    `apply_direct`'s ``values`` takes one value per node, so a one-element array is that door's own form, and reads."""
    with pytest.raises(TypeError, match=r"it takes a finite real number a float holds"):
        _at_a_node(value)


def test_a_neumann_node_refuses_a_form_as_any_node_does():
    """Its zero-flux check comes first, so it reads a value only once that value has the form: an array has no truth."""
    with pytest.raises(TypeError, match=r"it takes a finite real number a float holds"):
        NodeBC(nodes=[2], bc_type=GraphBCType.NEUMANN, value=np.array([1.0, 2.0]))


@pytest.mark.parametrize("too_large", [10**400, 10**5000], ids=["10**400", "10**5000"])
def test_a_number_no_float_holds_is_refused_by_name_on_each_path(too_large):
    """Each was Python's unnamed ``OverflowError`` but the graph's constant, which keeps its own refusal. The refusal
    prints its type: ``10**5000`` raises ``ValueError`` on being printed, past Python's limit on an int's digits."""
    where = {"kind": "a value", "where": "a grid"}
    with pytest.raises(NotImplementedError, match=r"a value has a value of type int no float holds"):
        _read_constant(too_large, **where)
    with pytest.raises(NotImplementedError, match=r"of type int no float holds"):
        boundary_value(lambda t, x: too_large, t=0.0, x=np.zeros(1), **where)
    with pytest.raises(NotImplementedError, match=r"of type int no float holds"):
        GraphApplicator(4).set_dirichlet_nodes([2], value=lambda t, node: too_large).apply(np.zeros(4), t=0.0)
    with pytest.raises(TypeError, match=r"a real number no float holds"):
        _at_a_node(too_large)


def test_a_non_finite_refusal_says_what_to_pass():
    with pytest.raises(NotImplementedError, match=r"non-finite solution: pass a finite real number"):
        _at_a_node(math.inf)
    with pytest.raises(NotImplementedError, match=r"non-finite solution: return one real number"):
        boundary_value(lambda t, x: math.inf, kind="a value", t=0.0, x=np.zeros(1), where="a grid")

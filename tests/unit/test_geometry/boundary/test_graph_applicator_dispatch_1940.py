"""`GraphApplicator` must reject an unknown field type and count degrees combinatorially.

Both defects share a shape: something that looks like a check is not one. `Literal` is an annotation,
not a runtime guard; `adj.sum(axis=1)` looks like a degree and is a strength. #1940
"""

from __future__ import annotations

import pytest

import numpy as np
import scipy.sparse as sp

from mfgarchon.geometry.boundary.applicator_graph import (
    GraphApplicator,
    GraphBCType,
    NodeBC,
)


def _path_graph_geometry(weight: float):
    """`0 - 1 - 2 - 3`, so the leaves are unambiguously nodes 0 and 3 whatever the edge weight."""

    matrix = sp.lil_matrix((4, 4))
    for i, j in ((0, 1), (1, 2), (2, 3)):
        matrix[i, j] = weight
        matrix[j, i] = weight

    class _NetworkData:
        adjacency_matrix = matrix.tocsr()

    class _Geometry:
        network_data = _NetworkData()

    return _Geometry()


@pytest.mark.parametrize("weight", [1.0, 0.5, 3.0])
def test_leaf_detection_does_not_depend_on_edge_weight(weight):
    """`adj.sum(axis=1)` is the weighted degree -- the node strength. On this graph at weight 0.5 it
    is `[0.5, 1.0, 1.0, 0.5]`, so `degrees == 1` selected nodes 1 and 2: the two NON-leaves, and the
    boundary condition was applied to the interior.

    Three weights, one of which (1.0) is the degenerate case where strength and degree coincide --
    that row passes either way and is here to show the parametrisation is not simply always-red.
    """
    assert GraphApplicator._detect_leaf_nodes(_path_graph_geometry(weight)) == [0, 3]


@pytest.mark.parametrize("weight", [1.0, 0.5, 3.0])
def test_low_degree_detection_does_not_depend_on_edge_weight(weight):
    """Same defect in the sibling detector. At weight 0.5, `degrees <= 1` selected all four nodes."""
    assert GraphApplicator._detect_low_degree_nodes(_path_graph_geometry(weight), threshold=1) == [0, 3]


def test_the_detectors_agree_with_each_other_at_threshold_one():
    """`_detect_leaf_nodes` and `_detect_low_degree_nodes(1)` are the same question asked twice.
    They now share one degree computation; this asserts they cannot drift apart again."""
    geometry = _path_graph_geometry(0.5)

    assert GraphApplicator._detect_leaf_nodes(geometry) == GraphApplicator._detect_low_degree_nodes(
        geometry, threshold=1
    )


def _pinned_applicator() -> GraphApplicator:
    applicator = GraphApplicator(num_nodes=4)
    applicator.add_node_bc(NodeBC(name="pin", nodes=[0], bc_type=GraphBCType.DIRICHLET, value=5.0))
    return applicator


@pytest.mark.parametrize("field_type", ["hjb", "VALUE", "u", "", "Density"])
def test_an_unrecognised_field_type_raises_rather_than_no_opping(field_type):
    """`field_type: Literal["value", "density"]` is a type annotation, not a runtime check. Every
    branch tests `== "value"` or `== "density"`, so any other string matched nothing and the field
    came back unmodified with no error -- including `"VALUE"`, which differs only in case."""
    with pytest.raises(ValueError, match="field_type must be"):
        _pinned_applicator().apply(np.array([1.0, 2.0, 3.0, 4.0]), field_type=field_type)


def test_the_two_accepted_field_types_still_do_their_separate_jobs():
    """Control for the guard above. A Dirichlet node pin belongs to the VALUE field and not to the
    density (#1471, adjoint duality), so these two must differ -- a guard that rejected everything,
    or one that made both branches behave alike, would fail here."""
    applicator = _pinned_applicator()
    field = np.array([1.0, 2.0, 3.0, 4.0])

    value_result = applicator.apply(field.copy(), field_type="value")
    density_result = applicator.apply(field.copy(), field_type="density")

    assert value_result[0] == pytest.approx(5.0), "the value field must take the Dirichlet pin"
    assert density_result[0] == pytest.approx(1.0), "the density field must not take a value pin"

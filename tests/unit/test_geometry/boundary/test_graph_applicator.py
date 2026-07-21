"""Issue #1471: first unit coverage for GraphApplicator and the graph-geometry node-BC attachment.

GraphApplicator becomes load-bearing once node-BC ownership moves to GraphGeometry (the network HJB
Dirichlet pin now flows through ``apply_hjb``). It previously had zero tests.
"""

import pytest

import numpy as np

from mfgarchon.geometry.boundary.applicator_graph import (
    GraphApplicator,
    GraphBCConfig,
    GraphBCType,
    NodeBC,
)


def _app(node_bcs, num_nodes=9):
    return GraphApplicator.from_config(GraphBCConfig(node_bcs=node_bcs), num_nodes=num_nodes)


def test_dirichlet_pins_value_not_density():
    """DIRICHLET is a value (HJB) pin — it must not touch the density (FP) field (Issue #1471 gate)."""
    app = _app([NodeBC(nodes=[0, 8], bc_type=GraphBCType.DIRICHLET, value=5.0)])
    u = np.full(9, 2.0)
    u_bc = app.apply_hjb(u.copy(), t=0.0)
    assert u_bc[0] == 5.0
    assert u_bc[8] == 5.0
    assert u_bc[4] == 2.0, "interior node untouched"
    assert u[0] == 2.0, "input must not be mutated (copied)"
    m = np.full(9, 0.1)
    m_bc = app.apply_fp(m.copy(), t=0.0)
    assert np.array_equal(m_bc, m), "a Dirichlet value pin must be a no-op on the density field"


def test_absorbing_is_a_dual_exit_node():
    """Issue #1478: an ABSORBING/exit node is one physical BC with dual operations — the FP density is
    absorbed (m -> 0) and the HJB value carries the exit cost (Dirichlet u = value)."""
    app = _app([NodeBC(nodes=[4], bc_type=GraphBCType.ABSORBING, value=5.0)])
    m_bc = app.apply_fp(np.full(9, 0.2), t=0.0)
    assert m_bc[4] == 0.0, "FP: exit node mass leaves (m -> 0)"
    assert m_bc[0] == 0.2
    u_bc = app.apply_hjb(np.full(9, 3.0), t=0.0)
    assert u_bc[4] == 5.0, "HJB: exit node value pinned to the exit cost (Dirichlet)"
    assert u_bc[0] == 3.0


def test_source_injects_density_only():
    app = _app([NodeBC(nodes=[2], bc_type=GraphBCType.SOURCE, value=3.0)])
    m_bc = app.apply_fp(np.full(9, 0.1), t=0.0)
    assert m_bc[2] == 3.0
    u_bc = app.apply_hjb(np.full(9, 2.0), t=0.0)
    assert u_bc[2] == 2.0, "source must not modify the value field"


def test_callable_value_and_2d_time_broadcast():
    app = _app([NodeBC(nodes=[0], bc_type=GraphBCType.DIRICHLET, value=lambda n, t: 7.0 + t)])
    u_2d = np.full((5, 9), 2.0)
    u_bc = app.apply_hjb(u_2d.copy(), t=0.5)
    assert np.allclose(u_bc[:, 0], 7.5), "2D (Nt, num_nodes) time-broadcast column-pin with callable value"


def test_out_of_range_node_skipped():
    app = _app([NodeBC(nodes=[0, 99], bc_type=GraphBCType.DIRICHLET, value=5.0)], num_nodes=9)
    u_bc = app.apply_hjb(np.full(9, 2.0), t=0.0)
    assert u_bc[0] == 5.0, "in-range node pinned; out-of-range node 99 skipped without IndexError"


def test_graph_geometry_owns_node_bc_and_maze_inherits():
    """Issue #1471: node-BC attaches at GraphGeometry, so both NetworkGeometry and MazeGeometry
    inherit it (graph is the highest abstraction over both; graphon is separate)."""
    pytest.importorskip("igraph")
    from mfgarchon.geometry.graph.maze_generator import MazeGeometry
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    cfg = GraphBCConfig(node_bcs=[NodeBC(nodes=[0], bc_type=GraphBCType.DIRICHLET, value=1.0)])

    net = GridNetwork(width=3, height=3, boundary_conditions=cfg)
    assert net.has_explicit_boundary_conditions() is True
    assert net.get_boundary_conditions() is cfg

    net_none = GridNetwork(width=3, height=3)
    assert net_none.has_explicit_boundary_conditions() is False
    assert net_none.get_boundary_conditions() is None

    maze = MazeGeometry(rows=3, cols=3, boundary_conditions=cfg)
    assert maze.has_explicit_boundary_conditions() is True, "maze must inherit the GraphGeometry attachment"
    assert maze.get_boundary_conditions() is cfg

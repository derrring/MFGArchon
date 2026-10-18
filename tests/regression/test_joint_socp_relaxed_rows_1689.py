"""#1689 PR 2a: ``joint_socp`` refuses an interior row its SOCP cannot place, and ``socp_allow_relaxed_rows=True``
gives main's relaxed rows, bit for bit on the capture machine.

The cloud is the smallest of a search that makes main relax rows: 120 Sobol points in [0, 2] x [0, 1] with the band
0.7 < x < 1.3 emptied, as exp09's cull empties the band beside its wall, plus 8 boundary nodes per side. The rows
beside the band have neighbours on one side only, and five of them have no exact joint SOCP.

ADMISSION (#2227): class 2, an oracle against pre-change output. ``fixtures/joint_socp_relaxed_rows_1689.npz`` is
main aec3d361's own output on this cloud, captured before the change: the cloud, delta and k, and for each relaxed
row its neighbour indices, L, D and the eps_M its relaxed SOCP returned, recorded by wrapping
``solve_relaxed_joint_socp_at_stencil`` because main keeps only the maximum. The row set and the indices are
compared exactly; L, D and eps_M at rtol 0, atol 1e-12, a floor for cross-platform last bits, as
``test_solver_golden_baseline.py`` does. On the capture machine the difference is 0.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_FIXTURE = Path(__file__).parent / "fixtures" / "joint_socp_relaxed_rows_1689.npz"


def _solver(ref, *, allow_relaxed: bool) -> HJBGFDMSolver:
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=1.0
        ),
        domain=TensorProductGrid(bounds=[(0, 2.0), (0, 1.0)], Nx_points=[9, 5], boundary_conditions=no_flux_bc(2)),
        conditions=Conditions(m_initial=lambda x: 0.5, u_terminal=lambda x: 0.0, T=1.0),
        Nt=4,
    )
    return HJBGFDMSolver(
        problem,
        collocation_points=ref["points"],
        boundary_indices=ref["boundary_indices"],
        delta=float(ref["delta"]),
        k_neighbors=int(ref["k_neighbors"]),
        derivative_method="taylor",
        taylor_order=2,
        weight_function="wendland",
        adaptive_neighborhoods=True,
        monotonicity_scheme="joint_socp",
        monotonicity_application="precompute",
        socp_allow_relaxed_rows=allow_relaxed,
    )


def test_opted_in_relaxed_rows_are_mains(mfg_caplog):
    ref = np.load(_FIXTURE)
    rows = [int(r) for r in ref["relaxed_rows"]]
    with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm"):
        stencils = _solver(ref, allow_relaxed=True)._joint_socp_stencils
    assert sorted(stencils.relaxed_rows) == rows
    for r in rows:
        row = stencils.stencils[r]
        np.testing.assert_array_equal(row.neighbor_indices, ref[f"row{r}_neighbor_indices"])
        np.testing.assert_allclose(row.L, ref[f"row{r}_L"], rtol=0, atol=1e-12, err_msg=f"row {r}: L")
        np.testing.assert_allclose(row.D, ref[f"row{r}_D"], rtol=0, atol=1e-12, err_msg=f"row {r}: D")
        assert abs(stencils.relaxed_rows[r] - float(ref[f"row{r}_eps_M"])) <= 1e-12, f"row {r}: eps_M"
    assert stencils.stats["n_feasible"] == int(ref["n_interior"]) - len(rows), "a relaxed row counted as feasible"
    report = [m for m in mfg_caplog.messages if "socp_allow_relaxed_rows=True" in m]
    assert len(report) == 1, mfg_caplog.messages
    assert f"{len(rows)} interior row(s)" in report[0]
    assert f"max eps_M = {float(ref['max_eps_M']):.4g}" in report[0]


def test_without_the_opt_in_the_rows_are_refused_by_name():
    ref = np.load(_FIXTURE)
    with pytest.raises(ValueError, match=r"have no joint-SOCP stencil") as refusal:
        _solver(ref, allow_relaxed=False)
    message = str(refusal.value)
    assert f"{len(ref['relaxed_rows'])} interior row(s)" in message
    for r in ref["relaxed_rows"]:
        assert f"{int(r)} at " in message, f"row {int(r)} is not named"
    assert "socp_max_stencil_enlargements" in message
    assert "socp_allow_relaxed_rows=True" in message


@pytest.mark.parametrize("allow_relaxed", [False, True])
def test_a_row_its_neighbourhood_does_not_contain_is_refused(allow_relaxed):
    """A node within ``visibility_margin`` of ``obstacle_sdf`` is removed from its own neighbourhood by the visibility
    filter, so the joint SOCP builds no row there. Main skipped it silently, and at solve time it took bare
    Wendland-Taylor weights. Found by #2741's review 1: the census's 2-D cloud (11 x 11, interior nodes jittered by up
    to 0.03, seed 0, delta 0.3) with a disc of radius 0.03 centred 0.05 to the right of the interior node nearest
    (0.5, 0.5), and ``visibility_margin=0.03``. The refusal names that node with its reason, and the opt-in, which
    accepts relaxed rows only, does not lift it."""
    grid = np.linspace(0.0, 1.0, 11)
    xs, ys = np.meshgrid(grid, grid, indexing="ij")
    points = np.column_stack([xs.ravel(), ys.ravel()])
    on_boundary = (points[:, 0] % 1 == 0) | (points[:, 1] % 1 == 0)
    points[~on_boundary] += np.random.default_rng(0).uniform(-0.03, 0.03, (int((~on_boundary).sum()), 2))
    interior = np.flatnonzero(~on_boundary)
    node = int(interior[np.argmin(np.linalg.norm(points[interior] - 0.5, axis=1))])
    centre = points[node] + np.array([0.05, 0.0])

    def disc(x):
        return np.linalg.norm(np.atleast_2d(np.asarray(x, dtype=float)) - centre, axis=1) - 0.03

    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=1.0
        ),
        domain=TensorProductGrid(bounds=[(0, 1.0), (0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(2)),
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
        Nt=4,
    )
    with pytest.raises(ValueError, match=r"have no joint-SOCP stencil") as refusal:
        HJBGFDMSolver(
            problem,
            collocation_points=points,
            boundary_indices=np.flatnonzero(on_boundary),
            delta=0.3,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            adaptive_neighborhoods=True,
            monotonicity_scheme="joint_socp",
            monotonicity_application="precompute",
            obstacle_sdf=disc,
            visibility_margin=0.03,
            socp_allow_relaxed_rows=allow_relaxed,
        )
    message = str(refusal.value)
    assert f"{node} at " in message, f"node {node} is not named: {message}"
    assert "its neighbourhood does not contain it" in message
    assert "move it out of the margin, or reduce visibility_margin" in message


def test_ghost_nodes_with_joint_socp_are_refused_by_name():
    """Ghost nodes keep the PDE row at a Neumann boundary node; joint_socp builds monotone rows for interior nodes only,
    so that row would carry bare Wendland-Taylor weights. Refused by name. At main it raised IndexError inside the
    boundary buffer #1689 deleted; without the refusal it now constructs and solves on those rows."""
    xs = np.linspace(0.0, 1.0, 11)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel()])
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.6
        ),
        domain=TensorProductGrid(bounds=[(0, 1.0), (0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(2)),
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.2),
        Nt=4,
    )
    with pytest.raises(ValueError, match=r"use_ghost_nodes=True\) is refused.*interior rows only"):
        HJBGFDMSolver(
            problem,
            collocation_points=points,
            boundary_indices=np.flatnonzero(np.min(np.abs(np.column_stack([points, 1.0 - points])), axis=1) < 1e-12),
            delta=0.3,
            monotonicity_scheme="joint_socp",
            use_ghost_nodes=True,
        )

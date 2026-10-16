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

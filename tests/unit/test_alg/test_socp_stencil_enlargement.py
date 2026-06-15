"""Issue #1106: SOCP-infeasibility-triggered adaptive stencil enlargement.

Some GFDM stencils near walls / corners / obstacle edges are *geometrically*
infeasible for the joint SOCP: Taylor consistency (``A^T L = e_lap``,
``A^T D = e_grad``) + M-matrix (``L_off >= 0``) + per-edge cone
(``||D_j|| <= C h_i L_j``) have an empty feasible set at the base
k_neighbors / delta, so C-bisection cannot recover them (the infeasibility is
*directional*, not a cone-magnitude issue). Penalty pressure does not fix this
(PR #1105). The fix adds Taylor degrees of freedom: when a stencil is infeasible
after C-bisection, add next-nearest neighbors and retry, up to a capped number
of enlargement steps.

These tests pin:
  (a) a stencil infeasible at base size becomes feasible after enlargement;
  (b) enlargement reduces the number of stencils that hit the relaxed fallback;
  (c) the enlarged stencil's stored (L, D, neighbor_indices) lengths agree
      (the precompute/runtime single-source contract that the HJB-GFDM
      assembly asserts on — Issue #1102 dual-source bug class);
  (d) enlargement OFF (default) is byte-identical to not passing the option;
  (e) cloud exhaustion and invalid configuration are handled.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

pytest.importorskip("cvxpy")

from mfgarchon.alg.numerical.gfdm_components.joint_socp import PrecomputedJointSocpStencils
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Deterministic irregular cloud with controllable (small) base stencils.
# Building neighborhoods directly as exactly-k nearest gives full control over
# the base stencil size, so infeasibility is reproducible (the solver's
# NeighborhoodBuilder otherwise returns whole-ball stencils that are too rich).
# ---------------------------------------------------------------------------


def _irregular_cloud(seed: int = 3, n: int = 160):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(n, 2))
    interior = np.array([i for i in range(n) if 0.15 < pts[i, 0] < 0.85 and 0.15 < pts[i, 1] < 0.85])
    return pts, interior


def _knn_neighborhoods(pts: np.ndarray, k: int) -> dict:
    """Base neighborhoods = exactly the k nearest points (center included)."""
    tree = cKDTree(pts)
    nbh = {}
    for i in range(len(pts)):
        _, idx = tree.query(pts[i], k=k)
        nbh[i] = {"indices": np.asarray(idx)}
    return nbh


def _make_stencils(pts, interior, nbh, *, enlarge, relaxed=False, delta=0.4, **kw):
    return PrecomputedJointSocpStencils(
        points=pts,
        interior_indices=interior,
        delta=delta,
        neighborhoods=nbh,
        cone_constant_C=8.0,
        eps_pos=0.0,
        cone_constant_C_max=8.0,
        use_relaxed_fallback=relaxed,
        max_stencil_enlargements=enlarge,
        enlargement_step=2,
        **kw,
    )


# ---------------------------------------------------------------------------
# (a) infeasible-at-base -> feasible-after-enlargement
# ---------------------------------------------------------------------------


def test_enlargement_flips_infeasible_stencils_to_feasible():
    """With the relaxed fallback OFF (so infeasibility is exposed), enabling
    enlargement strictly increases the feasible count and decreases the
    infeasible count, and records the enlargements it used."""
    pts, interior = _irregular_cloud()
    nbh = _knn_neighborhoods(pts, k=7)

    base = _make_stencils(pts, interior, nbh, enlarge=0, relaxed=False)
    enl = _make_stencils(pts, interior, nbh, enlarge=3, relaxed=False)

    assert base.stats["n_infeasible"] > 0, "test cloud must produce genuine base infeasibility"
    assert enl.stats["n_feasible"] > base.stats["n_feasible"], (
        f"enlargement should add feasible stencils: base {base.stats['n_feasible']} "
        f"-> enlarged {enl.stats['n_feasible']}"
    )
    assert enl.stats["n_infeasible"] < base.stats["n_infeasible"]
    assert enl.stats["n_enlarged"] > 0
    assert 1 <= enl.stats["max_enlargement_steps"] <= 3


def test_specific_stencil_infeasible_at_base_feasible_when_enlarged():
    """Identify a concrete stencil that is infeasible at its base (wall-starved)
    size and feasible only after enlargement; verify the recovered weights are a
    valid M-matrix Laplacian on the enlarged neighbor set."""
    pts, interior = _irregular_cloud()
    nbh = _knn_neighborhoods(pts, k=7)

    base = _make_stencils(pts, interior, nbh, enlarge=0, relaxed=False)
    enl = _make_stencils(pts, interior, nbh, enlarge=3, relaxed=False)

    base_infeasible = {int(i) for i in interior if not base.has_stencil(int(i))}
    recovered = sorted(i for i in base_infeasible if enl.has_stencil(i))
    assert recovered, "no infeasible-at-base stencil was recovered by enlargement"

    i = recovered[0]
    sd = enl.stencils[i]
    base_len = len(nbh[i]["indices"])
    assert len(sd.neighbor_indices) > base_len, (
        f"recovered stencil {i} must have grown beyond base size {base_len}, got {len(sd.neighbor_indices)}"
    )
    # Center present, Laplacian consistency (sum-zero) and M-matrix sign.
    assert i in set(int(x) for x in sd.neighbor_indices)
    assert np.isclose(sd.L.sum(), 0.0, atol=1e-8)
    L_off = np.delete(sd.L, sd.center_in_neighbors)
    assert np.all(L_off >= -1e-8)


# ---------------------------------------------------------------------------
# (b) enlargement reduces relaxed-fallback usage
# ---------------------------------------------------------------------------


def test_enlargement_reduces_relaxed_fallback_count():
    """With the relaxed fallback ON (the HJB-GFDM default), enlargement converts
    would-be relaxed (slack-active) stencils into exact-feasible ones, so the
    relaxed-fallback count strictly drops."""
    pts, interior = _irregular_cloud()
    nbh = _knn_neighborhoods(pts, k=7)

    base = _make_stencils(pts, interior, nbh, enlarge=0, relaxed=True)
    enl = _make_stencils(pts, interior, nbh, enlarge=3, relaxed=True)

    assert base.stats["n_relaxed_fallback"] > 0
    assert enl.stats["n_relaxed_fallback"] < base.stats["n_relaxed_fallback"], (
        f"enlargement should reduce relaxed-fallback usage: "
        f"{base.stats['n_relaxed_fallback']} -> {enl.stats['n_relaxed_fallback']}"
    )
    # Enlargement recovers exact-feasible stencils, so the exact-feasible set
    # never shrinks and the residual hard-infeasible set never grows.
    # (On these deliberately starved k=7 stencils even the always-feasible
    # relaxed SOCP occasionally fails to converge, so n_infeasible is not
    # guaranteed zero — the point is that enlargement only helps.)
    assert enl.stats["n_feasible"] >= base.stats["n_feasible"]
    assert enl.stats["n_infeasible"] <= base.stats["n_infeasible"]


# ---------------------------------------------------------------------------
# (c) precompute/runtime single-source contract: stored lengths agree
# ---------------------------------------------------------------------------


def test_enlarged_stencil_weight_index_lengths_agree():
    """Every stored stencil (enlarged or not) must have L, D and neighbor_indices
    of mutually consistent length — this is exactly the invariant the HJB-GFDM
    differentiation/Jacobian assembly asserts on (Issue #1102 / #1106)."""
    pts, interior = _irregular_cloud()
    nbh = _knn_neighborhoods(pts, k=7)
    enl = _make_stencils(pts, interior, nbh, enlarge=3, relaxed=True)

    n_grown = 0
    for i, sd in enl.stencils.items():
        n = len(sd.neighbor_indices)
        assert sd.L.shape == (n,), f"point {i}: L shape {sd.L.shape} != ({n},)"
        assert sd.D.shape == (2, n), f"point {i}: D shape {sd.D.shape} != (2, {n})"
        assert 0 <= sd.center_in_neighbors < n
        assert int(sd.neighbor_indices[sd.center_in_neighbors]) == i
        wd = enl.get_weights_dict(i)
        assert wd["grad_weights"].shape[1] == len(wd["lap_weights"]) == len(wd["neighbor_indices"])
        if n > len(nbh[i]["indices"]):
            n_grown += 1
    assert n_grown > 0, "expected at least one stencil to have grown via enlargement"


# ---------------------------------------------------------------------------
# (d) enlargement OFF is byte-identical to not passing the option
# ---------------------------------------------------------------------------


def test_enlargement_off_is_byte_identical_to_default():
    """max_stencil_enlargements=0 (and the no-arg default) must produce the exact
    same stencils and stats — the paper / default path is unchanged."""
    pts, interior = _irregular_cloud()
    nbh = _knn_neighborhoods(pts, k=7)

    default = PrecomputedJointSocpStencils(
        points=pts,
        interior_indices=interior,
        delta=0.4,
        neighborhoods=nbh,
        cone_constant_C=8.0,
        eps_pos=0.0,
        cone_constant_C_max=8.0,
        use_relaxed_fallback=True,
    )
    off = _make_stencils(pts, interior, nbh, enlarge=0, relaxed=True)

    assert default.stats["n_feasible"] == off.stats["n_feasible"]
    assert default.stats["n_relaxed_fallback"] == off.stats["n_relaxed_fallback"]
    assert off.stats["n_enlarged"] == 0
    assert set(default.stencils) == set(off.stencils)
    for i in default.stencils:
        np.testing.assert_array_equal(default.stencils[i].L, off.stencils[i].L)
        np.testing.assert_array_equal(default.stencils[i].D, off.stencils[i].D)
        np.testing.assert_array_equal(default.stencils[i].neighbor_indices, off.stencils[i].neighbor_indices)


# ---------------------------------------------------------------------------
# (e) edge / failure modes
# ---------------------------------------------------------------------------


def test_enlargement_handles_cloud_exhaustion():
    """A tiny cloud where the whole cloud is already in the stencil must not
    crash: _enlarge_stencil returns None and the stencil falls through to the
    relaxed fallback / infeasible count."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.3, 0.7], [0.7, 0.3]])
    interior = np.array([4])  # center point; base stencil = all 7 points
    nbh = {i: {"indices": np.arange(len(pts))} for i in range(len(pts))}
    # No crash even with a large enlargement budget; cloud has nothing to add.
    s = _make_stencils(pts, interior, nbh, enlarge=5, relaxed=False, delta=2.0)
    assert s.stats["n_enlarged"] == 0  # nothing could be added
    assert s.stats["n_interior"] == 1


def test_invalid_enlargement_step_raises():
    """enlargement_step < 1 with enlargement enabled is a configuration error."""
    pts, interior = _irregular_cloud(n=40)
    nbh = _knn_neighborhoods(pts, k=7)
    with pytest.raises(ValueError, match="enlargement_step"):
        PrecomputedJointSocpStencils(
            points=pts,
            interior_indices=interior,
            delta=0.4,
            neighborhoods=nbh,
            max_stencil_enlargements=2,
            enlargement_step=0,
        )


# ---------------------------------------------------------------------------
# Solver-level: enlargement flows through the runtime consumption paths
# without a matmul-shape mismatch (Path 1 differentiation matrices, Path 2
# per-point derivatives). This is the end-to-end precompute/runtime contract.
# ---------------------------------------------------------------------------


def _solver_with_small_stencils(enlarge: int):
    import sys

    sys.path.insert(0, "tests/unit/test_alg")
    from test_joint_socp_mirror_symmetry import _MockProblem  # noqa: PLC0415

    from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
    from mfgarchon.geometry import Hyperrectangle
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    rng = np.random.default_rng(11)
    n = 120
    interior_pts = rng.uniform(0.1, 0.9, size=(n, 2))
    bx = []
    for t in np.linspace(0, 1, 16):
        bx += [[t, 0.0], [t, 1.0], [0.0, t], [1.0, t]]
    pts = np.vstack([interior_pts, np.array(bx)])
    bdry = np.arange(n, len(pts))
    geom = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))
    bc = BoundaryConditions(
        segments=[
            BCSegment(name=nm, bc_type=BCType.NO_FLUX, boundary=b)
            for nm, b in [("l", "x_min"), ("r", "x_max"), ("b", "y_min"), ("t", "y_max")]
        ],
        dimension=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = HJBGFDMSolver(
            _MockProblem(geom),
            collocation_points=pts,
            boundary_indices=bdry,
            delta=0.12,
            k_neighbors=7,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            collocation_geometry=geom,
            adaptive_neighborhoods=False,
            boundary_conditions=bc,
            monotonicity_scheme="joint_socp",
            socp_max_stencil_enlargements=enlarge,
            socp_enlargement_step=2,
        )
    return solver, pts


def test_solver_enlargement_end_to_end_runtime_consistent():
    """At the HJB-GFDM solver level: enabling enlargement reduces relaxed-fallback
    usage, propagates the enlarged neighbor sets into the stored SOCP stencils,
    and the runtime consumption paths (differentiation-matrix assembly + per-point
    derivative override) build without a matmul-shape mismatch — the assertions
    added in those paths would fire if the precompute/runtime contract broke."""
    base, _ = _solver_with_small_stencils(enlarge=0)
    solver, pts = _solver_with_small_stencils(enlarge=3)

    bstats = base._joint_socp_stencils.stats
    stats = solver._joint_socp_stencils.stats
    assert stats["n_enlarged"] > 0, "solver-level enlargement did not fire"
    assert stats["n_relaxed_fallback"] < bstats["n_relaxed_fallback"]

    # At least one SOCP stencil grew beyond the operator's base stencil.
    socp = solver._joint_socp_stencils
    op = solver._gfdm_operator
    grew = 0
    for i in socp._interior_indices:
        i = int(i)
        if not socp.has_stencil(i):
            continue
        base_len = len(op.get_derivative_weights(i)["neighbor_indices"])
        if len(socp.stencils[i].neighbor_indices) > base_len:
            grew += 1
    assert grew > 0, "no stored SOCP stencil reflects an enlarged neighbor set"

    # Path 1: differentiation-matrix assembly (asserts length contract internally).
    solver._build_differentiation_matrices()
    assert solver._D_lap is not None and len(solver._D_grad) == 2

    # Path 2: per-point derivative override on the grown stencils.
    u = np.random.default_rng(0).standard_normal(len(pts))
    for i in socp._interior_indices[:20]:
        derivs = solver.approximate_derivatives(u, int(i))
        assert derivs  # non-empty dict, no matmul-size error


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

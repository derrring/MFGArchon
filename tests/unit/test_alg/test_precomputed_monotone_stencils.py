"""Direct unit tests for PrecomputedMonotoneStencils.

Pre-#1102, the class accepted only ``operator``: at runtime,
``HJBGFDMSolver.approximate_derivatives`` contracts the precomputed
``L_w`` against ``b = u_neighbors - u_center`` built on
``self.neighborhoods[i]["indices"]``. When ``adaptive_neighborhoods=True``
enlarged the runtime neighborhood (e.g. 53 → 522 at corner buffer points
on a 1200-point Stage C cloud), the two stencils diverged and
``L_w @ b`` raised ``ValueError: matmul: size N is different from K``.

Audit 2026-05-10 D.1 flagged this class as having no dedicated tests. The
suite below closes that gap and locks in the post-fix invariants:

1. Legacy path (neighborhoods=None) reproduces pre-fix behaviour.
2. Required-arg validation: neighborhoods= without points/delta raises.
3. Matched-indices path: neighborhoods= with same indices as op produces
   equivalent stencils to the legacy path (M-matrix property preserved).
4. Enlarged-stencil path: neighborhoods= with strictly larger index sets
   produces L_w whose length matches the enlarged stencil. This is the
   load-bearing property that resolves #1102.
5. Integration regression: HJBGFDMSolver with adaptive_neighborhoods=True
   and monotonicity_scheme="qp_m_matrix" completes construction without
   the precompute-vs-runtime shape mismatch firing.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.gfdm_components.precomputed_stencils import (
    PrecomputedMonotoneStencils,
)

# ---------------------------------------------------------------------------
# Minimal mock TaylorOperator — enough to drive _precompute legacy path.
# Mirrors the dict shape returned by TaylorOperator.get_derivative_weights.
# ---------------------------------------------------------------------------


class _MockTaylorOperator:
    def __init__(self, points: np.ndarray, neighborhoods: dict, delta: float):
        """Build a stand-in operator with Wendland-LSQ unconstrained weights
        at each point, matching the post-filter neighborhoods.
        """
        from mfgarchon.alg.numerical.gfdm_components.joint_socp import (
            build_taylor_matrix_1d,
            build_taylor_matrix_2d,
            wendland_stencil_weights,
        )

        self.points = points
        self.dimension = points.shape[1] if points.ndim == 2 else 1
        self._weights_cache: dict[int, dict] = {}
        for i, nh in neighborhoods.items():
            nbr = np.asarray(nh["indices"])
            offsets = points[nbr] - points[i]
            if self.dimension == 1:
                offsets_1d = offsets.reshape(-1)
                A, _ = build_taylor_matrix_1d(offsets_1d)
                w = wendland_stencil_weights(offsets_1d, delta)
                e_lap = np.array([0.0, 0.0, 1.0])
            else:
                A, _ = build_taylor_matrix_2d(offsets)
                w = wendland_stencil_weights(offsets, delta)
                e_lap = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
            W = np.diag(w)
            ATA = A.T @ W @ A
            try:
                coeffs = np.linalg.solve(ATA, e_lap)
                lap = (W @ A) @ coeffs
            except np.linalg.LinAlgError:
                lap = np.zeros(len(nbr))
            center_in = int(np.where(nbr == i)[0][0]) if i in nbr.tolist() else -1
            self._weights_cache[int(i)] = {
                "neighbor_indices": nbr,
                "lap_weights": lap,
                "center_idx_in_neighbors": center_in,
            }

    def get_derivative_weights(self, point_idx: int) -> dict | None:
        return self._weights_cache.get(int(point_idx))


def _build_2d_grid_with_neighborhoods(nx: int = 5, ny: int = 5, delta: float = 1.5):
    """Construct a small 2D Cartesian cloud with k-NN-like neighborhoods.

    All boundary points share the same fixed neighbor count, simulating
    the pre-adaptive op state. Returns (points, op_neighborhoods, boundary_mask).
    """
    xs = np.linspace(0.0, float(nx - 1), nx)
    ys = np.linspace(0.0, float(ny - 1), ny)
    pts = np.array([[x, y] for x in xs for y in ys])
    n = len(pts)
    # k-NN with k=8 — every point has 8 closest neighbors plus itself.
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    op_nh: dict[int, dict] = {}
    for i in range(n):
        _, idx = tree.query(pts[i], k=9)  # includes self
        op_nh[i] = {"indices": np.asarray(idx, dtype=int)}
    # Boundary mask: outermost ring
    is_boundary = np.zeros(n, dtype=bool)
    for i, p in enumerate(pts):
        if p[0] in (xs[0], xs[-1]) or p[1] in (ys[0], ys[-1]):
            is_boundary[i] = True
    return pts, op_nh, is_boundary


# ---------------------------------------------------------------------------
# 1. Legacy path is unchanged
# ---------------------------------------------------------------------------


def test_legacy_path_unchanged_no_neighborhoods():
    """neighborhoods=None falls back to op.get_derivative_weights() (pre-#1102)."""
    pts, op_nh, is_b = _build_2d_grid_with_neighborhoods()
    op = _MockTaylorOperator(pts, op_nh, delta=1.5)

    precomp = PrecomputedMonotoneStencils(operator=op, is_boundary=is_b, tolerance=1e-6)

    assert precomp.stats["n_boundary"] == int(is_b.sum())
    for i in np.where(is_b)[0]:
        sd = precomp.stencils.get(int(i))
        assert sd is not None, f"Missing stencil at boundary point {i}"
        # Legacy stencil indices come from op directly.
        assert np.array_equal(sd.neighbor_indices, op_nh[int(i)]["indices"])


# ---------------------------------------------------------------------------
# 2. Required-arg validation
# ---------------------------------------------------------------------------


def test_neighborhoods_without_points_raises():
    pts, op_nh, is_b = _build_2d_grid_with_neighborhoods()
    op = _MockTaylorOperator(pts, op_nh, delta=1.5)
    with pytest.raises(ValueError, match="points= and delta= are also required"):
        PrecomputedMonotoneStencils(operator=op, is_boundary=is_b, neighborhoods=op_nh)


def test_neighborhoods_without_delta_raises():
    pts, op_nh, is_b = _build_2d_grid_with_neighborhoods()
    op = _MockTaylorOperator(pts, op_nh, delta=1.5)
    with pytest.raises(ValueError, match="points= and delta= are also required"):
        PrecomputedMonotoneStencils(operator=op, is_boundary=is_b, neighborhoods=op_nh, points=pts)


# ---------------------------------------------------------------------------
# 3. Matched-indices: neighborhoods= with same indices as op produces
#    M-matrix-compliant stencils on the same indices.
# ---------------------------------------------------------------------------


def test_matched_neighborhoods_produces_equivalent_stencils():
    """When neighborhoods== matches op's indices, both paths produce stencils
    on the same indices and both satisfy the M-matrix property after QP.

    Bit-exact equality is not asserted: the Wendland-LSQ recomputation in
    the new path may differ from op's stored unconstrained weights by the
    LSQ-conditioning of the operator (different SVD truncation/conditioning
    paths). The load-bearing invariant is: same indices, M-matrix-compliant.
    """
    pts, op_nh, is_b = _build_2d_grid_with_neighborhoods()
    op = _MockTaylorOperator(pts, op_nh, delta=1.5)

    legacy = PrecomputedMonotoneStencils(operator=op, is_boundary=is_b)
    fresh = PrecomputedMonotoneStencils(
        operator=op,
        is_boundary=is_b,
        neighborhoods=op_nh,
        points=pts,
        delta=1.5,
    )

    assert legacy.stats["n_boundary"] == fresh.stats["n_boundary"]
    for i in np.where(is_b)[0]:
        sd_l = legacy.stencils.get(int(i))
        sd_f = fresh.stencils.get(int(i))
        assert sd_l is not None
        assert sd_f is not None
        # Same stencil indices.
        assert np.array_equal(sd_l.neighbor_indices, sd_f.neighbor_indices), (
            f"point {i}: indices differ between legacy and fresh path"
        )
        # M-matrix property holds in fresh path (sum-to-zero and off-diagonal
        # non-negative — modulo tolerance).
        if sd_f.center_in_neighbors is not None:
            off = np.delete(sd_f.weights, sd_f.center_in_neighbors)
            assert np.all(off >= -1e-6), f"point {i}: fresh path off-diagonal weights have negative entries"
            assert abs(np.sum(sd_f.weights)) < 1e-6, f"point {i}: fresh path weights do not sum to zero"


# ---------------------------------------------------------------------------
# 4. Enlarged stencil: post-adaptive indices strictly larger than op
# ---------------------------------------------------------------------------


def test_enlarged_neighborhoods_produces_correct_length_weights():
    """When neighborhoods[i] has strictly more indices than op (simulating
    adaptive δ-enlargement), L_w must have the enlarged length and remain
    M-matrix compliant after QP. This is the property that fails pre-#1102.
    """
    pts, op_nh, is_b = _build_2d_grid_with_neighborhoods()
    op = _MockTaylorOperator(pts, op_nh, delta=1.5)

    # Simulate adaptive enlargement: at each boundary point, take all points
    # within radius 2.5*delta instead of k=8. Strictly enlarges every
    # boundary stencil for this small grid.
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    enlarged_nh: dict[int, dict] = {}
    for i in range(len(pts)):
        idx = np.asarray(tree.query_ball_point(pts[i], r=3.0), dtype=int)
        enlarged_nh[i] = {"indices": idx}

    precomp = PrecomputedMonotoneStencils(
        operator=op,
        is_boundary=is_b,
        neighborhoods=enlarged_nh,
        points=pts,
        delta=1.5,
    )

    for i in np.where(is_b)[0]:
        i = int(i)
        sd = precomp.stencils.get(i)
        assert sd is not None, f"Missing stencil at boundary point {i}"
        n_enlarged = len(enlarged_nh[i]["indices"])
        n_legacy = len(op_nh[i]["indices"])
        # Sanity check on the test fixture: enlargement must actually enlarge.
        assert n_enlarged > n_legacy, (
            f"Test fixture broken at point {i}: r=3 not larger than k=8 ({n_enlarged} <= {n_legacy})"
        )
        # The fix: weights and indices share the enlarged length.
        assert len(sd.weights) == n_enlarged, (
            f"point {i}: L_w length {len(sd.weights)} != enlarged stencil "
            f"length {n_enlarged}. This is the #1102 shape-mismatch invariant."
        )
        assert len(sd.neighbor_indices) == n_enlarged
        # M-matrix invariants still hold on the enlarged stencil.
        if sd.center_in_neighbors is not None:
            off = np.delete(sd.weights, sd.center_in_neighbors)
            assert np.all(off >= -1e-6), f"point {i}: enlarged-stencil weights violate M-matrix off-diagonal"
            assert abs(np.sum(sd.weights)) < 1e-6, f"point {i}: enlarged-stencil weights do not sum to zero"


# ---------------------------------------------------------------------------
# 5. Integration regression — HJBGFDMSolver path
# ---------------------------------------------------------------------------


class _MockProblem:
    def __init__(self, geometry):
        self.geometry = geometry
        self.dimension = 2
        self.Nx = 9
        self.Nt = 5
        self.Dx = 0.1
        self.Dt = 0.2
        self.sigma = 0.1
        self.T = 1.0
        self.lambda_ = 1.0
        self.is_custom = False
        self.hamiltonian_class = None
        self.f_potential = None

    def H(self, x_idx, m_at_x, p_values, t_idx):
        return 0.5 * sum(v**2 for v in p_values.values() if isinstance(v, (int, float)))

    def get_hjb_hamiltonian_jacobian_contrib(self, *a, **kw):
        return None

    def get_hjb_residual_m_coupling_term(self, *a, **kw):
        return None

    def dH_dp(self, *a, **kw):
        return None


def test_solver_constructs_with_adaptive_neighborhoods_and_qp_m_matrix():
    """HJBGFDMSolver(adaptive_neighborhoods=True, monotonicity_scheme="qp_m_matrix")
    must complete construction including the PrecomputedMonotoneStencils
    initialisation, without runtime/precomp size mismatch.

    The crash mode pre-#1102 is at the runtime override site
    (``approximate_derivatives``); the fix here ensures L_w is built on
    the same stencil the runtime sees, removing the shape mismatch by
    construction. This test exercises the construction path; the
    runtime contraction is exercised by the call from ``_make_solver``
    setting up Taylor matrices end-to-end.
    """
    from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
    from mfgarchon.geometry import Hyperrectangle
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    LX, LY = 6.0, 6.0
    # Irregular cloud: small jitter to make adaptive enlargement bite at
    # some corner buffer points without being too pathological.
    rng = np.random.default_rng(0)
    nx, ny = 7, 7
    xs = np.linspace(0.0, LX, nx)
    ys = np.linspace(0.0, LY, ny)
    interior = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if 0 < ix < nx - 1 and 0 < iy < ny - 1:
                interior.append([x + rng.uniform(-0.1, 0.1), y + rng.uniform(-0.1, 0.1)])
    interior = np.asarray(interior)
    eps = 1e-7
    boundary = []
    for x in xs:
        boundary.append([x, eps])
        boundary.append([x, LY - eps])
    for y in ys:
        boundary.append([eps, y])
        boundary.append([LX - eps, y])
    boundary = np.asarray(boundary)
    pts = np.vstack([interior, boundary])
    bdry_idx = np.arange(len(interior), len(pts))

    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(name="right", bc_type=BCType.NO_FLUX, boundary="x_max"),
        ],
        dimension=2,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = HJBGFDMSolver(
            _MockProblem(geom),
            collocation_points=pts,
            boundary_indices=bdry_idx,
            delta=1.5,
            k_neighbors=12,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            collocation_geometry=geom,
            adaptive_neighborhoods=True,
            boundary_conditions=bc,
            monotonicity_scheme="qp_m_matrix",
            monotonicity_application="precompute",
        )

    # Construction succeeded. Now verify the load-bearing invariant: for
    # every precomputed stencil, the stored weights length matches the
    # runtime neighborhood length. Pre-#1102 these diverged at adaptive-
    # enlarged points and the runtime override would raise.
    precomp = s._precomputed_stencils
    assert precomp is not None, "qp_m_matrix + precompute should build stencils"
    for i, sd in precomp.stencils.items():
        runtime_nh = s.neighborhoods[i]["indices"]
        assert len(sd.weights) == len(runtime_nh), (
            f"point {i}: precomp L_w length {len(sd.weights)} != runtime "
            f"neighborhood length {len(runtime_nh)}. #1102 invariant violated."
        )

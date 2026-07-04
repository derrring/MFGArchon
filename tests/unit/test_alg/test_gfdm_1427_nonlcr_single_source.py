"""Issue #1427: non-LCR GFDM differentiation weights are single-sourced through the
adaptive-aware NeighborhoodBuilder, not the operator's pre-adaptive weights.

Two guarantees:
1. **Byte-identity on the default path** (adaptive_neighborhoods=False): routing non-LCR
   points through the builder produces weights bit-for-bit equal to the operator path, so the
   Weak-GFDM paper baseline is unchanged. This is the pinning test that fails if the fork reopens.
2. **Correctness on the adaptive path**: when adaptive enlargement fires for a non-LCR point, the
   built D_lap/D_grad consume the ENLARGED neighborhood (self.neighborhoods[i]); the operator's
   pre-adaptive weights would have silently diverged (the #1427 bug).
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.geometry import Hyperrectangle
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions


class _MockProblem:
    def __init__(self, geometry, dimension=2, sigma=0.3, T=1.0, Nt=10):
        self.geometry = geometry
        self.dimension = dimension
        self.Nx, self.Nt, self.Dx, self.Dt = 9, Nt, 0.1, T / Nt
        self.sigma, self.T, self.lambda_ = sigma, T, 1.0
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


def _bc():
    return BoundaryConditions(
        segments=[
            BCSegment(name=f"s_{ax}_{end}", bc_type=BCType.NO_FLUX, boundary=f"{ax}_{end}")
            for ax in ("x", "y")
            for end in ("min", "max")
        ],
        dimension=2,
    )


def _uniform_cloud(LX=4.0, LY=4.0, nx=5, ny=5, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = np.linspace(0.0, LX, nx), np.linspace(0.0, LY, ny)
    interior = [
        [x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05)]
        for ix, x in enumerate(xs)
        for iy, y in enumerate(ys)
        if 0 < ix < nx - 1 and 0 < iy < ny - 1
    ]
    eps = 1e-7
    boundary = [[x, eps] for x in xs] + [[x, LY - eps] for x in xs]
    boundary += [[eps, y] for y in ys[1:-1]] + [[LX - eps, y] for y in ys[1:-1]]
    pts = np.vstack([np.asarray(interior), np.asarray(boundary)])
    bdry_idx = np.arange(len(interior), len(pts))
    return pts, bdry_idx, Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))


def _build(pts, bdry, geom, *, adaptive, mode="hybrid", delta=1.5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(
            _MockProblem(geom),
            collocation_points=pts,
            boundary_indices=bdry,
            delta=delta,
            k_neighbors=12,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            collocation_geometry=geom,
            adaptive_neighborhoods=adaptive,
            boundary_conditions=_bc(),
            neighborhood_mode=mode,
        )


def _lcr_set(solver):
    if solver._use_local_coordinate_rotation and solver._boundary_handler is not None:
        return set(solver._boundary_handler.boundary_rotations.keys())
    return set()


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_nonlcr_weights_byte_identical_adaptive_off(seed):
    """Pinning test: builder path == operator path bit-for-bit on the paper's default config.

    If this fails, the #1427 fix has drifted the Weak-GFDM baseline — the fork must stay byte-identical
    here (the only behavioural change is on the adaptive_neighborhoods=True path).
    """
    solver = _build(*_uniform_cloud(seed=seed), adaptive=False)
    op, nb, lcr = solver._gfdm_operator, solver._neighborhood_builder, _lcr_set(solver)
    assert nb is not None

    compared = 0
    for i in range(solver.n_points):
        if i in lcr:
            continue
        wb = nb.compute_derivative_weights_from_taylor(i)
        wo = op.get_derivative_weights(i)
        if wb is None and wo is None:
            continue
        assert (wb is None) == (wo is None), f"point {i}: builder/operator disagree on availability"
        compared += 1
        np.testing.assert_array_equal(wb["neighbor_indices"], wo["neighbor_indices"])
        np.testing.assert_array_equal(wb["grad_weights"], wo["grad_weights"])
        np.testing.assert_array_equal(wb["lap_weights"], wo["lap_weights"])
    assert compared > 0, "no non-LCR points compared — test is vacuous"


def _hole_cloud():
    """A cloud with one interior point whose local neighbors are thinned, forcing adaptive
    delta enlargement for that non-LCR point."""
    xs = np.linspace(0.0, 6.0, 7)
    ys = np.linspace(0.0, 6.0, 7)
    interior, sparse = [], np.array([3.0, 3.0])
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if not (0 < ix < 6 and 0 < iy < 6):
                continue
            p = np.array([x, y])
            # thin out the ring immediately around the center so it must reach farther
            if 0 < np.linalg.norm(p - sparse) < 1.6 and not np.allclose(p, sparse):
                continue
            interior.append([x, y])
    eps = 1e-7
    boundary = [[x, eps] for x in xs] + [[x, 6.0 - eps] for x in xs]
    boundary += [[eps, y] for y in ys[1:-1]] + [[6.0 - eps, y] for y in ys[1:-1]]
    pts = np.vstack([np.asarray(interior), np.asarray(boundary)])
    bdry_idx = np.arange(len(interior), len(pts))
    return pts, bdry_idx, Hyperrectangle(np.array([[0.0, 6.0], [0.0, 6.0]]))


def test_adapted_nonlcr_routes_to_enlarged_neighborhood():
    """When adaptive enlargement fires for a non-LCR point, the built weights use the ENLARGED
    neighborhood; the operator's pre-adaptive weights would diverge (the #1427 divergence)."""
    # radius-mode (knn/hybrid take k-nearest by count and never enlarge) with a delta that
    # leaves interior points under-filled, so adaptive delta enlargement genuinely fires.
    solver = _build(*_hole_cloud(), adaptive=True, mode="radius", delta=1.2)
    op, nb, lcr = solver._gfdm_operator, solver._neighborhood_builder, _lcr_set(solver)

    adapted = [i for i in range(solver.n_points) if i not in lcr and nb.neighborhoods[i].get("adapted", False)]
    if not adapted:
        pytest.skip("no non-LCR adaptive enlargement triggered on this cloud")

    checked = 0
    for i in adapted:
        wb = nb.compute_derivative_weights_from_taylor(i)
        if wb is None:
            continue
        # the builder (single source) must use exactly self.neighborhoods[i] — the enlarged set
        np.testing.assert_array_equal(wb["neighbor_indices"], nb.neighborhoods[i]["indices"])
        wo = op.get_derivative_weights(i)
        # the operator's pre-adaptive stencil is a DIFFERENT (smaller/unenlarged) set — this is the
        # divergence #1427 closes; consuming the operator here would drop the enlarged neighbors.
        if wo is not None:
            assert not np.array_equal(np.asarray(wo["neighbor_indices"]), np.asarray(nb.neighborhoods[i]["indices"])), (
                f"point {i}: operator stencil unexpectedly equals the enlarged set — no divergence to fix"
            )
        checked += 1
    assert checked > 0, "adapted points found but none had SVD weights to check"

"""Tests for monotonicity_scheme / monotonicity_application canonical API.

v0.18.0 introduced the two-axis API replacing qp_optimization_level=.
v0.25.0 (Issue #1070) removed qp_optimization_level= from the constructor
signature; passing it now raises TypeError. These tests verify the canonical
API works correctly and that the removed parameter is no longer accepted.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# joint_socp requires cvxpy; skip those tests when the optional dep is absent.
try:
    import cvxpy  # noqa: F401

    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False
_requires_cvxpy = pytest.mark.skipif(not _HAS_CVXPY, reason="cvxpy not installed; joint_socp tests skipped")


def _problem_2d_quasi_uniform():
    """2D problem with N=11x11 quasi-uniform interior + boundary indices.

    Layout: 121 collocation points on [0,1]^2, with 4 corners + edges as boundary.
    Returns (problem, points, boundary_indices).
    """
    bc = no_flux_bc(dimension=2)
    domain = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=bc)
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=H,
    )
    problem = MFGProblem(geometry=domain, T=1.0, Nt=10, sigma=0.5, components=components)
    pts = problem.geometry.get_spatial_grid()
    if pts.ndim == 1:
        pts = np.atleast_2d(pts).T
    bdry = []
    for i, p in enumerate(pts):
        if min(p[0], 1.0 - p[0], p[1], 1.0 - p[1]) < 1e-9:
            bdry.append(i)
    return problem, pts, np.array(bdry)


# ---------------------------------------------------------------------------
# Mapping table (legacy → new) per docstring of HJBGFDMSolver
# ---------------------------------------------------------------------------
LEGACY_TO_NEW = {
    "none": ("none", None),  # application=None → "ignored"
    "auto": ("qp_m_matrix", "adaptive"),
    "always": ("qp_m_matrix", "always"),
    "precompute": ("qp_m_matrix", "precompute"),
}


@pytest.fixture(scope="module")
def setup():
    return _problem_2d_quasi_uniform()


# ---------------------------------------------------------------------------
# Equivalence tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("legacy_value", list(LEGACY_TO_NEW.keys()))
def test_scheme_application_match(setup, legacy_value):
    """For each canonical (scheme, application) pair, verify the solver stores the
    expected self.monotonicity_scheme, self.monotonicity_application, and the
    internal self.qp_optimization_level attribute."""
    problem, pts, bdry = setup
    new_scheme, new_app = LEGACY_TO_NEW[legacy_value]

    s = HJBGFDMSolver(
        problem,
        collocation_points=pts,
        boundary_indices=bdry,
        delta=0.3,
        monotonicity_scheme=new_scheme,
        monotonicity_application=new_app,
    )

    assert s.monotonicity_scheme == new_scheme, f"scheme mismatch for {legacy_value}"
    # Check internal qp_optimization_level attribute (set from scheme, not from the
    # removed parameter — this is an internal implementation detail, not public API).
    if new_scheme == "none":
        assert s.qp_optimization_level == "none"
    elif new_scheme == "qp_m_matrix" and new_app == "adaptive":
        assert s.qp_optimization_level == "auto"
    elif new_scheme == "qp_m_matrix":
        assert s.qp_optimization_level == new_app
    assert s.hjb_method_name is not None, "hjb_method_name should be set"


@pytest.mark.parametrize("legacy_value", list(LEGACY_TO_NEW.keys()))
def test_stencil_weights_canonical(setup, legacy_value):
    """Verify that canonical (scheme, application) constructs a solver with valid
    stencil weights (behavioral guard for the new API path)."""
    problem, pts, bdry = setup
    new_scheme, new_app = LEGACY_TO_NEW[legacy_value]

    s = HJBGFDMSolver(
        problem,
        collocation_points=pts,
        boundary_indices=bdry,
        delta=0.3,
        monotonicity_scheme=new_scheme,
        monotonicity_application=new_app,
    )

    op = s._gfdm_operator
    n_pts = pts.shape[0]
    for i in range(n_pts):
        w = op.get_derivative_weights(i)
        if w is None:
            continue
        assert "neighbor_indices" in w
        assert "lap_weights" in w
        assert "grad_weights" in w


def test_mutual_exclusion(setup):
    """v0.25.0: qp_optimization_level= no longer in the signature; passing it (even
    alongside monotonicity_scheme=) raises TypeError (unexpected kwarg)."""
    problem, pts, bdry = setup
    with pytest.raises(TypeError, match="qp_optimization_level"):
        HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=0.3,
            monotonicity_scheme="qp_m_matrix",
            qp_optimization_level="auto",
        )


def test_invalid_scheme_value(setup):
    """Passing a legacy bundle value to monotonicity_scheme= raises ValueError."""
    problem, pts, bdry = setup
    with pytest.raises(ValueError, match="monotonicity_scheme must be one of"):
        HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=0.3,
            monotonicity_scheme="auto",  # Wrong axis — "auto" is application, not scheme
        )


def test_invalid_application_value(setup):
    """Passing an unrecognized application value raises ValueError."""
    problem, pts, bdry = setup
    with pytest.raises(ValueError, match="monotonicity_application must be one of"):
        HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=0.3,
            monotonicity_scheme="qp_m_matrix",
            monotonicity_application="bogus",
        )


@_requires_cvxpy
def test_default_application_per_scheme(setup):
    """When monotonicity_application=None, scheme-recommended default is used."""
    problem, pts, bdry = setup
    # qp_m_matrix → adaptive
    s = HJBGFDMSolver(
        problem, collocation_points=pts, boundary_indices=bdry, delta=0.3, monotonicity_scheme="qp_m_matrix"
    )
    assert s.monotonicity_application == "adaptive"
    # joint_socp → precompute (will warn since not yet implemented)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = HJBGFDMSolver(
            problem, collocation_points=pts, boundary_indices=bdry, delta=0.3, monotonicity_scheme="joint_socp"
        )
    assert s.monotonicity_application == "precompute"


def test_removed_qp_optimization_level_raises_type_error(setup):
    """v0.25.0 removal (Issue #1070): qp_optimization_level= is no longer in the
    __init__ signature; passing it raises TypeError, not DeprecationWarning."""
    problem, pts, bdry = setup
    with pytest.raises(TypeError, match="qp_optimization_level"):
        HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=0.3,
            qp_optimization_level="auto",
        )


@_requires_cvxpy
def test_joint_socp_precompute_active(setup):
    """joint_socp with precompute application: PrecomputedJointSocpStencils
    is built at construction; reports feasibility stats."""
    problem, pts, bdry = setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=0.3,
            monotonicity_scheme="joint_socp",
            adaptive_neighborhoods=True,
        )
    # New attribute exists with feasibility stats
    assert s._joint_socp_stencils is not None, "Expected _joint_socp_stencils to be initialized for joint_socp scheme"
    stats = s._joint_socp_stencils.stats
    # On a quasi-uniform 11x11 grid, all interior nodes should be SOCP-feasible
    # (paper Theorem `thm:joint_socp_feasibility`)
    assert stats["n_feasible"] == stats["n_interior"], (
        f"Expected all {stats['n_interior']} interior nodes feasible, got {stats['n_feasible']}"
    )
    # Application defaults to precompute for joint_socp
    assert s.monotonicity_application == "precompute"
    # Internally: legacy qp_optimization_level aliased to "precompute" — joint_socp
    # IS a precompute application, and this routes the HJB Newton through the
    # per-point Hamiltonian path, matching the legacy precompute_socp_weights +
    # patch_operator workflow numerically.
    assert s.qp_optimization_level == "precompute"


@_requires_cvxpy
def test_joint_socp_weights_satisfy_constraints(setup):
    """Joint SOCP weights at every feasible stencil satisfy:
    (a) 2nd-order Taylor consistency (A^T L = e_lap, A^T D = e_grad)
    (b) M-matrix on -Δ_h (L_off ≥ 0)
    (c) Per-edge cone (||D_j||_2 ≤ C h_i L_j)
    """
    from mfgarchon.alg.numerical.gfdm_components.joint_socp import build_taylor_matrix_2d

    problem, pts, bdry = setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=0.3,
            monotonicity_scheme="joint_socp",
            adaptive_neighborhoods=True,
        )

    socp = s._joint_socp_stencils
    for i in s._joint_socp_stencils._interior_indices[:20]:  # sample 20 stencils
        i = int(i)
        if not socp.has_stencil(i):
            continue
        sd = socp.stencils[i]
        offsets = pts[sd.neighbor_indices] - pts[i]
        A, _ = build_taylor_matrix_2d(offsets)

        # 2nd-order consistency
        e_lap = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(A.T @ sd.L, e_lap, atol=1e-7, err_msg=f"L consistency at i={i}")
        e_dx = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        e_dy = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(A.T @ sd.D[0], e_dx, atol=1e-7, err_msg=f"D[0] consistency at i={i}")
        np.testing.assert_allclose(A.T @ sd.D[1], e_dy, atol=1e-7, err_msg=f"D[1] consistency at i={i}")

        # M-matrix
        L_off = np.delete(sd.L, sd.center_in_neighbors)
        assert np.all(L_off >= -1e-9), f"L_off must be ≥ 0 at i={i}: min(L_off)={L_off.min()}"

        # Per-edge cone. C-bisection may have raised C beyond the solver default
        # for marginally infeasible stencils, so the constraint to check at
        # stencil i is the C *actually achieved* there, not the solver-level
        # default. `achieved_C[i]` records the per-stencil bound used.
        C_i = socp.achieved_C.get(i, socp._C)
        h_i = float(np.median(np.linalg.norm(offsets[offsets.any(axis=1)], axis=1)))
        for j in range(len(sd.neighbor_indices)):
            if j == sd.center_in_neighbors:
                continue
            if sd.L[j] <= 1e-12:
                continue
            kappa = h_i * np.linalg.norm(sd.D[:, j]) / sd.L[j]
            assert kappa <= C_i + 1e-7, f"Cone violated at i={i}, j={j}: kappa={kappa:.4e} > C_i={C_i}"


@_requires_cvxpy
def test_joint_socp_unsupported_application_warns(setup):
    """joint_socp with adaptive/always application warns (precompute is the
    only supported strategy in v0.18.0)."""
    problem, pts, bdry = setup
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=0.3,
            monotonicity_scheme="joint_socp",
            monotonicity_application="adaptive",
            adaptive_neighborhoods=True,
        )
    fb = [w for w in caught if "joint_socp" in str(w.message) and "precompute" in str(w.message)]
    assert len(fb) >= 1, "Expected fallback warning when joint_socp + non-precompute application is requested"

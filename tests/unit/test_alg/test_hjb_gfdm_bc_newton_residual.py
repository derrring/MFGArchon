"""Regression tests for HJB GFDM BC Newton-residual semantics (Issue #1116).

The pre-#1116 BC dispatcher set `residual_bc[i]` at boundary rows to the BC
target value (e.g., `g_D` for Dirichlet, `g_N` for Neumann) rather than the
Newton residual `F_bc(u_current) - target`. Jacobian rows were correctly
assembled as `∂F_bc/∂u`, so `(J_bc, r_bc)` was structurally inconsistent:
J·d ≠ 0 while ∂r_bc/∂u ≡ 0 (boundary FD identically zero across all
directions).

This module asserts the post-fix invariant: a directional finite-difference
of the residual matches the analytic Jacobian at machine precision, on both
interior AND boundary rows, for problems where the pre-fix code visibly broke
(mixed BC + non-trivial initial state).

See docs/bug_reports/2026_05_hjb_bc_newton_mismatch.md for full evidence and
the diagnostic protocol this test codifies.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


def _make_separable_lq_hamiltonian():
    """Standard LQ Hamiltonian: H(p, m) = |p|²/2 + m, dH/dp = p."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: np.asarray(m),
        coupling_dm=lambda m: np.ones_like(np.asarray(m)),
    )


def _mixed_bc_walls_and_exit(LX: float, LY: float):
    """Mixed BC: no-flux walls except a Dirichlet exit window on x_max.

    Same topology as Stage C v3 (the configuration that surfaced #1116).
    """
    return BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(
                name="right_below", bc_type=BCType.NO_FLUX,
                boundary="x_max", region={"y": (0.0, 0.4 * LY)},
            ),
            BCSegment(
                name="right_above", bc_type=BCType.NO_FLUX,
                boundary="x_max", region={"y": (0.6 * LY, LY)},
            ),
            BCSegment(
                name="exit", bc_type=BCType.DIRICHLET, value=0.0,
                boundary="x_max", region={"y": (0.4 * LY, 0.6 * LY)},
                priority=1,
            ),
        ],
        dimension=2,
    )


def _build_mixed_bc_solver(Nx: int = 9, LX: float = 1.0, LY: float = 1.0):
    """Construct an HJBGFDMSolver on a structured 2D grid with mixed BC."""
    geometry = TensorProductGrid(
        bounds=[(0.0, LX), (0.0, LY)],
        Nx=[Nx, Nx],
        boundary_conditions=_mixed_bc_walls_and_exit(LX, LY),
    )

    def m_initial(x):
        x_arr = np.asarray(x)
        return float(np.exp(-10.0 * np.sum((x_arr - 0.5) ** 2)))

    components = MFGComponents(
        m_initial=m_initial,
        u_terminal=lambda x: 0.0,
        hamiltonian=_make_separable_lq_hamiltonian(),
    )

    problem = MFGProblem(geometry=geometry, T=1.0, Nt=4, components=components)

    collocation_points = geometry.get_spatial_grid()
    solver = HJBGFDMSolver(
        problem,
        collocation_points=collocation_points,
        delta=2.0 * (LX / (Nx - 1)),
        k_neighbors=12,
        derivative_method="taylor",
        taylor_order=2,
        weight_function="wendland",
    )
    return solver, problem


def _residual_and_jacobian_with_bc(solver, u_current, u_n_plus_1, m_n_plus_1, time_idx):
    """Compute (J_bc, r_bc) for a given u_current via the same path Newton uses.

    Uses the per-point cache path (`_compute_hjb_residual_with_cache` +
    `_compute_hjb_jacobian_sparse`). This is the path active when SOCP is on,
    and the path under which the residual-semantics bug fired in Stage C.
    """
    all_derivs = solver._approximate_all_derivatives_cached(u_current)
    residual = solver._compute_hjb_residual_with_cache(
        u_current, u_n_plus_1, m_n_plus_1, time_idx, all_derivs,
    )
    jacobian_sparse = solver._compute_hjb_jacobian_sparse(
        u_current, m_n_plus_1, time_idx, all_derivs,
    )
    jacobian_bc, residual_bc = solver._apply_boundary_conditions_to_sparse_system(
        jacobian_sparse, residual, time_idx, u_current,
    )
    return jacobian_bc, residual_bc


# ---------------------------------------------------------------------------
# Gate A — FD Jacobian directional check, mixed BC, non-trivial u
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fd_jacobian_matches_analytic_on_all_rows(eps, seed):
    """∂r/∂u via FD must equal analytic J at machine precision on every row.

    Pre-fix: boundary rows had ∂r/∂u ≡ 0 (residual was a constant in u), so
    |FD| was identically zero while |J·d| was order 1. This test would have
    failed catastrophically: boundary `rel_diff_total ≈ 1`.

    Post-fix: boundary rows have ∂r/∂u = w (the normal-derivative stencil or
    e_i for Dirichlet), matching the Jacobian to first-order in ε.
    """
    solver, problem = _build_mixed_bc_solver(Nx=9)
    n = solver.n_points

    # Non-trivial state that does NOT satisfy any BC trivially. This is the
    # crux: u_init ≠ 0 means Dirichlet violation; ∇u ≠ 0 along walls means
    # Neumann violation. Pre-fix code preserved both violations indefinitely.
    points = solver.collocation_points
    rng = np.random.default_rng(seed)
    u_current = (
        0.3 * points[:, 0] * points[:, 1]
        + 0.1 * np.sin(2 * np.pi * points[:, 0])
        + 0.05 * rng.standard_normal(n)
    )
    u_n_plus_1 = u_current + 0.01 * rng.standard_normal(n)
    m_n_plus_1 = 1.0 + 0.1 * rng.standard_normal(n)
    time_idx = problem.Nt - 1

    # Reference (J, r) at u_current
    J_bc, r_bc = _residual_and_jacobian_with_bc(
        solver, u_current, u_n_plus_1, m_n_plus_1, time_idx,
    )

    # Direction: random unit vector
    d = rng.standard_normal(n)
    d /= float(np.linalg.norm(d))

    # FD: (r(u + ε d) - r(u)) / ε, evaluated through the SAME BC path
    u_pert = u_current + eps * d
    _, r_bc_pert = _residual_and_jacobian_with_bc(
        solver, u_pert, u_n_plus_1, m_n_plus_1, time_idx,
    )
    Jd_fd = (r_bc_pert - r_bc) / eps
    Jd_analytic = J_bc.dot(d)

    # Split by row class
    bd_idx = np.asarray(solver.boundary_indices, dtype=np.int64)
    int_mask = np.ones(n, dtype=bool)
    if bd_idx.size > 0:
        int_mask[bd_idx] = False

    int_rel = (
        float(np.linalg.norm(Jd_fd[int_mask] - Jd_analytic[int_mask]))
        / max(float(np.linalg.norm(Jd_analytic[int_mask])), 1e-30)
    )
    bd_rel = (
        float(np.linalg.norm(Jd_fd[~int_mask] - Jd_analytic[~int_mask]))
        / max(float(np.linalg.norm(Jd_analytic[~int_mask])), 1e-30)
    )
    full_rel = (
        float(np.linalg.norm(Jd_fd - Jd_analytic))
        / max(float(np.linalg.norm(Jd_analytic)), 1e-30)
    )

    # ε=1e-6 gives O(ε * |H_uu|) truncation; with |u|, |∇u| ~ O(1) and quadratic
    # Hamiltonian, this stays ≤ 1e-4. ε=1e-5 stays ≤ 1e-3.
    tol = 1e-4 if eps <= 1e-6 else 1e-3
    assert int_rel < tol, (
        f"Interior FD rel-diff {int_rel:.3e} > {tol:.0e} (eps={eps}, seed={seed}). "
        f"Indicates the analytic Jacobian on interior rows is wrong."
    )
    assert bd_rel < tol, (
        f"Boundary FD rel-diff {bd_rel:.3e} > {tol:.0e} (eps={eps}, seed={seed}). "
        f"Pre-#1116 this would be ≈ 1 because r_bc was the BC target (constant in u). "
        f"|FD on boundary|={np.linalg.norm(Jd_fd[~int_mask]):.3e}, "
        f"|J·d on boundary|={np.linalg.norm(Jd_analytic[~int_mask]):.3e}."
    )
    assert full_rel < tol, f"Full FD rel-diff {full_rel:.3e} > {tol:.0e}"


# ---------------------------------------------------------------------------
# Gate B — 1D Dirichlet bc=0 + u_init=0: bit-exact Newton step
# ---------------------------------------------------------------------------


def test_1d_dirichlet_bc_zero_zero_init_residual_is_zero_at_boundary():
    """For Dirichlet g_D=0 starting from u[bd]=0, Newton residual at BC is 0.

    Pre-fix: residual_bc[bd] = bc_target = 0 (incidentally correct).
    Post-fix: residual_bc[bd] = u_current[bd] - bc_target = 0 - 0 = 0 (correct).

    Bit-exact equivalence for this specific case is the basis for the claim in
    the bug report that 1D LQ Dirichlet=0 EOC studies do not need re-running.
    """
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx=[21],
        boundary_conditions=BoundaryConditions(
            segments=[
                BCSegment(name="left", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min"),
                BCSegment(name="right", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max"),
            ],
            dimension=1,
        ),
    )

    def m_init_1d(x):
        return float(np.exp(-10.0 * (np.asarray(x) - 0.5) ** 2))

    components = MFGComponents(
        m_initial=m_init_1d,
        u_terminal=lambda x: 0.0,
        hamiltonian=_make_separable_lq_hamiltonian(),
    )
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=4, components=components)
    collocation_points = geometry.get_spatial_grid()
    solver = HJBGFDMSolver(
        problem,
        collocation_points=collocation_points,
        delta=0.2,
        k_neighbors=5,
        derivative_method="taylor",
        taylor_order=2,
        weight_function="wendland",
    )

    n = solver.n_points
    # u_init with exact zeros at the two Dirichlet endpoints
    u_current = np.zeros(n)
    u_current[1:-1] = 0.3 * np.sin(np.pi * collocation_points[1:-1, 0])

    u_n_plus_1 = u_current.copy()
    m_n_plus_1 = np.ones(n)

    _, r_bc = _residual_and_jacobian_with_bc(
        solver, u_current, u_n_plus_1, m_n_plus_1, time_idx=problem.Nt - 1,
    )

    for i in solver.boundary_indices:
        assert abs(float(r_bc[int(i)])) < 1e-15, (
            f"Dirichlet residual at i={i} (u={u_current[int(i)]}, target=0) "
            f"should be exactly 0 — got {r_bc[int(i)]}. This breaks the "
            f"bit-exact-equivalence claim for 1D LQ regression."
        )


def test_1d_dirichlet_bc_nonzero_uses_violation():
    """For Dirichlet g_D=u_target, Newton residual at BC encodes u-u_target.

    Asserts the fix went in correctly: residual is current minus target, not
    just target.
    """
    target_value = 2.5
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx=[21],
        boundary_conditions=BoundaryConditions(
            segments=[
                BCSegment(name="left", bc_type=BCType.DIRICHLET, value=target_value, boundary="x_min"),
                BCSegment(name="right", bc_type=BCType.DIRICHLET, value=target_value, boundary="x_max"),
            ],
            dimension=1,
        ),
    )

    def m_init_1d(x):
        return float(np.exp(-10.0 * (np.asarray(x) - 0.5) ** 2))

    components = MFGComponents(
        m_initial=m_init_1d,
        u_terminal=lambda x: 0.0,
        hamiltonian=_make_separable_lq_hamiltonian(),
    )
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=4, components=components)
    collocation_points = geometry.get_spatial_grid()
    solver = HJBGFDMSolver(
        problem,
        collocation_points=collocation_points,
        delta=0.2,
        k_neighbors=5,
        derivative_method="taylor",
        taylor_order=2,
        weight_function="wendland",
    )

    n = solver.n_points
    # Initial u: random offset from the target value at BC points
    u_current = np.linspace(0.0, 1.0, n) + 0.5  # u_bd ≠ target
    u_n_plus_1 = u_current.copy()
    m_n_plus_1 = np.ones(n)

    _, r_bc = _residual_and_jacobian_with_bc(
        solver, u_current, u_n_plus_1, m_n_plus_1, time_idx=problem.Nt - 1,
    )

    for i in solver.boundary_indices:
        i_int = int(i)
        expected = float(u_current[i_int]) - target_value
        actual = float(r_bc[i_int])
        assert abs(actual - expected) < 1e-14, (
            f"Dirichlet BC residual at i={i_int} should be u-target = "
            f"{u_current[i_int]} - {target_value} = {expected}, got {actual}. "
            f"Pre-#1116 this returned bc_target ({target_value}) instead."
        )


# ---------------------------------------------------------------------------
# Gate C — Neumann bc=0 with non-zero w·u_init: BC violation surfaces in residual
# ---------------------------------------------------------------------------


def test_neumann_bc_residual_encodes_current_violation():
    """For Neumann/NO_FLUX, residual is w·u_current - bc_target.

    Pre-fix this was just bc_target (0 for no-flux), so Newton never saw the
    BC violation. Post-fix the current violation is what the Newton step is
    designed to drive to zero.
    """
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx=[21],
        boundary_conditions=BoundaryConditions(
            segments=[
                BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
                BCSegment(name="right", bc_type=BCType.NO_FLUX, boundary="x_max"),
            ],
            dimension=1,
        ),
    )

    def m_init_1d(x):
        return float(np.exp(-10.0 * (np.asarray(x) - 0.5) ** 2))

    components = MFGComponents(
        m_initial=m_init_1d,
        u_terminal=lambda x: 0.0,
        hamiltonian=_make_separable_lq_hamiltonian(),
    )
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=4, components=components)
    collocation_points = geometry.get_spatial_grid()
    solver = HJBGFDMSolver(
        problem,
        collocation_points=collocation_points,
        delta=0.2,
        k_neighbors=5,
        derivative_method="taylor",
        taylor_order=2,
        weight_function="wendland",
    )

    n = solver.n_points
    # u with deliberately non-zero normal derivative at both boundaries: a
    # linear ramp. Old code would see residual=0 here; new code reports the
    # actual ∂_n u violation.
    u_current = collocation_points[:, 0].copy()
    u_n_plus_1 = u_current.copy()
    m_n_plus_1 = np.ones(n)

    J_bc, r_bc = _residual_and_jacobian_with_bc(
        solver, u_current, u_n_plus_1, m_n_plus_1, time_idx=problem.Nt - 1,
    )

    boundary_indices = list(solver.boundary_indices)
    assert len(boundary_indices) > 0
    # Linear u has |w·u_current| O(1); residual must reflect that, not be 0.
    boundary_residual_norm = float(np.linalg.norm(
        np.array([r_bc[int(i)] for i in boundary_indices])
    ))
    assert boundary_residual_norm > 0.1, (
        f"Neumann BC residual on linear u (w·u ≠ 0) must be non-zero, "
        f"got |r_bc[bd]|={boundary_residual_norm:.3e}. Pre-#1116 this was "
        f"identically 0 because residual_bc was hardcoded to bc_target=0."
    )

    # And it must equal the analytic violation w·u via the Jacobian row.
    for i in boundary_indices:
        i_int = int(i)
        row = J_bc.getrow(i_int).toarray().flatten()
        analytic_violation = float(row @ u_current)
        assert abs(float(r_bc[i_int]) - analytic_violation) < 1e-12, (
            f"Neumann residual at i={i_int} should equal w·u_current = "
            f"{analytic_violation}, got {float(r_bc[i_int])}."
        )

"""Issue #1074: per-stencil M-matrix verification for joint_socp.

The paper's `thm:discrete_comparison` claim hinges on each SOCP-feasible
stencil having M-matrix property:

1. Sum-zero on Laplacian:        ``sum(L) == 0``
2. Off-diagonal non-negative:    ``L[off] >= 0``  (so center is non-positive)
3. Per-edge cone bound:          ``||D[:, j]|| <= (C/h_i) * L[j]``  for each j

These are enforced by `solve_joint_socp_at_stencil` constraints (see
`gfdm_components/joint_socp.py:174`). This test verifies they hold empirically
for representative configurations. If the test fails, the paper claim has an
empirical counter-example.

The full assembled-matrix M-matrix property (Issue #1074 long-term ask) is now
also covered (tests at the bottom). Per-stencil SOCP feasibility does NOT imply
the assembled HJB iteration matrix ``I/dt - D*L + alpha*D_grad`` is an M-matrix:
the signed drift term flips an off-diagonal positive once
``|alpha| > alpha_crit = D * min_edge(L_ij / ||D_grad_ij||)`` (a Peclet-like
condition; ``critical_drift_for_dmp``). Note this off-diagonal *sign* condition is
**dt-independent** — a smaller dt restores diagonal dominance but not the sign. So
the discrete maximum principle holds only in the diffusion-dominated regime
``|alpha| <= alpha_crit``; the runtime ``check_dmp`` guard warns when a solve
exceeds it.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# joint_socp requires cvxpy
try:
    import cvxpy  # noqa: F401

    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


pytestmark = pytest.mark.skipif(not _HAS_CVXPY, reason="cvxpy not installed; joint_socp tests skipped")


def _make_solver(sigma: float, n_x: int = 21):
    """Construct an HJBGFDMSolver with joint_socp + precompute on a 1D grid."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem

    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[n_x],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: np.ones_like(np.asarray(m)),
        ),
    )
    problem = MFGProblem(geometry=geometry, T=0.5, Nt=10, sigma=sigma, components=components)
    bounds = problem.geometry.get_bounds()
    x_coords = np.linspace(bounds[0][0], bounds[1][0], n_x)
    collocation_points = x_coords.reshape(-1, 1)
    return HJBGFDMSolver(
        problem,
        collocation_points,
        delta=0.15,
        monotonicity_scheme="joint_socp",
        monotonicity_application="precompute",
    )


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
def test_socp_stencil_laplacian_consistency(sigma):
    """Issue #1074: sum(L) == 0 for every SOCP-feasible stencil (Laplacian
    consistency / sum-zero condition)."""
    solver = _make_solver(sigma)
    stencils = solver._joint_socp_stencils
    assert stencils is not None, "joint_socp stencils not precomputed"
    assert len(stencils.stencils) > 0, "no SOCP-feasible stencils produced"

    for idx, stencil in stencils.stencils.items():
        L = stencil.L
        assert np.isclose(L.sum(), 0.0, atol=1e-9), (
            f"point {idx}: sum(L) = {L.sum():.3e}, expected 0 (Laplacian consistency)"
        )


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
def test_socp_stencil_off_diagonal_nonnegative(sigma):
    """Issue #1074: L[j] >= 0 for off-diagonal j (M-matrix property part 1)."""
    solver = _make_solver(sigma)
    stencils = solver._joint_socp_stencils
    assert stencils is not None

    for idx, stencil in stencils.stencils.items():
        L = stencil.L
        center = stencil.center_in_neighbors
        L_off = np.delete(L, center)
        # Allow tiny SOCP slop (eps_pos default = 0)
        assert np.all(L_off >= -1e-9), (
            f"point {idx}: min(L_off) = {L_off.min():.3e}, expected >= 0 (M-matrix off-diagonal sign)"
        )


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
def test_socp_stencil_center_nonpositive(sigma):
    """Issue #1074: L[center] <= 0 (M-matrix property part 2, follows from
    sum-zero + off-diagonal non-negative)."""
    solver = _make_solver(sigma)
    stencils = solver._joint_socp_stencils
    assert stencils is not None

    for idx, stencil in stencils.stencils.items():
        L = stencil.L
        center = stencil.center_in_neighbors
        L_center = L[center]
        assert L_center <= 1e-9, f"point {idx}: L_center = {L_center:.3e}, expected <= 0"


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
def test_socp_stencil_per_edge_cone_bound(sigma):
    """Issue #1074: per-edge cone ||D[:, j]|| <= (C/h_i) * L[j] for each off-j.

    This is the core constraint linking gradient stencils to Laplacian stencils
    that closes the discrete comparison principle proof.
    """
    solver = _make_solver(sigma)
    stencils = solver._joint_socp_stencils
    assert stencils is not None

    C = stencils._C  # cone constant used during construction
    points = stencils._points

    n_violations = 0
    max_violation_ratio = 0.0
    for idx, stencil in stencils.stencils.items():
        L = stencil.L
        D = stencil.D  # shape (dimension, n_neighbors)
        # Recompute h_i (median nonzero neighbor distance, matches construction)
        nbr_idx = stencil.neighbor_indices
        offsets = points[nbr_idx] - points[idx]
        dists = np.linalg.norm(offsets, axis=-1)
        nz = dists[dists > 1e-12]
        h_i = float(np.median(nz)) if len(nz) > 0 else stencils._delta
        bound_per_j = (C / h_i) * L  # shape (n,)

        # Per-edge norm of D[:, j]
        D_norms = np.linalg.norm(D, axis=0)  # shape (n,)

        # Off-diagonal indices
        center = stencil.center_in_neighbors
        off_mask = np.ones(len(L), dtype=bool)
        off_mask[center] = False

        # For off-edges, the cone constraint applies
        for j in np.where(off_mask)[0]:
            if L[j] < 1e-12:
                # If L[j] ≈ 0, the bound becomes ≈ 0 — D[:, j] must also be ≈ 0
                if D_norms[j] > 1e-9:
                    n_violations += 1
                    max_violation_ratio = max(max_violation_ratio, D_norms[j] / 1e-9)
            else:
                ratio = D_norms[j] / bound_per_j[j]
                if ratio > 1.0 + 1e-6:  # tolerate tiny SOCP slop
                    n_violations += 1
                    max_violation_ratio = max(max_violation_ratio, ratio)

    assert n_violations == 0, (
        f"sigma={sigma}: {n_violations} per-edge cone violations, "
        f"max ||D[:,j]|| / ((C/h)*L[j]) = {max_violation_ratio:.4f} (expected <= 1)"
    )


def test_socp_at_least_some_feasible():
    """Issue #1074 sanity: at σ=1 (low-Pe regime), most stencils should be
    SOCP-feasible. If 0/many feasible, joint_socp is producing nothing useful."""
    solver = _make_solver(sigma=1.0, n_x=21)
    stencils = solver._joint_socp_stencils
    assert stencils is not None
    n_feasible = stencils.stats["n_feasible"]
    n_infeasible = stencils.stats["n_infeasible"]
    total = n_feasible + n_infeasible
    assert total > 0, "no stencils attempted"
    # At low Pe with σ=1, expect majority feasibility. Stage 1/2 paper
    # baseline reports >85% at N>=75.
    assert n_feasible > 0, (
        f"sigma=1 low-Pe: 0/{total} stencils SOCP-feasible — joint_socp scheme producing no usable stencils"
    )


# ---------------------------------------------------------------------------
# Issue #1074 — ASSEMBLED-matrix M-matrix property (the long-term ask).
# The per-stencil checks above do NOT imply the assembled HJB iteration matrix
# (I/dt - D·L + α·D_grad) is an M-matrix. These characterize when it is.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.3, 0.5, 1.0])
def test_assembled_m_matrix_holds_at_zero_drift(sigma):
    """Issue #1074: at zero drift the assembled interior matrix (I/dt - D·L) IS an M-matrix.

    This is the positive half — the SOCP-monotone Laplacian gives the diffusion-part discrete
    maximum principle. (The 2 boundary rows carry BC structure applied separately and are
    excluded via interior_indices.)"""
    from mfgarchon.alg.numerical.gfdm_components.monotonicity_enforcer import verify_assembled_m_matrix

    solver = _make_solver(sigma=sigma, n_x=21)
    grad_u = np.zeros((solver.n_points, 1))
    J = solver._compute_hjb_jacobian_vectorized(grad_u)
    report = verify_assembled_m_matrix(J, interior_indices=solver.interior_indices)
    assert report["is_m_matrix"], f"σ={sigma}: assembled interior matrix not M-matrix at zero drift: {report}"


@pytest.mark.parametrize("sigma", [0.3, 0.5, 1.0])
def test_assembled_m_matrix_breaks_under_strong_drift(sigma):
    """Issue #1074: per-stencil SOCP feasibility does NOT imply the assembled M-matrix.

    A large enough drift flips an off-diagonal positive via the α·D_grad term (the empirical
    counter-example to an *unconditional* DMP claim). Pinning this catches a future change that
    silently claims unconditional DMP. The threshold is predicted by critical_drift_for_dmp."""
    from mfgarchon.alg.numerical.gfdm_components.monotonicity_enforcer import (
        critical_drift_for_dmp,
        verify_assembled_m_matrix,
    )
    from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

    solver = _make_solver(sigma=sigma, n_x=21)
    solver._compute_hjb_jacobian_vectorized(np.zeros((solver.n_points, 1)))  # build operators
    diffusion_coeff = diffusion_from_volatility(sigma)
    alpha_crit = critical_drift_for_dmp(
        solver._D_lap, solver._D_grad, diffusion_coeff, interior_indices=solver.interior_indices
    )
    assert alpha_crit > 0.0  # SOCP cone bound keeps the threshold strictly positive

    # λ=1 ⇒ |α| = |grad|. Drift well past α_crit must break the assembled M-matrix.
    strong = np.full((solver.n_points, 1), 5.0 * alpha_crit)
    J = solver._compute_hjb_jacobian_vectorized(strong)
    report = verify_assembled_m_matrix(J, interior_indices=solver.interior_indices)
    assert not report["is_m_matrix"], f"σ={sigma}: expected M-matrix violation at |drift|=5·α_crit, got {report}"
    assert report["n_positive_offdiag"] > 0


@pytest.mark.parametrize("sigma", [0.3, 0.5, 1.0, 1.5])
def test_vectorized_jacobian_byte_identical_to_inline_lq_formula(sigma):
    """Issue #1071: ``_compute_hjb_jacobian_vectorized`` now routes ∂H/∂p through the
    single-source assembler (``assemble_hjb_jacobian_diag`` → ``H_class.evaluate_dp``)
    instead of the inline ``p/λ`` re-derivation.

    Pin: the assembled CSR is byte-identical (atol=0, dtype-exact) to the explicit inline
    LQ Jacobian ``(1/dt)I + Σ_d diag(p_d/λ) @ D_grad[d] - (σ²/2)·D_lap`` it replaced. The
    M-matrix property asserts above are necessary but insufficient — they pass for any
    structurally-similar matrix; this locks the λ / diffusion convention against a silent
    assembler drift (sign flip, stray m-term, wrong diffusion coefficient)."""
    from scipy.sparse import diags, eye

    from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

    solver = _make_solver(sigma=sigma, n_x=21)
    rng = np.random.default_rng(1071)
    grad_u = rng.standard_normal((solver.n_points, 1))

    J = solver._compute_hjb_jacobian_vectorized(grad_u)  # builds operators lazily

    # Independent reference: the explicit inline LQ Jacobian this consolidation replaced.
    n = solver.n_points
    dt = solver.problem.T / solver.problem.Nt
    lam = solver._control_cost_lambda()
    D = diffusion_from_volatility(solver._get_sigma_value(None))
    ref = (1.0 / dt) * eye(n, format="csr")
    for d in range(solver.dimension):
        ref = ref + diags(grad_u[:, d] / lam, format="csr") @ solver._D_grad[d]
    ref = ref - D * solver._D_lap

    Jc, rc = J.tocsr(), ref.tocsr()
    Jc.sort_indices()
    rc.sort_indices()
    assert Jc.shape == rc.shape
    assert np.array_equal(Jc.indptr, rc.indptr), "CSR indptr drift"
    assert np.array_equal(Jc.indices, rc.indices), "CSR indices drift"
    maxabsdiff = float(np.max(np.abs((Jc - rc).data))) if (Jc - rc).nnz else 0.0
    assert np.array_equal(Jc.data, rc.data), (
        f"σ={sigma}: assembled Jacobian not byte-identical to inline LQ formula (maxabsdiff={maxabsdiff:.3e})"
    )


def test_runtime_dmp_guard_warns_only_when_violated():
    """Issue #1074 runtime guard: check_dmp=True warns when the drift exceeds α_crit; it is
    silent below the threshold and a no-op when check_dmp=False (the default — numerically inert)."""
    import logging

    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    gfdm_logger = logging.getLogger("mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm")
    gfdm_logger.addHandler(handler)
    try:
        solver = _make_solver_check_dmp(sigma=0.3, check_dmp=True)
        solver._compute_hjb_jacobian_vectorized(np.zeros((solver.n_points, 1)))  # build operators + α_crit
        x = np.linspace(0.0, 1.0, solver.n_points)

        records.clear()
        solver._maybe_warn_dmp(1e-4 * x)  # |drift| ≪ α_crit
        assert not any("DMP" in m for m in records), "guard warned in the diffusion-dominated regime"

        records.clear()
        solver._dmp_warned = False
        solver._maybe_warn_dmp(100.0 * x)  # |drift| ≫ α_crit
        assert any("Issue #1074" in m for m in records), "guard did not warn under strong drift"

        off = _make_solver_check_dmp(sigma=0.3, check_dmp=False)
        off._compute_hjb_jacobian_vectorized(np.zeros((off.n_points, 1)))
        records.clear()
        off._maybe_warn_dmp(100.0 * x)
        assert not any("DMP" in m for m in records), "guard fired with check_dmp=False (should be inert)"
    finally:
        gfdm_logger.removeHandler(handler)


def _make_solver_check_dmp(sigma: float, check_dmp: bool, n_x: int = 21):
    """`_make_solver` variant exposing the check_dmp flag (Issue #1074 runtime guard)."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem

    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n_x], boundary_conditions=no_flux_bc(dimension=1))
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: np.ones_like(np.asarray(m)),
        ),
    )
    problem = MFGProblem(geometry=geometry, T=0.5, Nt=10, sigma=sigma, components=components)
    x_coords = np.linspace(0.0, 1.0, n_x).reshape(-1, 1)
    return HJBGFDMSolver(
        problem,
        x_coords,
        delta=0.15,
        monotonicity_scheme="joint_socp",
        monotonicity_application="precompute",
        check_dmp=check_dmp,
    )


# ---------------------------------------------------------------------------
# Issue #1253 — DMP guard dead on joint_socp solve path; alpha_crit over-estimated
# ---------------------------------------------------------------------------


def test_dmp_guard_fires_on_real_solve_path():
    """Issue #1253 2026-06-10: _maybe_warn_dmp must fire on the normal Newton solve path.

    Before the fix: _D_grad is None on the joint_socp/precompute per-point path because
    _compute_hjb_jacobian_sparse never calls _build_differentiation_matrices; the guard
    silently returned. After the fix: _build_differentiation_matrices() is called lazily
    inside _maybe_warn_dmp so the guard actually runs.

    Pinning condition: a joint_socp solve with check_dmp=True in advection-dominated
    1D regime (small sigma, steep u_terminal) must emit the DMP warning via
    solve_hjb_system(), WITHOUT any manual pre-build of _D_grad.
    """
    import logging

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    gfdm_logger = logging.getLogger("mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm")
    gfdm_logger.addHandler(handler)
    try:
        # Small sigma → small alpha_crit (D = sigma^2/2 ≈ 0.005).
        # Steep u_terminal → large grad_u → max_alpha ≫ alpha_crit from first Newton step.
        n_x = 21
        solver = _make_solver_check_dmp(sigma=0.1, check_dmp=True, n_x=n_x)

        # Verify operators have NOT been pre-built (this is the bug condition: guard
        # would silently return without the fix).
        assert solver._D_grad is None, "_D_grad must be None before solve to test the real path"

        n_pts = solver.n_points
        x_vals = solver.collocation_points[:, 0]
        # Steep linear terminal: gradient ≈ 10, far exceeds alpha_crit ≈ D/h ≈ 0.005/0.05 ≈ 0.1
        u_terminal = 10.0 * x_vals

        m_density = np.ones((solver.problem.Nt + 1, n_pts))  # uniform density, no coupling effect

        # Run through the REAL solve path. _solve_timestep → per-point branch
        # → _maybe_warn_dmp is called with _D_grad=None (pre-fix: silent return).
        solver.solve_hjb_system(
            M_density=m_density,
            U_terminal=u_terminal,
        )

        assert any("DMP" in m or "Issue #1074" in m for m in records), (
            "DMP warning must fire when drift > alpha_crit on the real solve path "
            "(Issue #1253: guard was silently skipped when _D_grad=None)"
        )
    finally:
        gfdm_logger.removeHandler(handler)


def test_critical_drift_includes_negative_lap_edges():
    """Issue #1253 2026-06-10: critical_drift_for_dmp must include edges where L_ij <= 0
    but gradient weight is nonzero (relaxed-SOCP slack rows).

    When L_ij < 0 and D_grad_ij ≠ 0, the assembled off-diagonal D*L_ij term is
    negative → the entry is positive at zero drift → alpha_crit must be ≤ 0.
    The old code iterated only l_row > atol and returned a positive finite threshold,
    producing a false-negative DMP warning.
    """
    from scipy.sparse import csr_matrix

    from mfgarchon.alg.numerical.gfdm_components.monotonicity_enforcer import critical_drift_for_dmp

    # Hand-crafted 3-point system: interior point i=1, neighbors j=0, j=2.
    # L[1,0]=−0.1 (negative, relaxed-SOCP slack), L[1,2]=0.5, L[1,1] adjusted for row sum=0.
    # D_grad[1,0]=1.0, D_grad[1,2]=1.0 (nonzero gradient on both edges).
    n = 3
    L_data = np.zeros((n, n))
    L_data[1, 0] = -0.1  # negative off-diagonal — relaxed SOCP slack
    L_data[1, 2] = 0.5
    L_data[1, 1] = -L_data[1, 0] - L_data[1, 2]  # row sum = 0

    G_data = np.zeros((n, n))
    G_data[1, 0] = 1.0
    G_data[1, 2] = 1.0
    G_data[1, 1] = -G_data[1, 0] - G_data[1, 2]

    d_lap = csr_matrix(L_data)
    d_grad = [csr_matrix(G_data)]
    diffusion_coeff = 1.0
    interior = np.array([1])

    alpha_crit = critical_drift_for_dmp(d_lap, d_grad, diffusion_coeff, interior_indices=interior)

    # Before fix: iterates only L_ij > atol → j=2 only → alpha_crit = 1.0*0.5/1.0 = 0.5
    # After fix: includes j=0 (L_ij=-0.1, g_norm=1.0) → alpha_crit = 1.0*(-0.1)/1.0 = -0.1
    assert alpha_crit <= 0.0, (
        f"alpha_crit={alpha_crit:.4f}: negative-L edge (L=-0.1, g=1.0) must drive alpha_crit <= 0; "
        "old code returned +0.5, masking a pre-existing M-matrix violation (Issue #1253)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

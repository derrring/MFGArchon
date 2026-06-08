"""Direct unit tests for HJBHowardSolver.

HJBHowardSolver graduates the research-side howard_patch.py family
(exp08 1D/2D, exp09, exp11) into mfgarchon proper. Replaces the Newton
inner loop of HJBGFDMSolver._solve_timestep when the Hamiltonian is
strictly convex in p. Resolves Issue #1118 (Newton stalls on |∇u|²
stiffness).

Suite covers:

1. Construction validation: requires SOCP-precomputed stencil_provider.
2. Discretisation enum validation.
3. 1D LQ closed-form: backward sweep reproduces the analytical Riccati
   profile for pure LQ (single-agent, no MFG coupling, no potential).
4. Newton-stall reproducer (Issue #1118): pure LQ regime where Newton
   inner bottoms out at Armijo MIN_ALPHA — Howard converges.
5. Each discretisation option runs to completion (upwind_projection,
   upwind_per_axis, central).
6. 2D smoke test on irregular cloud.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

pytest.importorskip("cvxpy")

from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_howard import HJBHowardSolver
from mfgarchon.geometry import Hyperrectangle
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

# ---------------------------------------------------------------------------
# Minimal mock problem (mirrors test_joint_socp_mirror_symmetry pattern)
# ---------------------------------------------------------------------------


class _MockProblem:
    def __init__(self, geometry, sigma=0.3, T=1.0, Nt=10, dimension=2):
        self.geometry = geometry
        self.dimension = dimension
        self.Nx = 9
        self.Nt = Nt
        self.Dx = 0.1
        self.Dt = T / Nt
        self.sigma = sigma
        self.T = T
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


def _make_2d_cloud(LX=4.0, LY=4.0, nx=5, ny=5, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, LX, nx)
    ys = np.linspace(0.0, LY, ny)
    interior = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if 0 < ix < nx - 1 and 0 < iy < ny - 1:
                interior.append([x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05)])
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
    return pts, bdry_idx, geom


def _make_1d_cloud(LX=2.0, n_int=11):
    interior = np.linspace(0.2, LX - 0.2, n_int).reshape(-1, 1)
    boundary = np.array([[1e-7], [LX - 1e-7]])
    pts = np.vstack([interior, boundary])
    bdry_idx = np.arange(len(interior), len(pts))
    geom = Hyperrectangle(np.array([[0.0, LX]]))
    return pts, bdry_idx, geom


def _make_gfdm_solver(pts, bdry, geom, problem, scheme="joint_socp", k_neighbors=12, inner_solver="newton"):
    bc = BoundaryConditions(
        segments=[
            BCSegment(name=f"side_{d}_{end}", bc_type=BCType.NO_FLUX, boundary=f"{ax}_{end}")
            for d in range(problem.dimension)
            for ax in (["x", "y", "z"][d],)
            for end in ("min", "max")
        ],
        dimension=problem.dimension,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=1.5,
            k_neighbors=k_neighbors,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            collocation_geometry=geom,
            adaptive_neighborhoods=False,
            boundary_conditions=bc,
            monotonicity_scheme=scheme,
            monotonicity_application="precompute",
            inner_solver=inner_solver,
        )


class _LQHam:
    """Minimal LQ Hamiltonian H = |p|²/2 exposing dp(x, m, p, t) = p (so α* = -dp = -p).

    Used to validate the integrated `inner_solver='howard'` path, which derives α* from
    `problem.hamiltonian_class.dp` (Issue #1118). Matches the explicit `lambda x,p,m,t: -p`
    the standalone Howard tests pass.
    """

    def dp(self, x, m, p, t=0.0):
        return np.asarray(p, dtype=float)


# ---------------------------------------------------------------------------
# 1. Construction validation
# ---------------------------------------------------------------------------


def test_construction_requires_joint_socp_stencils():
    """stencil_provider without _joint_socp_stencils raises."""
    _pts, _bdry, geom = _make_2d_cloud()
    problem = _MockProblem(geom)

    class _StubProvider:
        _joint_socp_stencils = None

    with pytest.raises(RuntimeError, match="_joint_socp_stencils"):
        HJBHowardSolver(
            problem,
            stencil_provider=_StubProvider(),
            alpha_star=lambda x, p, m, t: -p,
        )


def test_construction_rejects_unknown_discretisation():
    pts, bdry, geom = _make_2d_cloud()
    problem = _MockProblem(geom)
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem)
    with pytest.raises(ValueError, match="discretisation must be one of"):
        HJBHowardSolver(
            problem,
            stencil_provider=gfdm,
            alpha_star=lambda x, p, m, t: -p,
            discretisation="not_a_real_scheme",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 2. 1D LQ closed-form (Riccati profile)
# ---------------------------------------------------------------------------


def test_1d_lq_closed_form_riccati():
    """Pure LQ in 1D, mfgarchon convention.

    HJB residual form is `-u_t + H - (σ²/2)Δu = 0` (NAMING_CONVENTIONS.md
    § HJB Equation Conventions). With u(T, x) = 0.5(x - x_c)² (so terminal
    coefficient P(T) = 0.5), σ = 0, the Riccati ODE reads P'(t) = 2 P².
    Closed form for T = 1: P(t) = 1 / (4 - 2t), so P(0) = 0.25.

    Analytic ratio at off-centre probe ``|x - x_c| = 1``:
        u(0)/u(T) = P(0)/P(T) = 0.25 / 0.5 = 0.5

    See NAMING_CONVENTIONS.md § HJB Equation Conventions § Worked example
    for the full derivation. Probe MUST be off-centre (at centre, the
    quadratic vanishes and the test is uninformative for any P(t)).

    Tolerance is loose (~30%) because:
    - SOCP-corrected D_grad on boundary-buffer interior nodes is
      LSQ-fitted, not exact FD.
    - n_int=11 is a coarse 1D grid.
    - No-flux Neumann-by-extension BC introduces a small wall artifact.

    The load-bearing check is Riccati ordering + coefficient ballpark,
    not high-precision quadrature.
    """
    LX = 4.0
    pts, bdry, geom = _make_1d_cloud(LX=LX, n_int=11)
    problem = _MockProblem(geom, sigma=0.0, T=1.0, Nt=20, dimension=1)
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem, k_neighbors=5)

    # Terminal: U(T, x) = 0.5 * x² → G_s = 0.5
    x_pts = pts[:, 0]
    x_c = LX / 2
    U_T = 0.5 * (x_pts - x_c) ** 2

    # Use central D_grad: validates linear-algebra under canonical convention,
    # not the upwind-builder accuracy. Default upwind_projection is first-order
    # LSQ with bias `(1/2)·Σh_j³/Σh_j² · u''`. On this coarse 1D grid
    # (n_int=11, h≈0.36), that bias ≈ 0.6·u'' is comparable to |u'|=1.08,
    # so upwind loses fidelity. On 2D irregular clouds with k=12 averaging
    # over multiple directions + σ>0 diffusion (the regime Howard graduated
    # from per Issue #1118), the bias mitigates and upwind works — that
    # regime is covered by test_each_discretisation_completes and the 2D
    # smoke test below. See `hjb_howard.py` module docstring for the
    # convergence hypothesis (Bokanowski-Maroso-Zidani 2009).
    howard = HJBHowardSolver(
        problem,
        stencil_provider=gfdm,
        alpha_star=lambda x, p, m, t: -p,  # H = |p|²/2 → α* = -p
        discretisation="central",
        max_iter=30,
        tol=1e-6,
        volatility_field=0.0,
    )
    U = howard.solve_hjb_system(M_density=None, U_terminal=U_T)

    # Check Riccati at an OFF-center probe. The quadratic vanishes at x_c,
    # so probing the center would always give 0 regardless of P(t). Pick a
    # point with non-zero (x-x_c)² so the Riccati coefficient is observable.
    assert np.all(np.isfinite(U)), "Howard produced non-finite U"
    # Probe at x ≈ x_c - 1.0 (interior, off-center): (x-x_c)² = 1.0
    probe_offset = 1.0
    probe_idx = int(np.argmin(np.abs(x_pts - (x_c - probe_offset))))
    U_T_probe = float(U_T[probe_idx])
    U_0_probe = float(U[0, probe_idx])
    assert U_T_probe > 0.1, f"Test fixture broken: probe at x={x_pts[probe_idx]:.3f} sees U_T={U_T_probe:.4f}"
    # Analytical ratio P(0)/P(T) = 0.25 / 0.5 = 0.5 under mfgarchon's HJB
    # convention `-u_t + H - σ²Δu/2 = 0`. Probe at |x-x_c|=1, T=1, G_s=0.5.
    # See NAMING_CONVENTIONS.md § HJB Equation Conventions § Worked example.
    # Loose bound 0.3-0.85: coarse 1D grid + Neumann-by-extension wall artifact.
    ratio = U_0_probe / U_T_probe
    assert 0.3 < ratio < 0.85, (
        f"Riccati ordering broken: U(0)/U(T) at probe x={x_pts[probe_idx]:.3f} = {ratio:.3f}, "
        f"expected ~0.5 for mfgarchon LQ convention (P(0)/P(T) = 0.25/0.5). "
        f"See NAMING_CONVENTIONS.md § HJB Equation Conventions."
    )


# ---------------------------------------------------------------------------
# 3. Newton-stall reproducer (Issue #1118)
# ---------------------------------------------------------------------------


def test_howard_advances_where_newton_would_stall():
    """The temporal-plateau symptom of Issue #1118 manifests as U(t, x) ≈
    U(T, x) for all t after stall. Howard must produce U that varies
    monotonically backward in time.

    Pure LQ regime, single backward sweep.
    """
    LX = 4.0
    pts, bdry, geom = _make_1d_cloud(LX=LX, n_int=15)
    problem = _MockProblem(geom, sigma=0.0, T=1.0, Nt=10, dimension=1)
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem, k_neighbors=5)

    x_pts = pts[:, 0]
    U_T = (x_pts - LX / 2) ** 2  # quadratic terminal cost

    # Use central discretisation for 1D smooth-LQ validation (see the
    # Riccati test docstring for why upwind builders bias on coarse 1D
    # grids; the upwind regime is exercised by the 2D smoke test).
    howard = HJBHowardSolver(
        problem,
        stencil_provider=gfdm,
        alpha_star=lambda x, p, m, t: -p,
        discretisation="central",
        volatility_field=0.0,
    )
    U = howard.solve_hjb_system(M_density=None, U_terminal=U_T)

    # Temporal monotonicity at an OFF-center probe: U[Nt] > U[Nt-1] > ... > U[0].
    # The quadratic vanishes at x_c so center-probe would be trivially zero
    # and fail to distinguish plateau from convergence.
    probe_offset = 1.0
    probe_idx = int(np.argmin(np.abs(x_pts - (LX / 2 - probe_offset))))
    profile = np.array([float(U[nt, probe_idx]) for nt in range(problem.Nt + 1)])

    # Strictly decreasing backward (allowing tiny rounding).
    diffs = np.diff(profile)
    assert np.all(diffs >= -1e-6), (
        f"Temporal-plateau or non-monotone profile detected. diffs = {diffs}; profile = {profile}"
    )
    # And total cost-to-go shrinkage is non-trivial (NOT a plateau).
    assert profile[-1] - profile[0] > 0.1 * profile[-1], (
        f"Temporal plateau: U(0) ≈ U(T) ({profile[0]:.4f} vs {profile[-1]:.4f}). "
        f"This is the Issue #1118 symptom Howard must fix."
    )


# ---------------------------------------------------------------------------
# 4. Each discretisation option runs to completion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("discretisation", ["upwind_projection", "upwind_per_axis", "central"])
def test_each_discretisation_completes(discretisation):
    """All three A_adv assembly options produce finite U on a small 2D cloud.

    `central` is included for comparison only and may not converge on
    advection-dominant regimes, but should run without crashing.
    """
    pts, bdry, geom = _make_2d_cloud(nx=5, ny=5)
    problem = _MockProblem(geom, sigma=0.3, T=1.0, Nt=5, dimension=2)
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem)

    x_c = np.array([2.0, 2.0])
    U_T = 0.5 * np.sum((pts - x_c[None, :]) ** 2, axis=1)

    howard = HJBHowardSolver(
        problem,
        stencil_provider=gfdm,
        alpha_star=lambda x, p, m, t: -p,
        discretisation=discretisation,
        max_iter=15,
    )
    U = howard.solve_hjb_system(M_density=None, U_terminal=U_T)
    assert np.all(np.isfinite(U)), f"discretisation={discretisation} produced NaN/Inf"
    assert U.shape == (problem.Nt + 1, len(pts))
    # Terminal preserved bit-for-bit.
    assert np.allclose(U[problem.Nt], U_T)


# ---------------------------------------------------------------------------
# 5. 2D smoke + running_cost callable
# ---------------------------------------------------------------------------


def test_2d_smoke_with_running_cost_callable():
    """Running cost callable correctly enters the RHS. Use a constant
    running cost: U is shifted by `T · const` relative to the no-cost case.
    Just check that supplying running_cost produces a different result.
    """
    pts, bdry, geom = _make_2d_cloud(nx=4, ny=4)
    problem = _MockProblem(geom, sigma=0.2, T=1.0, Nt=5, dimension=2)
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem)

    x_c = np.array([2.0, 2.0])
    U_T = 0.5 * np.sum((pts - x_c[None, :]) ** 2, axis=1)

    base = HJBHowardSolver(
        problem,
        stencil_provider=gfdm,
        alpha_star=lambda x, p, m, t: -p,
    ).solve_hjb_system(M_density=None, U_terminal=U_T)

    n = len(pts)
    rc_const = 0.5
    with_rc = HJBHowardSolver(
        problem,
        stencil_provider=gfdm,
        alpha_star=lambda x, p, m, t: -p,
        running_cost=lambda t_idx: rc_const * np.ones(n),
    ).solve_hjb_system(M_density=None, U_terminal=U_T)

    # Adding a positive running cost shifts U(t<T, x) upward.
    interior_idx = np.array([i for i in range(n) if np.linalg.norm(pts[i] - x_c) < 0.7])
    if len(interior_idx) > 0:
        diff_at_t0 = float(np.mean(with_rc[0, interior_idx]) - np.mean(base[0, interior_idx]))
        assert diff_at_t0 > 0, f"Constant running cost did not increase U(0) at center; diff={diff_at_t0:.4f}"


# ---------------------------------------------------------------------------
# 7. Integrated path: HJBGFDMSolver(inner_solver='howard') (Issue #1118 PR1)
# ---------------------------------------------------------------------------


def test_integrated_howard_inner_solver_lq_1d():
    """inner_solver='howard' wired into HJBGFDMSolver.solve_hjb_system reproduces the 1D LQ
    Riccati profile — proving the integrated path derives alpha* = -hamiltonian_class.dp,
    delegates the backward sweep to Howard, and round-trips the collocation output. Same
    fixture and loose bound as test_1d_lq_closed_form_riccati (the standalone solver)."""
    LX = 4.0
    pts, bdry, geom = _make_1d_cloud(LX=LX, n_int=11)
    problem = _MockProblem(geom, sigma=0.0, T=1.0, Nt=20, dimension=1)
    problem.hamiltonian_class = _LQHam()  # H = |p|²/2 → dp = p → α* = -p
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem, k_neighbors=5, inner_solver="howard")

    x_pts = pts[:, 0]
    x_c = LX / 2
    U_T = 0.5 * (x_pts - x_c) ** 2

    U = gfdm.solve_hjb_system(M_density=None, U_terminal=U_T)  # delegates to Howard internally

    assert np.all(np.isfinite(U)), "integrated Howard path produced non-finite U"
    probe_idx = int(np.argmin(np.abs(x_pts - (x_c - 1.0))))
    U_T_probe = float(U_T[probe_idx])
    assert U_T_probe > 0.1, "fixture broken: probe is at the quadratic's vertex"
    ratio = float(U[0, probe_idx]) / U_T_probe
    assert 0.3 < ratio < 0.85, (
        f"integrated inner_solver='howard' Riccati ratio U(0)/U(T)={ratio:.3f}, expected ~0.5 "
        f"(matches the standalone test_1d_lq_closed_form_riccati)."
    )


def test_inner_solver_rejects_unknown_value():
    """An unknown inner_solver value fails fast at construction."""
    pts, bdry, geom = _make_1d_cloud()
    problem = _MockProblem(geom, dimension=1)
    with pytest.raises(ValueError, match="inner_solver must be"):
        _make_gfdm_solver(pts, bdry, geom, problem, k_neighbors=5, inner_solver="bogus")


def test_integrated_howard_requires_joint_socp_stencils():
    """inner_solver='howard' on a non-SOCP scheme raises at solve time (no _joint_socp_stencils)."""
    pts, bdry, geom = _make_1d_cloud()
    problem = _MockProblem(geom, sigma=0.0, T=1.0, Nt=5, dimension=1)
    problem.hamiltonian_class = _LQHam()
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem, scheme="none", k_neighbors=5, inner_solver="howard")
    U_T = 0.5 * (pts[:, 0] - 2.0) ** 2
    with pytest.raises(ValueError, match="SOCP-precomputed"):
        gfdm.solve_hjb_system(M_density=None, U_terminal=U_T)


def test_integrated_howard_requires_hamiltonian_class():
    """inner_solver='howard' without problem.hamiltonian_class raises (cannot derive α*)."""
    pts, bdry, geom = _make_1d_cloud()
    problem = _MockProblem(geom, sigma=0.0, T=1.0, Nt=5, dimension=1)  # hamiltonian_class=None
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem, k_neighbors=5, inner_solver="howard")
    U_T = 0.5 * (pts[:, 0] - 2.0) ** 2
    with pytest.raises(ValueError, match="hamiltonian_class"):
        gfdm.solve_hjb_system(M_density=None, U_terminal=U_T)


def _make_howard_gfdm_with_bc(bc, sigma=0.3, Nt=5):
    """Construct an inner_solver='howard' HJBGFDMSolver on the 1D cloud with the given BC."""
    pts, bdry, geom = _make_1d_cloud()
    problem = _MockProblem(geom, sigma=sigma, T=0.5, Nt=Nt, dimension=1)
    problem.hamiltonian_class = _LQHam()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gfdm = HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=1.5,
            k_neighbors=5,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            collocation_geometry=geom,
            adaptive_neighborhoods=False,
            boundary_conditions=bc,
            monotonicity_scheme="joint_socp",
            monotonicity_application="precompute",
            inner_solver="howard",
        )
    return gfdm, pts, bdry


def test_integrated_howard_dirichlet_value():
    """Issue #1118 PR2a: inner_solver='howard' now honors the prescribed Dirichlet VALUE via
    the shared value-form BC rows, not the old hardcoded b=0. Uses a NONZERO g_D and a uniform
    Dirichlet BC (the previously-silently-misclassified case): the backward steps must pin the
    boundary to g_D, not 0 and not the terminal value."""
    g_D = 2.0
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="x_min", bc_type=BCType.DIRICHLET, value=g_D, boundary="x_min"),
            BCSegment(name="x_max", bc_type=BCType.DIRICHLET, value=g_D, boundary="x_max"),
        ],
        dimension=1,
    )
    gfdm, pts, bdry = _make_howard_gfdm_with_bc(bc)
    U_T = 0.5 * (pts[:, 0] - 2.0) ** 2  # terminal != g_D at the boundary
    U = gfdm.solve_hjb_system(M_density=None, U_terminal=U_T)
    assert np.all(np.isfinite(U))
    assert np.allclose(U[0, bdry], g_D, atol=1e-6), (
        f"Dirichlet boundary not honored: U[0, boundary]={U[0, bdry]} != g_D={g_D} "
        f"(would be ~0 with the old hardcoded b=0)."
    )


def test_integrated_howard_rejects_robin_nonzero_alpha():
    """ROBIN with alpha != 0 has no normal-derivative row representation (it would drop the
    alpha*u term); it must keep raising even after #1118 PR2b enabled ROBIN(alpha=0). The
    Howard guard now admits 'robin', so the fail-loud comes from _bc_row_for_point during
    the solve, with a message naming alpha."""
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="x_min", bc_type=BCType.ROBIN, alpha=1.0, beta=1.0, value=0.0, boundary="x_min"),
            BCSegment(name="x_max", bc_type=BCType.ROBIN, alpha=1.0, beta=1.0, value=0.0, boundary="x_max"),
        ],
        dimension=1,
    )
    gfdm, pts, _ = _make_howard_gfdm_with_bc(bc)
    U_T = 0.5 * (pts[:, 0] - 2.0) ** 2
    with pytest.raises(NotImplementedError, match=r"alpha"):
        gfdm.solve_hjb_system(M_density=None, U_terminal=U_T)


# ---------------------------------------------------------------------------
# 8. Part 1: per-solve BC refresh (Issue #1118)
#
# GFDM snapshots boundary conditions + the preclassified segment map at
# __init__; the coupling layer resolves providers per Picard iteration by
# swapping geometry.boundary_conditions (FDM re-reads it each solve, GFDM did
# not). Without the refresh the adjoint coupling value would freeze at the
# construction-time BC — silent-wrong in the >1000x-impact regime.
# ---------------------------------------------------------------------------


def _neumann_per_face_1d(value):
    """Mixed (per-face) Neumann BC so the value flows through _bc_segment_per_point."""
    return BoundaryConditions(
        segments=[
            BCSegment(name="x_min", bc_type=BCType.NEUMANN, value=value, boundary="x_min"),
            BCSegment(name="x_max", bc_type=BCType.NEUMANN, value=value, boundary="x_max"),
        ],
        dimension=1,
    )


def _make_geom_sourced_gfdm_1d(bc, inner_solver="howard"):
    """Construct a 1D GFDM solver that reads BC from geometry (no explicit param).

    Returns (gfdm, geom, pts, bdry). Mutating geom.boundary_conditions then calling
    a solve (or _refresh_boundary_conditions_if_changed) mimics the coupling layer's
    per-iteration using_resolved_bc swap.
    """
    pts, bdry, geom = _make_1d_cloud()
    geom.boundary_conditions = bc
    problem = _MockProblem(geom, dimension=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gfdm = HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=1.5,
            k_neighbors=8,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            collocation_geometry=geom,
            adaptive_neighborhoods=False,
            monotonicity_scheme="joint_socp",
            monotonicity_application="precompute",
            inner_solver=inner_solver,
        )
    return gfdm, geom, pts, bdry


def test_gfdm_refreshes_geometry_bc_per_solve():
    """Geometry-sourced BC: a value swapped after construction must reach the value-form
    BC rows (Howard path). This FAILS on the pre-#1118 snapshot (frozen at construction)."""
    gfdm, geom, _pts, bdry = _make_geom_sourced_gfdm_1d(_neumann_per_face_1d(0.3))
    assert gfdm._bc_from_geometry is True
    bi = int(bdry[0])
    assert gfdm._bc_segment_per_point[bi].value == pytest.approx(0.3)
    assert gfdm._value_form_bc_rows(0)[bi][1] == pytest.approx(0.3)

    # Coupling layer swaps the geometry BC (resolved-per-Picard analogue).
    geom.boundary_conditions = _neumann_per_face_1d(0.9)
    gfdm._refresh_boundary_conditions_if_changed()

    assert gfdm.boundary_conditions is geom.boundary_conditions
    assert gfdm._bc_segment_per_point[bi].value == pytest.approx(0.9)
    assert gfdm._value_form_bc_rows(0)[bi][1] == pytest.approx(0.9), (
        "value-form BC target did not track the resolved geometry BC — stale snapshot."
    )


def test_gfdm_explicit_param_bc_not_refreshed():
    """An explicit boundary_conditions= argument is the caller's static choice and must
    NOT be overridden by the geometry, even if the geometry BC changes."""
    pts, bdry, geom = _make_1d_cloud()
    geom.boundary_conditions = _neumann_per_face_1d(0.3)
    problem = _MockProblem(geom, dimension=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gfdm = HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=1.5,
            k_neighbors=8,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            collocation_geometry=geom,
            adaptive_neighborhoods=False,
            monotonicity_scheme="joint_socp",
            monotonicity_application="precompute",
            inner_solver="howard",
            boundary_conditions=_neumann_per_face_1d(0.5),
        )
    assert gfdm._bc_from_geometry is False
    bi = int(bdry[0])
    assert gfdm._bc_segment_per_point[bi].value == pytest.approx(0.5)

    geom.boundary_conditions = _neumann_per_face_1d(0.9)
    gfdm._refresh_boundary_conditions_if_changed()

    assert gfdm._bc_segment_per_point[bi].value == pytest.approx(0.5), (
        "explicit-param BC was wrongly overridden by the geometry on refresh."
    )


def test_gfdm_refresh_noop_when_bc_unchanged():
    """No provider / no swap: refresh leaves the snapshot object identity intact (no churn)."""
    gfdm, _geom, _pts, _bdry = _make_geom_sourced_gfdm_1d(_neumann_per_face_1d(0.3))
    snapshot = gfdm.boundary_conditions
    gfdm._refresh_boundary_conditions_if_changed()
    assert gfdm.boundary_conditions is snapshot


# ---------------------------------------------------------------------------
# 9. ROBIN / adjoint-consistent BC for inner_solver='howard' (Issue #1118 PR2b)
#
# The adjoint-consistent BC is ROBIN(alpha=0, beta=1) whose resolved scalar is
# g = -sigma^2/2 * d ln(m)/dn. PR2b routes it through _build_neumann_bc_row
# (n.grad u = g), lifts the Howard guard for "robin", and fail-louds on the
# unsupported-but-reachable forms (alpha != 0, beta != 1, unresolved provider).
# Part 1's per-solve refresh transports the per-Picard resolved float in.
# ---------------------------------------------------------------------------

_AC_SIGMA = 0.3
_AC_DX = 0.1
_AC_REG = 1e-10


class _ProviderGeom:
    """Minimal geometry carrying only the grid spacing the provider state needs.

    Decouples the provider's density-array spacing (_AC_DX) from the GFDM cloud
    spacing: the provider indexes the density array, not the collocation cloud.
    """

    dimension = 1

    def get_grid_spacing(self):
        return [_AC_DX]


def _adjoint_robin_bc(sigma=_AC_SIGMA):
    from mfgarchon.geometry.boundary.providers import AdjointConsistentProvider

    return BoundaryConditions(
        segments=[
            BCSegment(
                name="left_ac",
                bc_type=BCType.ROBIN,
                alpha=0.0,
                beta=1.0,
                value=AdjointConsistentProvider(side="left", diffusion=sigma),
                boundary="x_min",
            ),
            BCSegment(
                name="right_ac",
                bc_type=BCType.ROBIN,
                alpha=0.0,
                beta=1.0,
                value=AdjointConsistentProvider(side="right", diffusion=sigma),
                boundary="x_max",
            ),
        ],
        dimension=1,
    )


def _expected_g(m, sigma=_AC_SIGMA):
    """Closed-form adjoint value g = -sigma^2/2 * d ln(m)/dn at each face (outward normal)."""
    ln = np.log(m + _AC_REG)
    g_left = -(sigma**2) / 2.0 * (-(ln[1] - ln[0]) / _AC_DX)
    g_right = -(sigma**2) / 2.0 * ((ln[-1] - ln[-2]) / _AC_DX)
    return g_left, g_right


def test_howard_robin_row_target_tracks_resolved_adjoint_g():
    """LOAD-BEARING: two DIFFERENT FP densities -> resolve the AdjointConsistentProvider ->
    Part-1 refresh -> read the value-form BC target from _value_form_bc_rows. The target MUST
    move between the two densities AND equal the hand-computed closed form each time. FAILS if
    g were frozen (refresh/row-builder stale): a frozen g gives g_iter2 == g_iter1, which the
    `!=` assertion rejects. A happy-path 'it ran' test would pass even with a frozen g."""
    prov_geom = _ProviderGeom()
    bc = _adjoint_robin_bc()
    gfdm, geom, _pts, bdry = _make_geom_sourced_gfdm_1d(bc, inner_solver="howard")
    assert gfdm._bc_from_geometry is True
    bi_left, bi_right = int(bdry[0]), int(bdry[1])  # lower 1e-7 -> x_min, upper -> x_max

    def resolve_refresh_and_read(m):
        resolved = bc.with_resolved_providers({"m_current": m, "geometry": prov_geom, "diffusion": _AC_SIGMA})
        assert resolved is not bc  # new object -> Part-1 refresh fires
        assert resolved.segments[0].bc_type is BCType.ROBIN  # stays ROBIN, value -> float
        geom.boundary_conditions = resolved
        gfdm._refresh_boundary_conditions_if_changed()
        assert gfdm.boundary_conditions is geom.boundary_conditions
        rows = gfdm._value_form_bc_rows(0)
        return rows[bi_left][1], rows[bi_right][1]

    m1 = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25])
    m2 = m1[::-1].copy()  # reversed slope -> g flips on both faces

    g1_left, g1_right = resolve_refresh_and_read(m1)
    e1_left, e1_right = _expected_g(m1)
    assert g1_left == pytest.approx(e1_left, abs=1e-9), (g1_left, e1_left)
    assert g1_right == pytest.approx(e1_right, abs=1e-9), (g1_right, e1_right)

    g2_left, g2_right = resolve_refresh_and_read(m2)
    e2_left, e2_right = _expected_g(m2)
    assert g2_left == pytest.approx(e2_left, abs=1e-9), (g2_left, e2_left)
    assert g2_right == pytest.approx(e2_right, abs=1e-9), (g2_right, e2_right)

    assert g2_left != g1_left, "g_left frozen across iterations (refresh/row-builder stale)"
    assert g2_right != g1_right, "g_right frozen across iterations (refresh/row-builder stale)"


def test_howard_robin_nonzero_beta_rejected():
    """ROBIN(alpha=0, beta!=1) needs a 1/beta scaling the delegated Neumann row does not apply;
    it is reachable via the BCSegment API, so it must fail loud, not be silently mis-solved."""
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="x_min", bc_type=BCType.ROBIN, alpha=0.0, beta=2.0, value=0.0, boundary="x_min"),
            BCSegment(name="x_max", bc_type=BCType.ROBIN, alpha=0.0, beta=2.0, value=0.0, boundary="x_max"),
        ],
        dimension=1,
    )
    gfdm, pts, _ = _make_howard_gfdm_with_bc(bc)
    U_T = 0.5 * (pts[:, 0] - 2.0) ** 2
    with pytest.raises(NotImplementedError, match=r"beta"):
        gfdm.solve_hjb_system(M_density=None, U_terminal=U_T)


def test_howard_unresolved_provider_fails_loud():
    """A raw (unresolved) AdjointConsistentProvider reaching Howard means the coupling layer
    failed to resolve it; the converted guard must fail loud (AssertionError), not silently
    solve against a meaningless provider object."""
    from mfgarchon.geometry.boundary.providers import AdjointConsistentProvider

    bc = BoundaryConditions(
        segments=[
            BCSegment(
                name="left_ac",
                bc_type=BCType.ROBIN,
                alpha=0.0,
                beta=1.0,
                value=AdjointConsistentProvider(side="left", diffusion=0.3),
                boundary="x_min",
            ),
            BCSegment(
                name="right_ac",
                bc_type=BCType.ROBIN,
                alpha=0.0,
                beta=1.0,
                value=AdjointConsistentProvider(side="right", diffusion=0.3),
                boundary="x_max",
            ),
        ],
        dimension=1,
    )
    gfdm, pts, _ = _make_howard_gfdm_with_bc(bc)  # explicit BC, providers NOT resolved
    U_T = 0.5 * (pts[:, 0] - 2.0) ** 2
    with pytest.raises(AssertionError, match=r"unresolved BCValueProvider"):
        gfdm.solve_hjb_system(M_density=None, U_terminal=U_T)


def test_howard_honors_resolved_robin_in_solution():
    """Howard must HONOR the resolved Robin row in the solved value function, not just build it:
    the value-form boundary equation row @ U == g holds at the solved (non-terminal) slices.
    Complements the load-bearing test (which checks the row is BUILT correctly). Verifies the
    value-form RHS=target is consumed by the inner solve (Constraint c) without depending on the
    Newton path (whose stall on this regime is the very motivation for Howard, #1118)."""
    from mfgarchon.geometry.boundary.providers import AdjointConsistentProvider

    sigma = _AC_SIGMA
    m = np.linspace(1.0, 0.25, 10)
    state = {"m_current": m, "geometry": _ProviderGeom(), "diffusion": sigma}
    g_left = AdjointConsistentProvider(side="left", diffusion=sigma).compute(state)
    g_right = AdjointConsistentProvider(side="right", diffusion=sigma).compute(state)
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="x_min", bc_type=BCType.ROBIN, alpha=0.0, beta=1.0, value=float(g_left), boundary="x_min"),
            BCSegment(name="x_max", bc_type=BCType.ROBIN, alpha=0.0, beta=1.0, value=float(g_right), boundary="x_max"),
        ],
        dimension=1,
    )
    gfdm, pts, bdry = _make_howard_gfdm_with_bc(bc, sigma=sigma)
    U_T = 0.5 * (pts[:, 0] - 2.0) ** 2
    U = gfdm.solve_hjb_system(M_density=None, U_terminal=U_T)
    assert np.all(np.isfinite(U))

    bi_left, bi_right = int(bdry[0]), int(bdry[1])
    rows = gfdm._value_form_bc_rows(0)
    row_l, tgt_l = rows[bi_left]
    row_r, tgt_r = rows[bi_right]
    u0 = np.asarray(U[0]).ravel()  # solved initial-time slice (terminal slice U[-1] is imposed)
    assert row_l @ u0 == pytest.approx(tgt_l, abs=1e-8), "Howard did not honor the left Robin row"
    assert row_r @ u0 == pytest.approx(tgt_r, abs=1e-8), "Howard did not honor the right Robin row"

"""Tests for the conservative finite-volume Fokker-Planck solver (Issue #422).

Four gates:
    1. Mass conservation to machine precision (no-flux 1D + 2D, free + advective-diffusive).
    2. Positivity under advection (MUSCL limiter prevents negative-density ringing).
    3. O(dx^2) convergence for MUSCL (and O(dx) for 1st-order upwind) vs a closed-form
       advected-diffused Gaussian.
    4. Agreement with the conservative divergence-upwind FDM solver on a smooth problem,
       shrinking under refinement.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver, FPFVMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc


# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------
def _components():
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0,
        coupling_dm=lambda m: 0.0,
    )
    return MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0)


def make_problem_1d(N, T, Nt, sigma, bounds=(0.0, 1.0), bc="no_flux"):
    bc_obj = {"no_flux": no_flux_bc(dimension=1), "periodic": periodic_bc(dimension=1)}[bc]
    geom = TensorProductGrid(bounds=[bounds], Nx_points=[N], boundary_conditions=bc_obj)
    prob = MFGProblem(geometry=geom, T=T, Nt=Nt, sigma=sigma, components=_components())
    return prob, geom


def make_problem_2d(Nx, Ny, T, Nt, sigma, bounds=((0.0, 1.0), (0.0, 1.0))):
    geom = TensorProductGrid(bounds=list(bounds), Nx_points=[Nx, Ny], boundary_conditions=no_flux_bc(dimension=2))
    prob = MFGProblem(geometry=geom, T=T, Nt=Nt, sigma=sigma, components=_components())
    return prob, geom


def normalized_gaussian_1d(x, x0, s0):
    g = np.exp(-((x - x0) ** 2) / (2 * s0**2))
    dx = x[1] - x[0]
    return g / (g.sum() * dx)


def analytic_gaussian_adv_diff(x, t, x0, s0, v0, diffusion):
    """Closed-form solution of m_t + v0 m_x = D m_xx on the line (Gaussian IC)."""
    var = s0**2 + 2.0 * diffusion * t
    return np.exp(-((x - x0 - v0 * t) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


# ===========================================================================
# Gate 1: Mass conservation to machine precision
# ===========================================================================
@pytest.mark.parametrize("reconstruction", ["upwind", "muscl"])
def test_gate1_mass_conservation_1d_free(reconstruction):
    """Pure diffusion (no drift), no-flux 1D: mass conserved to machine precision."""
    N, T, Nt, sigma = 81, 0.2, 40, 0.4
    prob, geom = make_problem_1d(N, T, Nt, sigma)
    x = geom.coordinates[0]
    dx = x[1] - x[0]
    m0 = normalized_gaussian_1d(x, 0.5, 0.12)

    solver = FPFVMSolver(prob, reconstruction=reconstruction)
    M = solver.solve_fp_system(m0)

    mass = M.sum(axis=1) * dx
    drift = float(np.max(np.abs(mass - mass[0])))
    assert drift < 1e-12, f"[{reconstruction}] free-diffusion mass drift {drift:.2e}"


@pytest.mark.parametrize("reconstruction", ["upwind", "muscl"])
def test_gate1_mass_conservation_1d_advective_diffusive(reconstruction):
    """Advection + diffusion, no-flux 1D: mass conserved to machine precision."""
    N, T, Nt, sigma = 81, 0.2, 40, 0.3
    prob, geom = make_problem_1d(N, T, Nt, sigma)
    x = geom.coordinates[0]
    dx = x[1] - x[0]
    m0 = normalized_gaussian_1d(x, 0.5, 0.12)
    drift = 0.6 * np.ones((Nt + 1, N))  # constant velocity, node-centered

    solver = FPFVMSolver(prob, reconstruction=reconstruction)
    M = solver.solve_fp_system(m0, drift_field=drift)

    mass = M.sum(axis=1) * dx
    mass_drift = float(np.max(np.abs(mass - mass[0])))
    assert mass_drift < 1e-12, f"[{reconstruction}] adv-diff mass drift {mass_drift:.2e}"


@pytest.mark.parametrize("reconstruction", ["upwind", "muscl"])
def test_gate1_mass_conservation_2d(reconstruction):
    """Advection + diffusion, no-flux 2D: mass conserved to machine precision."""
    Nx, Ny, T, Nt, sigma = 31, 31, 0.15, 30, 0.3
    prob, geom = make_problem_2d(Nx, Ny, T, Nt, sigma)
    x, y = geom.coordinates
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    g = np.exp(-(((X - 0.5) ** 2) + (Y - 0.5) ** 2) / (2 * 0.12**2))
    m0 = g / (g.sum() * dx * dy)

    drift = np.zeros((Nt + 1, Nx, Ny, 2))
    drift[..., 0] = 0.5  # vx
    drift[..., 1] = -0.3  # vy

    solver = FPFVMSolver(prob, reconstruction=reconstruction)
    M = solver.solve_fp_system(m0, drift_field=drift)

    mass = M.sum(axis=(1, 2)) * dx * dy
    mass_drift = float(np.max(np.abs(mass - mass[0])))
    assert mass_drift < 1e-12, f"[{reconstruction}] 2D mass drift {mass_drift:.2e}"


def test_gate1_mass_conservation_periodic():
    """Periodic 1D advection-diffusion: wrap-face flux keeps mass exact."""
    N, T, Nt, sigma = 80, 0.3, 60, 0.25
    prob, geom = make_problem_1d(N, T, Nt, sigma, bounds=(0.0, 1.0), bc="periodic")
    x = geom.coordinates[0]
    dx = x[1] - x[0]
    m0 = normalized_gaussian_1d(x, 0.5, 0.1)
    drift = 0.8 * np.ones((Nt + 1, N))

    solver = FPFVMSolver(prob, reconstruction="muscl")
    M = solver.solve_fp_system(m0, drift_field=drift)

    mass = M.sum(axis=1) * dx
    mass_drift = float(np.max(np.abs(mass - mass[0])))
    assert mass_drift < 1e-12, f"periodic mass drift {mass_drift:.2e}"


# ===========================================================================
# Gate 2: Positivity under advection (MUSCL limiter)
# ===========================================================================
def test_gate2_positivity_muscl_sharp_advection():
    """A sharp top-hat advected (pure advection) stays non-negative under MUSCL."""
    N, T, Nt = 200, 0.5, 200
    prob, geom = make_problem_1d(N, T, Nt, sigma=1e-9, bounds=(0.0, 1.0), bc="periodic")
    x = geom.coordinates[0]
    m0 = np.where((x > 0.3) & (x < 0.5), 1.0, 0.0)  # discontinuous, sharp
    drift = 0.7 * np.ones((Nt + 1, N))

    solver = FPFVMSolver(prob, reconstruction="muscl")
    M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=0.0)

    assert M.min() >= -1e-14, f"MUSCL produced negative density: {M.min():.2e}"


def test_gate2_positivity_muscl_peaked_advection_diffusion():
    """A peaked Gaussian under advection + diffusion stays non-negative under MUSCL."""
    N, T, Nt, sigma = 160, 0.3, 150, 0.15
    prob, geom = make_problem_1d(N, T, Nt, sigma, bounds=(-1.0, 1.0))
    x = geom.coordinates[0]
    m0 = normalized_gaussian_1d(x, 0.0, 0.06)  # sharp peak
    drift = 0.5 * np.ones((Nt + 1, N))

    solver = FPFVMSolver(prob, reconstruction="muscl")
    M = solver.solve_fp_system(m0, drift_field=drift)

    assert M.min() >= -1e-14, f"MUSCL produced negative density: {M.min():.2e}"


# ===========================================================================
# Gate 3: O(dx^2) convergence (MUSCL) vs O(dx) (upwind)
# ===========================================================================
def _convergence_slope(reconstruction, grids):
    x0, s0, v0, sigma = 0.0, 0.3, 0.4, 0.2
    diffusion = 0.5 * sigma**2
    T = 0.5
    bounds = (-2.5, 2.5)
    L = bounds[1] - bounds[0]

    errors, dxs = [], []
    for N in grids:
        dx = L / (N - 1)
        Nt = max(20, int(np.ceil(T / (0.4 * dx**2))))
        prob, geom = make_problem_1d(N, T, Nt, sigma, bounds=bounds)
        x = geom.coordinates[0]
        m0 = analytic_gaussian_adv_diff(x, 0.0, x0, s0, v0, diffusion)
        drift = v0 * np.ones((Nt + 1, N))

        solver = FPFVMSolver(prob, reconstruction=reconstruction)
        M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=sigma)

        m_exact = analytic_gaussian_adv_diff(x, T, x0, s0, v0, diffusion)
        err = np.sqrt(dx * np.sum((M[-1] - m_exact) ** 2))
        errors.append(err)
        dxs.append(dx)

    log_dx = np.log(np.array(dxs))
    log_err = np.log(np.array(errors))
    slope = np.polyfit(log_dx, log_err, 1)[0]
    return slope, errors


def test_gate3_convergence_muscl_second_order():
    grids = [41, 61, 81, 121, 161]
    slope, errors = _convergence_slope("muscl", grids)
    assert slope > 1.6, f"MUSCL convergence slope {slope:.2f} (errors={errors})"


def test_gate3_convergence_upwind_first_order():
    grids = [41, 61, 81, 121, 161]
    slope, errors = _convergence_slope("upwind", grids)
    assert 0.7 < slope < 1.4, f"upwind convergence slope {slope:.2f} (errors={errors})"


def test_gate3_muscl_beats_upwind():
    """At a fixed resolution MUSCL is strictly more accurate than upwind."""
    grids = [81]
    _, err_muscl = _convergence_slope("muscl", grids)
    _, err_upwind = _convergence_slope("upwind", grids)
    assert err_muscl[0] < err_upwind[0]


# ===========================================================================
# Gate 4: Agreement with the conservative divergence-upwind FDM solver
# ===========================================================================
def _fvm_fdm_maxdiff(N):
    # FVM-upwind and FDM-divergence-upwind share the same 1st-order upwind spatial operator;
    # their difference is the time discretization (Strang-split explicit-advection/implicit-
    # diffusion vs fully-implicit), which is O(dt). Refine dt together with dx (Nt proportional
    # to N) so the combined discretization difference shrinks.
    T, sigma = 0.25, 0.3
    Nt = 4 * (N - 1)
    bounds = (-1.5, 1.5)
    v0 = 0.5
    prob, geom = make_problem_1d(N, T, Nt, sigma, bounds=bounds)
    x = geom.coordinates[0]
    dx = x[1] - x[0]
    m0 = normalized_gaussian_1d(x, -0.3, 0.25)
    drift = v0 * np.ones((Nt + 1, N))

    fvm = FPFVMSolver(prob, reconstruction="upwind")
    M_fvm = fvm.solve_fp_system(m0, drift_field=drift, volatility_field=sigma)

    fdm = FPFDMSolver(prob, advection_scheme="divergence_upwind")
    M_fdm = fdm.solve_fp_system(m0, drift_field=drift, volatility_field=sigma)

    return float(np.max(np.abs(M_fvm[-1] - M_fdm[-1]))), dx


def test_gate4_fvm_fdm_agreement_shrinks():
    diff_coarse, _ = _fvm_fdm_maxdiff(61)
    diff_fine, _ = _fvm_fdm_maxdiff(121)
    peak = 1.0 / np.sqrt(2 * np.pi * 0.25**2)  # rough scale of the density peak
    assert diff_coarse < 0.15 * peak, f"coarse FVM-FDM max-diff too large: {diff_coarse:.3e}"
    assert diff_fine < diff_coarse, f"FVM-FDM diff did not shrink: {diff_coarse:.3e} -> {diff_fine:.3e}"


# ===========================================================================
# Registration / dispatch
# ===========================================================================
def test_scheme_registration():
    from mfgarchon.types import NumericalScheme

    assert NumericalScheme("fvm_upwind") == NumericalScheme.FVM_UPWIND
    assert NumericalScheme("fvm_muscl") == NumericalScheme.FVM_MUSCL
    # Mass-exact by construction -> no renormalization required.
    assert not NumericalScheme.FVM_UPWIND.requires_renormalization()
    assert not NumericalScheme.FVM_MUSCL.requires_renormalization()


def test_factory_creates_fvm_pair():
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfgarchon.factory import create_paired_solvers
    from mfgarchon.types import NumericalScheme

    prob, _ = make_problem_1d(41, 0.1, 10, 0.2)
    hjb, fp = create_paired_solvers(prob, NumericalScheme.FVM_MUSCL)
    assert isinstance(hjb, HJBFDMSolver)
    assert isinstance(fp, FPFVMSolver)
    assert fp.reconstruction == "muscl"

    _hjb_u, fp_u = create_paired_solvers(prob, NumericalScheme.FVM_UPWIND)
    assert fp_u.reconstruction == "upwind"


def test_potential_field_drives_coupling_velocity():
    """potential_field U -> alpha = -coupling*grad(U); matches the FDM divergence-upwind stencil."""
    N, T, Nt, sigma = 81, 0.2, 40, 0.3
    prob, geom = make_problem_1d(N, T, Nt, sigma)
    x = geom.coordinates[0]
    dx = x[1] - x[0]
    m0 = normalized_gaussian_1d(x, 0.5, 0.12)
    U = np.tile((x**2).reshape(1, -1), (Nt + 1, 1))  # smooth quadratic potential

    solver = FPFVMSolver(prob, reconstruction="upwind")
    M = solver.solve_fp_system(m0, potential_field=U)
    mass = M.sum(axis=1) * dx
    assert float(np.max(np.abs(mass - mass[0]))) < 1e-12
    assert M.min() >= -1e-14


def test_drift_and_potential_mutually_exclusive():
    prob, geom = make_problem_1d(41, 0.1, 10, 0.2)
    x = geom.coordinates[0]
    m0 = normalized_gaussian_1d(x, 0.5, 0.12)
    with pytest.raises(ValueError, match="at most one"):
        FPFVMSolver(prob).solve_fp_system(m0, drift_field=np.zeros((11, 41)), potential_field=np.zeros((11, 41)))


def test_varying_volatility_not_implemented():
    prob, geom = make_problem_1d(41, 0.1, 10, 0.2)
    x = geom.coordinates[0]
    m0 = normalized_gaussian_1d(x, 0.5, 0.12)
    bad_sigma = np.linspace(0.1, 0.5, 41)
    with pytest.raises(NotImplementedError, match="scalar"):
        FPFVMSolver(prob).solve_fp_system(m0, volatility_field=bad_sigma)

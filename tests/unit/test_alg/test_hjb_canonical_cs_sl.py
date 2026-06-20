#!/usr/bin/env python3
"""Unit tests for the canonical Carlini-Silva SL with implicit-alpha* DPP (Issue #1058).

``diffusion_method="canonical_cs"`` solves the IMPLICIT dynamic-programming principle: at
each grid node the optimal control alpha* is the per-point minimizer of the DPP objective

    phi(alpha) = (dt/2)|alpha|^2 - dt*h(x_i, m_i) + (1/2d) sum_k [I_h(y_k^+) + I_h(y_k^-)],
    u^n(x_i)  = min_alpha phi(alpha),

with y_k^pm = x_i + alpha*dt +/- sigma_k*sqrt(dt)*e_k, h = H(x, m, p=0, t) (single-source),
and I_h the linear (monotone Q1) interpolant. This is the CS 2014 hypothesis under which the
scheme is unconditionally stable -- distinct from the explicit-alpha* path
(``diffusion_method="stochastic"``, alpha* = -grad u at the grid node).

Gates:
  1. Convergence: 1D first-order HJB (sigma=0, H=|p|^2/2) converges to the analytic
     Hopf-Lax solution u(0,x) = x^2/(2(1+T)) under refinement.
  2. Unconditional stability: a time step far above the explicit-CFL limit stays bounded,
     where the explicit (operator-split) scheme blows up.
  3. Implicit vs explicit: on a stiff problem at large dt, the canonical (implicit-alpha*)
     result is far closer to the converged reference than the explicit-alpha* path.
  4. Existing paths byte-identical: covered by the existing
     ``test_hjb_semi_lagrangian.py`` suite (the new method is purely additive).

Reference implementation:
  mfg-research/.../exp08_towel_2d_validation/_preflight_1d/cs_sl_canonical_implicit_1d.py
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _make_problem(bounds, Nx, T, Nt, diffusion, terminal_fn, coupling=None):
    """1D MFGProblem with separable quadratic-control Hamiltonian H = |p|^2/2 + h."""
    geometry = TensorProductGrid(
        dimension=1, bounds=[bounds], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1)
    )
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=coupling,
            coupling_dm=(lambda m: 1.0) if coupling is not None else None,
        ),
        m_initial=lambda x: 1.0,
        u_terminal=terminal_fn,
    )
    problem = MFGProblem(geometry=geometry, T=T, Nt=Nt, diffusion=diffusion, components=components)
    return problem, geometry


def _grid(geometry):
    return geometry.get_spatial_grid().flatten()


class TestCanonicalCSInitialization:
    """Construction-time validation of the canonical_cs scheme."""

    def test_linear_accepted(self):
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(
            geometry=geometry,
            T=1.0,
            Nt=30,
            components=MFGComponents(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
            ),
        )
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear", diffusion_method="canonical_cs")
        assert solver.diffusion_method == "canonical_cs"
        assert solver.interpolation_method == "linear"

    @pytest.mark.parametrize("interp", ["cubic", "quintic", "nearest"])
    def test_nonlinear_interpolation_rejected(self, interp):
        """CS 2014 stability requires monotone (Q1/linear) interpolation -> fail fast."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(
            geometry=geometry,
            T=1.0,
            Nt=30,
            components=MFGComponents(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
            ),
        )
        with pytest.raises(ValueError, match=r"canonical_cs.*requires interpolation_method='linear'"):
            HJBSemiLagrangianSolver(problem, interpolation_method=interp, diffusion_method="canonical_cs")

    def test_apply_diffusion_raises(self):
        """canonical_cs bakes diffusion into the DPP; reaching _apply_diffusion is a bug."""
        problem, _ = _make_problem((0.0, 1.0), 31, 1.0, 30, 0.045, lambda x: 0.0)
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear", diffusion_method="canonical_cs")
        with pytest.raises(NotImplementedError, match="should not be called"):
            solver._apply_diffusion(np.zeros(31), 0.01)


class TestCanonicalCSCorrectness:
    """Sanity + gate 1 (convergence to analytic solution)."""

    def test_constant_terminal_preserved(self):
        """H = |p|^2/2 (h=0), constant terminal -> constant value (alpha*=0 minimizes phi)."""
        problem, geom = _make_problem((-1.0, 1.0), 31, 0.5, 10, 0.045, lambda x: 3.0)
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear", diffusion_method="canonical_cs")
        Nx = len(_grid(geom))
        U = solver.solve_hjb_system(
            M_density=np.ones((problem.Nt + 1, Nx)),
            U_terminal=np.full(Nx, 3.0),
            U_coupling_prev=np.zeros((problem.Nt + 1, Nx)),
        )
        np.testing.assert_allclose(U[0], 3.0, atol=1e-8)

    def test_gate1_convergence_to_hopf_lax(self):
        """Gate 1: sigma=0, H=|p|^2/2, g(x)=x^2/2 -> analytic u(0,x)=x^2/(2(1+T)).

        The canonical scheme's per-node DPP minimization is the discrete Hopf-Lax operator;
        composing it backward must converge to the exact viscosity solution under refinement.
        """
        T = 1.0
        errors = []
        for Nx, Nt in [(41, 20), (81, 40), (161, 80)]:
            problem, geom = _make_problem((-3.0, 3.0), Nx, T, Nt, 0.0, lambda x: 0.5 * x[0] ** 2)
            x = _grid(geom)
            solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear", diffusion_method="canonical_cs")
            U = solver.solve_hjb_system(
                M_density=np.ones((Nt + 1, Nx)),
                U_terminal=0.5 * x**2,
                U_coupling_prev=np.zeros((Nt + 1, Nx)),
            )
            assert np.all(np.isfinite(U)), f"non-finite at Nx={Nx}"
            u_exact = x**2 / (2.0 * (1.0 + T))
            interior = (x > -2.0) & (x < 2.0)  # exclude boundary-reflection layer
            errors.append(float(np.sqrt(np.mean((U[0][interior] - u_exact[interior]) ** 2))))

        # Monotone decrease under refinement, near first-order, and small in absolute terms.
        assert errors[0] > errors[1] > errors[2], f"not decreasing: {errors}"
        assert errors[-1] < 1e-2, f"finest L2 too large: {errors[-1]:.3e}"
        # Roughly O(dx): each grid doubling should roughly halve the error.
        assert errors[0] / errors[1] > 1.5, f"slow convergence: {errors}"

    def test_gate1b_convergence_with_diffusion(self):
        """Gate 1 (sigma>0): the diffusive LQ HJB converges to the analytic solution.

        For H = |p|^2/2 with terminal g(x)=x^2/2, the quadratic ansatz u(t,x) = a(t)x^2 + b(t)
        gives a(0) = 1/(2(1+T)) -- the x^2 coefficient is diffusion-independent (diffusion only
        shifts b). We verify both (i) the fitted interior x^2 coefficient converges to that
        analytic value, and (ii) Cauchy self-convergence (successive resolutions agree to a
        shrinking tolerance). A fine explicit-gradient scheme (ADI) is deliberately NOT used as
        the reference: it is the inaccurate scheme this issue exists to replace (it undershoots
        the analytic value badly on this LQ problem).
        """
        T, sigma = 0.5, 0.3
        c_exact = 1.0 / (2.0 * (1.0 + T))  # analytic x^2 coefficient at t=0
        coeff_rel_errors = []
        solutions = {}
        for Nx, Nt in [(31, 30), (61, 60), (121, 120)]:
            prob, geom = _make_problem((-3.0, 3.0), Nx, T, Nt, sigma**2 / 2, lambda x: 0.5 * x[0] ** 2)
            x = _grid(geom)
            U = HJBSemiLagrangianSolver(
                prob, interpolation_method="linear", diffusion_method="canonical_cs"
            ).solve_hjb_system(np.ones((Nt + 1, Nx)), 0.5 * x**2, np.zeros((Nt + 1, Nx)))
            assert np.all(np.isfinite(U))
            solutions[Nx] = (x, U[0])
            interior = (x > -2.0) & (x < 2.0)  # interior fit avoids the no-flux boundary layer
            c_fit = float(np.sum(U[0][interior] * x[interior] ** 2) / np.sum(x[interior] ** 4))
            coeff_rel_errors.append(abs(c_fit - c_exact) / c_exact)

        # (i) fitted x^2 coefficient converges monotonically to the analytic value.
        assert coeff_rel_errors[0] > coeff_rel_errors[1] > coeff_rel_errors[2], (
            f"x^2 coefficient not converging to analytic: rel_errors={coeff_rel_errors}"
        )
        assert coeff_rel_errors[-1] < 0.08, f"finest coefficient off by {coeff_rel_errors[-1]:.2%}"

        # (ii) Cauchy self-convergence vs the finest grid.
        xf, uf = solutions[121]
        cauchy = []
        for Nx in [31, 61]:
            x, u = solutions[Nx]
            interior = (x > -2.0) & (x < 2.0)
            cauchy.append(float(np.sqrt(np.mean((u[interior] - np.interp(x[interior], xf, uf)) ** 2))))
        assert cauchy[0] > cauchy[1], f"not self-converging: {cauchy}"
        assert cauchy[-1] < 3e-2, f"finest Cauchy gap too large: {cauchy[-1]:.3e}"


class TestCanonicalCSStability:
    """Gate 2: unconditional stability at a time step far above the explicit-CFL limit."""

    def test_gate2_large_dt_bounded(self):
        T, sigma = 1.0, 0.5
        Nx = 41
        g = lambda x: np.exp(-20.0 * x[0] ** 2)  # noqa: E731 (steep terminal)
        # Only a handful of steps over the whole horizon -> dt >> CFL.
        Nt = 2
        problem, geom = _make_problem((-2.0, 2.0), Nx, T, Nt, sigma**2 / 2, g)
        x = _grid(geom)
        gx = np.exp(-20.0 * x**2)
        dt = T / Nt
        dx = float(x[1] - x[0])
        # Explicit-diffusion stability bound dt < dx^2/(2*D); confirm we are well above it.
        diffusion_cfl = dt / (dx**2 / (2.0 * (sigma**2 / 2)))
        assert diffusion_cfl > 5.0, f"test mis-specified; CFL ratio only {diffusion_cfl:.1f}"

        M = np.ones((Nt + 1, Nx))
        U = HJBSemiLagrangianSolver(
            problem, interpolation_method="linear", diffusion_method="canonical_cs"
        ).solve_hjb_system(M, gx, np.zeros((Nt + 1, Nx)))

        assert np.all(np.isfinite(U)), "canonical_cs produced NaN/Inf at large dt"
        # Maximum principle: with h=0, u^n stays within [min g, max g].
        assert np.max(np.abs(U)) <= np.max(np.abs(gx)) + 1e-9, f"unbounded: max|U|={np.max(np.abs(U)):.3e}"

        # The explicit (operator-split) scheme at the same dt, with substepping disabled,
        # is unstable here -- demonstrating the contrast.
        U_explicit = HJBSemiLagrangianSolver(
            problem,
            interpolation_method="linear",
            diffusion_method="explicit",
            enable_adaptive_substepping=False,
            check_cfl=False,
        ).solve_hjb_system(M, gx, np.zeros((Nt + 1, Nx)))
        explicit_blew_up = (not np.all(np.isfinite(U_explicit))) or (
            np.max(np.abs(U_explicit)) > 100.0 * np.max(np.abs(gx))
        )
        assert explicit_blew_up, (
            f"explicit scheme unexpectedly stable at large dt: max|U|={np.max(np.abs(U_explicit)):.3e}"
        )


class TestCanonicalCSImplicitVsExplicit:
    """Gate 3: implicit-alpha* (canonical) beats explicit-alpha* (stochastic) on stiff problems."""

    def test_gate3_canonical_beats_explicit_at_large_dt(self):
        T, sigma, k = 1.0, 0.3, 2.0
        g = lambda x: np.tanh(k * x[0])  # noqa: E731 (stiff terminal -> large gradients)

        # Reference: a fine canonical solve (gate 1 establishes this scheme converges to the
        # analytic viscosity solution).
        Nx_ref, Nt_ref = 81, 80
        prob_ref, geom_ref = _make_problem((-2.0, 2.0), Nx_ref, T, Nt_ref, sigma**2 / 2, g)
        xr = _grid(geom_ref)
        ref0 = HJBSemiLagrangianSolver(
            prob_ref, interpolation_method="linear", diffusion_method="canonical_cs"
        ).solve_hjb_system(np.ones((Nt_ref + 1, Nx_ref)), np.tanh(k * xr), np.zeros((Nt_ref + 1, Nx_ref)))[0]

        # Coarse solve at a large time step (Nt=5 over T=1 -> dt=0.2).
        Nx, Nt = 41, 5
        prob, geom = _make_problem((-2.0, 2.0), Nx, T, Nt, sigma**2 / 2, g)
        x = _grid(geom)
        gx = np.tanh(k * x)
        M = np.ones((Nt + 1, Nx))
        ref_on_x = np.interp(x, xr, ref0)
        interior = (x > -1.5) & (x < 1.5)

        U_canonical = HJBSemiLagrangianSolver(
            prob, interpolation_method="linear", diffusion_method="canonical_cs"
        ).solve_hjb_system(M, gx, np.zeros((Nt + 1, Nx)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver_explicit = HJBSemiLagrangianSolver(
                prob,
                interpolation_method="linear",
                diffusion_method="stochastic",  # explicit-alpha* = -grad u at the grid node
                enable_adaptive_substepping=False,
                check_cfl=False,
            )
        U_explicit = solver_explicit.solve_hjb_system(M, gx, np.zeros((Nt + 1, Nx)))

        err_canonical = float(np.sqrt(np.mean((U_canonical[0][interior] - ref_on_x[interior]) ** 2)))
        err_explicit = float(np.sqrt(np.mean((U_explicit[0][interior] - ref_on_x[interior]) ** 2)))

        assert np.all(np.isfinite(U_canonical)), "canonical produced NaN/Inf"
        # The implicit-alpha* DPP is meaningfully closer to the reference than the explicit
        # at-grid alpha*. Issue #1413: the explicit (stochastic) path is now the CORRECTED
        # Lax-Oleinik scheme, so this gap (~6-7x here) reflects the genuine implicit-vs-explicit
        # alpha* difference at large dt — NOT the former sign/foot bug, which made the explicit
        # >10x worse (and ~24% off even at lambda=1; see Issue #575/#1413).
        assert err_canonical < err_explicit, f"canonical {err_canonical:.3e} !< explicit {err_explicit:.3e}"
        assert err_explicit > 3.0 * err_canonical, (
            f"explicit not meaningfully worse: canonical={err_canonical:.3e}, explicit={err_explicit:.3e}"
        )
        assert err_canonical < 0.1, f"canonical itself inaccurate at large dt: {err_canonical:.3e}"


class TestCanonicalCSMultiDimensional:
    """Dispatch-surface coverage: the nD per-node vector minimization runs and stays bounded."""

    def test_2d_runs_and_bounded(self):
        bc = no_flux_bc(dimension=2)
        grid = TensorProductGrid(
            dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=bc
        )
        problem = MFGProblem(
            geometry=grid,
            T=0.2,
            Nt=4,
            diffusion=0.3**2 / 2,
            components=MFGComponents(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
            ),
        )
        X, Y = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11), indexing="ij")
        U_terminal = 0.5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear", diffusion_method="canonical_cs")
        U = solver.solve_hjb_system(
            M_density=np.ones((5, 11, 11)),
            U_terminal=U_terminal,
            U_coupling_prev=np.zeros((5, 11, 11)),
        )
        assert U.shape == (5, 11, 11)
        assert np.all(np.isfinite(U))
        # With h=0 the maximum principle bounds u by the terminal data.
        assert np.max(np.abs(U)) <= np.max(np.abs(U_terminal)) + 1e-9

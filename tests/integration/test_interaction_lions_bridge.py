"""Integration tests for the interaction Lions bridge and ring equilibrium.

Issue #1023, Phase 2.

Gate 3 (lions-bridge equivalence): create_lions_source with an EnergyFunctional
produces the same HJB source as the hand-rolled energy(m)-lambda + FD path and
the optimized create_nonlocal_source path (backward compatibility).

Gate 4 (ring-equilibrium demo): a coupled HJB-FP solve with a repulsive
interaction kernel plus a central attractive potential depletes the centre and
pushes density outward, relative to the attractive-only baseline. This is the
non-local towel-on-the-beach signature that local f(m) cannot produce.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.coupling.lions_correction import (
    create_lions_source,
    create_nonlocal_source,
)
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.config import MFGSolverConfig
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.operators.interaction import (
    CombinedEnergy,
    ConvolutionCouplingOperator,
    GaussianKernel,
    PotentialEnergy,
    QuadraticInteractionEnergy,
)
from mfgarchon.utils.functional_calculus import FiniteDifferenceFunctionalDerivative
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


class TestGate3LionsBridgeEquivalence:
    """The analytic EnergyFunctional path matches the legacy FD / nonlocal paths."""

    def test_analytic_matches_nonlocal_source_exactly(self):
        N = 60
        x = np.linspace(0.0, 1.0, N)
        dx = x[1] - x[0]
        kernel = GaussianKernel(amplitude=1.3, length_scale=0.1)

        conv = ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx], use_fft=False)
        energy = QuadraticInteractionEnergy(conv)
        source_analytic = create_lions_source(energy)

        W = kernel.matrix(x)  # raw K(x_i, x_j) matrix
        source_nonlocal = create_nonlocal_source(W, grid_spacing=dx)

        m = np.sin(np.pi * x) + 1.2
        v = np.zeros(N)
        r_analytic = source_analytic(x, m, v, 0.0)
        r_nonlocal = source_nonlocal(x, m, v, 0.0)
        # Both equal (W @ m) * dx to machine precision.
        np.testing.assert_allclose(r_analytic, r_nonlocal, atol=1e-12)

    def test_analytic_matches_fd_lambda_path(self):
        N = 60
        x = np.linspace(0.0, 1.0, N)
        dx = x[1] - x[0]
        kernel = GaussianKernel(amplitude=1.3, length_scale=0.1)

        conv = ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx], use_fft=False)
        energy = QuadraticInteractionEnergy(conv)
        source_analytic = create_lions_source(energy)

        # Legacy path: hand-rolled energy lambda + finite-difference derivative.
        W = kernel.matrix(x)

        def energy_lambda(m):
            m = m.ravel()
            return 0.5 * np.sum(m * (W @ m)) * dx

        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")
        source_fd = create_lions_source(energy_lambda, fd)

        m = np.sin(np.pi * x) + 1.2
        v = np.zeros(N)
        r_analytic = source_analytic(x, m, v, 0.0)
        r_fd = source_fd(x, m, v, 0.0)
        rel = np.max(np.abs(r_analytic - r_fd)) / np.max(np.abs(r_analytic))
        assert rel < 1e-6

    def test_fd_path_requires_functional_derivative(self):
        """Backward compat: plain callable without FD instance raises clearly."""

        def energy_lambda(m):
            return 0.5 * np.sum(m**2)

        with pytest.raises(ValueError):
            create_lions_source(energy_lambda)

    def test_time_space_array_uses_last_slice(self):
        """Source handles both (Nx,) slice and (Nt+1, Nx) trajectory inputs."""
        N = 30
        x = np.linspace(0.0, 1.0, N)
        dx = x[1] - x[0]
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        source = create_lions_source(QuadraticInteractionEnergy(conv))

        m_slice = np.sin(np.pi * x) + 1.0
        m_traj = np.tile(m_slice, (5, 1))  # (Nt+1, Nx), constant in time
        r_slice = source(x, m_slice, np.zeros(N), 0.0)
        r_traj = source(x, m_traj, np.zeros(N), 0.0)
        np.testing.assert_allclose(r_slice, r_traj, atol=1e-12)


def _ring_problem(grid_only=False, amp=5.0, length_scale=0.15, bowl=4.0):
    """Build a 1D towel-on-the-beach problem.

    Central attractive potential (bowl, cost-signed) always present; the
    repulsive interaction is added only when ``grid_only`` is False.
    """
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0,
        coupling_dm=lambda m: 0.0,
    )

    def m_initial(xx):
        return np.exp(-((xx - 0.5) ** 2) / (2 * 0.12**2))

    components = MFGComponents(hamiltonian=H, u_terminal=lambda xx: 0.0, m_initial=m_initial)
    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[21 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=0.5,
        Nt=4,
        sigma=0.2,
        components=components,
    )
    g = problem.geometry.get_spatial_grid().ravel()
    dx = g[1] - g[0]
    potential = PotentialEnergy(bowl * (g - 0.5) ** 2)  # attractive bowl (cost away from centre)
    if grid_only:
        energy = potential
    else:
        conv = ConvolutionCouplingOperator(
            GaussianKernel(amplitude=amp, length_scale=length_scale),
            grid_shape=(len(g),),
            spacings=[dx],
            use_fft=True,
        )
        energy = CombinedEnergy([QuadraticInteractionEnergy(conv), potential])
    problem.source_term_hjb = create_lions_source(energy)
    return problem, g


def _solve_terminal_density(problem, g, iters=3):
    hjb = HJBFDMSolver(problem)
    fp = FPFDMSolver(problem)
    iterator = FixedPointIterator(problem, hjb, fp, config=MFGSolverConfig(max_iterations=iters))
    result = iterator.solve()
    M = result.M if hasattr(result, "M") else result[1]
    m_terminal = M[-1].ravel()
    m_terminal = m_terminal / np.trapezoid(m_terminal, g)
    return m_terminal


class TestGate4RingEquilibrium:
    """Non-local repulsion depletes the centre and spreads density outward."""

    @pytest.mark.slow  # coupled FixedPointIterator solve; deselected on PR-CI (30-min budget)
    def test_central_depletion_and_outward_spread(self):
        prob_attract, g = _ring_problem(grid_only=True)
        m_attract = _solve_terminal_density(prob_attract, g)

        prob_ring, g = _ring_problem(grid_only=False)
        m_ring = _solve_terminal_density(prob_ring, g)

        assert np.all(np.isfinite(m_attract))
        assert np.all(np.isfinite(m_ring))

        ci = int(np.argmin(np.abs(g - 0.5)))
        # Central depletion: the repulsive non-local coupling lowers the centre.
        assert m_ring[ci] < 0.7 * m_attract[ci]

        # Outward spread: variance about the centre increases with interaction.
        var_attract = np.trapezoid(m_attract * (g - 0.5) ** 2, g)
        var_ring = np.trapezoid(m_ring * (g - 0.5) ** 2, g)
        assert var_ring > 1.5 * var_attract

        # Density stays a non-negative normalized measure.
        assert np.all(m_ring >= -1e-9)
        assert np.trapezoid(m_ring, g) == pytest.approx(1.0)

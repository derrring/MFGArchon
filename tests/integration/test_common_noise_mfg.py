"""
Integration tests for Common Noise MFG solver.

Tests the CommonNoiseMFGSolver with simple analytical cases where solutions
can be verified or compared with deterministic limits.
"""

import pytest

import numpy as np

from mfgarchon.core import MFGComponents
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _minimal_components():
    """Minimal MFGComponents for test initialization (Issues #670, #673)."""
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.1 * m,
        coupling_dm=lambda m: 0.1 * np.ones_like(m),
    )
    return MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: np.zeros_like(x),
        m_initial=lambda x: np.ones_like(x),
    )


@pytest.mark.slow
class TestCommonNoiseMFGSolver:
    """Test suite for Common Noise MFG solver."""

    def test_solver_initialization(self):
        """Test solver can be initialized with valid problem."""
        from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver

        # Create simple stochastic problem
        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)

        def simple_hamiltonian(x, p, m, theta):
            return 0.5 * p**2 + 0.1 * m

        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1)
        )  # Nx=21 -> 22 points
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=simple_hamiltonian,
            components=_minimal_components(),
        )

        # Create solver
        solver = CommonNoiseMFGSolver(problem, num_noise_samples=10, variance_reduction=False, parallel=False)

        assert solver.K == 10
        assert not solver.variance_reduction
        assert not solver.parallel

    def test_solver_requires_common_noise(self):
        """Test that solver raises error if problem has no common noise."""
        from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver
        from mfgarchon.core import MFGProblem

        # Regular MFG problem without noise
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1)
        )  # Nx=21 -> 22 points
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=11, components=_minimal_components())

        # Should raise ValueError
        with pytest.raises(ValueError, match="must have common noise"):
            CommonNoiseMFGSolver(problem, num_noise_samples=10)

        # This test would verify that when noise variance → 0,
        # the common noise solution converges to deterministic solution

        # This test would verify CommonNoiseMFGResult structure:
        # - u_mean, m_mean shapes
        # - u_std, m_std shapes
        # - MC error estimates
        # - Confidence intervals

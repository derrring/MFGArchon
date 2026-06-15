"""
Integration tests for Lions derivative correction bridge.

Tests that the functional calculus infrastructure correctly feeds
into the HJB source term pipeline via create_lions_source and
create_nonlocal_source.
"""

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
from mfgarchon.utils.functional_calculus import FiniteDifferenceFunctionalDerivative
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _make_problem(Nx: int = 51, source_term_hjb=None) -> MFGProblem:
    """Create a 1D MFG problem with optional source term."""
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=0.5,
        Nt=10,
        sigma=0.3,
        components=components,
        source_term_hjb=source_term_hjb,
    )


class TestCreateLionsSource:
    """Test create_lions_source bridge function."""

    def test_returns_callable(self):
        """Bridge should return a callable with correct signature."""
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-4)

        def energy(m):
            return 0.5 * np.sum(m**2) * 0.02

        source = create_lions_source(energy, fd)
        assert callable(source)

    def test_source_signature(self):
        """Source should accept (x, m, v, t) and return array."""
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-4)

        def energy(m):
            return 0.5 * np.sum(m**2) * 0.02

        source = create_lions_source(energy, fd)
        x = np.linspace(0, 1, 50)
        m = np.ones(50) / 50
        v = np.zeros(50)
        result = source(x, m, v, 0.0)

        assert isinstance(result, np.ndarray)
        assert result.shape == (50,)

    def test_quadratic_functional_derivative(self):
        """For F[m] = (1/2) int m^2 dx, delta F/delta m = m * dx."""
        Nx = 50
        dx = 1.0 / Nx
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-4, method="central")

        def energy(m):
            return 0.5 * np.sum(m**2) * dx

        source = create_lions_source(energy, fd)
        x = np.linspace(0, 1, Nx)
        m = np.ones(Nx) * 2.0  # uniform density = 2

        result = source(x, m, np.zeros(Nx), 0.0)

        # Analytical: delta F/delta m[m](x_i) = m(x_i) * dx = 2 * dx
        expected = m * dx
        np.testing.assert_allclose(result, expected, rtol=1e-2)

    def test_linear_functional_derivative(self):
        """For F[m] = int V(x) m(x) dx, delta F/delta m = V(x) * dx."""
        Nx = 50
        dx = 1.0 / Nx
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-4, method="central")

        x = np.linspace(0, 1, Nx)
        V = np.sin(np.pi * x)

        def energy(m):
            return np.sum(V * m) * dx

        source = create_lions_source(energy, fd)
        m = np.ones(Nx)

        result = source(x, m, np.zeros(Nx), 0.0)

        # Analytical: delta F/delta m = V(x_i) * dx
        expected = V * dx
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-10)


class TestCreateNonlocalSource:
    """Test create_nonlocal_source optimized path."""

    def test_returns_callable(self):
        Nx = 30
        x = np.linspace(0, 1, Nx)
        W = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * 0.1**2))
        source = create_nonlocal_source(W, grid_spacing=x[1] - x[0])
        assert callable(source)

    def test_convolution_result(self):
        """W @ m should give the convolution integral."""
        Nx = 50
        dx = 1.0 / Nx
        x = np.linspace(0, 1, Nx)

        # Identity kernel: W = I -> (W @ m) = m
        W = np.eye(Nx)
        source = create_nonlocal_source(W, grid_spacing=dx)
        m = np.sin(np.pi * x) + 1.0

        result = source(x, m, np.zeros(Nx), 0.0)
        np.testing.assert_allclose(result, m * dx, atol=1e-12)

    def test_gaussian_kernel_symmetry(self):
        """Gaussian kernel convolution should be symmetric for symmetric m."""
        Nx = 51
        dx = 1.0 / (Nx - 1)
        x = np.linspace(0, 1, Nx)

        W = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * 0.1**2))
        source = create_nonlocal_source(W, grid_spacing=dx)

        # Symmetric density
        m = np.exp(-((x - 0.5) ** 2) / (2 * 0.15**2))

        result = source(x, m, np.zeros(Nx), 0.0)

        # Result should be approximately symmetric around 0.5
        np.testing.assert_allclose(result, result[::-1], atol=1e-6)

    def test_matches_lions_source(self):
        """Nonlocal source should match create_lions_source for same kernel."""
        Nx = 30
        dx = 1.0 / Nx
        x = np.linspace(0, 1, Nx)

        W = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * 0.2**2))

        # Direct path
        source_direct = create_nonlocal_source(W, grid_spacing=dx)

        # FD path
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-4, method="central")

        def energy(m):
            return 0.5 * np.sum(m * (W @ m)) * dx

        source_fd = create_lions_source(energy, fd)

        m = np.sin(np.pi * x) + 1.5

        result_direct = source_direct(x, m, np.zeros(Nx), 0.0)
        result_fd = source_fd(x, m, np.zeros(Nx), 0.0)

        # Should be close (FD has epsilon error)
        np.testing.assert_allclose(result_direct, result_fd, rtol=0.05)


class TestLionsCorrectionEndToEnd:
    """Test Lions correction flows through the full MFG solve pipeline."""

    def test_source_term_affects_solution(self):
        """MFG solve with Lions correction should differ from without."""
        Nx = 31
        problem_plain = _make_problem(Nx=Nx)

        # Get actual grid from geometry to match solver dimensions
        actual_grid = problem_plain.geometry.get_spatial_grid().ravel()
        dx = actual_grid[1] - actual_grid[0]

        # Gaussian interaction kernel sized to actual grid
        W = np.exp(-((actual_grid[:, None] - actual_grid[None, :]) ** 2) / (2 * 0.2**2))
        source = create_nonlocal_source(W, grid_spacing=dx)

        problem_lions = _make_problem(Nx=Nx, source_term_hjb=source)

        # Solve both
        hjb1, fp1 = HJBFDMSolver(problem_plain), FPFDMSolver(problem_plain)
        hjb2, fp2 = HJBFDMSolver(problem_lions), FPFDMSolver(problem_lions)

        config = MFGSolverConfig(max_iterations=5)
        iter1 = FixedPointIterator(problem_plain, hjb1, fp1, config=config)
        iter2 = FixedPointIterator(problem_lions, hjb2, fp2, config=config)

        result1 = iter1.solve()
        result2 = iter2.solve()

        U1 = result1.U if hasattr(result1, "U") else result1[0]
        U2 = result2.U if hasattr(result2, "U") else result2[0]

        # Solutions should differ due to nonlocal coupling
        assert not np.allclose(U1, U2, atol=1e-4)

    def test_solution_is_finite(self):
        """MFG with Lions correction should produce finite solution."""
        Nx = 31
        problem_tmp = _make_problem(Nx=Nx)
        actual_grid = problem_tmp.geometry.get_spatial_grid().ravel()
        dx = actual_grid[1] - actual_grid[0]

        W = 0.1 * np.exp(-((actual_grid[:, None] - actual_grid[None, :]) ** 2) / (2 * 0.2**2))
        source = create_nonlocal_source(W, grid_spacing=dx)

        problem = _make_problem(Nx=Nx, source_term_hjb=source)
        hjb, fp = HJBFDMSolver(problem), FPFDMSolver(problem)
        config = MFGSolverConfig(max_iterations=5)
        iterator = FixedPointIterator(problem, hjb, fp, config=config)

        result = iterator.solve()
        U = result.U if hasattr(result, "U") else result[0]
        M = result.M if hasattr(result, "M") else result[1]
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(M))

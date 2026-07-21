"""
Pytest configuration and shared fixtures for MFGarchon test suite.

This module provides common fixtures, test configuration, and utilities
used across the entire test suite.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

import numpy as np

# Import main package components
from mfgarchon import MFGProblem
from mfgarchon.config import MFGSolverConfig
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.factory import lq_mfg_initial_density, lq_mfg_terminal_cost
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# =============================================================================
# Default Components for Testing (Issue #670, #673: explicit specification required)
# =============================================================================


def _default_hamiltonian():
    """Default class-based Hamiltonian for tests (Issue #673)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_test_components(Lx: float = 1.0) -> MFGComponents:
    """
    Default MFGComponents for shared test fixtures.

    Uses LQ MFG problem components for well-tested behavior.

    Args:
        Lx: Domain length for terminal cost scaling

    Returns:
        MFGComponents with Gaussian initial density and quadratic terminal cost
    """
    return MFGComponents(
        hamiltonian=_default_hamiltonian(),
        m_initial=lq_mfg_initial_density(),
        u_terminal=lq_mfg_terminal_cost(Lx=Lx),
    )


# =============================================================================
# Test Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest.

    Markers are declared in pytest.ini and only there. This function used to re-declare twelve of
    them with `addinivalue_line`, which made `pytest.ini` look like the source while the real
    registration happened here -- deleting a declaration from the ini removed nothing, and
    `scripts/check_markers.py` reported a clean census over seven markers that were still live
    (#1706).
    """


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        # Add markers based on test file paths
        test_path = str(item.fspath)

        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/performance/" in test_path:
            item.add_marker(pytest.mark.performance)
        elif "/mathematical/" in test_path:
            item.add_marker(pytest.mark.mathematical)

        # Mark slow tests based on name patterns
        if "large" in item.name or "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Problem Fixtures
# =============================================================================


@pytest.fixture
def tiny_problem():
    """Very small problem for quick tests."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[6], boundary_conditions=no_flux_bc(dimension=1)
    )  # Nx=5 -> 6 points
    return MFGProblem(
        geometry=geometry,
        Nt=3,
        T=0.1,
        components=_default_test_components(Lx=1.0),
    )


@pytest.fixture
def small_problem():
    """Small problem for unit tests."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)
    )  # Nx=10 -> 11 points
    return MFGProblem(
        geometry=geometry,
        Nt=5,
        T=0.5,
        components=_default_test_components(Lx=1.0),
    )


@pytest.fixture
def medium_problem():
    """Medium problem for integration tests."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[26], boundary_conditions=no_flux_bc(dimension=1)
    )  # Nx=25 -> 26 points
    return MFGProblem(
        geometry=geometry,
        Nt=12,
        T=1.0,
        components=_default_test_components(Lx=1.0),
    )


@pytest.fixture
def large_problem():
    """Large problem for performance tests."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1)
    )  # Nx=50 -> 51 points
    return MFGProblem(
        geometry=geometry,
        Nt=25,
        T=2.0,
        components=_default_test_components(Lx=1.0),
    )


@pytest.fixture(
    params=[
        {"Nx_points": 11, "Nt": 5, "T": 0.5},  # Nx=10 -> 11 points
        {"Nx_points": 16, "Nt": 8, "T": 1.0},  # Nx=15 -> 16 points
        {"Nx_points": 21, "Nt": 10, "T": 1.5},  # Nx=20 -> 21 points
    ]
)
def parametrized_problem(request):
    """Parametrized problem fixture for testing multiple configurations."""
    params = request.param
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[params["Nx_points"]], boundary_conditions=no_flux_bc(dimension=1)
    )
    return MFGProblem(
        geometry=geometry,
        Nt=params["Nt"],
        T=params["T"],
        components=_default_test_components(Lx=1.0),
    )


@pytest.fixture(params=[0.1, 0.5, 1.0, 2.0])
def diffusion_coefficient(request):
    """Parametrized diffusion coefficient for testing."""
    return request.param


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default MFGSolverConfig for testing."""
    return MFGSolverConfig()


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def deterministic_arrays():
    """Deterministic arrays for reproducible tests."""
    np.random.seed(42)  # Fixed seed for reproducibility
    # Note: shapes match small_problem (Nx=10→11 points) and medium_problem (Nx=25→26 points)
    return {
        "U_small": np.random.rand(6, 11),  # (Nt+1, Nx+1) for small problem
        "M_small": np.random.rand(6, 11),
        "U_medium": np.random.rand(13, 26),  # (Nt+1, Nx+1) for medium problem
        "M_medium": np.random.rand(13, 26),
    }


@pytest.fixture
def valid_test_matrices():
    """Valid test matrices with proper physical properties."""
    np.random.seed(123)

    # Create density matrix with mass conservation
    M = np.random.rand(11, 21)
    M = np.maximum(M, 0)  # Ensure non-negativity
    # Normalize each time slice to conserve mass
    for t in range(M.shape[0]):
        if np.sum(M[t, :]) > 0:
            M[t, :] /= np.sum(M[t, :])

    # Create value function matrix
    U = np.random.rand(11, 21) * 10 - 5  # Range [-5, 5]

    return {"U": U, "M": M}


@pytest.fixture
def boundary_conditions():
    """Standard boundary condition configurations."""
    return {
        "dirichlet": {"type": "dirichlet", "value": 0.0},
        "neumann": {"type": "neumann", "derivative": 0.0},
        "periodic": {"type": "periodic"},
        "mixed": {"type": "mixed", "left": "dirichlet", "right": "neumann"},
    }


# =============================================================================
# File System Fixtures
# =============================================================================


@pytest.fixture
def temp_directory():
    """Temporary directory for file operations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_output_dir(temp_directory):
    """Test output directory with subdirectories."""
    output_dir = temp_directory / "test_output"
    output_dir.mkdir()

    # Create subdirectories
    (output_dir / "results").mkdir()
    (output_dir / "plots").mkdir()
    (output_dir / "reports").mkdir()

    return output_dir


# =============================================================================
# Solver Fixtures
# =============================================================================


@pytest.fixture(params=["fixed_point"])
def solver_type(request):
    """Parametrized solver type for testing."""
    return request.param


@pytest.fixture
def solver_factory():
    """Factory function for creating solvers."""

    def _create_solver(problem, config=None):
        from mfgarchon.factory import create_solver

        return create_solver(problem, config=config)

    return _create_solver


# =============================================================================
# Mathematical Properties Fixtures
# =============================================================================


@pytest.fixture
def tolerance_levels():
    """Different tolerance levels for testing convergence."""
    return {"strict": 1e-8, "normal": 1e-6, "relaxed": 1e-4, "loose": 1e-2}


@pytest.fixture
def convergence_criteria():
    """Different convergence criteria configurations."""
    return {
        "standard": {"relative_tolerance": True, "absolute_tolerance": False},
        "absolute": {"relative_tolerance": False, "absolute_tolerance": True},
        "combined": {"relative_tolerance": True, "absolute_tolerance": True},
    }


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def performance_timer():
    """Timer utility for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()
            return self.elapsed

        @property
        def elapsed(self):
            if self.start_time is None:
                return 0
            end = self.end_time or time.time()
            return end - self.start_time

    return Timer()


@pytest.fixture
def memory_tracker():
    """Memory usage tracker for performance testing."""
    import psutil

    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process()
            self.start_memory = None
            self.peak_memory = 0

        def start(self):
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory

        def update(self):
            current_memory = self.process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current_memory)
            return current_memory

        @property
        def current_mb(self):
            return self.process.memory_info().rss / 1024 / 1024

        @property
        def increase_mb(self):
            if self.start_memory is None:
                return 0
            return self.current_mb - self.start_memory

    return MemoryTracker()


# =============================================================================
# Mock and Stub Fixtures
# =============================================================================


@pytest.fixture
def mock_convergence_result():
    """Mock convergence result for testing."""
    return {
        "converged": True,
        "iterations": 15,
        "final_error": 1e-7,
        "error_history": [1e-1, 3e-2, 8e-3, 2e-3, 5e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7],
        "convergence_rate": 0.3,
        "execution_time": 2.5,
    }


@pytest.fixture
def mock_failed_convergence():
    """Mock failed convergence result for testing error handling."""
    return {
        "converged": False,
        "iterations": 50,
        "final_error": 1e-3,
        "error_history": [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3],
        "convergence_rate": None,
        "execution_time": 10.0,
        "failure_reason": "Maximum iterations reached",
    }


# =============================================================================
# Parameterized Test Data
# =============================================================================


@pytest.fixture(
    params=[
        (10, 5, 0.5),  # Small problem
        (20, 10, 1.0),  # Medium problem
        (30, 15, 1.5),  # Large problem
    ]
)
def problem_dimensions(request):
    """Parametrized problem dimensions (Nx, Nt, T)."""
    return request.param


@pytest.fixture(
    params=[
        {"max_iterations": 10, "tolerance": 1e-4},
        {"max_iterations": 20, "tolerance": 1e-6},
        {"max_iterations": 50, "tolerance": 1e-8},
    ]
)
def newton_parameters(request):
    """Parametrized Newton solver parameters."""
    return request.param


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def reference_solutions():
    """Reference solutions for validation (computed once per session)."""
    # This would load or compute reference solutions
    # For now, return empty dict - implement as needed
    return {}


@pytest.fixture(scope="session")
def performance_baselines():
    """Performance baselines for regression testing."""
    return {
        "small_problem_time": 1.0,  # seconds
        "medium_problem_time": 5.0,  # seconds
        "memory_per_dof": 1e-6,  # MB per degree of freedom
    }


# =============================================================================
# Cleanup Utilities
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_numpy_state():
    """Automatically cleanup numpy random state after each test."""
    yield
    # Reset numpy random state to avoid test interference
    np.random.seed(None)


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress known warnings during testing."""
    import warnings

    # Suppress specific warnings that are expected during testing
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="mfgarchon.*")
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

    yield

    # Reset warning filters
    warnings.resetwarnings()


# =============================================================================
# Test Data Validation
# =============================================================================


def validate_mfg_solution(U, M, problem):
    """Validate that U and M arrays represent a valid MFG solution."""
    # Check dimensions
    Nx_points = problem.geometry.get_grid_shape()[0]
    expected_shape = (problem.Nt + 1, Nx_points)
    assert U.shape == expected_shape, f"U shape {U.shape} != expected {expected_shape}"
    assert M.shape == expected_shape, f"M shape {M.shape} != expected {expected_shape}"

    # Check for NaN/Inf
    assert not np.any(np.isnan(U)), "U contains NaN values"
    assert not np.any(np.isnan(M)), "M contains NaN values"
    assert not np.any(np.isinf(U)), "U contains Inf values"
    assert not np.any(np.isinf(M)), "M contains Inf values"

    # Check physical properties
    assert np.all(M >= -1e-10), f"M contains negative values: min={np.min(M)}"

    # Check mass conservation (approximately)
    dx = problem.geometry.get_grid_spacing()[0]
    initial_mass = np.sum(problem.m_initial) * dx  # Issue #670: unified naming
    for t in range(problem.Nt + 1):
        current_mass = np.sum(M[t, :]) * dx
        mass_error = abs(current_mass - initial_mass)
        assert mass_error < 0.1, f"Mass conservation violated at t={t}: error={mass_error}"


# Export validation function for use in tests
__all__ = ["validate_mfg_solution"]

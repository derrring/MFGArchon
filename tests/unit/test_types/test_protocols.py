#!/usr/bin/env python3
"""
Unit tests for mfgarchon/types/protocols.py

Tests Protocol interfaces including:
- MFGProblem (runtime-checkable protocol)
- MFGSolver (runtime-checkable protocol)
- MFGResult (runtime-checkable protocol)
- SolverConfig (runtime-checkable protocol)
- Protocol compliance using isinstance() checks
- Duck typing behavior
"""

import pytest

import numpy as np

from mfgarchon.types.protocols import (
    MFGProblem,
    MFGResult,
    MFGSolver,
    SolverConfig,
)

# ===================================================================
# Test MFGProblem Protocol
# ===================================================================


@pytest.mark.unit
def test_mfg_problem_protocol_compliance():
    """Test MFGProblem protocol with compliant class."""

    class SimpleProblem:
        def get_domain_bounds(self) -> tuple[float, float]:
            return (0.0, 1.0)

        def get_time_horizon(self) -> float:
            return 1.0

        def evaluate_hamiltonian(self, x: float, p: float, m: float, t: float) -> float:
            return 0.5 * p**2

        def get_initial_density(self):
            return np.ones(51) / 51

        def get_initial_value_function(self):
            return np.zeros(51)

    problem = SimpleProblem()

    # Protocol is runtime-checkable
    assert isinstance(problem, MFGProblem)


@pytest.mark.unit
def test_mfg_problem_protocol_duck_typing():
    """Test MFGProblem protocol with duck-typed object."""

    class DuckTypedProblem:
        """No explicit protocol inheritance, but implements interface."""

        def get_domain_bounds(self):
            return (-1.0, 1.0)

        def get_time_horizon(self):
            return 2.0

        def evaluate_hamiltonian(self, x, p, m, t):
            return p**2 + 0.1 * m

        def get_initial_density(self):
            return np.exp(-(np.linspace(-1, 1, 101) ** 2))

        def get_initial_value_function(self):
            return np.zeros(101)

    problem = DuckTypedProblem()

    # Duck typing works with runtime_checkable protocols
    assert isinstance(problem, MFGProblem)
    assert problem.get_domain_bounds() == (-1.0, 1.0)
    assert problem.get_time_horizon() == 2.0


@pytest.mark.unit
def test_mfg_problem_protocol_incomplete():
    """Test MFGProblem protocol with incomplete implementation."""

    class IncompleteProblem:
        def get_domain_bounds(self):
            return (0.0, 1.0)

        # Missing other methods

    problem = IncompleteProblem()

    # Incomplete implementation does not match protocol
    assert not isinstance(problem, MFGProblem)


@pytest.mark.unit
def test_mfg_problem_protocol_methods():
    """Test MFGProblem protocol method signatures."""

    class TestProblem:
        def get_domain_bounds(self):
            return (0.0, 1.0)

        def get_time_horizon(self):
            return 1.0

        def evaluate_hamiltonian(self, x, p, m, t):
            # Test with actual computation
            return 0.5 * p**2 + 0.1 * x * m

        def get_initial_density(self):
            # Gaussian density
            x = np.linspace(0, 1, 51)
            m0 = np.exp(-((x - 0.5) ** 2) / 0.1)
            return m0 / np.sum(m0)

        def get_initial_value_function(self):
            # Terminal cost
            x = np.linspace(0, 1, 51)
            return (x - 0.5) ** 2

    problem = TestProblem()
    assert isinstance(problem, MFGProblem)

    # Test method calls
    xmin, xmax = problem.get_domain_bounds()
    assert xmin < xmax
    assert problem.get_time_horizon() > 0
    h = problem.evaluate_hamiltonian(0.5, 1.0, 0.5, 0.5)
    assert isinstance(h, float)
    m0 = problem.get_initial_density()
    assert m0.shape == (51,)
    u0 = problem.get_initial_value_function()
    assert u0.shape == (51,)


# ===================================================================
# Test MFGSolver Protocol
# ===================================================================


@pytest.mark.unit
def test_mfg_solver_protocol_compliance():
    """Test MFGSolver protocol with compliant class."""

    class DummyResult:
        @property
        def u(self):
            return np.zeros((31, 51))

        @property
        def m(self):
            return np.ones((31, 51)) / 51

        @property
        def converged(self):
            return True

        @property
        def iterations(self):
            return 10

        def plot_solution(self, **kwargs):
            pass

        def export_data(self, filename):
            pass

    class SimpleSolver:
        def solve(self, problem, **kwargs):
            return DummyResult()

    solver = SimpleSolver()

    # Protocol is runtime-checkable
    assert isinstance(solver, MFGSolver)


@pytest.mark.unit
def test_mfg_solver_protocol_duck_typing():
    """Test MFGSolver protocol with duck-typed solver."""

    class DuckTypedSolver:
        """No explicit protocol inheritance."""

        def solve(self, problem, **kwargs):
            # Return None for testing
            return None

    solver = DuckTypedSolver()

    # Duck typing works
    assert isinstance(solver, MFGSolver)


@pytest.mark.unit
def test_mfg_solver_protocol_solve_method():
    """Test MFGSolver protocol solve method."""

    class MockProblem:
        def get_domain_bounds(self):
            return (0.0, 1.0)

        def get_time_horizon(self):
            return 1.0

        def evaluate_hamiltonian(self, x, p, m, t):
            return 0.5 * p**2

        def get_initial_density(self):
            return np.ones(51) / 51

        def get_initial_value_function(self):
            return np.zeros(51)

    class MockResult:
        @property
        def u(self):
            return np.zeros((31, 51))

        @property
        def m(self):
            return np.ones((31, 51)) / 51

        @property
        def converged(self):
            return True

        @property
        def iterations(self):
            return 5

        def plot_solution(self, **kwargs):
            pass

        def export_data(self, filename):
            pass

    class TestSolver:
        def solve(self, problem, **kwargs):
            # Accept kwargs (unused in this mock)
            return MockResult()

    solver = TestSolver()
    problem = MockProblem()

    assert isinstance(solver, MFGSolver)
    assert isinstance(problem, MFGProblem)

    # Solve with kwargs
    result = solver.solve(problem, max_iterations=50, tolerance=1e-6)
    assert result is not None


@pytest.mark.unit
def test_mfg_solver_protocol_incomplete():
    """Test MFGSolver protocol with incomplete implementation."""

    class IncompleteSolver:
        def other_method(self):
            pass

        # Missing solve method

    solver = IncompleteSolver()

    # Incomplete implementation
    assert not isinstance(solver, MFGSolver)


# ===================================================================
# Test MFGResult Protocol
# ===================================================================


@pytest.mark.unit
def test_mfg_result_protocol_compliance():
    """Test MFGResult protocol with compliant class."""

    class SimpleResult:
        @property
        def u(self):
            return np.zeros((31, 51))

        @property
        def m(self):
            return np.ones((31, 51))

        @property
        def converged(self):
            return True

        @property
        def iterations(self):
            return 10

        def plot_solution(self, **kwargs):
            pass

        def export_data(self, filename):
            pass

    result = SimpleResult()

    # Protocol is runtime-checkable
    assert isinstance(result, MFGResult)


@pytest.mark.unit
def test_mfg_result_protocol_properties():
    """Test MFGResult protocol properties."""

    class TestResult:
        def __init__(self):
            self._u = np.random.rand(31, 51)
            self._m = np.random.rand(31, 51)
            self._converged = True
            self._iterations = 42

        @property
        def u(self):
            return self._u

        @property
        def m(self):
            return self._m

        @property
        def converged(self):
            return self._converged

        @property
        def iterations(self):
            return self._iterations

        def plot_solution(self, **kwargs):
            # Mock implementation
            dpi = kwargs.get("dpi", 150)
            return dpi

        def export_data(self, filename):
            # Mock implementation
            return filename

    result = TestResult()
    assert isinstance(result, MFGResult)

    # Test property access
    assert result.u.shape == (31, 51)
    assert result.m.shape == (31, 51)
    assert result.converged is True
    assert result.iterations == 42

    # Test method calls
    dpi = result.plot_solution(dpi=300)
    assert dpi == 300
    fname = result.export_data("output.npz")
    assert fname == "output.npz"


@pytest.mark.unit
def test_mfg_result_protocol_not_converged():
    """Test MFGResult protocol with non-converged result."""

    class NotConvergedResult:
        @property
        def u(self):
            return np.zeros((31, 51))

        @property
        def m(self):
            return np.ones((31, 51))

        @property
        def converged(self):
            return False

        @property
        def iterations(self):
            return 100

        def plot_solution(self, **kwargs):
            pass

        def export_data(self, filename):
            pass

    result = NotConvergedResult()
    assert isinstance(result, MFGResult)
    assert result.converged is False
    assert result.iterations == 100


@pytest.mark.unit
def test_mfg_result_protocol_incomplete():
    """Test MFGResult protocol with incomplete implementation."""

    class IncompleteResult:
        @property
        def u(self):
            return np.zeros((31, 51))

        # Missing other properties and methods

    result = IncompleteResult()

    # Incomplete implementation
    assert not isinstance(result, MFGResult)


# ===================================================================
# Test SolverConfig Protocol
# ===================================================================


@pytest.mark.unit
def test_solver_config_protocol_compliance():
    """Test SolverConfig protocol with compliant class."""

    class SimpleConfig:
        @property
        def max_iterations(self):
            return 100

        @property
        def tolerance(self):
            return 1e-6

        def get_parameter(self, name, default=None):
            params = {"damping": 0.5, "verbose": True}
            return params.get(name, default)

    config = SimpleConfig()

    # Protocol is runtime-checkable
    assert isinstance(config, SolverConfig)


@pytest.mark.unit
def test_solver_config_protocol_properties():
    """Test SolverConfig protocol properties."""

    class TestConfig:
        def __init__(self):
            self._max_iterations = 200
            self._tolerance = 1e-8
            self._params = {"learning_rate": 0.01, "batch_size": 32}

        @property
        def max_iterations(self):
            return self._max_iterations

        @property
        def tolerance(self):
            return self._tolerance

        def get_parameter(self, name, default=None):
            return self._params.get(name, default)

    config = TestConfig()
    assert isinstance(config, SolverConfig)

    # Test properties
    assert config.max_iterations == 200
    assert config.tolerance == 1e-8

    # Test parameter access
    assert config.get_parameter("learning_rate") == 0.01
    assert config.get_parameter("batch_size") == 32
    assert config.get_parameter("unknown", default=42) == 42


@pytest.mark.unit
def test_solver_config_protocol_get_parameter_default():
    """Test SolverConfig protocol get_parameter with defaults."""

    class MinimalConfig:
        @property
        def max_iterations(self):
            return 50

        @property
        def tolerance(self):
            return 1e-5

        def get_parameter(self, name, default=None):
            # Always return default for testing
            return default

    config = MinimalConfig()
    assert isinstance(config, SolverConfig)

    # Test default values
    assert config.get_parameter("nonexistent") is None
    assert config.get_parameter("nonexistent", default=123) == 123
    assert config.get_parameter("nonexistent", default="value") == "value"


@pytest.mark.unit
def test_solver_config_protocol_incomplete():
    """Test SolverConfig protocol with incomplete implementation."""

    class IncompleteConfig:
        @property
        def max_iterations(self):
            return 100

        # Missing tolerance and get_parameter

    config = IncompleteConfig()

    # Incomplete implementation
    assert not isinstance(config, SolverConfig)


# ===================================================================
# Test Protocol Integration
# ===================================================================


@pytest.mark.unit
def test_protocol_integration_full_workflow():
    """Test integration of all protocols in a workflow."""

    # Define compliant classes
    class IntegrationProblem:
        def get_domain_bounds(self):
            return (0.0, 1.0)

        def get_time_horizon(self):
            return 1.0

        def evaluate_hamiltonian(self, x, p, m, t):
            return 0.5 * p**2

        def get_initial_density(self):
            return np.ones(51) / 51

        def get_initial_value_function(self):
            return np.zeros(51)

    class IntegrationConfig:
        @property
        def max_iterations(self):
            return 100

        @property
        def tolerance(self):
            return 1e-6

        def get_parameter(self, name, default=None):
            return default

    class IntegrationResult:
        @property
        def u(self):
            return np.random.rand(31, 51)

        @property
        def m(self):
            return np.random.rand(31, 51)

        @property
        def converged(self):
            return True

        @property
        def iterations(self):
            return 25

        def plot_solution(self, **kwargs):
            pass

        def export_data(self, filename):
            pass

    class IntegrationSolver:
        def solve(self, problem, **kwargs):
            return IntegrationResult()

    # Create instances
    problem = IntegrationProblem()
    config = IntegrationConfig()
    solver = IntegrationSolver()

    # Verify protocol compliance
    assert isinstance(problem, MFGProblem)
    assert isinstance(config, SolverConfig)
    assert isinstance(solver, MFGSolver)

    # Execute workflow
    result = solver.solve(
        problem,
        max_iterations=config.max_iterations,
        tolerance=config.tolerance,
    )

    # Verify result
    assert isinstance(result, MFGResult)
    assert result.converged is True
    assert result.iterations <= config.max_iterations


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_docstring():
    """Test module has comprehensive docstring."""
    from mfgarchon.types import protocols

    assert protocols.__doc__ is not None
    assert "Core Type Protocols" in protocols.__doc__

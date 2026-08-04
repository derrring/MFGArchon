#!/usr/bin/env python3
"""
Unit tests for mfgarchon/alg/numerical/fp_solvers/base_fp.py

Tests BaseFPSolver abstract base class including:
- Abstract base class structure and instantiation
- Initialization with MFG problem
- Abstract method enforcement (solve_fp_system)
- Attribute initialization (problem, fp_method_name, backend)
- Concrete subclass implementation patterns
- Integration with MFGProblem types
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


# ===================================================================
# Mock Problem for Testing
# ===================================================================


class MockMFGProblem(MFGProblem):
    """Minimal mock MFG problem for testing."""

    def __init__(self):
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[101],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        super().__init__(
            geometry=geometry,
            T=1.0,
            Nt=100,
            sigma=0.1,
            components=_default_components(),
        )
        self.dim = 1
        self.dx = 0.01


# ===================================================================
# Test Abstract Base Class Structure
# ===================================================================


@pytest.mark.unit
def test_base_fp_solver_is_abstract():
    """Test BaseFPSolver cannot be instantiated directly."""
    problem = MockMFGProblem()

    with pytest.raises(TypeError) as exc_info:
        BaseFPSolver(problem)

    assert "abstract" in str(exc_info.value).lower()
    assert "solve_fp_system" in str(exc_info.value)


@pytest.mark.unit
def test_base_fp_solver_requires_solve_fp_system():
    """Test BaseFPSolver requires solve_fp_system implementation."""

    # Incomplete implementation - missing solve_fp_system
    class IncompleteFPSolver(BaseFPSolver):
        pass

    problem = MockMFGProblem()

    with pytest.raises(TypeError) as exc_info:
        IncompleteFPSolver(problem)

    assert "solve_fp_system" in str(exc_info.value)


@pytest.mark.unit
def test_base_fp_solver_has_abstract_method():
    """Test solve_fp_system is marked as abstract method."""
    assert hasattr(BaseFPSolver, "solve_fp_system")
    assert hasattr(BaseFPSolver.solve_fp_system, "__isabstractmethod__")
    assert BaseFPSolver.solve_fp_system.__isabstractmethod__


# ===================================================================
# Test Concrete Subclass Implementation
# ===================================================================


class ConcreteFPSolver(BaseFPSolver):
    """Minimal concrete FP solver for testing."""

    def __init__(self, problem: MFGProblem):
        super().__init__(problem)
        self.fp_method_name = "ConcreteFP"

    def solve_fp_system(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """Minimal implementation returning zeros."""
        Nt, Nx = U_solution_for_drift.shape
        return np.zeros((Nt, Nx))


@pytest.mark.unit
def test_concrete_fp_solver_instantiation():
    """Test concrete FP solver can be instantiated."""
    problem = MockMFGProblem()
    solver = ConcreteFPSolver(problem)

    assert isinstance(solver, BaseFPSolver)
    assert isinstance(solver, ConcreteFPSolver)


@pytest.mark.unit
def test_concrete_fp_solver_inherits_base_methods():
    """Test concrete solver inherits base class methods."""
    problem = MockMFGProblem()
    solver = ConcreteFPSolver(problem)

    # Has solve_fp_system method
    assert hasattr(solver, "solve_fp_system")
    assert callable(solver.solve_fp_system)


# ===================================================================
# Test Initialization
# ===================================================================


@pytest.mark.unit
def test_base_fp_solver_init_with_problem():
    """Test BaseFPSolver initialization stores problem."""
    problem = MockMFGProblem()
    solver = ConcreteFPSolver(problem)

    assert solver.problem is problem
    assert isinstance(solver.problem, MFGProblem)


@pytest.mark.unit
def test_base_fp_solver_init_default_attributes():
    """Test BaseFPSolver initialization sets all expected attributes."""
    problem = MockMFGProblem()

    # Use base class __init__ through concrete subclass
    class AttributeTestSolver(BaseFPSolver):
        def solve_fp_system(self, m_initial_condition, U_solution_for_drift):
            return np.array([])

    solver = AttributeTestSolver(problem)

    assert hasattr(solver, "problem")
    assert hasattr(solver, "fp_method_name")
    assert hasattr(solver, "backend")
    assert solver.fp_method_name == "BaseFP"  # Default from base class


# ===================================================================
# Test solve_fp_system Method Signature
# ===================================================================


@pytest.mark.unit
def test_solve_fp_system_signature():
    """Test solve_fp_system has correct signature."""
    problem = MockMFGProblem()
    solver = ConcreteFPSolver(problem)

    # Should accept two arrays
    m_init = np.ones(10)
    U_solution = np.ones((20, 10))

    result = solver.solve_fp_system(m_init, U_solution)
    assert isinstance(result, np.ndarray)


@pytest.mark.unit
def test_solve_fp_system_returns_ndarray():
    """Test solve_fp_system returns numpy array."""
    problem = MockMFGProblem()
    solver = ConcreteFPSolver(problem)

    m_init = np.ones(5)
    U_solution = np.ones((10, 5))

    result = solver.solve_fp_system(m_init, U_solution)
    assert isinstance(result, np.ndarray)


@pytest.mark.unit
def test_solve_fp_system_shape_matches_input():
    """Test solve_fp_system returns array with correct shape."""
    problem = MockMFGProblem()
    solver = ConcreteFPSolver(problem)

    Nt, Nx = 20, 10
    m_init = np.ones(Nx)
    U_solution = np.ones((Nt, Nx))

    result = solver.solve_fp_system(m_init, U_solution)
    assert result.shape == (Nt, Nx)


# ===================================================================
# Test Custom Concrete Implementations
# ===================================================================


class CustomFPSolver(BaseFPSolver):
    """Custom FP solver that returns input for testing."""

    def __init__(self, problem: MFGProblem, method_name: str = "CustomFP"):
        super().__init__(problem)
        self.fp_method_name = method_name

    def solve_fp_system(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """Return U_solution unchanged for testing."""
        return U_solution_for_drift


@pytest.mark.unit
def test_custom_fp_solver_with_method_name():
    """Test custom FP solver with specified method name."""
    problem = MockMFGProblem()
    solver = CustomFPSolver(problem, method_name="TestMethod")

    assert solver.fp_method_name == "TestMethod"


@pytest.mark.unit
def test_custom_fp_solver_solve_implementation():
    """Test custom FP solver solve implementation."""
    problem = MockMFGProblem()
    solver = CustomFPSolver(problem)

    m_init = np.ones(5)
    U_solution = np.random.rand(10, 5)

    result = solver.solve_fp_system(m_init, U_solution)
    assert np.array_equal(result, U_solution)


# ===================================================================
# Test Backend Attribute
# ===================================================================


class BackendTestSolver(BaseFPSolver):
    """FP solver that sets backend for testing."""

    def __init__(self, problem: MFGProblem, backend: str | None = None):
        super().__init__(problem)
        self.backend = backend

    def solve_fp_system(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        return np.zeros(U_solution_for_drift.shape)


@pytest.mark.unit
def test_backend_can_be_set():
    """Test backend attribute can be set in subclass."""
    problem = MockMFGProblem()
    solver = BackendTestSolver(problem, backend="numpy")

    assert solver.backend == "numpy"


@pytest.mark.unit
def test_backend_default_none():
    """Test backend defaults to None."""
    problem = MockMFGProblem()
    solver = BackendTestSolver(problem)

    assert solver.backend is None


@pytest.mark.unit
def test_backend_can_be_modified():
    """Test backend can be modified after initialization."""
    problem = MockMFGProblem()
    solver = BackendTestSolver(problem)

    solver.backend = "pytorch"
    assert solver.backend == "pytorch"


# ===================================================================
# Test Problem Access
# ===================================================================


@pytest.mark.unit
def test_solver_accesses_problem_attributes():
    """Test solver can access problem attributes."""
    problem = MockMFGProblem()
    solver = ConcreteFPSolver(problem)

    assert solver.problem.dim == 1
    assert solver.problem.T == 1.0
    assert solver.problem.dt == 0.01


@pytest.mark.unit
def test_solver_problem_is_same_instance():
    """Test solver stores same problem instance."""
    problem = MockMFGProblem()
    solver = ConcreteFPSolver(problem)

    assert solver.problem is problem


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_base_fp_solver_has_docstring():
    """Test BaseFPSolver has comprehensive docstring."""
    assert BaseFPSolver.__doc__ is not None
    assert "Fokker-Planck" in BaseFPSolver.__doc__
    assert "FP equation" in BaseFPSolver.__doc__


@pytest.mark.unit
def test_solve_fp_system_has_docstring():
    """Test solve_fp_system has comprehensive docstring."""
    assert BaseFPSolver.solve_fp_system.__doc__ is not None
    assert "density" in BaseFPSolver.solve_fp_system.__doc__
    assert "drift" in BaseFPSolver.solve_fp_system.__doc__


# ===================================================================
# Test Multiple Inheritance Scenarios
# ===================================================================


class MixinClass:
    """Mixin class for testing multiple inheritance."""

    def extra_method(self):
        return "mixin"


class MixedFPSolver(MixinClass, BaseFPSolver):
    """FP solver with mixin for testing."""

    def solve_fp_system(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        return np.zeros(U_solution_for_drift.shape)


@pytest.mark.unit
def test_multiple_inheritance_with_base_fp():
    """Test BaseFPSolver works with multiple inheritance."""
    problem = MockMFGProblem()
    solver = MixedFPSolver(problem)

    assert isinstance(solver, BaseFPSolver)
    assert isinstance(solver, MixinClass)
    assert solver.extra_method() == "mixin"


# ===================================================================
# Test Edge Cases
# ===================================================================


@pytest.mark.unit
def test_solve_fp_system_with_empty_arrays():
    """Test solve_fp_system handles empty arrays."""
    problem = MockMFGProblem()
    solver = ConcreteFPSolver(problem)

    m_init = np.array([])
    U_solution = np.empty((0, 0))

    result = solver.solve_fp_system(m_init, U_solution)
    assert result.shape == (0, 0)


@pytest.mark.unit
def test_solve_fp_system_with_large_arrays():
    """Test solve_fp_system handles large arrays."""
    problem = MockMFGProblem()
    solver = ConcreteFPSolver(problem)

    Nt, Nx = 1000, 500
    m_init = np.ones(Nx)
    U_solution = np.ones((Nt, Nx))

    result = solver.solve_fp_system(m_init, U_solution)
    assert result.shape == (Nt, Nx)

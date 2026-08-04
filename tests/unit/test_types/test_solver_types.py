#!/usr/bin/env python3
"""
Unit tests for mfgarchon/types/solver_types.py

Tests solver type definitions including:
- Solver return types (SolverReturnTuple, JAXSolverReturn)
- Solver state types (SolverState, ComplexSolverState, IntermediateResult)
- Configuration types (ParameterDict, SolverOptions, ConfigValue)
- Callback types (ErrorCallback, ProgressCallback, ConvergenceCallback)
- GFDM types (MultiIndexTuple, DerivativeDict, GradientDict, StencilResult)
- Metadata types (MetadataDict, ConvergenceMetadata)
- Mathematical function types (HamiltonianFunction, LagrangianFunction, etc.)
- Solver protocols (NewtonSolver, LinearSolver)
- Exception classes (SolverError, ConvergenceError, ConfigurationError)
"""

import pytest

import numpy as np

from mfgarchon.types.solver_types import (
    ConfigurationError,
    ConvergenceError,
    SolverError,
)

# ===================================================================
# Test Solver Return Types
# ===================================================================


@pytest.mark.unit
def test_solver_return_tuple_type():
    """Test SolverReturnTuple type alias usage."""
    from mfgarchon.types.solver_types import SolverReturnTuple

    # Typical solver return
    U = np.zeros((31, 51))
    M = np.ones((31, 51)) / 51
    info = {"converged": True, "iterations": 10, "residual": 1e-6}

    result: SolverReturnTuple = (U, M, info)

    assert len(result) == 3
    assert result[0].shape == (31, 51)
    assert result[1].shape == (31, 51)
    assert isinstance(result[2], dict)
    assert result[2]["converged"] is True


@pytest.mark.unit
def test_jax_solver_return_type():
    """Test JAXSolverReturn type alias usage."""
    from mfgarchon.types.solver_types import JAXSolverReturn

    # JAX solver return format (using numpy arrays as mock)
    U_jax = np.zeros((31, 51))
    M_jax = np.ones((31, 51))
    converged = True
    iterations = 42
    residual = 1e-8

    result: JAXSolverReturn = (U_jax, M_jax, converged, iterations, residual)

    assert len(result) == 5
    assert result[2] is True  # converged
    assert result[3] == 42  # iterations
    assert result[4] == 1e-8  # residual


# ===================================================================
# Test Solver State Types
# ===================================================================


@pytest.mark.unit
def test_solver_state_tuple():
    """Test SolverState as tuple."""
    from mfgarchon.types.solver_types import SolverState

    u = np.random.rand(31, 51)
    m = np.random.rand(31, 51)

    state: SolverState = (u, m)

    assert isinstance(state, tuple)
    assert len(state) == 2


@pytest.mark.unit
def test_solver_state_dict():
    """Test SolverState as dict."""
    from mfgarchon.types.solver_types import SolverState

    state: SolverState = {
        "u": np.zeros((31, 51)),
        "m": np.ones((31, 51)),
        "iteration": 5,
        "residual": 1e-5,
    }

    assert isinstance(state, dict)
    assert "u" in state
    assert "m" in state


@pytest.mark.unit
def test_complex_solver_state():
    """Test ComplexSolverState type."""
    from mfgarchon.types.solver_types import ComplexSolverState

    # Simple tuple state
    state1: ComplexSolverState = (np.zeros((31, 51)), np.ones((31, 51)))
    assert isinstance(state1, tuple)

    # Dict state with metadata
    state2: ComplexSolverState = {
        "u": np.zeros((31, 51)),
        "m": np.ones((31, 51)),
        "residual": 1e-6,
        "iteration": 10,
    }
    assert isinstance(state2, dict)

    # Custom object state
    class CustomState:
        def __init__(self):
            self.u = np.zeros((31, 51))
            self.m = np.ones((31, 51))

    state3: ComplexSolverState = CustomState()
    assert hasattr(state3, "u")


@pytest.mark.unit
def test_intermediate_result_types():
    """Test IntermediateResult type variations."""
    from mfgarchon.types.solver_types import IntermediateResult

    # Simple array
    result1: IntermediateResult = np.random.rand(31, 51)
    assert isinstance(result1, np.ndarray)

    # Tuple of arrays
    result2: IntermediateResult = (np.zeros((31, 51)), np.ones((31, 51)))
    assert isinstance(result2, tuple)

    # Dict of arrays
    result3: IntermediateResult = {
        "u_new": np.zeros((31, 51)),
        "m_new": np.ones((31, 51)),
    }
    assert isinstance(result3, dict)


# ===================================================================
# Test Configuration Types
# ===================================================================


@pytest.mark.unit
def test_parameter_dict():
    """Test ParameterDict type."""
    from mfgarchon.types.solver_types import ParameterDict

    params: ParameterDict = {
        "dt": 0.01,
        "dx": 0.02,
        "max_iterations": 100,
        "method": "newton",
        "use_line_search": True,
    }

    assert isinstance(params["dt"], float)
    assert isinstance(params["max_iterations"], int)
    assert isinstance(params["method"], str)
    assert isinstance(params["use_line_search"], bool)


@pytest.mark.unit
def test_solver_options():
    """Test SolverOptions type with None values."""
    from mfgarchon.types.solver_types import SolverOptions

    options: SolverOptions = {
        "tolerance": 1e-6,
        "max_iter": 100,
        "verbose": True,
        "callback": None,  # Optional callback
        "preconditioner": None,
    }

    assert options["callback"] is None
    assert options["tolerance"] == 1e-6


@pytest.mark.unit
def test_config_value_flexible():
    """Test ConfigValue flexible type."""
    from mfgarchon.types.solver_types import ConfigValue

    # Various value types
    val1: ConfigValue = 3.14  # float
    val2: ConfigValue = 42  # int
    val3: ConfigValue = "method_name"  # str
    val4: ConfigValue = True  # bool
    val5: ConfigValue = None  # None
    val6: ConfigValue = [1.0, 2.0, 3.0]  # list
    val7: ConfigValue = {"nested": "config"}  # dict

    def square_func(x):
        return x**2

    val8: ConfigValue = square_func  # Callable

    assert isinstance(val1, float)
    assert isinstance(val2, int)
    assert isinstance(val3, str)
    assert isinstance(val4, bool)
    assert val5 is None
    assert isinstance(val6, list)
    assert isinstance(val7, dict)
    assert callable(val8)


# ===================================================================
# Test Callback Types
# ===================================================================


@pytest.mark.unit
def test_error_callback():
    """Test ErrorCallback type."""
    from mfgarchon.types.solver_types import ErrorCallback

    def error_handler(exc: Exception) -> None:
        print(f"Error: {exc}")

    callback: ErrorCallback = error_handler
    assert callable(callback)

    # None is also valid
    no_callback: ErrorCallback = None
    assert no_callback is None


@pytest.mark.unit
def test_progress_callback():
    """Test ProgressCallback type."""
    from mfgarchon.types.solver_types import ProgressCallback

    progress_data = []

    def progress_handler(iteration: int, residual: float) -> None:
        progress_data.append((iteration, residual))

    callback: ProgressCallback = progress_handler
    callback(1, 1e-3)
    callback(2, 1e-4)

    assert len(progress_data) == 2
    assert progress_data[0] == (1, 1e-3)


@pytest.mark.unit
def test_convergence_callback():
    """Test ConvergenceCallback type."""
    from mfgarchon.types.solver_types import ConvergenceCallback

    result_data = {}

    def convergence_handler(converged: bool, iterations: int, residual: float) -> None:
        result_data["converged"] = converged
        result_data["iterations"] = iterations
        result_data["residual"] = residual

    callback: ConvergenceCallback = convergence_handler
    callback(True, 50, 1e-8)

    assert result_data["converged"] is True
    assert result_data["iterations"] == 50


# ===================================================================
# Test GFDM-Specific Types
# ===================================================================


@pytest.mark.unit
def test_multi_index_tuple():
    """Test MultiIndexTuple type."""
    from mfgarchon.types.solver_types import MultiIndexTuple

    # Second derivative in x: ∂²/∂x²
    index1: MultiIndexTuple = (2, 0)
    assert len(index1) == 2

    # Mixed derivative: ∂²/∂x∂y
    index2: MultiIndexTuple = (1, 1)
    assert len(index2) == 2

    # Higher dimensions
    index3: MultiIndexTuple = (2, 1, 0)
    assert len(index3) == 3


@pytest.mark.unit
def test_derivative_dict():
    """Test DerivativeDict type."""
    from mfgarchon.types.solver_types import DerivativeDict

    derivatives: DerivativeDict = {
        (2, 0): 0.5,  # ∂²/∂x²
        (0, 2): 0.3,  # ∂²/∂y²
        (1, 1): -0.1,  # ∂²/∂x∂y
    }

    assert derivatives[(2, 0)] == 0.5
    assert len(derivatives) == 3


@pytest.mark.unit
def test_gradient_dict():
    """Test GradientDict type."""
    from mfgarchon.types.solver_types import GradientDict

    gradient: GradientDict = {"dx": 1.5, "dy": -0.8, "dz": 0.3}

    assert gradient["dx"] == 1.5
    assert gradient["dy"] == -0.8
    assert "dz" in gradient


@pytest.mark.unit
def test_stencil_result():
    """Test StencilResult type."""
    from mfgarchon.types.solver_types import StencilResult

    stencil1 = np.array([1.0, -2.0, 1.0])
    stencil2 = np.array([0.5, 0.0, -0.5])

    result: StencilResult = [
        (stencil1, True),  # Success
        (stencil2, False),  # Failure
    ]

    assert len(result) == 2
    assert result[0][1] is True  # First stencil succeeded
    assert result[1][1] is False  # Second stencil failed


# ===================================================================
# Test Metadata Types
# ===================================================================


@pytest.mark.unit
def test_metadata_dict():
    """Test MetadataDict type."""
    from mfgarchon.types.solver_types import MetadataDict

    metadata: MetadataDict = {
        "dt": 0.01,
        "iterations": 50,
        "method": "newton",
        "converged": True,
        "solution": np.zeros((31, 51)),
        "optional_field": None,
    }

    assert metadata["dt"] == 0.01
    assert metadata["converged"] is True
    assert metadata["optional_field"] is None


@pytest.mark.unit
def test_convergence_metadata():
    """Test ConvergenceMetadata type."""
    from mfgarchon.types.solver_types import ConvergenceMetadata

    conv_info: ConvergenceMetadata = {
        "converged": True,
        "iterations": 42,
        "residual": 1e-8,
        "reason": "tolerance_met",
        "residual_history": [1e-2, 1e-4, 1e-6, 1e-8],
    }

    assert conv_info["converged"] is True
    assert len(conv_info["residual_history"]) == 4


# ===================================================================
# Test Mathematical Function Types
# ===================================================================


@pytest.mark.unit
def test_hamiltonian_function():
    """Test HamiltonianFunction type."""
    from mfgarchon.types.solver_types import HamiltonianFunction

    def quadratic_hamiltonian(x: float, p: float, m: float, t: float) -> float:
        return 0.5 * p**2 + 0.1 * m

    H: HamiltonianFunction = quadratic_hamiltonian

    result = H(0.5, 1.0, 0.8, 0.0)
    assert isinstance(result, float)
    assert result == 0.5 * 1.0**2 + 0.1 * 0.8


@pytest.mark.unit
def test_lagrangian_function():
    """Test LagrangianFunction type."""
    from mfgarchon.types.solver_types import LagrangianFunction

    def quadratic_lagrangian(x: float, v: float, m: float, t: float) -> float:
        return 0.5 * v**2

    L: LagrangianFunction = quadratic_lagrangian

    result = L(0.5, 2.0, 0.8, 0.0)
    assert result == 2.0


@pytest.mark.unit
def test_density_function():
    """Test DensityFunction type."""
    from mfgarchon.types.solver_types import DensityFunction

    def gaussian_density(x: float) -> float:
        return np.exp(-((x - 0.5) ** 2) / 0.1)

    rho: DensityFunction = gaussian_density

    result = rho(0.5)
    assert result == 1.0  # Peak at center


@pytest.mark.unit
def test_value_function():
    """Test ValueFunction type."""
    from mfgarchon.types.solver_types import ValueFunction

    def quadratic_value(x: float) -> float:
        return x**2

    g: ValueFunction = quadratic_value

    assert g(2.0) == 4.0
    assert g(3.0) == 9.0


# ===================================================================
# Test Solver Protocols
# ===================================================================


@pytest.mark.unit
def test_newton_solver_protocol():
    """Test NewtonSolver protocol usage."""

    class MockNewtonSolver:
        def solve_step(self, u_current: np.ndarray, rhs: np.ndarray) -> np.ndarray:
            # Simple mock: return rhs
            return rhs

    solver = MockNewtonSolver()

    # Test usage (protocols are not @runtime_checkable in source)
    u = np.ones(51)
    rhs = np.zeros(51)
    result = solver.solve_step(u, rhs)
    assert result.shape == (51,)
    assert hasattr(solver, "solve_step")


# ===================================================================
# Test Exception Classes
# ===================================================================


@pytest.mark.unit
def test_solver_error():
    """Test SolverError exception."""
    with pytest.raises(SolverError) as exc_info:
        raise SolverError("Generic solver error")

    assert "Generic solver error" in str(exc_info.value)
    assert isinstance(exc_info.value, Exception)


@pytest.mark.unit
def test_convergence_error():
    """Test ConvergenceError exception."""
    with pytest.raises(ConvergenceError) as exc_info:
        raise ConvergenceError("Solver did not converge after 100 iterations")

    assert "did not converge" in str(exc_info.value)
    assert isinstance(exc_info.value, SolverError)  # Inherits from SolverError


@pytest.mark.unit
def test_configuration_error():
    """Test ConfigurationError exception."""
    with pytest.raises(ConfigurationError) as exc_info:
        raise ConfigurationError("Invalid solver configuration: negative dt")

    assert "Invalid solver configuration" in str(exc_info.value)
    assert isinstance(exc_info.value, SolverError)  # Inherits from SolverError


@pytest.mark.unit
def test_exception_hierarchy():
    """Test exception inheritance hierarchy."""
    # ConvergenceError is a SolverError
    assert issubclass(ConvergenceError, SolverError), "ConvergenceError should inherit from SolverError"
    assert isinstance(ConvergenceError("test"), SolverError)

    # ConfigurationError is a SolverError
    assert issubclass(ConfigurationError, SolverError), "ConfigurationError should inherit from SolverError"
    assert isinstance(ConfigurationError("test"), SolverError)

    # Both should also be Exception subclasses
    assert issubclass(SolverError, Exception)


# ===================================================================
# Test Legacy Compatibility
# ===================================================================


@pytest.mark.unit
def test_legacy_solver_return():
    """Test LegacySolverReturn backward compatibility and deprecation warning."""
    import warnings

    # Test that deprecation warning is emitted
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from mfgarchon.types.solver_types import LegacySolverReturn

        # Check warning was issued
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "LegacySolverReturn" in str(deprecation_warnings[-1].message)
        assert "SolverReturnTuple" in str(deprecation_warnings[-1].message)

    # LegacySolverReturn is an alias for SolverReturnTuple
    U = np.zeros((31, 51))
    M = np.ones((31, 51))
    info = {"converged": True}

    result: LegacySolverReturn = (U, M, info)

    # Should work with both type aliases
    assert isinstance(result, tuple)
    assert len(result) == 3


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_docstring():
    """Test module has comprehensive docstring."""
    from mfgarchon.types import solver_types

    assert solver_types.__doc__ is not None
    assert "Solver Type Definitions" in solver_types.__doc__

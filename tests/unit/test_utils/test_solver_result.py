"""Tests for mfgarchon.utils.solver_result module."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.utils.solver_result import (
    ConvergenceResult,
    MFGSolverResult,
    SolverResult,
    create_solver_result,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_arrays():
    """Create sample solution arrays."""
    U = np.random.randn(10, 20)
    M = np.random.rand(10, 20)
    return U, M


@pytest.fixture
def sample_errors():
    """Create sample error histories with convergence."""
    error_U = np.array([1.0, 0.5, 0.1, 0.01, 0.001])
    error_M = np.array([1.0, 0.6, 0.12, 0.015, 0.0015])
    return error_U, error_M


# ============================================================================
# Test: SolverResult Initialization
# ============================================================================


class TestSolverResultInit:
    """Test SolverResult initialization and validation."""

    def test_basic_initialization(self, sample_arrays, sample_errors):
        """Test basic initialization with required parameters."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = SolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
        )

        assert result.U is U
        assert result.M is M
        assert result.iterations == 5
        np.testing.assert_array_equal(result.error_history_U, error_U)
        np.testing.assert_array_equal(result.error_history_M, error_M)
        assert result.solver_name == "Unknown Solver"
        assert result.converged is False
        assert result.execution_time is None
        assert isinstance(result.metadata, dict)

    def test_full_initialization(self, sample_arrays, sample_errors):
        """Test initialization with all parameters."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        metadata = {"algorithm": "Newton", "step_size": 0.1}
        result = SolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
            solver_name="Test Solver",
            converged=True,
            execution_time=12.34,
            metadata=metadata,
        )

        assert result.solver_name == "Test Solver"
        assert result.converged is True
        assert result.execution_time == 12.34
        assert result.metadata == metadata

    def test_shape_validation(self, sample_errors):
        """Test shape validation raises error for mismatched arrays."""
        error_U, error_M = sample_errors
        U = np.random.randn(10, 20)
        M = np.random.rand(10, 15)  # Wrong shape

        with pytest.raises(ValueError, match="U and M shapes must match"):
            SolverResult(
                U=U,
                M=M,
                iterations=5,
                error_history_U=error_U,
                error_history_M=error_M,
            )

    def test_error_history_length_validation(self, sample_arrays):
        """Test error history length validation."""
        U, M = sample_arrays
        error_U = np.array([1.0, 0.5, 0.1])
        error_M = np.array([1.0, 0.5])  # Different length

        with pytest.raises(ValueError, match="Error history arrays must have same length"):
            SolverResult(
                U=U,
                M=M,
                iterations=3,
                error_history_U=error_U,
                error_history_M=error_M,
            )

    def test_error_history_trimming(self, sample_arrays):
        """Test automatic trimming when iterations < error history length."""
        U, M = sample_arrays
        error_U = np.array([1.0, 0.5, 0.1, 0.01, 0.001])
        error_M = np.array([1.0, 0.6, 0.12, 0.015, 0.0015])

        result = SolverResult(
            U=U,
            M=M,
            iterations=3,  # Less than error history length (5)
            error_history_U=error_U,
            error_history_M=error_M,
        )

        assert len(result.error_history_U) == 3
        assert len(result.error_history_M) == 3
        np.testing.assert_array_equal(result.error_history_U, error_U[:3])

    def test_iterations_exceeds_history_error(self, sample_arrays):
        """Test error when iterations exceed error history length."""
        U, M = sample_arrays
        error_U = np.array([1.0, 0.5, 0.1])
        error_M = np.array([1.0, 0.6, 0.12])

        with pytest.raises(ValueError, match=r"Iterations.*exceeds error history length"):
            SolverResult(
                U=U,
                M=M,
                iterations=10,  # More than error history length
                error_history_U=error_U,
                error_history_M=error_M,
            )


# ============================================================================
# Test: Backward Compatibility
# ============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_tuple_unpacking(self, sample_arrays, sample_errors):
        """Test tuple-like unpacking."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = SolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
        )

        # Unpack like a tuple
        u, m, iters, err_u, err_m = result

        assert u is U
        assert m is M
        assert iters == 5
        np.testing.assert_array_equal(err_u, error_U)
        np.testing.assert_array_equal(err_m, error_M)

    def test_length(self, sample_arrays, sample_errors):
        """Test __len__ returns 5 for compatibility."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = SolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
        )

        assert len(result) == 5

    def test_indexing(self, sample_arrays, sample_errors):
        """Test __getitem__ for tuple-like indexing."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = SolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
        )

        assert result[0] is U
        assert result[1] is M
        assert result[2] == 5
        np.testing.assert_array_equal(result[3], error_U)
        np.testing.assert_array_equal(result[4], error_M)


# ============================================================================
# Test: Properties
# ============================================================================


class TestProperties:
    """Test SolverResult properties."""

    def test_final_error_u(self, sample_arrays, sample_errors):
        """Test final_error_U property."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = SolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
        )

        assert result.final_error_U == error_U[-1]

    def test_final_error_m(self, sample_arrays, sample_errors):
        """Test final_error_M property."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = SolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
        )

        assert result.final_error_M == error_M[-1]

    def test_final_error_empty_history(self, sample_arrays):
        """Test final_error with empty history returns inf."""
        U, M = sample_arrays

        result = SolverResult(
            U=U,
            M=M,
            iterations=0,
            error_history_U=np.array([]),
            error_history_M=np.array([]),
        )

        assert result.final_error_U == float("inf")
        assert result.final_error_M == float("inf")

    def test_max_error(self, sample_arrays, sample_errors):
        """Test max_error property."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = SolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
        )

        assert result.max_error == max(error_U[-1], error_M[-1])


# ============================================================================
# Test: Methods
# ============================================================================


class TestMethods:
    """Test SolverResult methods."""

    def test_to_dict(self, sample_arrays, sample_errors):
        """Test to_dict conversion."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = SolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
            solver_name="Test Solver",
            converged=True,
            execution_time=10.5,
            metadata={"extra": "data"},
        )

        d = result.to_dict()

        assert d["iterations"] == 5
        assert d["solver_name"] == "Test Solver"
        assert d["converged"] is True
        assert d["execution_time"] == 10.5
        assert d["final_error_U"] == error_U[-1]
        assert d["final_error_M"] == error_M[-1]
        assert d["max_error"] == max(error_U[-1], error_M[-1])
        assert d["solution_shape"] == U.shape
        assert d["metadata"] == {"extra": "data"}
        np.testing.assert_array_equal(d["U"], U)
        np.testing.assert_array_equal(d["M"], M)

    def test_repr_converged(self, sample_arrays, sample_errors):
        """Test __repr__ for converged result."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = SolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
            solver_name="Test Solver",
            converged=True,
            execution_time=10.5,
        )

        repr_str = repr(result)
        assert "Test Solver" in repr_str
        assert "SUCCESS" in repr_str
        assert "5 iters" in repr_str
        assert "10.500s" in repr_str

    def test_repr_not_converged(self, sample_arrays):
        """Test __repr__ for non-converged result."""
        U, M = sample_arrays
        # Create error history matching iterations
        error_U = np.ones(100) * 0.1
        error_M = np.ones(100) * 0.1

        result = SolverResult(
            U=U,
            M=M,
            iterations=100,
            error_history_U=error_U,
            error_history_M=error_M,
            converged=False,
        )

        repr_str = repr(result)
        assert "WARNING" in repr_str
        assert "100 iters" in repr_str

    def test_repr_without_timing(self, sample_arrays, sample_errors):
        """Test __repr__ without execution time."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = SolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
        )

        repr_str = repr(result)
        # Should end with ")" but no timing suffix
        assert repr_str.endswith(")")
        # Should not have a timing value like "12.500s"
        assert not any(f"{t}s)" in repr_str for t in ["10.", "0.1", "1.2"])


# ============================================================================
# Test: ConvergenceResult
# ============================================================================


class TestConvergenceResult:
    """Test ConvergenceResult class."""

    def test_basic_initialization(self, sample_errors):
        """Test ConvergenceResult initialization."""
        error_U, error_M = sample_errors

        conv = ConvergenceResult(
            error_history_U=error_U,
            error_history_M=error_M,
            iterations_performed=5,
            converged=True,
            final_tolerance=1e-3,
        )

        assert conv.iterations_performed == 5
        assert conv.converged is True
        assert conv.final_tolerance == 1e-3
        assert conv.convergence_criteria == "L2_relative"

    def test_convergence_trend_converging(self):
        """Test convergence trend detection: converging."""
        error_U = np.array([1.0, 0.5, 0.2, 0.1, 0.05])
        error_M = np.array([1.0, 0.6, 0.3, 0.12, 0.06])

        conv = ConvergenceResult(
            error_history_U=error_U,
            error_history_M=error_M,
            iterations_performed=5,
            converged=True,
            final_tolerance=0.1,
        )

        assert conv.convergence_trend == "converging"

    def test_convergence_trend_diverging(self):
        """Test convergence trend detection: diverging."""
        error_U = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        error_M = np.array([0.12, 0.6, 1.2, 2.5, 6.0])

        conv = ConvergenceResult(
            error_history_U=error_U,
            error_history_M=error_M,
            iterations_performed=5,
            converged=False,
            final_tolerance=0.1,
        )

        assert conv.convergence_trend == "diverging"

    def test_convergence_trend_stagnating(self):
        """Test convergence trend detection: stagnating."""
        error_U = np.array([0.1, 0.1, 0.101, 0.099, 0.1])
        error_M = np.array([0.1, 0.1, 0.102, 0.098, 0.1])

        conv = ConvergenceResult(
            error_history_U=error_U,
            error_history_M=error_M,
            iterations_performed=5,
            converged=False,
            final_tolerance=0.01,
        )

        assert conv.convergence_trend == "stagnating"

    def test_convergence_trend_oscillating(self):
        """Test convergence trend detection: oscillating."""
        # Create oscillating pattern: neither converging, diverging, nor stagnating
        # Last 3 values: [0.4, 0.35, 0.38]
        # Not converging (not strictly decreasing)
        # Not diverging (0.38 < 0.35 * 1.2 = 0.42)
        # Not stagnating (max/min = 0.4/0.35 = 1.14 > 1.1)
        # Therefore: oscillating
        error_U = np.array([1.0, 0.6, 0.4, 0.35, 0.38])
        error_M = np.array([1.0, 0.65, 0.42, 0.37, 0.39])

        conv = ConvergenceResult(
            error_history_U=error_U,
            error_history_M=error_M,
            iterations_performed=5,
            converged=False,
            final_tolerance=0.1,
        )

        assert conv.convergence_trend == "oscillating"

    def test_convergence_trend_insufficient_data(self):
        """Test convergence trend with insufficient data."""
        error_U = np.array([1.0, 0.5])
        error_M = np.array([1.0, 0.6])

        conv = ConvergenceResult(
            error_history_U=error_U,
            error_history_M=error_M,
            iterations_performed=2,
            converged=False,
            final_tolerance=0.1,
        )

        assert conv.convergence_trend == "insufficient_data"

    def test_estimate_convergence_rate(self):
        """Test convergence rate estimation."""
        # Exponential decay: error ~ exp(-k*n)
        errors = np.exp(-0.5 * np.arange(10))
        conv = ConvergenceResult(
            error_history_U=errors,
            error_history_M=errors,
            iterations_performed=10,
            converged=True,
            final_tolerance=0.001,
        )

        rate = conv.estimate_convergence_rate()
        assert rate is not None
        assert 0.4 < rate < 0.6  # Should be close to 0.5

    def test_estimate_convergence_rate_insufficient_data(self):
        """Test convergence rate with insufficient data."""
        conv = ConvergenceResult(
            error_history_U=np.array([1.0]),
            error_history_M=np.array([1.0]),
            iterations_performed=1,
            converged=False,
            final_tolerance=0.1,
        )

        assert conv.estimate_convergence_rate() is None

    def test_estimate_convergence_rate_zero_errors(self):
        """Test convergence rate with zero errors."""
        conv = ConvergenceResult(
            error_history_U=np.array([0.0, 0.0, 0.0]),
            error_history_M=np.array([0.0, 0.0, 0.0]),
            iterations_performed=3,
            converged=True,
            final_tolerance=0.0,
        )

        assert conv.estimate_convergence_rate() is None


# ============================================================================
# Test: Factory Function
# ============================================================================


class TestCreateSolverResult:
    """Test create_solver_result factory function."""

    def test_basic_creation(self, sample_arrays, sample_errors):
        """Test basic factory creation."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = create_solver_result(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
        )

        assert isinstance(result, SolverResult)
        assert result.iterations == 5
        assert result.converged is False

    def test_auto_convergence_detection(self, sample_arrays, sample_errors):
        """Test automatic convergence detection."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        # Should detect convergence
        result = create_solver_result(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
            tolerance=0.01,
        )

        assert result.converged

    def test_convergence_analysis_in_metadata(self, sample_arrays, sample_errors):
        """Test convergence analysis added to metadata."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = create_solver_result(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
            tolerance=0.01,
        )

        assert "convergence_analysis" in result.metadata
        analysis = result.metadata["convergence_analysis"]
        assert isinstance(analysis, ConvergenceResult)
        assert analysis.converged

    def test_custom_metadata(self, sample_arrays, sample_errors):
        """Test custom metadata passed through."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        result = create_solver_result(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
            algorithm="Newton",
            step_size=0.1,
        )

        assert result.metadata["algorithm"] == "Newton"
        assert result.metadata["step_size"] == 0.1


# ============================================================================
# Test: Type Alias
# ============================================================================


class TestTypeAlias:
    """Test backward compatibility type alias."""

    def test_mfg_solver_result_alias(self, sample_arrays, sample_errors):
        """Test MFGSolverResult is alias for SolverResult."""
        U, M = sample_arrays
        error_U, error_M = sample_errors

        assert MFGSolverResult is SolverResult

        result = MFGSolverResult(
            U=U,
            M=M,
            iterations=5,
            error_history_U=error_U,
            error_history_M=error_M,
        )

        assert isinstance(result, SolverResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

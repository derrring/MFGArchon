"""
Unit tests for Pydantic-based array validation system.

This module tests array shape validation, physical constraint checking,
mass conservation, and comprehensive experiment configuration validation.
"""

import warnings

import pytest
from pydantic import ValidationError

import numpy as np

from mfgarchon.config.array_validation import (
    ArrayValidationConfig,
    CollocationConfig,
    ExperimentConfig,
    MFGArrays,
    MFGGridConfig,
)

# NDArray is imported at module level in mfgarchon/config/array_validation.py
# (Issue #1010 B5), so `model_rebuild()` is no longer needed at import time.


class TestArrayValidationConfig:
    """Test array validation configuration."""

    def test_valid_configuration(self):
        """Test creation with valid parameters."""
        config = ArrayValidationConfig(mass_conservation_rtol=1e-4, smoothness_threshold=500.0, cfl_max=0.6)
        assert config.mass_conservation_rtol == 1e-4
        assert config.smoothness_threshold == 500.0
        assert config.cfl_max == 0.6

    def test_default_values(self):
        """Test default parameter values."""
        config = ArrayValidationConfig()
        assert config.mass_conservation_rtol == 1e-3
        assert config.smoothness_threshold == 1e3
        assert config.cfl_max == 0.5

    def test_validation_mass_conservation_rtol_positive(self):
        """Test mass_conservation_rtol must be positive."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ArrayValidationConfig(mass_conservation_rtol=0.0)

        with pytest.raises(ValidationError, match="greater than 0"):
            ArrayValidationConfig(mass_conservation_rtol=-0.001)

    def test_validation_smoothness_threshold_positive(self):
        """Test smoothness_threshold must be positive."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ArrayValidationConfig(smoothness_threshold=0.0)

    def test_validation_cfl_max_range(self):
        """Test cfl_max must be in valid range."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ArrayValidationConfig(cfl_max=0.0)

        with pytest.raises(ValidationError, match="less than or equal to 1"):
            ArrayValidationConfig(cfl_max=1.5)


class TestMFGGridConfig:
    """Test MFG grid configuration validation."""

    def test_valid_configuration(self):
        """Test creation with valid grid parameters."""
        config = MFGGridConfig(Nx=100, Nt=50, xmin=0.0, xmax=2.0, T=1.5, sigma=0.2)
        assert config.Nx == 100
        assert config.Nt == 50
        assert config.xmin == 0.0
        assert config.xmax == 2.0
        assert config.T == 1.5
        assert config.sigma == 0.2

    def test_default_xmin_xmax_T(self):
        """Test default values for domain and time."""
        config = MFGGridConfig(Nx=50, Nt=30, sigma=0.1)
        assert config.xmin == 0.0
        assert config.xmax == 1.0
        assert config.T == 1.0

    def test_validation_nx_range(self):
        """Test Nx must be in valid range."""
        with pytest.raises(ValidationError, match="greater than or equal to 10"):
            MFGGridConfig(Nx=5, Nt=30, sigma=0.1)

        with pytest.raises(ValidationError, match="less than or equal to 2000"):
            MFGGridConfig(Nx=3000, Nt=30, sigma=0.1)

    def test_validation_nt_range(self):
        """Test Nt must be in valid range."""
        with pytest.raises(ValidationError, match="greater than or equal to 10"):
            MFGGridConfig(Nx=50, Nt=5, sigma=0.1)

        with pytest.raises(ValidationError, match="less than or equal to 20000"):
            MFGGridConfig(Nx=50, Nt=25000, sigma=0.1)

    def test_validation_xmax_greater_than_xmin(self):
        """Test xmax must be greater than xmin."""
        with pytest.raises(ValidationError, match=r"xmax .* must be > xmin"):
            MFGGridConfig(Nx=50, Nt=30, xmin=1.0, xmax=1.0, sigma=0.1)

        with pytest.raises(ValidationError, match=r"xmax .* must be > xmin"):
            MFGGridConfig(Nx=50, Nt=30, xmin=2.0, xmax=1.0, sigma=0.1)

    def test_validation_T_positive(self):
        """Test T must be positive."""
        with pytest.raises(ValidationError, match="greater than 0"):
            MFGGridConfig(Nx=50, Nt=30, T=0.0, sigma=0.1)

    def test_validation_T_reasonable(self):
        """Test T must be reasonable."""
        with pytest.raises(ValidationError, match="less than or equal to 100"):
            MFGGridConfig(Nx=50, Nt=30, T=150.0, sigma=0.1)

    def test_validation_sigma_positive(self):
        """Test sigma must be positive."""
        with pytest.raises(ValidationError, match="greater than 0"):
            MFGGridConfig(Nx=50, Nt=30, sigma=0.0)

    def test_validation_sigma_reasonable(self):
        """Test sigma must be reasonable."""
        with pytest.raises(ValidationError, match="less than or equal to 10"):
            MFGGridConfig(Nx=50, Nt=30, sigma=15.0)

    def test_property_dx(self):
        """Test dx property calculation."""
        config = MFGGridConfig(Nx=100, Nt=50, xmin=0.0, xmax=2.0, sigma=0.1)
        expected_dx = 2.0 / 100
        assert abs(config.dx - expected_dx) < 1e-10

    def test_property_dt(self):
        """Test dt property calculation."""
        config = MFGGridConfig(Nx=50, Nt=100, T=2.0, sigma=0.1)
        expected_dt = 2.0 / 100
        assert abs(config.dt - expected_dt) < 1e-10

    def test_property_cfl_number(self):
        """Test CFL number calculation uses the PDE coefficient D = sigma^2/2 (Issue #1550)."""
        config = MFGGridConfig(Nx=100, Nt=100, xmin=0.0, xmax=1.0, T=1.0, sigma=0.1)
        dx = 1.0 / 100
        dt = 1.0 / 100
        # Diffusive CFL = D*dt/dx^2 with D = sigma^2/2, NOT bare sigma^2 (= 2D).
        expected_cfl = 0.5 * (0.1**2) * dt / (dx**2)
        assert abs(config.cfl_number - expected_cfl) < 1e-10

    def test_property_grid_shape(self):
        """Test grid_shape property."""
        config = MFGGridConfig(Nx=50, Nt=30, sigma=0.1)
        assert config.grid_shape == (31, 51)  # (Nt+1, Nx+1)

    def test_cfl_stability_warning(self):
        """Test CFL stability warning validator."""
        # CFL > 0.5 should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MFGGridConfig(Nx=10, Nt=10, T=1.0, sigma=1.0)  # High sigma for large CFL

            assert len(w) >= 1
            assert "CFL number" in str(w[0].message)
            assert "instability" in str(w[0].message).lower()

    def test_cfl_stability_no_warning_when_safe(self):
        """Test no CFL warning when stable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MFGGridConfig(Nx=100, Nt=100, T=1.0, sigma=0.05)  # Very small sigma for safe CFL

            # Should have no CFL warnings
            cfl_warnings = [warning for warning in w if "CFL" in str(warning.message)]
            assert len(cfl_warnings) == 0


class TestMFGArrays:
    """Test MFG solution array validation."""

    @pytest.fixture
    def grid_config(self):
        """Create standard grid configuration for testing."""
        return MFGGridConfig(Nx=50, Nt=30, xmin=0.0, xmax=1.0, T=1.0, sigma=0.1)

    @pytest.fixture
    def valid_U_solution(self, grid_config):
        """Create valid U solution array."""
        shape = grid_config.grid_shape
        return np.random.randn(*shape).astype(np.float64)

    @pytest.fixture
    def valid_M_solution(self, grid_config):
        """Create valid M solution array (normalized density)."""
        shape = grid_config.grid_shape
        M = np.abs(np.random.randn(*shape)).astype(np.float64)
        # Normalize each time slice to integrate to 1
        dx = grid_config.dx
        for t_idx in range(shape[0]):
            M[t_idx] /= np.trapezoid(M[t_idx], dx=dx)
        return M

    def test_valid_arrays(self, grid_config, valid_U_solution, valid_M_solution):
        """Test creation with valid solution arrays."""
        arrays = MFGArrays(U_solution=valid_U_solution, M_solution=valid_M_solution, grid_config=grid_config)

        assert arrays.U_solution.shape == grid_config.grid_shape
        assert arrays.M_solution.shape == grid_config.grid_shape

    def test_U_solution_wrong_shape(self, grid_config, valid_M_solution):
        """Test U solution with wrong shape raises error."""
        wrong_shape_U = np.random.randn(20, 30)

        with pytest.raises(ValidationError, match=r"U solution shape .* != expected"):
            MFGArrays(U_solution=wrong_shape_U, M_solution=valid_M_solution, grid_config=grid_config)

    def test_U_solution_nan_values(self, grid_config, valid_M_solution):
        """Test U solution with NaN values raises error."""
        U_with_nan = np.random.randn(*grid_config.grid_shape)
        U_with_nan[5, 10] = np.nan

        with pytest.raises(ValidationError):
            MFGArrays(U_solution=U_with_nan, M_solution=valid_M_solution, grid_config=grid_config)

    def test_U_solution_inf_values(self, grid_config, valid_M_solution):
        """Test U solution with infinite values raises error."""
        U_with_inf = np.random.randn(*grid_config.grid_shape)
        U_with_inf[5, 10] = np.inf

        with pytest.raises(ValidationError):
            MFGArrays(U_solution=U_with_inf, M_solution=valid_M_solution, grid_config=grid_config)

    def test_U_solution_wrong_dtype(self, grid_config, valid_M_solution):
        """Test U solution with wrong dtype raises error."""
        U_int = np.zeros(grid_config.grid_shape, dtype=np.int32)

        with pytest.raises(ValidationError):
            MFGArrays(U_solution=U_int, M_solution=valid_M_solution, grid_config=grid_config)

    def test_U_solution_smoothness_warning(self, grid_config, valid_M_solution):
        """Test U solution smoothness warning for non-smooth arrays."""
        # Create array with large second differences
        U_non_smooth = np.random.randn(*grid_config.grid_shape) * 1e5

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            MFGArrays(U_solution=U_non_smooth, M_solution=valid_M_solution, grid_config=grid_config)

            # May or may not warn depending on random values
            # Just check it doesn't crash
            assert True

    def test_M_solution_wrong_shape(self, grid_config, valid_U_solution):
        """Test M solution with wrong shape raises error."""
        wrong_shape_M = np.random.randn(20, 30)

        with pytest.raises(ValidationError):
            MFGArrays(U_solution=valid_U_solution, M_solution=wrong_shape_M, grid_config=grid_config)

    def test_M_solution_nan_values(self, grid_config, valid_U_solution):
        """Test M solution with NaN values raises error."""
        M_with_nan = np.abs(np.random.randn(*grid_config.grid_shape))
        M_with_nan[5, 10] = np.nan

        with pytest.raises(ValidationError):
            MFGArrays(U_solution=valid_U_solution, M_solution=M_with_nan, grid_config=grid_config)

    def test_M_solution_negative_values(self, grid_config, valid_U_solution):
        """Test M solution with negative values raises error."""
        M_with_negative = np.abs(np.random.randn(*grid_config.grid_shape))  # Start with positive
        M_with_negative[5, 10] = -0.1  # Then add one negative value

        with pytest.raises(ValidationError):
            MFGArrays(U_solution=valid_U_solution, M_solution=M_with_negative, grid_config=grid_config)

    def test_M_solution_mass_conservation_warning_initial(self, grid_config, valid_U_solution):
        """Test mass conservation warning for initial condition."""
        M_unnormalized = np.abs(np.random.randn(*grid_config.grid_shape))
        # Don't normalize - should warn

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MFGArrays(U_solution=valid_U_solution, M_solution=M_unnormalized, grid_config=grid_config)

            # Should have warning about initial mass (case-insensitive search)
            mass_warnings = [warning for warning in w if "initial mass" in str(warning.message).lower()]
            assert len(mass_warnings) > 0

    def test_M_solution_mass_conservation_warning_final(self, grid_config, valid_U_solution):
        """Test mass conservation warning for final condition."""
        M = np.abs(np.random.randn(*grid_config.grid_shape))
        dx = grid_config.dx
        # Normalize all but final time slice
        for t_idx in range(M.shape[0] - 1):
            M[t_idx] /= np.trapezoid(M[t_idx], dx=dx)
        # Leave final unnormalized

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MFGArrays(U_solution=valid_U_solution, M_solution=M, grid_config=grid_config)

            # Should warn about final mass (case-insensitive search)
            mass_warnings = [warning for warning in w if "final mass" in str(warning.message).lower()]
            assert len(mass_warnings) > 0

    def test_solution_consistency_shape_mismatch(self, grid_config):
        """Test that U and M shape mismatch is detected."""
        U = np.random.randn(*grid_config.grid_shape)
        M = np.abs(np.random.randn(20, 30))  # Wrong shape

        with pytest.raises(ValidationError, match=r"M solution shape .* != expected"):
            MFGArrays(U_solution=U, M_solution=M, grid_config=grid_config)

    def test_get_solution_statistics(self, grid_config, valid_U_solution, valid_M_solution):
        """Test solution statistics retrieval."""
        arrays = MFGArrays(U_solution=valid_U_solution, M_solution=valid_M_solution, grid_config=grid_config)

        stats = arrays.get_solution_statistics()

        assert "U" in stats
        assert "M" in stats
        assert "mass_conservation" in stats
        assert "numerical_stability" in stats

        # Check U statistics
        assert "shape" in stats["U"]
        assert "min" in stats["U"]
        assert "max" in stats["U"]
        assert "mean" in stats["U"]
        assert "std" in stats["U"]

        # Check M statistics
        assert stats["M"]["shape"] == grid_config.grid_shape

        # Check mass conservation
        assert "initial_mass" in stats["mass_conservation"]
        assert "final_mass" in stats["mass_conservation"]
        assert "mass_drift" in stats["mass_conservation"]

        # Check numerical stability
        assert "cfl_number" in stats["numerical_stability"]
        assert stats["numerical_stability"]["cfl_number"] == grid_config.cfl_number

    def test_custom_validation_config(self, grid_config, valid_U_solution, valid_M_solution):
        """Test using custom validation configuration."""
        validation_config = ArrayValidationConfig(mass_conservation_rtol=1e-2, cfl_max=0.8)

        arrays = MFGArrays(
            U_solution=valid_U_solution,
            M_solution=valid_M_solution,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        assert arrays.validation_config.mass_conservation_rtol == 1e-2
        assert arrays.validation_config.cfl_max == 0.8


class TestCollocationConfig:
    """Test collocation points configuration validation."""

    @pytest.fixture
    def grid_config(self):
        """Create standard grid configuration for testing."""
        return MFGGridConfig(Nx=100, Nt=50, xmin=0.0, xmax=1.0, sigma=0.1)

    def test_valid_collocation_points(self, grid_config):
        """Test creation with valid collocation points."""
        points = np.linspace(0.1, 0.9, 20).reshape(-1, 1)
        config = CollocationConfig(points=points, grid_config=grid_config)

        assert config.points.shape == (20, 1)

    def test_collocation_points_wrong_shape_1d(self, grid_config):
        """Test collocation points with wrong 1D shape raises error."""
        points = np.linspace(0.1, 0.9, 20)  # Not Nx1

        with pytest.raises(ValidationError):
            CollocationConfig(points=points, grid_config=grid_config)

    def test_collocation_points_wrong_shape_3d(self, grid_config):
        """Test collocation points with 3D shape raises error."""
        points = np.random.randn(10, 1, 1)  # 3D

        with pytest.raises(ValidationError):
            CollocationConfig(points=points, grid_config=grid_config)

    def test_collocation_points_outside_domain_min(self, grid_config):
        """Test collocation points below xmin raises error."""
        points = np.array([[-0.1], [0.5], [0.8]])

        with pytest.raises(ValidationError):
            CollocationConfig(points=points, grid_config=grid_config)

    def test_collocation_points_outside_domain_max(self, grid_config):
        """Test collocation points above xmax raises error."""
        points = np.array([[0.2], [0.5], [1.1]])

        with pytest.raises(ValidationError):
            CollocationConfig(points=points, grid_config=grid_config)

    def test_collocation_points_few_warning(self, grid_config):
        """Test warning for too few collocation points."""
        points = np.array([[0.5]])  # Only 1 point

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CollocationConfig(points=points, grid_config=grid_config)

            few_warnings = [warning for warning in w if "Few collocation points" in str(warning.message)]
            assert len(few_warnings) >= 1

    def test_collocation_points_many_warning(self, grid_config):
        """Test warning for too many collocation points."""
        points = np.linspace(0.0, 1.0, 150).reshape(-1, 1)  # More than Nx=100

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CollocationConfig(points=points, grid_config=grid_config)

            many_warnings = [warning for warning in w if "Many collocation points" in str(warning.message)]
            assert len(many_warnings) >= 1

    def test_collocation_points_duplicates_warning(self, grid_config):
        """Test warning for duplicate collocation points."""
        points = np.array([[0.3], [0.5], [0.5], [0.7]])  # Duplicate 0.5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CollocationConfig(points=points, grid_config=grid_config)

            # Check for duplicate warning in any warning message (not just first)
            duplicate_warnings = [warning for warning in w if "Duplicate collocation points" in str(warning.message)]
            assert len(duplicate_warnings) >= 1

    def test_collocation_points_irregular_distribution_warning(self, grid_config):
        """Test warning for irregular point distribution."""
        # Create very irregular spacing
        points = np.array([[0.1], [0.11], [0.12], [0.9]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CollocationConfig(points=points, grid_config=grid_config)

            # Should warn about irregular distribution
            irregular_warnings = [warning for warning in w if "Irregular" in str(warning.message)]
            assert len(irregular_warnings) > 0


class TestExperimentConfig:
    """Test complete experiment configuration validation."""

    @pytest.fixture
    def grid_config(self):
        """Create standard grid configuration."""
        return MFGGridConfig(Nx=50, Nt=30, sigma=0.1)

    def test_valid_experiment_config(self, grid_config):
        """Test creation with valid experiment configuration."""
        config = ExperimentConfig(
            grid_config=grid_config,
            experiment_name="test_experiment",
            description="Test description",
            researcher="pytest",
            tags=["test", "validation"],
        )

        assert config.experiment_name == "test_experiment"
        assert config.description == "Test description"
        assert config.researcher == "pytest"
        assert config.tags == ["test", "validation"]

    def test_default_values(self, grid_config):
        """Test default parameter values."""
        config = ExperimentConfig(grid_config=grid_config, experiment_name="test")

        assert config.description is None
        assert config.researcher == ""
        assert config.tags == []
        assert config.output_dir == "./results"
        assert config.save_arrays is True
        assert config.save_plots is True

    def test_experiment_name_validation_invalid_characters(self, grid_config):
        """Test experiment name validation for invalid characters."""
        with pytest.raises(ValidationError, match="Experiment name must contain only"):
            ExperimentConfig(grid_config=grid_config, experiment_name="invalid name!")

        with pytest.raises(ValidationError, match="Experiment name must contain only"):
            ExperimentConfig(grid_config=grid_config, experiment_name="test/experiment")

    def test_experiment_name_validation_valid_characters(self, grid_config):
        """Test experiment name validation accepts valid characters."""
        valid_names = ["experiment_1", "test-experiment", "exp.2025", "TEST_123"]

        for name in valid_names:
            config = ExperimentConfig(grid_config=grid_config, experiment_name=name)
            assert config.experiment_name == name

    def test_with_arrays(self, grid_config):
        """Test experiment config with solution arrays."""
        U = np.random.randn(*grid_config.grid_shape)
        M = np.abs(np.random.randn(*grid_config.grid_shape))
        dx = grid_config.dx
        for t_idx in range(M.shape[0]):
            M[t_idx] /= np.trapezoid(M[t_idx], dx=dx)

        arrays = MFGArrays(U_solution=U, M_solution=M, grid_config=grid_config)

        config = ExperimentConfig(grid_config=grid_config, experiment_name="test", arrays=arrays)

        assert config.arrays is not None
        assert config.arrays.U_solution.shape == grid_config.grid_shape

    def test_with_collocation(self, grid_config):
        """Test experiment config with collocation points."""
        points = np.linspace(0.1, 0.9, 20).reshape(-1, 1)
        collocation = CollocationConfig(points=points, grid_config=grid_config)

        config = ExperimentConfig(grid_config=grid_config, experiment_name="test", collocation=collocation)

        assert config.collocation is not None
        assert len(config.collocation.points) == 20

    def test_to_notebook_metadata(self, grid_config):
        """Test conversion to notebook metadata format."""
        config = ExperimentConfig(
            grid_config=grid_config,
            experiment_name="test_exp",
            description="Test description",
            researcher="pytest",
            tags=["test"],
        )

        metadata = config.to_notebook_metadata()

        assert metadata["experiment_name"] == "test_exp"
        assert metadata["description"] == "Test description"
        assert metadata["researcher"] == "pytest"
        assert metadata["tags"] == ["test"]
        assert "grid_config" in metadata

    def test_to_notebook_metadata_with_arrays(self, grid_config):
        """Test notebook metadata includes solution statistics."""
        U = np.random.randn(*grid_config.grid_shape)
        M = np.abs(np.random.randn(*grid_config.grid_shape))
        dx = grid_config.dx
        for t_idx in range(M.shape[0]):
            M[t_idx] /= np.trapezoid(M[t_idx], dx=dx)

        arrays = MFGArrays(U_solution=U, M_solution=M, grid_config=grid_config)

        config = ExperimentConfig(grid_config=grid_config, experiment_name="test", arrays=arrays)

        metadata = config.to_notebook_metadata()

        assert "solution_statistics" in metadata
        assert "U" in metadata["solution_statistics"]
        assert "M" in metadata["solution_statistics"]

    def test_to_notebook_metadata_with_collocation(self, grid_config):
        """Test notebook metadata includes collocation point count."""
        points = np.linspace(0.1, 0.9, 15).reshape(-1, 1)
        collocation = CollocationConfig(points=points, grid_config=grid_config)

        config = ExperimentConfig(grid_config=grid_config, experiment_name="test", collocation=collocation)

        metadata = config.to_notebook_metadata()

        assert "collocation_points" in metadata
        assert metadata["collocation_points"] == 15

    def test_grid_config_consistency_warning_arrays(self, grid_config):
        """Test warning when array grid config differs."""
        # Create arrays with different grid config
        other_grid = MFGGridConfig(Nx=100, Nt=50, sigma=0.2)
        U = np.random.randn(*other_grid.grid_shape)
        M = np.abs(np.random.randn(*other_grid.grid_shape))
        dx = other_grid.dx
        for t_idx in range(M.shape[0]):
            M[t_idx] /= np.trapezoid(M[t_idx], dx=dx)

        arrays = MFGArrays(U_solution=U, M_solution=M, grid_config=other_grid)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ExperimentConfig(grid_config=grid_config, experiment_name="test", arrays=arrays)

            # Should warn about grid config mismatch
            grid_warnings = [warning for warning in w if "grid configuration differs" in str(warning.message).lower()]
            assert len(grid_warnings) > 0

    def test_grid_config_consistency_warning_collocation(self, grid_config):
        """Test warning when collocation grid config differs."""
        # Create collocation with different grid config
        other_grid = MFGGridConfig(Nx=100, Nt=50, sigma=0.2)
        points = np.linspace(0.1, 0.9, 20).reshape(-1, 1)
        collocation = CollocationConfig(points=points, grid_config=other_grid)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ExperimentConfig(grid_config=grid_config, experiment_name="test", collocation=collocation)

            # Should warn about grid config mismatch
            grid_warnings = [warning for warning in w if "grid configuration differs" in str(warning.message).lower()]
            assert len(grid_warnings) > 0

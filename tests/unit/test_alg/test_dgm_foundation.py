"""
Test cases for Deep Galerkin Methods (DGM) foundation.

This module tests the core DGM infrastructure including sampling strategies,
neural architectures, variance reduction techniques, and the MFG DGM solver.
"""

import pytest

import numpy as np

from mfgarchon.alg.neural.dgm.base_dgm import DGMConfig, DGMResult, VarianceReductionMethod
from mfgarchon.alg.neural.dgm.sampling import (
    MonteCarloSampler,
    QuasiMonteCarloSampler,
    adaptive_sampling,
)
from mfgarchon.alg.neural.dgm.variance_reduction import (
    ControlVariates,
    ImportanceSampling,
    adaptive_importance_distribution,
    create_control_variate_function,
)

pytestmark = pytest.mark.optional_torch

# Skip PyTorch tests if not available
pytorch_available = True
try:
    import torch

    from mfgarchon.alg.neural.dgm.architectures import (
        DeepGalerkinNetwork,
        HighDimMLP,
        ResidualDGMNetwork,
    )
except ImportError:
    pytorch_available = False


class TestDGMConfig:
    """Test DGM configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DGMConfig()

        assert config.hidden_layers == [256, 256, 256, 256]
        assert config.activation == "tanh"
        assert config.learning_rate == 1e-3
        assert config.num_interior_points == 10000
        assert config.sampling_strategy == "monte_carlo"
        # use_control_variates is deprecated (defaults to None), use variance_reduction instead
        assert config.variance_reduction == VarianceReductionMethod.CONTROL_VARIATES
        assert config.tolerance == 1e-5

    def test_custom_config(self):
        """Test custom configuration."""
        config = DGMConfig(
            hidden_layers=[128, 128],
            learning_rate=1e-4,
            num_interior_points=5000,
            sampling_strategy="quasi_monte_carlo",
        )

        assert config.hidden_layers == [128, 128]
        assert config.learning_rate == 1e-4
        assert config.num_interior_points == 5000
        assert config.sampling_strategy == "quasi_monte_carlo"


class TestDGMResult:
    """Test DGM result container."""

    def test_default_result(self):
        """Test default result values."""
        result = DGMResult()

        assert result.value_network is None
        assert result.density_network is None
        assert result.final_loss == np.inf
        assert result.converged is False
        assert result.num_epochs == 0
        assert result.dimension == 0


class TestMonteCarloSampler:
    """Test Monte Carlo sampling for high-dimensional domains."""

    def test_initialization(self):
        """Test sampler initialization."""
        domain_bounds = [(-1, 1), (-2, 2)]
        sampler = MonteCarloSampler(domain_bounds, dimension=2, seed=42)

        assert sampler.dimension == 2
        assert sampler.total_dimension == 3  # Including time
        assert len(sampler.domain_bounds) == 2

    def test_sample_interior(self):
        """Test interior domain sampling."""
        domain_bounds = [(-1, 1), (-1, 1)]
        sampler = MonteCarloSampler(domain_bounds, dimension=2, seed=42)

        points = sampler.sample_interior(100, time_bounds=(0, 1))

        assert points.shape == (100, 3)  # (t, x1, x2)
        assert np.all(points[:, 0] >= 0)
        assert np.all(points[:, 0] <= 1)
        assert np.all(points[:, 1] >= -1)
        assert np.all(points[:, 1] <= 1)
        assert np.all(points[:, 2] >= -1)
        assert np.all(points[:, 2] <= 1)

    def test_sample_boundary(self):
        """Test boundary sampling."""
        domain_bounds = [(-1, 1), (-1, 1)]
        sampler = MonteCarloSampler(domain_bounds, dimension=2, seed=42)

        boundary_points = sampler.sample_boundary(40, time_bounds=(0, 1))

        # Should have points on all 4 boundary faces (2 * dimension)
        assert boundary_points.shape[0] >= 40 - 4  # Allow for integer division
        assert boundary_points.shape[1] == 3

        # Check that at least some points are on boundaries
        on_boundary = (
            (np.abs(boundary_points[:, 1] + 1) < 1e-10)  # x1 = -1
            | (np.abs(boundary_points[:, 1] - 1) < 1e-10)  # x1 = 1
            | (np.abs(boundary_points[:, 2] + 1) < 1e-10)  # x2 = -1
            | (np.abs(boundary_points[:, 2] - 1) < 1e-10)  # x2 = 1
        )
        assert np.any(on_boundary)

    def test_sample_initial(self):
        """Test initial condition sampling."""
        domain_bounds = [(-1, 1), (-1, 1)]
        sampler = MonteCarloSampler(domain_bounds, dimension=2, seed=42)

        initial_points = sampler.sample_initial(50)

        assert initial_points.shape == (50, 2)  # Only spatial dimensions
        assert np.all(initial_points[:, 0] >= -1)
        assert np.all(initial_points[:, 0] <= 1)
        assert np.all(initial_points[:, 1] >= -1)
        assert np.all(initial_points[:, 1] <= 1)


class TestQuasiMonteCarloSampler:
    """Test Quasi-Monte Carlo sampling."""

    def test_sample_interior(self):
        """Test interior sampling produces valid points."""
        domain_bounds = [(-1, 1), (-1, 1)]
        sampler = QuasiMonteCarloSampler(domain_bounds, dimension=2, sequence_type="sobol")

        points = sampler.sample_interior(100, time_bounds=(0, 1))

        # Check shape: (num_points, dimension + time)
        assert points.shape == (100, 3)
        # Check time bounds
        assert np.all(points[:, 0] >= 0)
        assert np.all(points[:, 0] <= 1)
        # Check spatial bounds
        assert np.all(points[:, 1] >= -1)
        assert np.all(points[:, 1] <= 1)
        assert np.all(points[:, 2] >= -1)
        assert np.all(points[:, 2] <= 1)


class TestAdaptiveSampling:
    """Test adaptive sampling based on residual analysis."""

    def test_adaptive_sampling_high_residuals(self):
        """Test adaptive sampling with high residuals."""
        # Create sample points
        current_points = np.random.uniform(-1, 1, (50, 3))
        domain_bounds = [(-1, 1), (-1, 1)]

        # Mock residual function with high residuals at specific points
        def residual_function(points):
            # High residuals for points with x1 > 0.5
            return np.where(points[:, 1] > 0.5, 2.0, 0.1)

        new_points = adaptive_sampling(
            residual_function, current_points, threshold=1.0, max_new_points=100, domain_bounds=domain_bounds
        )

        assert len(new_points) > 0
        assert new_points.shape[1] == 3  # Same dimension as input points

        # Check bounds are respected
        assert np.all(new_points[:, 1] >= -1)
        assert np.all(new_points[:, 1] <= 1)
        assert np.all(new_points[:, 2] >= -1)
        assert np.all(new_points[:, 2] <= 1)

    def test_adaptive_sampling_low_residuals(self):
        """Test adaptive sampling with low residuals."""
        current_points = np.random.uniform(-1, 1, (50, 3))
        domain_bounds = [(-1, 1), (-1, 1)]

        # Mock residual function with all low residuals
        def residual_function(points):
            return np.ones(len(points)) * 0.01

        new_points = adaptive_sampling(
            residual_function, current_points, threshold=1.0, max_new_points=100, domain_bounds=domain_bounds
        )

        assert len(new_points) == 0


class TestControlVariates:
    """Test control variates for variance reduction."""

    def test_initialization(self):
        """Test control variate initialization."""

        def control_func(points):
            return np.sum(points**2, axis=1)

        cv = ControlVariates(control_function=control_func)

        assert cv.control_function is not None
        assert cv.control_coefficient == 0.0
        assert cv.is_calibrated is False

    def test_calibration(self):
        """Test control variate calibration."""

        # Simple control function
        def control_func(points):
            return np.sum(points**2, axis=1)

        # Target function correlated with control
        def target_func(points):
            return 2 * np.sum(points**2, axis=1) + np.random.normal(0, 0.1, len(points))

        cv = ControlVariates(control_function=control_func)
        sample_points = np.random.uniform(-1, 1, (1000, 3))

        cv.calibrate(sample_points, target_func)

        assert cv.is_calibrated is True
        assert cv.control_coefficient != 0.0

    def test_apply_variance_reduction(self):
        """Test variance reduction application."""

        def control_func(points):
            return np.sum(points**2, axis=1)

        cv = ControlVariates(control_function=control_func)
        cv.control_coefficient = 0.5
        cv.is_calibrated = True

        sample_points = np.random.uniform(-1, 1, (100, 3))
        function_values = np.random.normal(0, 1, 100)

        reduced_values = cv.apply(sample_points, function_values)

        assert len(reduced_values) == len(function_values)
        assert not np.array_equal(reduced_values, function_values)  # Should be different


class TestImportanceSampling:
    """Test importance sampling for efficient Monte Carlo."""

    def test_initialization(self):
        """Test importance sampling initialization."""
        is_sampler = ImportanceSampling()

        assert is_sampler.importance_function is None
        assert is_sampler.is_setup is False

    def test_setup_from_residuals(self):
        """Test setup from residual analysis."""
        sample_points = np.random.uniform(-1, 1, (100, 3))
        residuals = np.random.uniform(0, 1, 100)

        is_sampler = ImportanceSampling()
        is_sampler.setup_from_residuals(sample_points, residuals)

        assert is_sampler.is_setup is True
        assert is_sampler.importance_function is not None

    def test_uniform_sampling_fallback(self):
        """Test fallback to uniform sampling."""
        is_sampler = ImportanceSampling()
        domain_bounds = [(-1, 1), (-1, 1)]
        time_bounds = (0, 1)

        points, weights = is_sampler.sample(100, domain_bounds, time_bounds)

        assert points.shape == (100, 3)
        assert len(weights) == 100
        assert np.allclose(weights, 1.0 / 100)  # Uniform weights


class TestControlVariateFunctions:
    """Test control variate function creation."""

    def test_linear_quadratic_control(self):
        """Test linear-quadratic control variate."""
        control_func = create_control_variate_function("linear_quadratic")

        assert control_func is not None

        # Test function evaluation
        points = np.array([[0.5, 1.0, -1.0], [0.0, 0.0, 0.0]])
        values = control_func(points)

        assert len(values) == 2
        assert not np.isnan(values).any()

    def test_crowd_dynamics_control(self):
        """Test crowd dynamics control variate."""
        control_func = create_control_variate_function("crowd_dynamics")

        assert control_func is not None

        points = np.array([[0.5, 1.0, -1.0], [0.0, 0.0, 0.0]])
        values = control_func(points)

        assert len(values) == 2
        assert not np.isnan(values).any()

    def test_unknown_problem_type(self):
        """Test unknown problem type returns None."""
        control_func = create_control_variate_function("unknown_type")

        assert control_func is None


class TestAdaptiveImportanceDistribution:
    """Test adaptive importance distribution creation."""

    def test_normal_residuals(self):
        """Test importance distribution with normal residuals."""
        points = np.random.uniform(-1, 1, (100, 3))
        residuals = np.random.uniform(0, 1, 100)

        importance_func = adaptive_importance_distribution(points, residuals)

        assert importance_func is not None

        # Test evaluation
        query_points = np.random.uniform(-1, 1, (50, 3))
        importance_values = importance_func(query_points)

        assert len(importance_values) == 50
        assert np.all(importance_values > 0)  # Should be positive

    def test_zero_residuals(self):
        """Test importance distribution with zero residuals."""
        points = np.random.uniform(-1, 1, (100, 3))
        residuals = np.zeros(100)

        importance_func = adaptive_importance_distribution(points, residuals)

        query_points = np.random.uniform(-1, 1, (50, 3))
        importance_values = importance_func(query_points)

        # Should return uniform importance
        assert np.allclose(importance_values, 1.0)


@pytest.mark.skipif(not pytorch_available, reason="PyTorch not available")
class TestDGMArchitectures:
    """Test neural network architectures for DGM."""

    def test_deep_galerkin_network(self):
        """Test Deep Galerkin network architecture."""
        input_dim = 5  # 4D space + 1D time
        network = DeepGalerkinNetwork(input_dim, hidden_layers=[128, 128])

        # Test forward pass
        x = torch.randn(32, input_dim)
        output = network(x)

        assert output.shape == (32, 1)  # Scalar output
        assert not torch.isnan(output).any()

    def test_high_dim_mlp(self):
        """Test high-dimensional MLP."""
        input_dim = 10  # 9D space + 1D time
        network = HighDimMLP(input_dim, base_width=64, num_layers=4)

        x = torch.randn(16, input_dim)
        output = network(x)

        assert output.shape == (16, 1)
        assert not torch.isnan(output).any()

    def test_residual_dgm_network(self):
        """Test residual DGM network."""
        input_dim = 6  # 5D space + 1D time
        network = ResidualDGMNetwork(input_dim, hidden_dim=128, num_residual_blocks=4)

        x = torch.randn(24, input_dim)
        output = network(x)

        assert output.shape == (24, 1)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

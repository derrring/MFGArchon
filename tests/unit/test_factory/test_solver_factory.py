#!/usr/bin/env python3
"""
Unit tests for mfgarchon/factory/solver_factory.py

Tests solver factory patterns including:
- SolverFactoryConfig dataclass
- SolverFactory class methods
- Config update with kwargs
- Error handling and validation
"""

from unittest.mock import Mock, patch

import pytest

from mfgarchon.config import FPConfig, MFGSolverConfig
from mfgarchon.factory.solver_factory import (
    SolverFactory,
    SolverFactoryConfig,
    create_accurate_solver,
    create_amr_solver,
    create_basic_solver,
    create_fast_solver,
    create_research_solver,
    create_semi_lagrangian_solver,
    create_solver,
    create_standard_solver,
)

# ===================================================================
# Mock Problem for Testing
# ===================================================================


class MockMFGProblem:
    """Minimal mock MFG problem for testing solver factory."""

    def __init__(
        self,
        T=1.0,
        Nt=50,
        xmin=0.0,
        xmax=1.0,
        Nx=100,
        sigma=0.1,
        coupling_coefficient=0.5,
    ):
        self.T = T
        self.Nt = Nt  # Number of time INTERVALS (not points)
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx  # Number of space INTERVALS (not points)
        # Dx = domain_length / Nx (number of intervals)
        self.Dx = (xmax - xmin) / Nx if Nx > 0 else 0.0
        # Dt = T / Nt (number of intervals)
        self.Dt = T / Nt if Nt > 0 else 0.0
        self.sigma = sigma
        self.coupling_coefficient = coupling_coefficient


# ===================================================================
# Test SolverFactoryConfig Dataclass
# ===================================================================


@pytest.mark.unit
def test_solver_factory_config_defaults():
    """Test SolverFactoryConfig has correct default values."""
    config = SolverFactoryConfig()

    assert config.solver_type == "fixed_point"
    assert config.custom_config is None
    assert config.solver_kwargs == {}


@pytest.mark.unit
def test_solver_factory_config_custom_values():
    """Test SolverFactoryConfig with custom values."""
    custom_kwargs = {"max_iterations": 100}
    custom_config = MFGSolverConfig()
    config = SolverFactoryConfig(
        solver_type="fixed_point",
        custom_config=custom_config,
        solver_kwargs=custom_kwargs,
    )

    assert config.solver_type == "fixed_point"
    assert config.custom_config is custom_config
    assert config.solver_kwargs == custom_kwargs


@pytest.mark.unit
def test_solver_factory_config_post_init():
    """Test SolverFactoryConfig __post_init__ initializes solver_kwargs."""
    config = SolverFactoryConfig(solver_kwargs=None)
    assert config.solver_kwargs == {}


# ===================================================================
# Test SolverFactory._update_config_with_kwargs
# ===================================================================


@pytest.mark.unit
def test_update_config_with_kwargs_picard():
    """Test updating Picard config parameters."""
    base_config = MFGSolverConfig()
    updated = SolverFactory._update_config_with_kwargs(
        base_config,
        max_picard_iterations=150,
        picard_tolerance=1e-6,
    )

    assert updated.picard.max_iterations == 150
    assert updated.picard.tolerance == 1e-6
    # Base config should be unchanged (deepcopy)
    assert base_config.picard.max_iterations != 150


@pytest.mark.unit
def test_update_config_with_kwargs_newton():
    """Test updating Newton config parameters."""
    base_config = MFGSolverConfig()
    updated = SolverFactory._update_config_with_kwargs(
        base_config,
        max_newton_iterations=50,
        newton_tolerance=1e-7,
    )

    assert updated.hjb.newton.max_iterations == 50
    assert updated.hjb.newton.tolerance == 1e-7


@pytest.mark.unit
def test_update_config_with_kwargs_particles():
    """Test updating particle config parameters."""
    base_config = MFGSolverConfig()
    updated = SolverFactory._update_config_with_kwargs(
        base_config,
        num_particles=8000,
    )

    assert updated.fp.particle.num_particles == 8000


@pytest.mark.unit
def test_update_config_with_kwargs_num_particles_non_particle_fp():
    """Issue #1317: num_particles kwarg must not raise when fp.method has no
    particle sub-config.

    When ``fp.method`` is e.g. 'fdm'/'fem'/'network', ``fp.particle`` is None.
    The factory must guard the assignment (mirroring the hjb.gfdm/delta guard)
    and log a warning instead of raising ``AttributeError``.
    """
    base_config = MFGSolverConfig(fp=FPConfig(method="fdm"))
    # Precondition that triggers the bug: no particle sub-config present.
    assert base_config.fp.particle is None

    with patch("mfgarchon.factory.solver_factory.logger") as mock_logger:
        updated = SolverFactory._update_config_with_kwargs(
            base_config,
            num_particles=8000,
        )

    # Did not raise; particle sub-config stays None and a warning is logged.
    assert updated.fp.particle is None
    mock_logger.warning.assert_called_once()


@pytest.mark.unit
def test_update_config_with_kwargs_picard_damping():
    """Test updating picard damping factor."""
    base_config = MFGSolverConfig()
    updated = SolverFactory._update_config_with_kwargs(
        base_config,
        max_picard_iterations=200,
    )

    assert updated.picard.max_iterations == 200


# ===================================================================
# Test SolverFactory.create_solver Validation
# ===================================================================


@pytest.mark.unit
def test_create_solver_none_problem():
    """Test create_solver raises error for None problem."""
    with pytest.raises(ValueError) as exc_info:
        SolverFactory.create_solver(problem=None)

    assert "Problem cannot be None" in str(exc_info.value)
    assert "MFGProblem" in str(exc_info.value)


@pytest.mark.unit
def test_create_solver_invalid_solver_type():
    """Test create_solver raises error for invalid solver type."""
    problem = MockMFGProblem()

    with pytest.raises(ValueError) as exc_info:
        SolverFactory.create_solver(problem=problem, solver_type="invalid_type")

    assert "Unknown solver type" in str(exc_info.value)
    assert "invalid_type" in str(exc_info.value)
    assert "fixed_point" in str(exc_info.value)


@pytest.mark.unit
def test_create_solver_fixed_point_missing_solvers():
    """Test fixed_point solver requires hjb_solver and fp_solver."""
    problem = MockMFGProblem()

    with pytest.raises(ValueError) as exc_info:
        SolverFactory.create_solver(problem=problem, solver_type="fixed_point")

    assert "requires both hjb_solver and fp_solver" in str(exc_info.value)


# ===================================================================
# Test SolverFactory.create_solver with Mocks
# ===================================================================


@pytest.mark.unit
def test_create_solver_fixed_point_with_solvers():
    """Test creating fixed_point solver with provided solvers."""
    problem = MockMFGProblem()
    mock_hjb = Mock()
    mock_fp = Mock()

    with patch("mfgarchon.factory.solver_factory.FixedPointIterator") as MockIterator:
        MockIterator.return_value = Mock()

        solver = SolverFactory.create_solver(
            problem=problem,
            solver_type="fixed_point",
            hjb_solver=mock_hjb,
            fp_solver=mock_fp,
        )

        assert solver is not None
        MockIterator.assert_called_once()
        call_kwargs = MockIterator.call_args[1]
        assert call_kwargs["problem"] == problem
        assert call_kwargs["hjb_solver"] == mock_hjb
        assert call_kwargs["fp_solver"] == mock_fp


@pytest.mark.unit
def test_create_solver_custom_config():
    """Test create_solver with custom configuration."""
    problem = MockMFGProblem()
    mock_hjb = Mock()
    mock_fp = Mock()
    custom_config = MFGSolverConfig(convergence_tolerance=1e-8)

    with patch("mfgarchon.factory.solver_factory.FixedPointIterator") as MockIterator:
        MockIterator.return_value = Mock()

        solver = SolverFactory.create_solver(
            problem=problem,
            solver_type="fixed_point",
            hjb_solver=mock_hjb,
            fp_solver=mock_fp,
            config=custom_config,
        )

        assert solver is not None
        call_kwargs = MockIterator.call_args[1]
        assert call_kwargs["config"] is not None


# ===================================================================
# Test Convenience Function: create_solver
# ===================================================================


@pytest.mark.unit
def test_convenience_create_solver():
    """Test convenience create_solver function."""
    problem = MockMFGProblem()
    mock_hjb = Mock()
    mock_fp = Mock()

    with patch("mfgarchon.factory.solver_factory.SolverFactory.create_solver") as mock_create:
        mock_create.return_value = Mock()

        solver = create_solver(
            problem=problem,
            hjb_solver=mock_hjb,
            fp_solver=mock_fp,
        )

        assert solver is not None
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["problem"] == problem
        assert call_kwargs["hjb_solver"] == mock_hjb
        assert call_kwargs["fp_solver"] == mock_fp


# ===================================================================
# Test Removed Functions Raise NotImplementedError
# ===================================================================


@pytest.mark.unit
def test_create_basic_solver():
    """Test create_basic_solver raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        create_basic_solver()

    assert "has been removed" in str(exc_info.value)


@pytest.mark.unit
def test_create_standard_solver():
    """Test create_standard_solver raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        create_standard_solver()

    assert "has been removed" in str(exc_info.value)


@pytest.mark.unit
def test_create_fast_solver_removed():
    """Test create_fast_solver raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        create_fast_solver()

    assert "has been removed" in str(exc_info.value)


@pytest.mark.unit
def test_create_semi_lagrangian_solver():
    """Test create_semi_lagrangian_solver raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        create_semi_lagrangian_solver()

    assert "has been removed" in str(exc_info.value)


@pytest.mark.unit
def test_create_accurate_solver():
    """Test create_accurate_solver raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        create_accurate_solver()

    assert "has been removed" in str(exc_info.value)


@pytest.mark.unit
def test_create_research_solver():
    """Test create_research_solver raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        create_research_solver()

    assert "has been removed" in str(exc_info.value)


@pytest.mark.unit
def test_create_amr_solver():
    """Test create_amr_solver raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        create_amr_solver()

    assert "has been removed" in str(exc_info.value)


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_exports_solver_factory_class():
    """Test module exports SolverFactory class."""
    from mfgarchon.factory import solver_factory

    assert hasattr(solver_factory, "SolverFactory")
    assert isinstance(solver_factory.SolverFactory, type)


@pytest.mark.unit
def test_module_exports_convenience_functions():
    """Test module exports all convenience functions."""
    from mfgarchon.factory import solver_factory

    expected_functions = [
        "create_solver",
        "create_basic_solver",
        "create_standard_solver",
        "create_fast_solver",
        "create_semi_lagrangian_solver",
        "create_accurate_solver",
        "create_research_solver",
        "create_amr_solver",
    ]

    for func_name in expected_functions:
        assert hasattr(solver_factory, func_name)
        assert callable(getattr(solver_factory, func_name))

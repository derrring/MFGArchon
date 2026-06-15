#!/usr/bin/env python3
"""
Unit tests for General MFG Factory

Issue #673: Updated to class-based Hamiltonian API.

Tests the GeneralMFGFactory class for creating MFG problems from
various input formats (Hamiltonians, configs, files).
"""

import tempfile
from pathlib import Path

import pytest

import numpy as np

# Check if OmegaConf is available
try:
    import omegaconf

    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    omegaconf = None

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.factory.general_mfg_factory import (
    GeneralMFGFactory,
    create_general_mfg_problem,
    get_general_factory,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def factory():
    """Create GeneralMFGFactory instance."""
    return GeneralMFGFactory()


@pytest.fixture
def sample_hamiltonian():
    """Create sample class-based Hamiltonian."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


@pytest.fixture
def sample_functions():
    """Create sample initial/final condition functions."""

    def potential(x):
        return 0.5 * np.sum(x**2, axis=-1)

    def initial_density(x):
        return np.exp(-10 * np.sum((x - 0.5) ** 2, axis=-1))

    def final_value(x):
        return 0.5 * np.sum(x**2, axis=-1)

    return {
        "potential": potential,
        "initial_density": initial_density,
        "final_value": final_value,
    }


@pytest.fixture
def domain_config():
    """Create sample domain configuration."""
    return {"xmin": 0.0, "xmax": 1.0, "Nx": 20}


@pytest.fixture
def time_config():
    """Create sample time configuration."""
    return {"T": 1.0, "Nt": 10}


@pytest.fixture
def solver_config():
    """Create sample solver configuration."""
    return {"sigma": 0.1, "coupling_coefficient": 1.0}


# ============================================================================
# Test: Factory Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_factory_initialization():
    """Test GeneralMFGFactory can be instantiated."""
    factory = GeneralMFGFactory()

    assert factory is not None
    assert hasattr(factory, "function_registry")
    assert hasattr(factory, "hamiltonian_registry")
    assert isinstance(factory.function_registry, dict)
    assert isinstance(factory.hamiltonian_registry, dict)
    assert len(factory.function_registry) == 0
    assert len(factory.hamiltonian_registry) == 0


@pytest.mark.unit
@pytest.mark.fast
def test_get_general_factory():
    """Test get_general_factory convenience function."""
    factory = get_general_factory()

    assert isinstance(factory, GeneralMFGFactory)


# ============================================================================
# Test: Registration
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_register_single_function(factory):
    """Test registering a single function."""

    def test_func(x):
        return x**2

    factory.register_functions({"test_func": test_func})

    assert "test_func" in factory.function_registry
    assert factory.function_registry["test_func"] is test_func


@pytest.mark.unit
@pytest.mark.fast
def test_register_multiple_functions(factory, sample_functions):
    """Test registering multiple functions at once."""
    factory.register_functions(sample_functions)

    assert len(factory.function_registry) == len(sample_functions)
    for name in sample_functions:
        assert name in factory.function_registry


@pytest.mark.unit
@pytest.mark.fast
def test_register_functions_overwrites(factory):
    """Test that registering function with same name overwrites."""

    def func_v1(x):
        return x

    def func_v2(x):
        return x**2

    factory.register_functions({"func": func_v1})
    assert factory.function_registry["func"] is func_v1

    factory.register_functions({"func": func_v2})
    assert factory.function_registry["func"] is func_v2


@pytest.mark.unit
@pytest.mark.fast
def test_register_hamiltonian(factory, sample_hamiltonian):
    """Test registering a Hamiltonian."""
    factory.register_hamiltonian("test_H", sample_hamiltonian)

    assert "test_H" in factory.hamiltonian_registry
    assert factory.hamiltonian_registry["test_H"] is sample_hamiltonian


# ============================================================================
# Test: Create from Hamiltonian
# ============================================================================


@pytest.mark.unit
def test_create_from_hamiltonian_minimal(factory, sample_hamiltonian, domain_config, time_config, sample_functions):
    """Test creating MFG problem with minimal required components."""
    problem = factory.create_from_hamiltonian(
        hamiltonian=sample_hamiltonian,
        domain_config=domain_config,
        time_config=time_config,
        m_initial=sample_functions["initial_density"],
        u_terminal=sample_functions["final_value"],
    )

    assert problem is not None
    assert hasattr(problem, "T")
    assert time_config["T"] == problem.T
    assert hasattr(problem, "H")


@pytest.mark.unit
def test_create_from_hamiltonian_with_solver_config(
    factory, sample_hamiltonian, domain_config, time_config, solver_config, sample_functions
):
    """Test creating problem with solver configuration."""
    problem = factory.create_from_hamiltonian(
        hamiltonian=sample_hamiltonian,
        domain_config=domain_config,
        time_config=time_config,
        solver_config=solver_config,
        m_initial=sample_functions["initial_density"],
        u_terminal=sample_functions["final_value"],
    )

    assert problem is not None
    assert hasattr(problem, "sigma")


@pytest.mark.unit
def test_create_from_hamiltonian_with_optional_components(
    factory, sample_hamiltonian, domain_config, time_config, sample_functions
):
    """Test creating problem with optional components."""
    problem = factory.create_from_hamiltonian(
        hamiltonian=sample_hamiltonian,
        domain_config=domain_config,
        time_config=time_config,
        potential_func=sample_functions["potential"],
        m_initial=sample_functions["initial_density"],
        u_terminal=sample_functions["final_value"],
    )

    assert problem is not None
    assert hasattr(problem, "T")
    assert time_config["T"] == problem.T


@pytest.mark.unit
def test_create_from_hamiltonian_with_description(
    factory, sample_hamiltonian, domain_config, time_config, sample_functions
):
    """Test creating problem with description."""
    description = "Test MFG Problem"

    problem = factory.create_from_hamiltonian(
        hamiltonian=sample_hamiltonian,
        domain_config=domain_config,
        time_config=time_config,
        description=description,
        m_initial=sample_functions["initial_density"],
        u_terminal=sample_functions["final_value"],
    )

    assert problem is not None


# ============================================================================
# Test: Create from Config Dict
# ============================================================================


@pytest.mark.unit
def test_create_from_config_dict_simple(factory):
    """Test creating problem from simple config dict."""
    config = {
        "hamiltonian": {"type": "separable", "control_cost": 1.0, "coupling_coefficient": 1.0},
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 20},
        "time": {"T": 1.0, "Nt": 10},
        # 'solver' section is required (Issue #1284 / 2026-06-11 survey: fail-loud fix).
        "solver": {"sigma": 0.5, "coupling_coefficient": 1.0},
        "functions": {
            "m_initial": "lambda x: np.exp(-10 * (x - 0.5)**2)",
            "u_final": "lambda x: 0.5 * x**2",
        },
    }

    problem = factory.create_from_config_dict(config)
    assert problem is not None


@pytest.mark.unit
@pytest.mark.fast
def test_create_from_config_dict_validates_required_fields(factory):
    """Test that missing required fields are caught."""
    from mfgarchon.utils.validation import ValidationError

    incomplete_config = {
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 20}
        # Missing hamiltonian and time
    }

    with pytest.raises((ValueError, ValidationError)):
        factory.create_from_config_dict(incomplete_config)


# ============================================================================
# Test: Config Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_validate_config_valid(factory):
    """Test validation accepts valid config."""
    config = {
        "hamiltonian": {"type": "separable", "control_cost": 1.0},
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 20},
        "time": {"T": 1.0, "Nt": 10},
    }

    validated = factory.validate_config(config)

    assert isinstance(validated, dict)
    assert validated["valid"] is True
    assert len(validated["errors"]) == 0


@pytest.mark.unit
@pytest.mark.fast
def test_validate_config_missing_hamiltonian_section(factory):
    """Test validation catches missing hamiltonian section."""
    config = {
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 20},
        "time": {"T": 1.0, "Nt": 10},
    }

    validated = factory.validate_config(config)

    assert validated["valid"] is False
    assert "Missing required section: hamiltonian" in validated["errors"]


@pytest.mark.unit
@pytest.mark.fast
def test_validate_config_missing_domain(factory):
    """Test validation catches missing domain."""
    config = {
        "hamiltonian": {"type": "separable"},
        "time": {"T": 1.0, "Nt": 10},
    }

    validated = factory.validate_config(config)

    assert validated["valid"] is False
    assert "Missing required section: domain" in validated["errors"]


@pytest.mark.unit
@pytest.mark.fast
def test_validate_config_missing_time(factory):
    """Test validation catches missing time."""
    config = {
        "hamiltonian": {"type": "separable"},
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 20},
    }

    validated = factory.validate_config(config)

    assert validated["valid"] is False
    assert "Missing required section: time" in validated["errors"]


# ============================================================================
# Test: Template Creation
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not OMEGACONF_AVAILABLE, reason="OmegaConf not available")
def test_create_template_config(factory):
    """Test creating template configuration file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = Path(tmpdir) / "template_config.yaml"

        factory.create_template_config(str(template_path))

        # Check file was created
        assert template_path.exists()

        # Check file has content
        content = template_path.read_text()
        assert len(content) > 0
        assert "hamiltonian" in content.lower() or "domain" in content.lower()


# ============================================================================
# Test: Convenience Functions
# ============================================================================


@pytest.mark.unit
def test_create_general_mfg_problem_from_hamiltonian():
    """Test module-level convenience function."""
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )

    def initial_density(x):
        return np.exp(-10 * np.sum((x - 0.5) ** 2, axis=-1))

    def final_value(x):
        return 0.5 * np.sum(x**2, axis=-1)

    # Use convenience function with explicit parameters
    problem = create_general_mfg_problem(
        hamiltonian=hamiltonian,
        xmin=0.0,
        xmax=1.0,
        Nx=20,
        T=1.0,
        Nt=10,
        m_initial=initial_density,
        u_terminal=final_value,
    )

    assert problem is not None


# ============================================================================
# Test: Error Handling
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_create_from_hamiltonian_invalid_type(factory, domain_config, time_config):
    """Test error when hamiltonian is not HamiltonianBase."""
    with pytest.raises((ValueError, TypeError)):
        factory.create_from_hamiltonian(
            hamiltonian="not a hamiltonian",  # Invalid type
            domain_config=domain_config,
            time_config=time_config,
        )


@pytest.mark.unit
@pytest.mark.fast
def test_create_from_hamiltonian_missing_domain(factory, sample_hamiltonian, time_config):
    """Test error when domain config is missing."""
    with pytest.raises((TypeError, KeyError)):
        factory.create_from_hamiltonian(
            hamiltonian=sample_hamiltonian,
            domain_config=None,  # type: ignore
            time_config=time_config,
        )


@pytest.mark.unit
@pytest.mark.fast
def test_create_from_hamiltonian_legacy_domain_missing_nx(factory, sample_hamiltonian, time_config, sample_functions):
    """Legacy 1D domain spec (xmin/xmax/Lx) without Nx must fail fast with a
    clear ValueError naming Nx, not a bare KeyError (Issue #1363)."""
    with pytest.raises(ValueError, match="Nx"):
        factory.create_from_hamiltonian(
            hamiltonian=sample_hamiltonian,
            domain_config={"xmin": 0.0, "xmax": 1.0, "Lx": 1.0},  # no Nx
            time_config=time_config,
            m_initial=sample_functions["initial_density"],
            u_terminal=sample_functions["final_value"],
        )


@pytest.mark.unit
@pytest.mark.fast
def test_create_from_hamiltonian_missing_time(factory, sample_hamiltonian, domain_config):
    """Test error when time config is missing."""
    with pytest.raises((TypeError, KeyError)):
        factory.create_from_hamiltonian(
            hamiltonian=sample_hamiltonian,
            domain_config=domain_config,
            time_config=None,  # type: ignore
        )


# ============================================================================
# Test: Integration
# ============================================================================


@pytest.mark.unit
def test_end_to_end_problem_creation(
    factory, sample_hamiltonian, sample_functions, domain_config, time_config, solver_config
):
    """Test complete workflow from Hamiltonian to problem."""
    # Register Hamiltonian
    factory.register_hamiltonian("test_H", sample_hamiltonian)

    # Create problem
    problem = factory.create_from_hamiltonian(
        hamiltonian=sample_hamiltonian,
        domain_config=domain_config,
        time_config=time_config,
        solver_config=solver_config,
        potential_func=sample_functions["potential"],
        m_initial=sample_functions["initial_density"],
        u_terminal=sample_functions["final_value"],
    )

    # Verify problem has all components
    assert problem is not None
    assert hasattr(problem, "H")
    assert hasattr(problem, "dH_dm")
    assert hasattr(problem, "T")
    assert time_config["T"] == problem.T
    assert hasattr(problem, "sigma")
    assert problem.sigma == solver_config["sigma"]


@pytest.mark.unit
def test_multiple_problems_from_same_factory(factory, sample_hamiltonian, sample_functions, domain_config, time_config):
    """Test creating multiple problems from same factory instance."""
    # Create first problem
    problem1 = factory.create_from_hamiltonian(
        hamiltonian=sample_hamiltonian,
        domain_config=domain_config,
        time_config=time_config,
        m_initial=sample_functions["initial_density"],
        u_terminal=sample_functions["final_value"],
    )

    # Create second problem with different config
    problem2 = factory.create_from_hamiltonian(
        hamiltonian=sample_hamiltonian,
        domain_config={"xmin": 0.0, "xmax": 2.0, "Nx": 30},
        time_config={"T": 2.0, "Nt": 20},
        m_initial=sample_functions["initial_density"],
        u_terminal=sample_functions["final_value"],
    )

    # Both should be valid but different
    assert problem1 is not None
    assert problem2 is not None
    assert problem1.T != problem2.T


# ============================================================================
# Test: Function Registry Usage
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_function_registry_accessible_after_registration(factory):
    """Test registered functions can be accessed."""

    def custom_func(x):
        return x * 2

    factory.register_functions({"custom": custom_func})

    # Should be able to retrieve function
    retrieved = factory.function_registry["custom"]
    assert retrieved is custom_func
    assert retrieved(5) == 10

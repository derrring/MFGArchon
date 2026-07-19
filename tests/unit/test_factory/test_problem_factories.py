"""
Unit tests for unified problem factories (Phase 3.3).

Tests the new factory functions that support both unified MFGProblem
and legacy specialized problem classes.

NOTE: Uses TensorProductGrid (unified geometry API) for grid geometries.
For unstructured meshes, use Mesh2D/Mesh3D.

Updated: January 2026 - Issue #673: Class-based Hamiltonian API.
"""

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.factory import (
    create_crowd_problem,
    create_lq_problem,
    create_mfg_problem,
    create_standard_problem,
    create_stochastic_problem,
)
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _default_hamiltonian():
    """Default class-based Hamiltonian for tests (Issue #673)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


@pytest.fixture
def simple_domain():
    """Create simple 1D domain for testing using TensorProductGrid."""
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))


@pytest.fixture
def simple_2d_domain():
    """Create simple 2D domain for testing using TensorProductGrid."""
    return TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 21], boundary_conditions=no_flux_bc(dimension=2)
    )


def test_create_standard_problem(simple_domain):
    """Test standard MFG problem creation with class-based Hamiltonian (Issue #673)."""

    def terminal_cost(x):
        return x**2

    def initial_density(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    # Test unified API with class-based Hamiltonian
    problem = create_standard_problem(
        hamiltonian=_default_hamiltonian(),
        terminal_cost=terminal_cost,
        initial_density=initial_density,
        geometry=simple_domain,
        use_unified=True,
    )

    assert isinstance(problem, MFGProblem)
    assert problem.components is not None
    assert problem.components.hamiltonian is not None
    assert problem.components.problem_type == "standard"


def test_create_lq_problem(simple_domain):
    """Test Linear-Quadratic MFG problem creation (Issue #673)."""

    def terminal_cost(x):
        return x**2

    def initial_density(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    problem = create_lq_problem(
        geometry=simple_domain,
        terminal_cost=terminal_cost,
        initial_density=initial_density,
        running_cost_control=0.5,
        running_cost_congestion=1.0,
    )

    assert isinstance(problem, MFGProblem)
    assert problem.components is not None
    assert problem.components.hamiltonian is not None


def test_factory_respects_time_horizon_and_num_timesteps(simple_domain, simple_2d_domain):
    """Factories must honor time_horizon / num_timesteps.

    Regression: create_mfg_problem built MFGProblem(time_horizon=, num_timesteps=), but
    MFGProblem takes T / Nt, so both fell into **kwargs and were silently dropped (T, Nt
    defaulting to 1.0 / 51). Discriminating: reverting the T=/Nt= mapping makes these
    assertions fail (T=1.0, Nt=51 regardless of the requested values).
    """
    lq = create_lq_problem(
        geometry=simple_domain,
        terminal_cost=lambda x: 0.0,
        initial_density=lambda x: 1.0,
        time_horizon=2.5,
        num_timesteps=7,
    )
    assert (lq.T, lq.Nt) == (2.5, 7)

    ham = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    std = create_standard_problem(
        hamiltonian=ham,
        terminal_cost=lambda x: 0.0,
        initial_density=lambda x: 1.0,
        geometry=simple_domain,
        time_horizon=3.0,
        num_timesteps=9,
    )
    assert (std.T, std.Nt) == (3.0, 9)

    crowd = create_crowd_problem(
        geometry=simple_2d_domain,
        target_location=np.array([0.8, 0.8]),
        initial_density=lambda x: 1.0,
        time_horizon=1.5,
        num_timesteps=6,
    )
    assert (crowd.T, crowd.Nt) == (1.5, 6)


def test_create_crowd_problem(simple_2d_domain):
    """Test crowd dynamics MFG problem creation (Issue #673 class-based Hamiltonian)."""
    target = np.array([0.8, 0.8])

    def initial_density(x):
        return np.exp(-10 * np.linalg.norm(x - np.array([0.2, 0.2])) ** 2)

    problem = create_crowd_problem(
        geometry=simple_2d_domain,
        target_location=target,
        initial_density=initial_density,
    )

    assert isinstance(problem, MFGProblem)
    assert problem.components is not None
    # Issue #673: Potential is now embedded in SeparableHamiltonian, not separate potential_func
    assert problem.components.hamiltonian is not None
    # Potential is stored as private _potential attribute in SeparableHamiltonian
    assert problem.components.hamiltonian._potential is not None


def test_create_stochastic_problem(simple_domain):
    """Test stochastic MFG problem creation with class-based Hamiltonian (Issue #673)."""

    def terminal_cost(x):
        return x**2

    def initial_density(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    def common_noise(t):
        return np.sin(2 * np.pi * t)

    problem = create_stochastic_problem(
        hamiltonian=_default_hamiltonian(),
        terminal_cost=terminal_cost,
        initial_density=initial_density,
        geometry=simple_domain,
        noise_intensity=0.5,
        common_noise=common_noise,
    )

    assert isinstance(problem, MFGProblem)
    assert problem.components is not None
    # Stochastic parameters are stored in the parameters dict
    assert problem.components.parameters.get("noise_intensity") == 0.5
    assert problem.components.parameters.get("common_noise_func") is not None
    assert problem.components.problem_type == "stochastic"


def test_create_mfg_problem_with_components(simple_domain):
    """Test main factory function with explicit components (Issue #673)."""
    components = MFGComponents(
        hamiltonian=_default_hamiltonian(),
        u_terminal=lambda x: x**2,
        m_initial=lambda x: np.exp(-(x**2)),
        problem_type="standard",
    )

    problem = create_mfg_problem("standard", components, geometry=simple_domain)

    assert isinstance(problem, MFGProblem)
    assert problem.components.problem_type == "standard"


def test_backward_compatibility_warning(simple_domain):
    """Test that legacy API triggers deprecation warning."""

    def terminal_cost(x):
        return x**2

    def initial_density(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    # Test that use_unified=False triggers warning
    with pytest.warns(DeprecationWarning, match="deprecated"):
        problem = create_lq_problem(
            geometry=simple_domain,
            terminal_cost=terminal_cost,
            initial_density=initial_density,
            use_unified=False,
        )

    # Should still create a valid problem
    assert problem is not None


def test_problem_type_detection():
    """Test automatic problem type detection (Issue #673)."""

    domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))

    # Default components for Issue #670 Fail Fast requirement
    def m_initial(x):
        return np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze()

    def u_final(x):
        return 0.0

    # Standard MFG with class-based Hamiltonian
    components_standard = MFGComponents(
        hamiltonian=_default_hamiltonian(),
        m_initial=m_initial,
        u_terminal=u_final,
        problem_type="standard",
    )
    problem = MFGProblem(geometry=domain, components=components_standard)
    assert problem.components.problem_type == "standard"

    # Stochastic MFG (type set explicitly via problem_type)
    components_stochastic = MFGComponents(
        hamiltonian=_default_hamiltonian(),
        m_initial=m_initial,
        u_terminal=u_final,
        parameters={"noise_intensity": 0.5},
        problem_type="stochastic",
    )
    problem = MFGProblem(geometry=domain, components=components_stochastic)
    assert problem.components.problem_type == "stochastic"


def _variational_components():
    """MFGComponents for the variational-branch tests (Issue #1642, capability D1)."""
    return MFGComponents(
        hamiltonian=_default_hamiltonian(),
        u_terminal=lambda x: x**2,
        m_initial=lambda x: np.exp(-(x**2)),
        problem_type="variational",
    )


def test_variational_legacy_branch_raises_actionable_error(simple_domain):
    """Legacy variational branch must fail with a diagnostic, not ModuleNotFoundError.

    Before #1642 (D1) this branch imported ``mfgarchon.solvers.variational``, a module
    that has never existed, so a documented public call raised ModuleNotFoundError with
    no guidance. Routing it to the real
    ``mfgarchon.alg.optimization.variational_problem.VariationalMFGProblem`` is not a
    fix either: that class takes ``(xmin, xmax, Nx, ...)`` plus VariationalMFGComponents,
    so ``geometry`` would fall into ``**kwargs`` and the domain would silently default to
    [0, 1] with Nx=51. The branch therefore refuses.

    Catches: (a) a regression to the ghost import (ModuleNotFoundError is not a subclass
    of NotImplementedError, so the raises-check fails); (b) a "fix" that silently returns
    a problem on the wrong domain (no exception at all).
    """
    with (
        pytest.warns(DeprecationWarning, match="deprecated"),
        pytest.raises(NotImplementedError) as excinfo,
    ):
        create_mfg_problem(
            "variational",
            _variational_components(),
            geometry=simple_domain,
            use_unified=False,
        )

    message = str(excinfo.value)
    # The diagnostic must name where the real class lives and which issue blocks the stack.
    assert "mfgarchon.alg.optimization.variational_problem" in message
    assert "VariationalMFGComponents" in message
    assert "#1342" in message
    assert "use_unified=True" in message


def test_variational_unified_path_still_builds_a_problem(simple_domain):
    """Only the legacy specialized branch is narrowed; the unified path is untouched.

    Catches over-removal: narrowing the whole "variational" problem_type instead of the
    dead ``use_unified=False`` branch would break this.
    """
    problem = create_mfg_problem("variational", _variational_components(), geometry=simple_domain)

    assert isinstance(problem, MFGProblem)
    assert problem.components.problem_type == "variational"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
Unit tests for mfgarchon/core/mfg_problem.py

Tests MFG problem base class including:
- MFGComponents dataclass
- MFGProblem initialization and defaults
- Default mathematical functions
- Custom components handling
- Hamiltonian methods
- Boundary conditions
- Getter methods
"""

import pytest

import numpy as np

from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BoundaryConditions, dirichlet_bc, no_flux_bc

# ===========================================================================
# Test Helpers - Issue #670: m_initial/u_final now required in MFGComponents
# ===========================================================================


def default_geometry(bounds=None, Nx_points=None, dimension=1):
    """Create a default geometry with explicit boundary conditions."""
    if bounds is None:
        bounds = [(0.0, 1.0)] if dimension == 1 else [(0.0, 1.0)] * dimension
    if Nx_points is None:
        Nx_points = [11] if dimension == 1 else [11] * dimension
    return TensorProductGrid(
        bounds=bounds,
        Nx_points=Nx_points,
        boundary_conditions=no_flux_bc(dimension=dimension),
    )


def default_hamiltonian():
    """Create a default class-based Hamiltonian for testing.

    Issue #673: Hamiltonian required (no default in MFGComponents)

    Returns:
        SeparableHamiltonian: H = ½|p|²/λ - m²
    """
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )


def default_components():
    """Create default MFGComponents with Hamiltonian, m_initial, and u_final.

    Issue #670: m_initial/u_final required
    Issue #673: Hamiltonian required (no default)
    """
    return MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian at center
        u_terminal=lambda x: x**2,  # Quadratic terminal cost
    )


def create_test_problem(**kwargs):
    """Create a test MFGProblem with required m_initial/u_final (Issue #670).

    Uses default geometry and components if not provided.
    """
    if "geometry" not in kwargs:
        kwargs["geometry"] = default_geometry()
    if "components" not in kwargs:
        kwargs["components"] = default_components()
    return MFGProblem(**kwargs)


# ===================================================================
# Test MFGComponents Dataclass
# ===================================================================


@pytest.mark.unit
def test_mfg_components_defaults():
    """Test MFGComponents with required Hamiltonian and default optional values."""
    # Issue #673: Hamiltonian is now required
    H = default_hamiltonian()
    components = MFGComponents(hamiltonian=H)

    # Required: Hamiltonian must be set
    assert components._hamiltonian_class is H
    assert components.hamiltonian is H

    # Optional fields have None/empty defaults
    assert components.potential_func is None
    assert components.m_initial is None
    assert components.u_final is None
    assert components.boundary_conditions is None
    assert components.parameters == {}
    assert components.description == "MFG Problem"
    assert components.problem_type == "mfg"


@pytest.mark.unit
def test_mfg_components_custom_values():
    """Test MFGComponents with custom values (class-based Hamiltonian)."""
    # Issue #673: Class-based Hamiltonian API only
    H = default_hamiltonian()

    components = MFGComponents(
        hamiltonian=H,
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        parameters={"param1": 1.0},
        description="Custom Problem",
        problem_type="custom",
    )

    # Issue #673: Class-based Hamiltonian stored directly (no legacy func wrappers)
    assert components._hamiltonian_class is H
    assert components.parameters == {"param1": 1.0}
    assert components.description == "Custom Problem"
    assert components.problem_type == "custom"


# ===================================================================
# Test MFGProblem Initialization
# ===================================================================


@pytest.mark.unit
def test_mfg_problem_default_initialization():
    """Test MFGProblem with default parameters (Issue #670: requires m_initial/u_final)."""
    problem = create_test_problem()

    # Domain parameters (default_geometry uses Nx=11, so 10 intervals)
    bounds = problem.geometry.get_bounds()
    assert bounds[0][0] == 0.0
    assert bounds[1][0] == 1.0
    assert problem._get_domain_length() == 1.0
    assert problem.geometry.get_grid_shape()[0] == 11  # 11 points

    # Time parameters
    assert problem.T == 1.0
    assert problem.Nt == 51
    assert problem.dt == pytest.approx(1.0 / 51)

    # Physical parameters (None → 0 for diffusion/drift)
    assert problem.sigma == 0.0
    assert problem.volatility_field == 0.0
    assert problem.drift_field == 0.0
    assert problem.coupling_coefficient == 0.5

    # Custom components (now required)
    assert problem.components is not None  # Must have components now
    assert problem.is_custom is True  # Has custom m_initial/u_final


@pytest.mark.unit
def test_mfg_problem_custom_domain():
    """Test MFGProblem with custom domain parameters."""
    geometry = default_geometry(bounds=[(-1.0, 2.0)], Nx_points=[101])  # Nx=100 intervals
    problem = create_test_problem(geometry=geometry)

    bounds = problem.geometry.get_bounds()
    assert bounds[0][0] == -1.0
    assert bounds[1][0] == 2.0
    assert problem._get_domain_length() == 3.0
    assert problem.geometry.get_grid_shape()[0] - 1 == 100  # Nx intervals
    assert problem.geometry.get_grid_spacing()[0] == pytest.approx(3.0 / 100)
    assert problem.geometry.get_grid_shape()[0] == 101  # Nx+1 points


@pytest.mark.unit
def test_mfg_problem_custom_time():
    """Test MFGProblem with custom time parameters."""
    problem = create_test_problem(T=2.0, Nt=100)

    assert problem.T == 2.0
    assert problem.Nt == 100
    assert problem.dt == pytest.approx(2.0 / 100)
    assert len(problem.tSpace) == 101  # Nt + 1


@pytest.mark.unit
def test_mfg_problem_custom_coefficients():
    """Test MFGProblem with custom coefficients (Issue #811 convention)."""
    # sigma= provides SDE volatility directly
    problem = create_test_problem(sigma=0.5, coupling_coefficient=0.8)

    assert problem.sigma == 0.5
    assert problem.coupling_coefficient == 0.8


@pytest.mark.unit
def test_mfg_problem_spatial_grid():
    """Test MFGProblem spatial grid generation."""
    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[11])  # Nx=10 intervals
    problem = create_test_problem(geometry=geometry)

    assert problem.geometry.get_grid_shape()[0] == 11  # Nx+1 points
    x_grid = problem.geometry.get_spatial_grid()
    assert x_grid[0] == 0.0
    assert x_grid[-1] == 1.0
    # Check uniform spacing
    spacing = np.diff(x_grid)
    assert np.allclose(spacing, spacing[0])


@pytest.mark.unit
def test_mfg_problem_temporal_grid():
    """Test MFGProblem temporal grid generation."""
    problem = create_test_problem(T=1.0, Nt=10)

    assert len(problem.tSpace) == 11  # Nt + 1 points
    assert problem.tSpace[0] == 0.0
    assert problem.tSpace[-1] == 1.0
    # Check uniform spacing
    spacing = np.diff(problem.tSpace)
    assert np.allclose(spacing, spacing[0])


# ===================================================================
# Test Default Functions
# ===================================================================


@pytest.mark.unit
def test_mfg_problem_default_potential():
    """Test default potential function is initialized (Issue #670: requires m_initial/u_final)."""
    # Issue #671 will address: potential defaults to zero if not provided
    problem = create_test_problem()

    assert hasattr(problem, "f_potential")
    assert isinstance(problem.f_potential, np.ndarray)
    assert len(problem.f_potential) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    # Potential defaults to zero when not provided (Issue #671 will change this)
    assert np.allclose(problem.f_potential, 0.0)


@pytest.mark.unit
def test_mfg_problem_default_final_value():
    """Test final value function is initialized (Issue #670: must come from MFGComponents)."""
    problem = create_test_problem()

    assert hasattr(problem, "u_terminal")
    assert isinstance(problem.u_terminal, np.ndarray)
    assert len(problem.u_terminal) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    # Check final value is non-zero (set from default_components: x**2)
    assert not np.allclose(problem.u_terminal, 0.0)


@pytest.mark.unit
def test_mfg_problem_default_initial_density():
    """Test initial density function is initialized (Issue #670: must come from MFGComponents)."""
    problem = create_test_problem()

    assert hasattr(problem, "m_initial")
    assert isinstance(problem.m_initial, np.ndarray)
    assert len(problem.m_initial) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    # Check initial density is non-negative
    assert np.all(problem.m_initial >= 0.0)
    # Check it has some mass
    assert np.sum(problem.m_initial) > 0.0


# ===================================================================
# Test Custom Components
# ===================================================================


@pytest.mark.unit
def test_mfg_problem_with_custom_potential():
    """Test MFGProblem with custom potential function."""

    def custom_potential(x, t):
        return x**2

    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[11])
    # Issue #670: must provide m_initial and u_final
    # Issue #673: Hamiltonian required
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        potential_func=custom_potential,
        m_initial=lambda x: 1.0,  # Uniform
        u_terminal=lambda x: 0.0,  # Zero terminal cost
    )
    problem = MFGProblem(geometry=geometry, components=components)

    assert problem.is_custom is True
    assert problem.components is not None
    # Check potential was set using custom function
    expected = problem.geometry.get_spatial_grid() ** 2
    # Flatten both arrays for comparison (problem stores as 2D column vector)
    assert np.allclose(np.ravel(problem.f_potential), np.ravel(expected))


@pytest.mark.unit
def test_mfg_problem_with_custom_initial_density():
    """Test MFGProblem with custom initial density function."""

    def custom_initial(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[11])
    # Issue #670: must provide both m_initial and u_final
    # Issue #673: Hamiltonian required
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=custom_initial,
        u_terminal=lambda x: 0.0,  # Zero terminal cost
    )
    problem = MFGProblem(geometry=geometry, components=components)

    assert problem.is_custom is True
    # Check initial density was set using custom function and normalized
    expected_unnormalized = np.exp(-10 * (problem.geometry.get_spatial_grid() - 0.5) ** 2)
    # Normalize expected (same way as in MFGProblem.__init__)
    integral = np.sum(expected_unnormalized) * problem.geometry.get_grid_spacing()[0]
    expected = expected_unnormalized / integral
    # Flatten both arrays for comparison (problem stores as 2D column vector)
    assert np.allclose(np.ravel(problem.m_initial), np.ravel(expected))


@pytest.mark.unit
def test_mfg_problem_with_custom_final_value():
    """Test MFGProblem with custom final value function."""

    def custom_final(x):
        return np.sin(x * np.pi)

    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[11])
    # Issue #670: must provide both m_initial and u_final
    # Issue #673: Hamiltonian required
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        u_terminal=custom_final,
        m_initial=lambda x: 1.0,  # Uniform
    )
    problem = MFGProblem(geometry=geometry, components=components)

    assert problem.is_custom is True
    # Check final value was set using custom function
    expected = np.sin(problem.geometry.get_spatial_grid() * np.pi)
    # Flatten both arrays for comparison (problem stores as 2D column vector)
    assert np.allclose(np.ravel(problem.u_terminal), np.ravel(expected))


@pytest.mark.unit
def test_mfg_problem_with_custom_initial_density_2d():
    """Test MFGProblem with callable m_initial in 2D (Issue #777)."""

    def gaussian_2d(x):
        x_arr = np.asarray(x)
        return np.exp(-5 * np.sum(x_arr**2))

    geometry = default_geometry(
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        Nx_points=[11, 11],
        dimension=2,
    )
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=gaussian_2d,
        u_terminal=lambda x: 0.5 * np.sum(np.asarray(x) ** 2),
    )
    problem = MFGProblem(geometry=geometry, components=components)

    m_init = problem.get_m_initial()
    assert m_init.shape == (11, 11)
    assert np.all(np.isfinite(m_init))
    assert np.all(m_init >= 0)
    # Center (origin) should have highest density
    assert m_init[5, 5] == m_init.max()
    # Corner should have lowest density
    assert m_init[0, 0] == m_init.min()
    # Symmetry: radially symmetric IC should produce symmetric density
    assert np.allclose(m_init, np.flip(m_init, axis=0))
    assert np.allclose(m_init, np.flip(m_init, axis=1))


@pytest.mark.unit
def test_mfg_problem_with_custom_final_value_2d():
    """Test MFGProblem with callable u_terminal in 2D."""

    def quadratic_2d(x):
        x_arr = np.asarray(x)
        return 0.5 * np.sum(x_arr**2)

    geometry = default_geometry(
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        Nx_points=[11, 11],
        dimension=2,
    )
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=lambda x: 1.0,  # Uniform
        u_terminal=quadratic_2d,
    )
    problem = MFGProblem(geometry=geometry, components=components)

    u_term = problem.u_terminal
    grid_shape = problem.geometry.get_grid_shape()
    assert u_term.shape == grid_shape
    assert np.all(np.isfinite(u_term))
    # Center (origin) should have lowest terminal cost
    assert u_term[5, 5] == u_term.min()
    assert np.isclose(u_term[5, 5], 0.0, atol=1e-10)
    # Corner should have highest terminal cost
    assert np.isclose(u_term[0, 0], 0.5 * (1.0 + 1.0), atol=1e-10)


@pytest.mark.unit
def test_mfg_problem_validates_negative_m_initial():
    """Test that negative m_initial raises ValueError (Issue #672: Fail Fast)."""
    geometry = default_geometry()
    # Invalid: negative density
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=lambda x: x - 0.5,  # Negative for x < 0.5
        u_terminal=lambda x: 0.0,
    )

    with pytest.raises(ValueError, match="m_initial contains negative values"):
        MFGProblem(geometry=geometry, components=components)


@pytest.mark.unit
def test_mfg_problem_validates_zero_mass_m_initial():
    """Test that zero-mass m_initial raises ValueError (Issue #672: Fail Fast)."""
    geometry = default_geometry()
    # Invalid: zero everywhere
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=lambda x: 0.0,  # Zero mass
        u_terminal=lambda x: 0.0,
    )

    with pytest.raises(ValueError, match="m_initial has zero or negligible total mass"):
        MFGProblem(geometry=geometry, components=components)


# ===================================================================
# Test Hamiltonian Methods
# ===================================================================


@pytest.mark.unit
def test_hamiltonian_h_default():
    """Test default Hamiltonian H computation."""
    problem = create_test_problem(sigma=1.0, coupling_coefficient=0.5)

    x_idx = 5
    m_at_x = 1.0
    # Issue #673: Use derivs with tuple notation (p_values deprecated)
    derivs = {(1,): 0.1}  # 1D gradient = 0.1
    t_idx = 5

    H_val = problem.H(x_idx=x_idx, m_at_x=m_at_x, derivs=derivs, t_idx=t_idx)

    # Should return a finite value
    assert np.isfinite(H_val)
    assert isinstance(H_val, (float, np.floating))


@pytest.mark.unit
def test_hamiltonian_dh_dm_default():
    """Test default Hamiltonian derivative dH/dm."""
    problem = create_test_problem()

    x_idx = 5
    m_at_x = 1.0
    # Issue #673: Use derivs with tuple notation (p_values deprecated)
    derivs = {(1,): 0.1}  # 1D gradient = 0.1
    t_idx = 5

    dH_val = problem.dH_dm(x_idx=x_idx, m_at_x=m_at_x, derivs=derivs, t_idx=t_idx)

    # Should return a finite value
    assert np.isfinite(dH_val)
    assert isinstance(dH_val, (float, np.floating))


@pytest.mark.unit
def test_mfg_problem_custom_hamiltonian():
    """Test MFGProblem with custom Hamiltonian (class-based API)."""
    from mfgarchon.core.hamiltonian import HamiltonianBase

    # Issue #673: Use class-based Hamiltonian (function-based no longer supported)
    class CustomHamiltonian(HamiltonianBase):
        """Custom Hamiltonian: H = |p|² + m²"""

        def __call__(self, x, m, p, t=0.0):
            p_scalar = float(p[0]) if hasattr(p, "__len__") else float(p)
            return p_scalar**2 + m**2

        def dp(self, x, m, p, t=0.0):
            # dH/dp = 2p
            import numpy as np

            return 2.0 * np.atleast_1d(p)

        def dm(self, x, m, p, t=0.0):
            # dH/dm = 2m
            return 2.0 * m

    custom_H = CustomHamiltonian()

    # Issue #670: must provide m_initial and u_final
    components = MFGComponents(
        hamiltonian=custom_H,
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    problem = create_test_problem(components=components)

    # Should use custom Hamiltonian
    assert problem.is_custom is True
    # Issue #673: Use derivs with tuple notation (p_values no longer supported)
    # derivs={(1,): 0.15} means p = 0.15
    # H = p² + m² = 0.15² + 0.5² = 0.0225 + 0.25 = 0.2725
    H_value = problem.H(x_idx=10, m_at_x=0.5, derivs={(1,): 0.15}, t_idx=0)
    assert isinstance(H_value, float)
    assert H_value == pytest.approx(0.15**2 + 0.5**2)  # = 0.2725


# ===================================================================
# Test Boundary Conditions
# ===================================================================


@pytest.mark.unit
def test_get_boundary_conditions_default():
    """Test get_boundary_conditions returns default no_flux BC from geometry."""
    problem = create_test_problem()

    bc = problem.get_boundary_conditions()

    # Default BC from geometry is no_flux (mass-conserving)
    assert isinstance(bc, BoundaryConditions)
    assert bc.type == "no_flux"


@pytest.mark.unit
def test_get_boundary_conditions_custom():
    """Test get_boundary_conditions: geometry BC takes priority (Issue #674 SSOT).

    Note: With geometry-first API (Issue #674), the geometry's BC takes priority
    over components BC. This test verifies this priority order.
    """
    # Uses Dirichlet BC in components (lower priority)
    custom_bc = dirichlet_bc(value=0.0, dimension=1)
    # Issue #670: must provide m_initial and u_final
    # Issue #673: Hamiltonian required
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        boundary_conditions=custom_bc,
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    # create_test_problem uses default_geometry which has no_flux_bc
    problem = create_test_problem(components=components)

    bc = problem.get_boundary_conditions()

    # Issue #674: Geometry BC takes priority over components BC
    # The geometry has no_flux BC (from default_geometry), so that's what we get
    assert isinstance(bc, BoundaryConditions)
    assert bc.type == "no_flux"


# ===================================================================
# Test Getter Methods
# ===================================================================


@pytest.mark.unit
def test_get_potential_at_time():
    """Test get_potential_at_time returns array."""
    problem = create_test_problem(Nt=20)

    potential = problem.get_potential_at_time(t_idx=5)

    assert isinstance(potential, np.ndarray)
    assert len(potential) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    # Should match stored potential
    assert np.allclose(potential, problem.f_potential)


@pytest.mark.unit
def test_get_final_u():
    """Test get_final_u returns final value function (deprecated, calls get_u_terminal)."""
    problem = create_test_problem()

    u_final = problem.get_final_u()

    assert isinstance(u_final, np.ndarray)
    assert len(u_final) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    assert np.allclose(u_final, problem.u_terminal)


@pytest.mark.unit
def test_get_initial_m():
    """Test get_initial_m returns initial density."""
    problem = create_test_problem()

    m_initial = problem.get_initial_m()

    assert isinstance(m_initial, np.ndarray)
    assert len(m_initial) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    assert np.allclose(m_initial, problem.m_initial)


@pytest.mark.unit
def test_get_problem_info():
    """Test get_problem_info returns comprehensive info dict."""
    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[51])  # Nx=50 intervals
    problem = create_test_problem(geometry=geometry, T=1.0, Nt=100, sigma=0.5, coupling_coefficient=0.8)

    info = problem.get_problem_info()

    assert isinstance(info, dict)
    # Check key fields are present
    assert "domain" in info
    assert "time" in info
    assert "coefficients" in info
    assert "problem_type" in info

    # Check values
    assert info["domain"]["xmin"] == 0.0
    assert info["domain"]["xmax"] == 1.0
    assert info["domain"]["Nx"] == 50  # Intervals, from legacy info
    assert info["time"]["T"] == 1.0
    assert info["time"]["Nt"] == 100
    assert info["coefficients"]["sigma"] == 0.5
    assert info["coefficients"]["coupling_coefficient"] == 0.8


# ===================================================================
# Test Edge Cases
# ===================================================================


@pytest.mark.unit
def test_mfg_problem_zero_nt():
    """Test MFGProblem handles Nt=0 gracefully."""
    problem = create_test_problem(Nt=0)

    assert problem.Nt == 0
    assert problem.dt == 0.0
    assert len(problem.tSpace) == 1


@pytest.mark.unit
def test_mfg_problem_rejects_nonpositive_T():
    """Issue #1077: T <= 0 is a degenerate time horizon (dt = T/Nt = 0 -> all-zero
    integration) and must fail fast, consistent with MFGGridConfig's T-positive validation.
    (Nt=0 stays intentionally graceful, above.)"""
    for bad_T in (0.0, -1.0):
        with pytest.raises(ValueError, match=r"T .*must be > 0"):
            create_test_problem(T=bad_T)


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_exports_classes():
    """Test module exports main classes."""
    from mfgarchon.core import mfg_problem

    assert hasattr(mfg_problem, "MFGComponents")
    assert hasattr(mfg_problem, "MFGProblem")


@pytest.mark.unit
def test_module_exports_are_classes():
    """Test exported objects are actually classes."""
    assert isinstance(MFGComponents, type)
    assert isinstance(MFGProblem, type)


# ===================================================================
# Test Dual Geometry Support (Issue #257 Phase 3)
# ===================================================================


@pytest.mark.unit
def test_dual_geometry_specification():
    """Test MFGProblem with separate HJB and FP geometries (Issue #257)."""
    # Create two different 2D grids with BC
    hjb_grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[51, 51],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    fp_grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[21, 21],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    # Create 2D components
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    # Create problem with dual geometries
    problem = MFGProblem(
        hjb_geometry=hjb_grid,
        fp_geometry=fp_grid,
        time_domain=(1.0, 50),
        sigma=0.1,
        components=components,
    )

    # Check that both geometries are stored
    assert problem.hjb_geometry is hjb_grid
    assert problem.fp_geometry is fp_grid

    # Check that geometry projector was created
    assert problem.geometry_projector is not None
    assert problem.geometry_projector.hjb_geometry is hjb_grid
    assert problem.geometry_projector.fp_geometry is fp_grid


@pytest.mark.unit
def test_dual_geometry_backward_compatibility():
    """Test that unified geometry mode still works (backward compatibility)."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[31, 31],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    # Create problem with unified geometry (old API)
    problem = MFGProblem(geometry=grid, time_domain=(1.0, 50), sigma=0.1, components=components)

    # Check that both hjb_geometry and fp_geometry point to the same geometry
    assert problem.hjb_geometry is grid
    assert problem.fp_geometry is grid

    # Check that no projector is created for unified mode
    assert problem.geometry_projector is None


@pytest.mark.unit
def test_dual_geometry_error_on_partial_specification():
    """Test that specifying only one of hjb_geometry/fp_geometry raises error."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[31, 31],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    components = MFGComponents(hamiltonian=default_hamiltonian(), m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0)
    # Test with only hjb_geometry
    with pytest.raises(ValueError, match="both 'hjb_geometry' AND 'fp_geometry' must be specified"):
        MFGProblem(hjb_geometry=grid, time_domain=(1.0, 50), components=components)

    # Test with only fp_geometry
    with pytest.raises(ValueError, match="both 'hjb_geometry' AND 'fp_geometry' must be specified"):
        MFGProblem(fp_geometry=grid, time_domain=(1.0, 50), components=components)


@pytest.mark.unit
def test_dual_geometry_error_on_conflict():
    """Test that specifying both geometry and dual geometries raises error."""
    grid1 = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[31, 31],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    grid2 = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[21, 21],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    grid3 = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[11, 11],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    components = MFGComponents(hamiltonian=default_hamiltonian(), m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0)
    # Test conflict: can't specify both geometry and dual geometries
    with pytest.raises(ValueError, match=r"Specify EITHER 'geometry'.*OR.*'hjb_geometry', 'fp_geometry'"):
        MFGProblem(geometry=grid1, hjb_geometry=grid2, fp_geometry=grid3, time_domain=(1.0, 50), components=components)


@pytest.mark.unit
def test_dual_geometry_projector_attributes():
    """Test that geometry projector has correct attributes."""
    hjb_grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[51, 51],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    fp_grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[31, 31],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    components = MFGComponents(hamiltonian=default_hamiltonian(), m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0)
    problem = MFGProblem(hjb_geometry=hjb_grid, fp_geometry=fp_grid, time_domain=(1.0, 50), components=components)

    projector = problem.geometry_projector

    # Check that projector has the right geometries
    assert projector.hjb_geometry is hjb_grid
    assert projector.fp_geometry is fp_grid

    # Check that projector has selected appropriate methods
    assert projector.hjb_to_fp_method in ["grid_interpolation", "interpolation", "registry"]
    assert projector.fp_to_hjb_method in ["grid_restriction", "nearest", "registry"]


@pytest.mark.unit
def test_dual_geometry_with_1d_grids():
    """Test dual geometry with 1D grids."""
    # Create two 1D grids with different resolutions
    hjb_grid = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[101])  # Fine grid
    fp_grid = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[51])  # Coarse grid

    components = MFGComponents(hamiltonian=default_hamiltonian(), m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0)
    problem = MFGProblem(
        hjb_geometry=hjb_grid, fp_geometry=fp_grid, time_domain=(1.0, 50), sigma=0.1, components=components
    )

    # Verify dual geometry setup
    assert problem.hjb_geometry is hjb_grid
    assert problem.fp_geometry is fp_grid
    assert problem.geometry_projector is not None


@pytest.mark.unit
def test_dual_geometry_legacy_mode_compatibility():
    """Test that legacy 1D mode sets hjb_geometry and fp_geometry correctly."""
    # Legacy mode creates its own grid internally
    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[101])  # Nx=100 intervals
    problem = create_test_problem(geometry=geometry, T=1.0, Nt=50)

    # Check that hjb_geometry and fp_geometry are set (to the same unified geometry)
    assert problem.hjb_geometry is not None
    assert problem.fp_geometry is not None
    assert problem.hjb_geometry is problem.fp_geometry  # Unified mode
    assert problem.geometry_projector is None  # No projector for unified mode


@pytest.mark.unit
def test_legacy_1d_geometry_properties_removed():
    """Removed in v0.20.0: the deprecated 1D geometry read-properties raise AttributeError.

    Tier-3 removal of the legacy 1D geometry surface (deprecated since v0.17.0,
    Issue #435). Use the geometry-first API instead:
      xmin -> geometry.get_bounds()[0][0]; xmax -> geometry.get_bounds()[1][0];
      Lx -> bounds[1][0] - bounds[0][0]; Nx -> geometry.num_spatial_points - 1;
      dx -> geometry.get_grid_spacing()[0]; xSpace -> geometry.get_spatial_grid();
      _grid -> geometry.
    """
    problem = create_test_problem()
    for attr in ("xmin", "xmax", "Lx", "Nx", "dx", "xSpace", "_grid"):
        with pytest.raises(AttributeError):
            getattr(problem, attr)


# ===================================================================
# Test DiffusionField Support (Feature: diffusion-field-support)
# ===================================================================


@pytest.mark.unit
def test_volatility_field_none():
    """Test MFGProblem with no diffusion (deterministic). None → 0."""
    problem = create_test_problem(diffusion=None)

    assert problem.volatility_field == 0.0
    assert problem.sigma == 0.0
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_volatility_field_scalar():
    """Test MFGProblem with scalar diffusion D = sigma^2/2 (Issue #811)."""
    # diffusion=0.125 means D=0.125, so sigma = sqrt(2*0.125) = 0.5
    problem = create_test_problem(diffusion=0.125)

    assert problem.sigma == pytest.approx(0.5)
    # volatility_field stores SDE volatility (for solver compatibility)
    assert problem.volatility_field == pytest.approx(0.5)
    # diffusion property returns PDE coefficient D
    assert problem.diffusion == pytest.approx(0.125)
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_volatility_field_array():
    """Test MFGProblem with array diffusion coefficient (spatially varying)."""
    # Create a spatially varying diffusion array (11 points from default_geometry)
    sigma_array = np.linspace(0.1, 1.0, 11)

    problem = create_test_problem(sigma=sigma_array)

    # Array should be stored in volatility_field
    assert isinstance(problem.volatility_field, np.ndarray)
    assert np.array_equal(problem.volatility_field, sigma_array)

    # Scalar sigma should be the mean for backward compatibility
    assert problem.sigma == pytest.approx(np.mean(sigma_array))
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_volatility_field_callable():
    """Test MFGProblem with callable diffusion coefficient (state-dependent)."""

    def sigma_func(t, x, m):
        """State-dependent diffusion: higher diffusion in high-density regions."""
        return 0.1 + 0.5 * m

    problem = create_test_problem(sigma=sigma_func)

    # Callable should be stored in volatility_field
    assert callable(problem.volatility_field)
    assert problem.volatility_field is sigma_func

    # Scalar sigma should default to 1.0 for callable
    assert problem.sigma == 1.0
    assert problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_diffusion_primary_parameter():
    """Test that 'diffusion' converts D -> sigma (Issue #811)."""
    # diffusion=0.3 means D=0.3, sigma = sqrt(2*0.3) ~ 0.7746
    import math

    problem = create_test_problem(diffusion=0.3)

    assert problem.sigma == pytest.approx(math.sqrt(2 * 0.3))
    assert problem.diffusion == pytest.approx(0.3)
    # volatility_field stores converted volatility
    assert problem.volatility_field == pytest.approx(math.sqrt(2 * 0.3))


@pytest.mark.unit
def test_sigma_direct_sde_volatility():
    """Test that sigma= provides SDE volatility directly (Issue #811)."""
    # sigma= is NOT deprecated — it's the canonical SDE volatility input
    problem = create_test_problem(sigma=0.4)

    # sigma stored directly (no conversion)
    assert problem.sigma == 0.4
    assert problem.volatility_field == 0.4
    # diffusion property computes D = sigma^2/2
    assert problem.diffusion == pytest.approx(0.4**2 / 2.0)
    # volatility is alias for sigma
    assert problem.volatility == 0.4


@pytest.mark.unit
def test_volatility_alias_for_sigma():
    """Test that volatility= is an alias for sigma= (Issue #811)."""
    problem = create_test_problem(volatility=0.6)

    assert problem.sigma == 0.6
    assert problem.volatility == 0.6
    assert problem.diffusion == pytest.approx(0.6**2 / 2.0)


@pytest.mark.unit
def test_diffusion_sigma_mutual_exclusion():
    """Test that diffusion=, sigma=, volatility= are mutually exclusive."""
    # Two at once should raise
    with pytest.raises(ValueError, match="at most one"):
        create_test_problem(diffusion=0.1, sigma=0.5)

    with pytest.raises(ValueError, match="at most one"):
        create_test_problem(sigma=0.5, volatility=0.5)

    with pytest.raises(ValueError, match="at most one"):
        create_test_problem(diffusion=0.1, volatility=0.5)


@pytest.mark.unit
def test_diffusion_sigma_equivalence():
    """Test that diffusion=D and sigma=sqrt(2D) give identical results (Issue #811)."""
    import math

    D = 0.125  # PDE coefficient
    sigma = math.sqrt(2 * D)  # SDE volatility = 0.5

    p1 = create_test_problem(diffusion=D)
    p2 = create_test_problem(sigma=sigma)

    assert p1.sigma == pytest.approx(p2.sigma)
    assert p1.diffusion == pytest.approx(p2.diffusion)
    assert p1.volatility == pytest.approx(p2.volatility)


@pytest.mark.unit
def test_drift_field_none():
    """Test MFGProblem with no drift field (default). None → 0."""
    problem = create_test_problem()

    assert problem.drift_field == 0.0
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_drift_field_array():
    """Test MFGProblem with array drift field."""
    # Create a drift field array (default_geometry has 11 grid points)
    drift_array = np.ones(11) * 0.1  # Constant drift

    problem = create_test_problem(drift=drift_array)

    assert isinstance(problem.drift_field, np.ndarray)
    assert np.array_equal(problem.drift_field, drift_array)
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_drift_field_scalar():
    """Test MFGProblem with scalar drift (constant drift)."""
    problem = create_test_problem(drift=0.05)

    assert problem.drift_field == 0.05
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_drift_field_callable():
    """Test MFGProblem with callable drift field."""

    def drift_func(t, x, m):
        """Drift towards high-density regions."""
        return -0.1 * x

    problem = create_test_problem(drift=drift_func)

    assert callable(problem.drift_field)
    assert problem.drift_field is drift_func
    assert problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_get_diffusion_coefficient_field():
    """Test get_diffusion_coefficient_field returns CoefficientField wrapper."""
    problem = create_test_problem(sigma=0.5)

    coeff_field = problem.get_diffusion_coefficient_field()

    # Should return a CoefficientField instance
    from mfgarchon.utils.pde_coefficients import CoefficientField

    assert isinstance(coeff_field, CoefficientField)
    assert coeff_field.name == "diffusion"


@pytest.mark.unit
def test_get_drift_coefficient_field():
    """Test get_drift_coefficient_field returns CoefficientField wrapper."""
    # default_geometry has 11 grid points
    drift_array = np.ones(11) * 0.1
    problem = create_test_problem(drift=drift_array)

    coeff_field = problem.get_drift_coefficient_field()

    # Should return a CoefficientField instance
    from mfgarchon.utils.pde_coefficients import CoefficientField

    assert isinstance(coeff_field, CoefficientField)
    assert coeff_field.name == "drift"


@pytest.mark.unit
def test_has_state_dependent_coefficients_mixed():
    """Test has_state_dependent_coefficients with mixed coefficient types."""

    def sigma_func(t, x, m):
        return 0.1 + 0.5 * m

    # Scalar drift, callable volatility (default_geometry has 11 grid points)
    problem1 = create_test_problem(sigma=sigma_func, drift=np.zeros(11))
    assert problem1.has_state_dependent_coefficients()

    # Callable drift, scalar diffusion
    def drift_func(t, x, m):
        return -0.1 * x

    problem2 = create_test_problem(sigma=0.5, drift=drift_func)
    assert problem2.has_state_dependent_coefficients()

    # Both scalar
    problem3 = create_test_problem(sigma=0.5, drift=np.zeros(11))
    assert not problem3.has_state_dependent_coefficients()


@pytest.mark.unit
def test_volatility_field_with_geometry():
    """Test DiffusionField support works with geometry-based API."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[21, 21],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    def sigma_func(t, x, m):
        """2D state-dependent diffusion."""
        return 0.1 + 0.1 * np.sum(x**2)

    # Issue #670: must provide m_initial and u_final
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    problem = MFGProblem(geometry=grid, time_domain=(1.0, 50), sigma=sigma_func, components=components)

    assert callable(problem.volatility_field)
    assert problem.has_state_dependent_coefficients()
    assert problem.dimension == 2

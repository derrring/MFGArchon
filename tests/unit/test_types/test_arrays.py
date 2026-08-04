#!/usr/bin/env python3
"""
Unit tests for mfgarchon/types/arrays.py

Tests type alias definitions for array types used throughout MFGarchon including:
- Core solution arrays (SolutionArray, SpatialGrid, TimeGrid)
- Advanced array types (SpatialArray, TemporalArray)
- Multi-dimensional array types (Array1D, Array2D, Array3D)
- Specialized array types (StateArray, ParticleArray, WeightArray, DensityArray)
- Legacy compatibility aliases
"""

import pytest

import numpy as np

from mfgarchon.types.arrays import (  # noqa: TC001
    Array1D,
    Array2D,
    Array3D,
    DensityArray,
    ParticleArray,
    SolutionArray,
    SpatialArray,
    SpatialGrid,
    StateArray,
    TemporalArray,
    TimeGrid,
    WeightArray,
)

# ===================================================================
# Test Core Solution Arrays
# ===================================================================


@pytest.mark.unit
def test_solution_array_1d():
    """Test SolutionArray for 1D problem."""
    # 1D spatio-temporal solution (Nt+1, Nx+1)
    Nt, Nx = 30, 50
    arr: SolutionArray = np.random.rand(Nt + 1, Nx + 1)

    assert arr.shape == (31, 51)
    assert isinstance(arr, np.ndarray)


@pytest.mark.unit
def test_solution_array_2d():
    """Test SolutionArray for 2D problem."""
    # 2D spatio-temporal solution (Nt+1, Ny+1, Nx+1)
    Nt, Ny, Nx = 20, 40, 50
    arr: SolutionArray = np.random.rand(Nt + 1, Ny + 1, Nx + 1)

    assert arr.shape == (21, 41, 51)
    assert isinstance(arr, np.ndarray)


@pytest.mark.unit
def test_solution_array_zeros():
    """Test SolutionArray initialization with zeros."""
    arr: SolutionArray = np.zeros((31, 51))

    assert arr.shape == (31, 51)
    assert np.all(arr == 0.0)


# ===================================================================
# Test Grid Coordinate Arrays
# ===================================================================


@pytest.mark.unit
def test_spatial_grid_1d():
    """Test SpatialGrid for 1D domain."""
    # 1D spatial coordinates (Nx+1,)
    Nx = 50
    arr: SpatialGrid = np.linspace(0, 1, Nx + 1)

    assert arr.shape == (51,)
    assert arr[0] == 0.0
    assert arr[-1] == 1.0


@pytest.mark.unit
def test_spatial_grid_2d():
    """Test SpatialGrid for 2D domain."""
    # 2D spatial coordinates (Ny+1, Nx+1)
    Ny, Nx = 40, 50
    X, _Y = np.meshgrid(np.linspace(0, 1, Nx + 1), np.linspace(0, 1, Ny + 1))
    arr: SpatialGrid = X

    assert arr.shape == (41, 51)
    assert isinstance(arr, np.ndarray)


@pytest.mark.unit
def test_time_grid():
    """Test TimeGrid for temporal coordinates."""
    # Temporal coordinates (Nt+1,)
    Nt = 30
    T = 2.0
    arr: TimeGrid = np.linspace(0, T, Nt + 1)

    assert arr.shape == (31,)
    assert arr[0] == 0.0
    assert arr[-1] == 2.0


# ===================================================================
# Test Advanced Array Types
# ===================================================================


@pytest.mark.unit
def test_spatial_array_alias():
    """Test SpatialArray as alias for SpatialGrid."""
    arr: SpatialArray = np.linspace(0, 1, 51)

    assert arr.shape == (51,)
    assert isinstance(arr, np.ndarray)


@pytest.mark.unit
def test_temporal_array_alias():
    """Test TemporalArray as alias for TimeGrid."""
    arr: TemporalArray = np.linspace(0, 2, 31)

    assert arr.shape == (31,)
    assert isinstance(arr, np.ndarray)


# ===================================================================
# Test Multi-dimensional Array Types
# ===================================================================


@pytest.mark.unit
def test_array_1d():
    """Test Array1D type."""
    arr: Array1D = np.array([1, 2, 3, 4, 5])

    assert arr.ndim == 1
    assert arr.shape == (5,)


@pytest.mark.unit
def test_array_2d():
    """Test Array2D type."""
    arr: Array2D = np.array([[1, 2, 3], [4, 5, 6]])

    assert arr.ndim == 2
    assert arr.shape == (2, 3)


@pytest.mark.unit
def test_array_3d():
    """Test Array3D type."""
    arr: Array3D = np.random.rand(5, 4, 3)

    assert arr.ndim == 3
    assert arr.shape == (5, 4, 3)


# ===================================================================
# Test Specialized Array Types
# ===================================================================


@pytest.mark.unit
def test_state_array():
    """Test StateArray for neural network states."""
    # Variable shape depending on architecture
    hidden_dim = 128
    arr: StateArray = np.random.rand(hidden_dim)

    assert arr.shape == (128,)
    assert isinstance(arr, np.ndarray)


@pytest.mark.unit
def test_state_array_batched():
    """Test StateArray with batch dimension."""
    batch_size, hidden_dim = 32, 64
    arr: StateArray = np.random.rand(batch_size, hidden_dim)

    assert arr.shape == (32, 64)


@pytest.mark.unit
def test_particle_array():
    """Test ParticleArray for particle methods."""
    # Shape: (num_particles, spatial_dim)
    num_particles, spatial_dim = 100, 2
    arr: ParticleArray = np.random.rand(num_particles, spatial_dim)

    assert arr.shape == (100, 2)


@pytest.mark.unit
def test_particle_array_3d():
    """Test ParticleArray for 3D spatial problems."""
    num_particles, spatial_dim = 50, 3
    arr: ParticleArray = np.random.rand(num_particles, spatial_dim)

    assert arr.shape == (50, 3)


@pytest.mark.unit
def test_weight_array():
    """Test WeightArray for particle weights."""
    # Shape: (num_particles,)
    num_particles = 100
    arr: WeightArray = np.random.rand(num_particles)

    assert arr.shape == (100,)
    assert arr.ndim == 1


@pytest.mark.unit
def test_weight_array_normalized():
    """Test WeightArray with normalized weights."""
    num_particles = 50
    arr: WeightArray = np.ones(num_particles) / num_particles

    assert arr.shape == (50,)
    assert np.isclose(np.sum(arr), 1.0)


@pytest.mark.unit
def test_density_array_1d():
    """Test DensityArray for 1D density function."""
    # Shape: (Nx+1,) for 1D
    Nx = 100
    arr: DensityArray = np.random.rand(Nx + 1)

    assert arr.shape == (101,)


@pytest.mark.unit
def test_density_array_matches_spatial_grid():
    """Test DensityArray shape matches SpatialGrid."""
    Nx = 50
    x: SpatialGrid = np.linspace(0, 1, Nx + 1)
    m: DensityArray = np.exp(-((x - 0.5) ** 2) / 0.1)  # Gaussian density

    assert x.shape == m.shape


# ===================================================================
# Test Legacy Compatibility
# ===================================================================


@pytest.mark.unit
def test_spatial_coordinates_deprecated_alias():
    """Test SpatialCoordinates legacy alias emits deprecation warning."""
    import warnings

    # Test that deprecation warning is emitted
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Import locally to trigger warning
        import importlib

        import mfgarchon.types.arrays as arrays_module

        importlib.reload(arrays_module)
        _ = arrays_module.SpatialCoordinates

        # Check warning was issued
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "SpatialCoordinates" in str(deprecation_warnings[-1].message)
        assert "SpatialGrid" in str(deprecation_warnings[-1].message)

    # Test functionality still works (use the reloaded module's alias)
    arr: SpatialGrid = np.linspace(0, 1, 51)
    assert arr.shape == (51,)


@pytest.mark.unit
def test_temporal_coordinates_deprecated_alias():
    """Test TemporalCoordinates legacy alias emits deprecation warning."""
    import warnings

    # Test that deprecation warning is emitted
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import importlib

        import mfgarchon.types.arrays as arrays_module

        importlib.reload(arrays_module)
        _ = arrays_module.TemporalCoordinates

        # Check warning was issued
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "TemporalCoordinates" in str(deprecation_warnings[-1].message)
        assert "TimeGrid" in str(deprecation_warnings[-1].message)

    # Test functionality still works (use the reloaded module's alias)
    arr: TimeGrid = np.linspace(0, 2, 31)
    assert arr.shape == (31,)


# ===================================================================
# Test Type Alias Behavior
# ===================================================================


@pytest.mark.unit
def test_type_aliases_are_ndarray():
    """Test all type aliases are NDArray (numpy arrays)."""
    solution: SolutionArray = np.zeros((31, 51))
    spatial: SpatialGrid = np.linspace(0, 1, 51)
    temporal: TimeGrid = np.linspace(0, 1, 31)

    assert isinstance(solution, np.ndarray)
    assert isinstance(spatial, np.ndarray)
    assert isinstance(temporal, np.ndarray)


@pytest.mark.unit
def test_type_aliases_interchangeable():
    """Test type aliases are interchangeable (duck typing)."""
    # All are NDArray underneath
    arr = np.random.rand(10, 10)

    solution: SolutionArray = arr
    grid: SpatialGrid = arr
    array2d: Array2D = arr

    # All refer to same object
    assert solution is arr
    assert grid is arr
    assert array2d is arr


# ===================================================================
# Test Practical Usage Scenarios
# ===================================================================


@pytest.mark.unit
def test_mfg_solution_structure():
    """Test typical MFG solution structure."""
    # Problem parameters
    T, Nt = 1.0, 30
    xmin, xmax, Nx = 0.0, 1.0, 50

    # Create grids
    t: TimeGrid = np.linspace(0, T, Nt + 1)
    x: SpatialGrid = np.linspace(xmin, xmax, Nx + 1)

    # Create solution arrays
    u: SolutionArray = np.zeros((Nt + 1, Nx + 1))
    m: SolutionArray = np.zeros((Nt + 1, Nx + 1))

    assert t.shape == (31,)
    assert x.shape == (51,)
    assert u.shape == (31, 51)
    assert m.shape == (31, 51)


@pytest.mark.unit
def test_particle_method_structure():
    """Test typical particle method structure."""
    num_particles = 1000
    spatial_dim = 2

    # Particle positions and weights
    particles: ParticleArray = np.random.rand(num_particles, spatial_dim)
    weights: WeightArray = np.ones(num_particles) / num_particles

    assert particles.shape == (1000, 2)
    assert weights.shape == (1000,)
    assert np.isclose(np.sum(weights), 1.0)


@pytest.mark.unit
def test_neural_network_state():
    """Test neural network state array usage."""
    batch_size = 64
    state_dim = 128

    # Network state
    state: StateArray = np.random.randn(batch_size, state_dim)

    assert state.shape == (64, 128)
    assert isinstance(state, np.ndarray)


# ===================================================================
# Test Array Operations
# ===================================================================


@pytest.mark.unit
def test_array_arithmetic():
    """Test arithmetic operations on typed arrays."""
    u: SolutionArray = np.ones((31, 51))
    v: SolutionArray = np.ones((31, 51)) * 2

    result = u + v

    assert result.shape == (31, 51)
    assert np.all(result == 3.0)


@pytest.mark.unit
def test_array_slicing():
    """Test slicing operations on typed arrays."""
    u: SolutionArray = np.random.rand(31, 51)

    # Extract spatial slice at t=0
    u_initial: DensityArray = u[0, :]

    # Extract temporal evolution at x=25
    u_time_series = u[:, 25]

    assert u_initial.shape == (51,)
    assert u_time_series.shape == (31,)


@pytest.mark.unit
def test_array_broadcasting():
    """Test broadcasting with typed arrays."""
    # 2D solution array
    u: SolutionArray = np.ones((31, 51))

    # 1D spatial grid
    x: SpatialGrid = np.linspace(0, 1, 51)

    # Broadcasting multiplication
    result = u * x  # Broadcast x across time dimension

    assert result.shape == (31, 51)


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_docstring():
    """Test module has comprehensive docstring."""
    from mfgarchon.types import arrays

    assert arrays.__doc__ is not None
    assert "Array Type Definitions" in arrays.__doc__

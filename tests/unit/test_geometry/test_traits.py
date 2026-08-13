"""
Tests for geometry trait enums and protocols (Issue #732 Tier 1b).

Validates:
    - Enum values and completeness
    - Protocol isinstance checks on concrete geometry classes
    - Trait-first dispatch pattern
    - Non-trait-aware objects fail isinstance checks
"""

import pytest

import numpy as np

from mfgarchon.geometry.traits import (
    BoundaryAware,
    BoundaryDef,
    ConnectivityAware,
    ConnectivityType,
    StructureAware,
    StructureType,
    TraitAwareGeometry,
)

# =============================================================================
# Enum tests
# =============================================================================


@pytest.mark.unit
class TestConnectivityType:
    def test_values(self):
        assert ConnectivityType.IMPLICIT.value == "implicit"
        assert ConnectivityType.EXPLICIT.value == "explicit"
        assert ConnectivityType.DYNAMIC.value == "dynamic"

    def test_completeness(self):
        assert len(ConnectivityType) == 3


@pytest.mark.unit
class TestStructureType:
    def test_values(self):
        assert StructureType.STRUCTURED.value == "structured"
        assert StructureType.UNSTRUCTURED.value == "unstructured"

    def test_completeness(self):
        assert len(StructureType) == 2


@pytest.mark.unit
class TestBoundaryDef:
    def test_values(self):
        assert BoundaryDef.BOX.value == "box"
        assert BoundaryDef.MESH.value == "mesh"
        assert BoundaryDef.IMPLICIT.value == "implicit"
        assert BoundaryDef.NONE.value == "none"

    def test_completeness(self):
        assert len(BoundaryDef) == 4


# =============================================================================
# Protocol tests with mock geometry
# =============================================================================


class _MockTraitGeometry:
    """Minimal mock implementing all 3 trait properties."""

    @property
    def connectivity_type(self) -> ConnectivityType:
        return ConnectivityType.IMPLICIT

    @property
    def structure_type(self) -> StructureType:
        return StructureType.STRUCTURED

    @property
    def boundary_def(self) -> BoundaryDef:
        return BoundaryDef.BOX


class _MockPartialGeometry:
    """Mock implementing only StructureAware."""

    @property
    def structure_type(self) -> StructureType:
        return StructureType.UNSTRUCTURED


class _MockNoTraits:
    """Object with no trait properties."""


@pytest.mark.unit
class TestProtocolIsinstance:
    def test_full_trait_geometry(self):
        geo = _MockTraitGeometry()
        assert isinstance(geo, ConnectivityAware)
        assert isinstance(geo, StructureAware)
        assert isinstance(geo, BoundaryAware)
        assert isinstance(geo, TraitAwareGeometry)

    def test_partial_trait_geometry(self):
        geo = _MockPartialGeometry()
        assert not isinstance(geo, ConnectivityAware)
        assert isinstance(geo, StructureAware)
        assert not isinstance(geo, BoundaryAware)
        assert not isinstance(geo, TraitAwareGeometry)

    def test_no_traits(self):
        obj = _MockNoTraits()
        assert not isinstance(obj, ConnectivityAware)
        assert not isinstance(obj, StructureAware)
        assert not isinstance(obj, BoundaryAware)
        assert not isinstance(obj, TraitAwareGeometry)


# =============================================================================
# Concrete geometry tests
# =============================================================================


def _make_tensor_grid():
    """Create a minimal TensorProductGrid for testing."""
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import neumann_bc

    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=neumann_bc(dimension=1))


def _make_tensor_grid_2d():
    """Create a 2D TensorProductGrid for testing."""
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import neumann_bc

    return TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[5, 5],
        boundary_conditions=neumann_bc(dimension=2),
    )


@pytest.mark.unit
class TestTensorProductGridTraits:
    def test_connectivity(self):
        grid = _make_tensor_grid()
        assert isinstance(grid, ConnectivityAware)
        assert grid.connectivity_type == ConnectivityType.IMPLICIT

    def test_structure(self):
        grid = _make_tensor_grid()
        assert isinstance(grid, StructureAware)
        assert grid.structure_type == StructureType.STRUCTURED

    def test_boundary(self):
        grid = _make_tensor_grid()
        assert isinstance(grid, BoundaryAware)
        assert grid.boundary_def == BoundaryDef.BOX

    def test_composite(self):
        grid = _make_tensor_grid()
        assert isinstance(grid, TraitAwareGeometry)

    def test_2d_same_traits(self):
        grid = _make_tensor_grid_2d()
        assert isinstance(grid, TraitAwareGeometry)
        assert grid.connectivity_type == ConnectivityType.IMPLICIT
        assert grid.structure_type == StructureType.STRUCTURED
        assert grid.boundary_def == BoundaryDef.BOX


@pytest.mark.unit
class TestImplicitDomainTraits:
    def test_hyperrectangle_traits(self):
        from mfgarchon.geometry import Hyperrectangle

        domain = Hyperrectangle(bounds=np.array([[0.0, 1.0], [0.0, 1.0]]))
        assert isinstance(domain, TraitAwareGeometry)
        assert domain.connectivity_type == ConnectivityType.DYNAMIC
        assert domain.structure_type == StructureType.UNSTRUCTURED
        assert domain.boundary_def == BoundaryDef.IMPLICIT

    def test_hypersphere_traits(self):
        from mfgarchon.geometry import Hypersphere

        sphere = Hypersphere(center=np.array([0.0, 0.0]), radius=1.0)
        assert isinstance(sphere, TraitAwareGeometry)
        assert sphere.connectivity_type == ConnectivityType.DYNAMIC
        assert sphere.structure_type == StructureType.UNSTRUCTURED
        assert sphere.boundary_def == BoundaryDef.IMPLICIT


@pytest.mark.unit
class TestGraphGeometryTraits:
    def test_grid_network_traits(self):
        """GridNetwork should have Explicit/Unstructured/None traits."""
        try:
            from mfgarchon.geometry import GridNetwork

            net = GridNetwork(grid_shape=(3, 3))
        except Exception:
            pytest.skip("GridNetwork requires igraph or networkx")

        assert isinstance(net, TraitAwareGeometry)
        assert net.connectivity_type == ConnectivityType.EXPLICIT
        assert net.structure_type == StructureType.UNSTRUCTURED
        assert net.boundary_def == BoundaryDef.NONE


# =============================================================================
# Dispatch pattern tests
# =============================================================================


@pytest.mark.unit
class TestTraitDispatch:
    """Test the trait-first dispatch pattern from the architecture doc."""

    def test_structured_reshape(self):
        """Trait-based dispatch: reshape field for structured grids."""
        grid = _make_tensor_grid_2d()
        field = np.ones(grid.num_spatial_points)

        # Asserted rather than branched on: as an `if`, a regression in either trait
        # check skips the body and the test reports success having checked nothing.
        assert isinstance(grid, StructureAware)
        assert grid.structure_type == StructureType.STRUCTURED

        shape = grid.get_grid_shape()
        assert shape == (5, 5)
        # The reshape is only well-defined because these two agree.
        assert int(np.prod(shape)) == grid.num_spatial_points
        assert field.reshape(shape).shape == (5, 5)

    def test_trait_first_with_fallback(self):
        """Trait check first, then GeometryType fallback."""
        from mfgarchon.geometry.protocol import GeometryType

        grid = _make_tensor_grid()

        # Both paths evaluated unconditionally. As an if/elif only one ever runs, and the
        # legacy branch hardcodes True -- so the assertion held whether or not the trait
        # path worked. The claim worth making is that the two encodings agree.
        trait = isinstance(grid, StructureAware) and grid.structure_type == StructureType.STRUCTURED
        legacy = grid.geometry_type == GeometryType.CARTESIAN_GRID

        assert trait
        assert legacy
        assert trait == legacy

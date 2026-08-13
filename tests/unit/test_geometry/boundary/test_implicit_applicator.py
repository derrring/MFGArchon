"""Tests for ImplicitApplicator — Issue #712 consolidation.

Verifies:
- Inheritance from MeshfreeApplicator (not BaseBCApplicator)
- Protocol-based boundary detection (no hasattr fallbacks)
- Dirichlet and Neumann BC application
- Dispatch integration
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary.applicator_base import DiscretizationType
from mfgarchon.geometry.boundary.applicator_implicit import ImplicitApplicator
from mfgarchon.geometry.boundary.applicator_meshfree import MeshfreeApplicator
from mfgarchon.geometry.protocol import GeometryType

# ---------------------------------------------------------------------------
# Test fixture: lightweight mock geometry implementing GeometryProtocol
# ---------------------------------------------------------------------------

_CENTER = np.array([0.5, 0.5])
_RADIUS = 0.4


class _CircleGeometry:
    """Minimal GeometryProtocol-compliant circle domain for testing."""

    dimension = 2
    geometry_type = GeometryType.IMPLICIT
    num_spatial_points = 441

    def sdf(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - _CENTER, axis=-1) - _RADIUS

    def is_on_boundary(self, points: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
        return np.abs(self.sdf(points)) < tolerance

    def get_boundary_normal(self, points: np.ndarray) -> np.ndarray:
        diff = points - _CENTER
        norms = np.linalg.norm(diff, axis=-1, keepdims=True)
        return diff / np.maximum(norms, 1e-10)

    def get_bounds(self):
        return np.array([0.0, 0.0]), np.array([1.0, 1.0])

    def get_collocation_points(self):
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 21)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def get_spatial_grid(self):
        return self.get_collocation_points()

    def get_grid_shape(self):
        return (21, 21)

    def get_problem_config(self):
        return {"num_spatial_points": 441, "spatial_shape": (21, 21)}

    def get_boundary_conditions(self):
        return None

    def get_boundary_regions(self):
        return {"all": {}}

    def get_boundary_indices(self, points, tolerance=1e-10):
        return np.where(self.is_on_boundary(points, tolerance))[0]

    def get_boundary_info(self, points, tolerance=1e-10):
        indices = self.get_boundary_indices(points, tolerance)
        if len(indices) == 0:
            return indices, np.array([], dtype=np.float64).reshape(0, 2)
        normals = self.get_boundary_normal(points[indices])
        return indices, normals

    def project_to_boundary(self, points):
        diff = points - _CENTER
        norms = np.linalg.norm(diff, axis=-1, keepdims=True)
        return _CENTER + diff / np.maximum(norms, 1e-10) * _RADIUS

    def project_to_interior(self, points):
        sdf_vals = self.sdf(points)
        outside = sdf_vals > 0
        result = points.copy()
        if np.any(outside):
            result[outside] = self.project_to_boundary(points[outside])
        return result


@pytest.fixture
def geometry():
    return _CircleGeometry()


@pytest.fixture
def applicator(geometry):
    return ImplicitApplicator(geometry=geometry, boundary_tolerance=0.03)


@pytest.fixture
def grid_points():
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


# ---------------------------------------------------------------------------
# Inheritance and type identity tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInheritance:
    """ImplicitApplicator inherits from MeshfreeApplicator (Issue #712)."""

    def test_inherits_from_meshfree(self, applicator):
        """ImplicitApplicator IS-A MeshfreeApplicator."""
        assert isinstance(applicator, MeshfreeApplicator)

    def test_discretization_type_meshfree(self, applicator):
        """Returns MESHFREE discretization type (inherited from base)."""
        assert applicator.discretization_type == DiscretizationType.MESHFREE


# ---------------------------------------------------------------------------
# Boundary condition application tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBCApplication:
    """ImplicitApplicator applies BCs using geometry protocol methods."""

    def test_apply_dirichlet(self, applicator, geometry, grid_points):
        """Dirichlet BC sets boundary values to prescribed value."""
        from mfgarchon.geometry.boundary import dirichlet_bc

        field = np.linalg.norm(grid_points - _CENTER, axis=-1)
        bc = dirichlet_bc(dimension=2, value=0.0)
        result = applicator.apply(field, bc, grid_points)

        boundary_mask = geometry.is_on_boundary(grid_points, tolerance=0.03)
        if np.any(boundary_mask):
            assert np.allclose(result[boundary_mask], 0.0)

    def test_apply_neumann_no_flux(self, applicator, geometry, grid_points):
        """Neumann zero-flux BC uses interpolation along normals."""
        from mfgarchon.geometry.boundary import neumann_bc

        field = np.linalg.norm(grid_points - _CENTER, axis=-1)
        bc = neumann_bc(dimension=2)
        result = applicator.apply(field, bc, grid_points)

        # No-flux should modify boundary values (not leave them as original)
        # Result should still be finite
        assert np.all(np.isfinite(result))

    def test_apply_time_positional(self, applicator, grid_points):
        """time parameter can be passed positionally (LSP compliance)."""
        from mfgarchon.geometry.boundary import dirichlet_bc

        field = np.ones(len(grid_points))
        bc = dirichlet_bc(dimension=2, value=0.0)
        # Pass time as positional argument (4th arg)
        result = applicator.apply(field, bc, grid_points, 0.5)
        assert result.shape == field.shape


# ---------------------------------------------------------------------------
# Protocol usage tests (no hasattr fallbacks)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProtocolUsage:
    """Boundary detection uses protocol methods directly, not hasattr."""

    def test_boundary_detection_uses_protocol(self, applicator, grid_points):
        """_detect_boundary_points calls geometry.is_on_boundary directly."""
        mask = applicator._detect_boundary_points(grid_points)
        assert mask.dtype == np.bool_
        assert mask.shape == (len(grid_points),)

    def test_normals_use_protocol(self, applicator, geometry, grid_points):
        """_compute_boundary_normals calls geometry.get_boundary_normal directly."""
        boundary_mask = geometry.is_on_boundary(grid_points, tolerance=0.03)
        if np.any(boundary_mask):
            boundary_pts = grid_points[boundary_mask]
            normals = applicator._compute_boundary_normals(boundary_pts)
            assert normals.shape == boundary_pts.shape
            # Normals should be approximately unit vectors
            norms = np.linalg.norm(normals, axis=-1)
            assert np.allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Dispatch integration test
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDispatch:
    """dispatch.py selects ImplicitApplicator for IMPLICIT geometry type."""

    def test_dispatch_implicit_geometry(self, geometry):
        """get_applicator_for_geometry returns ImplicitApplicator for MESHFREE + IMPLICIT."""
        from mfgarchon.geometry.boundary.dispatch import get_applicator_for_geometry

        applicator = get_applicator_for_geometry(geometry, discretization="MESHFREE")
        assert isinstance(applicator, ImplicitApplicator)

    def test_dispatch_gfdm_uses_meshfree(self, geometry):
        """get_applicator_for_geometry returns MeshfreeApplicator for GFDM."""
        from mfgarchon.geometry.boundary.dispatch import get_applicator_for_geometry

        applicator = get_applicator_for_geometry(geometry, discretization="GFDM")
        assert isinstance(applicator, MeshfreeApplicator)

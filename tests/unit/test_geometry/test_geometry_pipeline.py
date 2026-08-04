"""
Basic tests for the geometry pipeline implementation.

These tests verify that the core mesh generation pipeline works correctly
without requiring external dependencies like Gmsh.
"""

import sys
from pathlib import Path

import pytest

import numpy as np

# Add MFGarchon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mfgarchon.geometry import Mesh2D, MeshData


class TestMeshData:
    """Test the MeshData container class."""

    def test_mesh_data_creation(self):
        """Test basic MeshData creation and validation."""
        vertices = np.array([[0, 0], [1, 0], [0, 1]])
        elements = np.array([[0, 1, 2]])

        mesh_data = MeshData(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_tags=np.array([1, 1, 1]),
            element_tags=np.array([1]),
            boundary_faces=np.array([[0, 1], [1, 2], [2, 0]]),
            dimension=2,
        )

        assert mesh_data.num_vertices == 3
        assert mesh_data.num_elements == 1
        assert mesh_data.dimension == 2
        assert mesh_data.element_type == "triangle"

    def test_mesh_data_bounds(self):
        """Test bounding box computation."""
        vertices = np.array([[0, 0], [2, 0], [1, 3]])
        elements = np.array([[0, 1, 2]])

        mesh_data = MeshData(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_tags=np.array([1, 1, 1]),
            element_tags=np.array([1]),
            boundary_faces=np.array([[0, 1], [1, 2], [2, 0]]),
            dimension=2,
        )

        min_coords, max_coords = mesh_data.bounds
        np.testing.assert_array_equal(min_coords, [0, 0])
        np.testing.assert_array_equal(max_coords, [2, 3])

    def test_triangle_area_computation(self):
        """Test triangle area computation."""
        # Simple right triangle with base=1, height=1, area=0.5
        vertices = np.array([[0, 0], [1, 0], [0, 1]])
        elements = np.array([[0, 1, 2]])

        mesh_data = MeshData(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_tags=np.array([1, 1, 1]),
            element_tags=np.array([1]),
            boundary_faces=np.array([[0, 1], [1, 2], [2, 0]]),
            dimension=2,
        )

        areas = mesh_data.compute_element_volumes()
        assert len(areas) == 1
        assert abs(areas[0] - 0.5) < 1e-10

    def test_invalid_dimension(self):
        """Test validation of mismatched dimensions."""
        vertices = np.array([[0, 0], [1, 0], [0, 1]])  # 2D vertices
        elements = np.array([[0, 1, 2]])

        # Should raise error when vertex dimension doesn't match specified dimension
        with pytest.raises(ValueError, match="Vertex coordinates dimension 2 doesn't match specified dimension 4"):
            MeshData(
                vertices=vertices,
                elements=elements,
                element_type="triangle",
                boundary_tags=np.array([1, 1, 1]),
                element_tags=np.array([1]),
                boundary_faces=np.array([[0, 1], [1, 2], [2, 0]]),
                dimension=4,  # Mismatch: 2D vertices but dimension=4
            )


class TestMesh2D:
    """Test the Mesh2D geometry class."""

    def test_rectangle_domain_creation(self):
        """Test rectangular domain creation."""
        domain = Mesh2D(domain_type="rectangle", bounds=(0.0, 2.0, 0.0, 1.0), mesh_size=0.5)

        assert domain.domain_type == "rectangle"
        assert domain.dimension == 2
        assert domain.xmin == 0.0
        assert domain.xmax == 2.0
        assert domain.ymin == 0.0
        assert domain.ymax == 1.0
        assert domain.mesh_size == 0.5

    def test_circle_domain_creation(self):
        """Test circular domain creation."""
        domain = Mesh2D(domain_type="circle", center=(0.5, 0.5), radius=0.4, mesh_size=0.1)

        assert domain.domain_type == "circle"
        assert domain.center == (0.5, 0.5)
        assert domain.radius == 0.4

    def test_polygon_domain_creation(self):
        """Test polygonal domain creation."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

        domain = Mesh2D(domain_type="polygon", vertices=vertices, mesh_size=0.2)

        assert domain.domain_type == "polygon"
        assert domain.vertices == vertices

    def test_polygon_validation(self):
        """Test polygon validation with insufficient vertices."""
        with pytest.raises(ValueError, match="at least 3 vertices"):
            Mesh2D(domain_type="polygon", vertices=[(0, 0), (1, 0)], mesh_size=0.2)  # Only 2 vertices

    def test_rectangle_bounds(self):
        """Test bounds computation for rectangular domain."""
        domain = Mesh2D(domain_type="rectangle", bounds=(1.0, 3.0, 2.0, 4.0), mesh_size=0.1)

        min_coords, max_coords = domain.bounds
        np.testing.assert_array_equal(min_coords, [1.0, 2.0])
        np.testing.assert_array_equal(max_coords, [3.0, 4.0])

    def test_circle_bounds(self):
        """Test bounds computation for circular domain."""
        domain = Mesh2D(domain_type="circle", center=(1.0, 2.0), radius=0.5, mesh_size=0.1)

        min_coords, max_coords = domain.bounds
        np.testing.assert_array_equal(min_coords, [0.5, 1.5])
        np.testing.assert_array_equal(max_coords, [1.5, 2.5])

    def test_polygon_bounds(self):
        """Test bounds computation for polygonal domain."""
        vertices = [(0, 1), (2, 0), (3, 3), (1, 2)]

        domain = Mesh2D(domain_type="polygon", vertices=vertices, mesh_size=0.1)

        min_coords, max_coords = domain.bounds
        np.testing.assert_array_equal(min_coords, [0, 0])
        np.testing.assert_array_equal(max_coords, [3, 3])


def test_geometry_package_import():
    """Test that the geometry package imports correctly."""
    from mfgarchon.geometry import BoundaryConditions, Mesh2D, MeshData

    # Test that classes are available
    assert MeshData is not None
    assert Mesh2D is not None
    assert BoundaryConditions is not None


if __name__ == "__main__":
    pytest.main([__file__])

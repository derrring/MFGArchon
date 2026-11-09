#!/usr/bin/env python3
"""
Geometry Utilities for MFG Problems

This module provides convenient aliases and utilities for geometric operations
commonly used in MFG problems, particularly for obstacle representation and
signed distance functions.

The actual implementations live in mfgarchon.geometry.implicit, but this module
makes them more discoverable as utilities.

Example:
    >>> from mfgarchon.utils.geometry import RectangleObstacle, CircleObstacle
    >>> from mfgarchon.utils.geometry import Union
    >>>
    >>> # Create obstacles
    >>> wall = RectangleObstacle(np.array([[0.4, 0.6], [0.0, 0.5]]))
    >>> pillar = CircleObstacle(center=np.array([0.7, 0.7]), radius=0.1)
    >>>
    >>> # Combine obstacles
    >>> obstacles = Union([wall, pillar])
    >>>
    >>> # Query signed distance
    >>> distance = obstacles.signed_distance(points)
"""

from __future__ import annotations

# Import from geometry.implicit for convenient access
from mfgarchon.geometry.implicit import Hyperrectangle, Hypersphere, ImplicitDomain
from mfgarchon.geometry.implicit.csg_operations import (
    ComplementDomain,
    DifferenceDomain,
    IntersectionDomain,
    UnionDomain,
)

# =============================================================================
# CONVENIENT ALIASES
# =============================================================================

# More intuitive names for common use cases
RectangleObstacle = Hyperrectangle
"""Alias for Hyperrectangle - represents axis-aligned rectangular obstacles."""

CircleObstacle = Hypersphere
"""Alias for Hypersphere - represents circular/spherical obstacles."""

BoxObstacle = Hyperrectangle
"""Alias for Hyperrectangle - emphasizes 3D box interpretation."""

SphereObstacle = Hypersphere
"""Alias for Hypersphere - emphasizes 3D sphere interpretation."""

# CSG operation aliases (shorter names)
Union = UnionDomain
"""Alias for UnionDomain - combine multiple domains."""

Intersection = IntersectionDomain
"""Alias for IntersectionDomain - intersection of domains."""

Difference = DifferenceDomain
"""Alias for DifferenceDomain - subtract one domain from another."""

Complement = ComplementDomain
"""Alias for ComplementDomain - complement of a domain."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_rectangle_obstacle(xmin: float, xmax: float, ymin: float, ymax: float) -> Hyperrectangle:
    """
    Create a 2D rectangular obstacle.

    Args:
        xmin, xmax: X-axis bounds
        ymin, ymax: Y-axis bounds

    Returns:
        Rectangular obstacle

    Example:
        >>> obstacle = create_rectangle_obstacle(0.4, 0.6, 0.2, 0.8)
        >>> is_inside = obstacle.contains(np.array([[0.5, 0.5]]))
    """
    import numpy as np

    return Hyperrectangle(np.array([[xmin, xmax], [ymin, ymax]]))


def create_circle_obstacle(center_x: float, center_y: float, radius: float) -> Hypersphere:
    """
    Create a 2D circular obstacle.

    Args:
        center_x, center_y: Center coordinates
        radius: Circle radius

    Returns:
        Circular obstacle

    Example:
        >>> obstacle = create_circle_obstacle(0.5, 0.5, 0.2)
        >>> distance = obstacle.signed_distance(np.array([[0.6, 0.6]]))
    """
    import numpy as np

    return Hypersphere(center=np.array([center_x, center_y]), radius=radius)


def create_box_obstacle(xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float) -> Hyperrectangle:
    """
    Create a 3D box obstacle.

    Args:
        xmin, xmax: X-axis bounds
        ymin, ymax: Y-axis bounds
        zmin, zmax: Z-axis bounds

    Returns:
        Box obstacle

    Example:
        >>> obstacle = create_box_obstacle(0.4, 0.6, 0.2, 0.8, 0.1, 0.9)
    """
    import numpy as np

    return Hyperrectangle(np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]]))


def create_sphere_obstacle(center_x: float, center_y: float, center_z: float, radius: float) -> Hypersphere:
    """
    Create a 3D spherical obstacle.

    Args:
        center_x, center_y, center_z: Center coordinates
        radius: Sphere radius

    Returns:
        Spherical obstacle

    Example:
        >>> obstacle = create_sphere_obstacle(0.5, 0.5, 0.5, 0.2)
    """
    import numpy as np

    return Hypersphere(center=np.array([center_x, center_y, center_z]), radius=radius)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "BoxObstacle",
    "CircleObstacle",
    "Complement",
    "Difference",
    # Primitive shapes
    "Hyperrectangle",
    "Hypersphere",
    # Base classes
    "ImplicitDomain",
    "Intersection",
    # Obstacle aliases (most common usage)
    "RectangleObstacle",
    "SphereObstacle",
    # CSG operations (short, clean names)
    "Union",
    "create_box_obstacle",
    "create_circle_obstacle",
    # Factory functions
    "create_rectangle_obstacle",
    "create_sphere_obstacle",
]

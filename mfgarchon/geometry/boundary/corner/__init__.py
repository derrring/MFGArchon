"""
Corner Handling for Boundary Conditions (Issue #521).

This module provides unified corner handling across all solver types:

Position-based (implicit diagonal reflection):
    >>> from mfgarchon.geometry.boundary.corner import reflect_positions, wrap_positions
    >>> reflected = reflect_positions(particles, bounds)

    Position functions process all dimensions simultaneously, producing
    diagonal reflection at corners. No strategy parameter needed.

Velocity-based (specular reflection with corner_strategy):
    >>> from mfgarchon.geometry.boundary.corner import reflect_velocity
    >>> v_new = reflect_velocity(position, velocity, bounds, corner_strategy="average")

    For Billiard dynamics and hard-sphere collisions. Uses specular reflection:
    v_new = v - 2(v·n)n

Normal-based (explicit strategy selection):
    >>> from mfgarchon.geometry import DomainGeometry
    >>> geom = DomainGeometry(bounds)
    >>> normal = geom.get_boundary_normal(point, corner_strategy="average")

    Normal queries are methods on DomainGeometry because they need access
    to geometry state. Available strategies: "average", "priority", "mollify".

Detection:
    >>> geom.is_near_corner(points)  # Boolean array
    >>> geom.get_boundary_faces_at_point(point)  # List of (dim, side) tuples

Module Structure:
    corner/
    ├── __init__.py      # This file - unified exports
    ├── position.py      # reflect_positions, absorb_at_walls, walls_reached; absorb_positions, wrap_positions (deprecated)
    ├── velocity.py      # reflect_velocity, reflect_velocity_with_normal
    └── strategies.py    # CornerStrategy enum, validation

Architecture Notes:
    - Position functions are STATELESS (pure functions on arrays)
    - Velocity functions are STATELESS (pure functions, include normal computation)
    - Normal queries need GEOMETRY STATE (methods on DomainGeometry)

See Also:
    - docs/development/CORNER_HANDLING_IMPLEMENTATION_STATUS.md
    - Issue #521 for corner handling architecture
"""

# =============================================================================
# Position-based functions (stateless, canonical implementation)
# For PERIODIC utilities, use mfgarchon.geometry.boundary.periodic instead
# =============================================================================
from .position import (
    absorb_at_walls,
    absorb_positions,  # Deprecated - a clamp; use absorb_at_walls for an absorbing wall
    reflect_positions,
    walls_reached,
    wrap_positions,  # Deprecated - use boundary.periodic.wrap_positions
)

# =============================================================================
# Strategy definitions
# =============================================================================
from .strategies import (
    DEFAULT_CORNER_STRATEGY,
    CornerStrategy,
    CornerStrategyLiteral,
    validate_corner_strategy,
)

# =============================================================================
# Velocity-based functions (specular reflection)
# =============================================================================
from .velocity import reflect_velocity, reflect_velocity_with_normal

__all__ = [
    # Position-based (for NON-PERIODIC boundaries)
    # For PERIODIC, use mfgarchon.geometry.boundary.periodic
    "reflect_positions",
    "absorb_at_walls",
    "walls_reached",
    "absorb_positions",  # Deprecated - a clamp; use absorb_at_walls for an absorbing wall
    "wrap_positions",  # Deprecated - use boundary.periodic.wrap_positions
    # Velocity-based (Billiard dynamics)
    "reflect_velocity",
    "reflect_velocity_with_normal",
    # Strategy definitions
    "CornerStrategy",
    "CornerStrategyLiteral",
    "DEFAULT_CORNER_STRATEGY",
    "validate_corner_strategy",
]

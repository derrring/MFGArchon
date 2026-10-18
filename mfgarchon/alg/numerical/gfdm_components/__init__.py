"""
GFDM Components - Composition-based architecture for GFDM solvers.

This package provides modular components for GFDM solvers,extracted from the original mixin-based hierarchy (Issue #545).

Components:
-----------
- GridCollocationMapper: Bidirectional grid ↔ collocation interpolation
- BoundaryHandler: Boundary operations, LCR, ghost nodes
- NeighborhoodBuilder: Stencil construction, Taylor matrices

Benefits of Composition:
------------------------
1. Testability: Components can be tested independently
2. Reusability: Components usable across FDM, FEM, GFDM solvers
3. Clarity: Explicit dependencies, clear data flow
4. Maintainability: Changes localized to specific components
5. No hasattr: All attributes initialized explicitly

See docs/development/PARTICLE_SOLVER_TEMPLATE.md for composition pattern guide.
"""

from .boundary_handler import BoundaryHandler
from .grid_collocation_mapper import GridCollocationMapper
from .neighborhood_builder import NeighborhoodBuilder

__all__ = [
    "BoundaryHandler",
    "GridCollocationMapper",
    "NeighborhoodBuilder",
]

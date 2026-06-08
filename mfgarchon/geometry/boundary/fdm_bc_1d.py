"""
1D FDM Boundary Conditions for MFG Problems.

.. deprecated:: 0.14.0
    This module is deprecated. Use the unified boundary condition API instead:

    **Old (deprecated):**
        from mfgarchon.geometry.boundary.fdm_bc_1d import BoundaryConditions
        bc = BoundaryConditions(type="periodic")

    **New (recommended):**
        from mfgarchon.geometry import periodic_bc
        bc = periodic_bc(dimension=1)

    The unified API from conditions.py supports all dimensions and mixed BCs.
    This module will be removed in v1.0.0.

This module provides simple boundary condition specification for 1D finite
difference methods. Uses left/right value pattern for 1D domain endpoints.

For multi-dimensional or segment-based BC specification, use conditions.py.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass
class BoundaryConditions:
    """
    Boundary condition configuration for 1D MFG problems.

    .. deprecated:: 0.14.0
        Use :class:`mfgarchon.geometry.boundary.conditions.BoundaryConditions` instead.
        This class will be removed in v1.0.0.

        Migration::

            # Old:
            from mfgarchon.geometry.boundary.fdm_bc_1d import BoundaryConditions
            bc = BoundaryConditions(type="no_flux")

            # New:
            from mfgarchon.geometry.boundary import no_flux_bc
            bc = no_flux_bc(dimension=1)
    """

    type: str  # 'periodic', 'dirichlet', 'neumann', 'no_flux', or 'robin'

    # Boundary values
    # For Dirichlet: value of u at boundary
    # For Neumann: value of du/dn at boundary
    # For no_flux: F(boundary) = 0 where F = v*m - D*dm/dx
    left_value: float | None = None
    right_value: float | None = None

    # Robin boundary condition parameters: αu + βdu/dn = g
    # α coefficients (multiplier of solution value)
    left_alpha: float | None = None  # coefficient of u at left boundary
    left_beta: float | None = None  # coefficient of du/dn at left boundary
    right_alpha: float | None = None  # coefficient of u at right boundary
    right_beta: float | None = None  # coefficient of du/dn at right boundary

    def __post_init__(self):
        """Validate boundary condition parameters."""
        warnings.warn(
            "fdm_bc_1d.BoundaryConditions is deprecated since v0.14.0. "
            "Use no_flux_bc(dimension=1), periodic_bc(dimension=1), etc. "
            "from mfgarchon.geometry.boundary instead. "
            "Will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.type == "robin":
            if any(
                v is None
                for v in [
                    self.left_alpha,
                    self.left_beta,
                    self.right_alpha,
                    self.right_beta,
                ]
            ):
                raise ValueError("Robin boundary conditions require alpha and beta coefficients")

    def is_periodic(self) -> bool:
        """Check if boundary conditions are periodic."""
        return self.type == "periodic"

    def is_dirichlet(self) -> bool:
        """Check if boundary conditions are Dirichlet."""
        return self.type == "dirichlet"

    def is_neumann(self) -> bool:
        """Check if boundary conditions are Neumann."""
        return self.type == "neumann"

    def is_no_flux(self) -> bool:
        """Check if boundary conditions are no-flux."""
        return self.type == "no_flux"

    def is_robin(self) -> bool:
        """Check if boundary conditions are Robin."""
        return self.type == "robin"

    def get_matrix_size(self, num_interior_points: int) -> int:
        """
        Get the size of the system matrix for these boundary conditions.

        Args:
            num_interior_points: Number of interior grid points (M)

        Returns:
            Size of the system matrix
        """
        if self.type == "periodic":
            return num_interior_points
        elif self.type == "dirichlet":
            return num_interior_points - 1
        elif self.type in ["neumann", "robin"]:
            return num_interior_points + 1
        elif self.type == "no_flux":
            return num_interior_points
        else:
            raise ValueError(f"Unknown boundary condition type: {self.type}")

    def validate_values(self):
        """Validate that required values are provided for the boundary condition type."""
        if self.type == "dirichlet":
            if self.left_value is None or self.right_value is None:
                raise ValueError("Dirichlet boundary conditions require left_value and right_value")

        elif self.type == "neumann":
            if self.left_value is None or self.right_value is None:
                raise ValueError("Neumann boundary conditions require left_value and right_value")

        elif self.type == "robin":
            required_params = [
                self.left_alpha,
                self.left_beta,
                self.left_value,
                self.right_alpha,
                self.right_beta,
                self.right_value,
            ]
            if any(param is None for param in required_params):
                raise ValueError(
                    "Robin boundary conditions require left_alpha, left_beta, left_value, "
                    "right_alpha, right_beta, and right_value"
                )

    def __str__(self) -> str:
        """String representation of boundary conditions."""
        if self.type == "periodic":
            return "Periodic"
        elif self.type == "dirichlet":
            return f"Dirichlet(left={self.left_value}, right={self.right_value})"
        elif self.type == "neumann":
            return f"Neumann(left={self.left_value}, right={self.right_value})"
        elif self.type == "no_flux":
            return "No-flux"
        elif self.type == "robin":
            return (
                f"Robin(left: {self.left_alpha}u + {self.left_beta}u' = {self.left_value}, "
                f"right: {self.right_alpha}u + {self.right_beta}u' = {self.right_value})"
            )
        else:
            return f"Unknown({self.type})"

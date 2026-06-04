"""
DEPRECATED: This module has moved to mfgarchon.alg.numerical.adjoint.

This module is deprecated since v0.17.0 and will be removed in v1.0.0.
Please update your imports:

    # Old (deprecated):
    from mfgarchon.geometry.boundary import create_adjoint_consistent_bc_1d

    # New (canonical):
    from mfgarchon.alg.numerical.adjoint import create_adjoint_consistent_bc_1d

See Issue #704 for details on the unified adjoint module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.utils.deprecation import deprecated
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

# Import from submodules directly (deprecated module, see Issue #704)
from .conditions import BoundaryConditions
from .types import BCSegment, BCType

if TYPE_CHECKING:
    from numpy.typing import NDArray


@deprecated(
    since="v0.17.0",
    replacement="Use mfgarchon.alg.numerical.adjoint.compute_boundary_log_density_gradient_1d instead.",
)
def compute_boundary_log_density_gradient_1d(
    m: NDArray[np.floating],
    dx: float,
    side: str,
    regularization: float = 1e-10,
) -> float:
    """
    DEPRECATED: Use mfgarchon.alg.numerical.adjoint.compute_boundary_log_density_gradient_1d.

    Compute dln(m)/dn at 1D boundary using one-sided finite differences.
    """
    m_safe = m + regularization
    ln_m = np.log(m_safe)

    if side == "left":
        grad_ln_m = -(ln_m[1] - ln_m[0]) / dx
    elif side == "right":
        grad_ln_m = (ln_m[-1] - ln_m[-2]) / dx
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side}")

    return float(grad_ln_m)


@deprecated(
    since="v0.17.0",
    replacement="Use mfgarchon.alg.numerical.adjoint.create_adjoint_consistent_bc_1d instead.",
)
def create_adjoint_consistent_bc_1d(
    m_current: NDArray[np.floating],
    dx: float,
    sigma: float,
    domain_bounds: NDArray[np.floating] | None = None,
    regularization: float = 1e-10,
) -> BoundaryConditions:
    """
    DEPRECATED: Use mfgarchon.alg.numerical.adjoint.create_adjoint_consistent_bc_1d.

    Create adjoint-consistent Robin BC for 1D HJB equation.
    """

    # Inline implementation to avoid circular import
    m_safe = m_current + regularization
    ln_m = np.log(m_safe)

    grad_ln_m_left = -(ln_m[1] - ln_m[0]) / dx
    grad_ln_m_right = (ln_m[-1] - ln_m[-2]) / dx

    diffusion_coeff = diffusion_from_volatility(sigma)
    value_left = -diffusion_coeff * grad_ln_m_left
    value_right = -diffusion_coeff * grad_ln_m_right

    segments = [
        BCSegment(
            name="left_adjoint_consistent",
            bc_type=BCType.ROBIN,
            alpha=0.0,
            beta=1.0,
            value=value_left,
            boundary="x_min",
            priority=1,
        ),
        BCSegment(
            name="right_adjoint_consistent",
            bc_type=BCType.ROBIN,
            alpha=0.0,
            beta=1.0,
            value=value_right,
            boundary="x_max",
            priority=1,
        ),
    ]

    return BoundaryConditions(
        segments=segments,
        dimension=1,
        domain_bounds=domain_bounds,
        default_bc=BCType.NEUMANN,
        default_value=0.0,
    )


@deprecated(
    since="v0.17.0",
    replacement="Use mfgarchon.alg.numerical.adjoint.compute_adjoint_consistent_bc_values instead.",
)
def compute_adjoint_consistent_bc_values(
    m_current: NDArray[np.floating],
    geometry: object,
    sigma: float,
    dimension: int = 1,
    regularization: float = 1e-10,
) -> BoundaryConditions:
    """
    DEPRECATED: Use mfgarchon.alg.numerical.adjoint.compute_adjoint_consistent_bc_values.

    Create adjoint-consistent Robin BC for HJB equation (dimension-agnostic).
    """

    if dimension == 1:
        dx = geometry.get_grid_spacing()[0]
        domain_bounds = getattr(geometry, "domain_bounds", None)
        return create_adjoint_consistent_bc_1d(
            m_current=m_current,
            dx=dx,
            sigma=sigma,
            domain_bounds=domain_bounds,
            regularization=regularization,
        )
    else:
        raise NotImplementedError(f"Adjoint-consistent BC not yet implemented for {dimension}D.")


# Backward compatibility alias (the target already warns via @deprecated)
compute_coupled_hjb_bc_values = compute_adjoint_consistent_bc_values


__all__ = [
    "create_adjoint_consistent_bc_1d",
    "compute_adjoint_consistent_bc_values",
    "compute_boundary_log_density_gradient_1d",
    "compute_coupled_hjb_bc_values",
]

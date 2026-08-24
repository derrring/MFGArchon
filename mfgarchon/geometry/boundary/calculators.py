"""
Concrete topology and calculator implementations for boundary conditions.

This module contains the concrete implementations of the Topology and
BoundaryCalculator protocols, plus the LinearConstraint dataclass.

Extracted from applicator_base.py (mechanical refactor, no logic changes).
"""

from __future__ import annotations

from dataclasses import dataclass

from mfgarchon.utils.deprecation import deprecated

from .ghost_cells import (
    ghost_cell_dirichlet,
    ghost_cell_fp_no_flux,
    ghost_cell_neumann,
    ghost_cell_robin,
)
from .protocols import FieldData, GridType

# =============================================================================
# Concrete Topology Implementations
# =============================================================================


class PeriodicTopology:
    """
    Periodic boundary topology.

    In periodic topology, boundaries wrap around: the ghost cell at the
    low boundary equals the interior value at the high boundary, and vice versa.

    This is a MEMORY/INDEXING concept, not a physics concept. The Calculator
    is NOT used for periodic boundaries - values come from wrap-around.
    """

    def __init__(self, dimension: int, shape: tuple[int, ...]):
        """
        Initialize periodic topology.

        Args:
            dimension: Spatial dimension (1, 2, 3, ...)
            shape: Grid shape (interior points)
        """
        if len(shape) != dimension:
            raise ValueError(f"Shape length {len(shape)} must match dimension {dimension}")
        self._dimension = dimension
        self._shape = shape

    @property
    def is_periodic(self) -> bool:
        return True

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __repr__(self) -> str:
        return f"PeriodicTopology(dimension={self._dimension}, shape={self._shape})"


class BoundedTopology:
    """
    Bounded (non-periodic) boundary topology.

    In bounded topology, boundaries are physical edges that require ghost
    values computed by a Calculator. The topology itself just marks that
    boundaries exist - the Calculator provides the values.

    This separation enables:
    - Same Calculator works with any bounded grid
    - Different Calculators can be swapped without changing topology
    """

    def __init__(self, dimension: int, shape: tuple[int, ...]):
        """
        Initialize bounded topology.

        Args:
            dimension: Spatial dimension (1, 2, 3, ...)
            shape: Grid shape (interior points)
        """
        if len(shape) != dimension:
            raise ValueError(f"Shape length {len(shape)} must match dimension {dimension}")
        self._dimension = dimension
        self._shape = shape

    @property
    def is_periodic(self) -> bool:
        return False

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __repr__(self) -> str:
        return f"BoundedTopology(dimension={self._dimension}, shape={self._shape})"


# =============================================================================
# Concrete Calculator Implementations
# =============================================================================


class DirichletCalculator:
    """
    Calculator for Dirichlet (fixed value) boundary conditions.

    Computes ghost cell value such that the boundary value equals the
    prescribed value g:
        u_boundary = (u_ghost + u_interior) / 2 = g  (cell-centered)
        => u_ghost = 2*g - u_interior

    Supports vectorized operations for efficient array processing.
    """

    def __init__(
        self,
        boundary_value: float = 0.0,
        grid_type: GridType = GridType.CELL_CENTERED,
    ):
        """
        Initialize Dirichlet calculator.

        Args:
            boundary_value: Prescribed value at boundary
            grid_type: Grid type (cell-centered or vertex-centered)
        """
        self._boundary_value = boundary_value
        self._grid_type = grid_type

    @property
    def boundary_value(self) -> float:
        return self._boundary_value

    @property
    def grid_type(self) -> GridType:
        return self._grid_type

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        **kwargs,
    ) -> T:
        """Compute ghost value for Dirichlet BC (vectorized)."""
        # NumPy broadcasting handles both scalar and array inputs
        return ghost_cell_dirichlet(interior_value, self._boundary_value, self._grid_type)

    def __repr__(self) -> str:
        return f"DirichletCalculator(boundary_value={self._boundary_value})"


class NeumannCalculator:
    """
    Calculator for Neumann (fixed flux) boundary conditions.

    Computes ghost cell value such that the normal derivative equals
    the prescribed flux g:
        du/dn = (u_ghost - u_interior) / dx = g      (both centrings)
        => u_ghost = u_interior + dx*g

    The separation is `dx` on either centring and du/dn already carries the wall's direction.
    This docstring said `2*dx` until #1936; that is the formula #1972 removed from the body it
    describes, and it was the fourth surviving copy of it.

    Supports vectorized operations for efficient array processing.
    """

    def __init__(
        self,
        flux_value: float = 0.0,
        grid_type: GridType = GridType.CELL_CENTERED,
    ):
        """
        Initialize Neumann calculator.

        Args:
            flux_value: Prescribed normal flux (du/dn)
            grid_type: Grid type
        """
        self._flux_value = flux_value
        self._grid_type = grid_type

    @property
    def flux_value(self) -> float:
        return self._flux_value

    @property
    def grid_type(self) -> GridType:
        return self._grid_type

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        **kwargs,
    ) -> T:
        """Compute ghost value for Neumann BC (vectorized)."""
        # No `side` and no centring: `_flux_value` is du/dn, which carries the direction, and the
        # ghost-to-interior separation is `dx` for both centrings (#1972). This wrapper previously
        # multiplied by an outward sign and reached the `2*dx` branch, so it disagreed with the live
        # applicator path by a factor of 2 AND by a sign at the min wall -- 0.0 against 1.5 for
        # u0=1, dx=0.25, value=2.
        return ghost_cell_neumann(interior_value, self._flux_value, dx)

    def __repr__(self) -> str:
        return f"NeumannCalculator(flux_value={self._flux_value})"


class RobinCalculator:
    """
    Calculator for Robin (mixed) boundary conditions.

    Computes ghost cell value for the Robin condition:
        alpha*u + beta*du/dn = g

    Supports vectorized operations for efficient array processing.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        rhs_value: float = 0.0,
        grid_type: GridType = GridType.CELL_CENTERED,
    ):
        """
        Initialize Robin calculator.

        Args:
            alpha: Coefficient on u (Dirichlet weight)
            beta: Coefficient on du/dn (Neumann weight)
            rhs_value: Right-hand side value g
            grid_type: Grid type
        """
        self._alpha = alpha
        self._beta = beta
        self._rhs_value = rhs_value
        self._grid_type = grid_type

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        **kwargs,
    ) -> T:
        """Compute ghost value for Robin BC (vectorized).

        `side` is accepted for the protocol and not used: `rhs_value` carries `du/dn`, which
        already has the wall's direction. Deriving an outward sign from `side` and passing it on
        is what #2063 removed -- this was the only caller in the package that did, and it was the
        only one getting a wrong answer, inverting the vertex-centred min wall.
        """
        # grid_type by KEYWORD, not position. Passed positionally it lands on whatever slot
        # follows `dx`, so reintroducing an `outward_normal_sign` parameter would silently rebind
        # GridType onto it and fall back to cell-centred -- mutation-tested: that restoration
        # leaves 51/51 green. The keyword makes the binding structural rather than lucky (#2063).
        return ghost_cell_robin(
            interior_value,
            self._rhs_value,
            self._alpha,
            self._beta,
            dx,
            grid_type=self._grid_type,
        )

    def __repr__(self) -> str:
        return f"RobinCalculator(alpha={self._alpha}, beta={self._beta}, rhs={self._rhs_value})"


class ZeroGradientCalculator:
    """
    Calculator for zero gradient (du/dn = 0) boundary conditions.

    Implements edge extension: ghost = interior, ensuring du/dn = 0.

    Physical meaning: The field has no gradient normal to the boundary.
    Use cases:
    - HJB value functions at reflective walls
    - Any field needing smooth extension at boundaries

    **For mass-conserving boundaries (FP equations), use ZeroFluxCalculator instead.**

    Supports vectorized operations for efficient array processing.
    """

    def __init__(self, grid_type: GridType = GridType.CELL_CENTERED):
        self._grid_type = grid_type

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        **kwargs,
    ) -> T:
        """Compute ghost value for zero gradient BC (edge extension, vectorized)."""
        # Simply return interior value - works for both scalar and array
        return interior_value

    def __repr__(self) -> str:
        return "ZeroGradientCalculator()"


# Backward compatibility alias (with deprecation warning)
class NoFluxCalculator(ZeroGradientCalculator):
    """
    Deprecated alias for ZeroGradientCalculator.

    .. deprecated:: 0.16.11
        Use :class:`ZeroGradientCalculator` instead for du/dn = 0.
        For mass-conserving flux BC (J*n = 0), use :class:`ZeroFluxCalculator`.
    """

    @deprecated(
        since="v0.16.11",
        replacement="Use ZeroGradientCalculator for du/dn = 0, or ZeroFluxCalculator for J*n = 0.",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LinearExtrapolationCalculator:
    """
    Calculator for linear extrapolation boundary conditions.

    Uses zero second derivative (d^2 u/dx^2 = 0) at boundary.
    Ghost = 2*u_0 - u_1

    Suitable for HJB problems with linear value growth at infinity.
    Supports vectorized operations for efficient array processing.
    """

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        second_interior_value: T | None = None,
        **kwargs,
    ) -> T:
        """
        Compute ghost value via linear extrapolation (vectorized).

        Args:
            interior_value: Value at point adjacent to boundary (u_0)
            dx: Grid spacing (not used, but part of protocol)
            side: Boundary side (not used, but part of protocol)
            second_interior_value: Value at second interior point (u_1)

        Returns:
            Ghost value = 2*u_0 - u_1
        """
        if second_interior_value is None:
            # #2059: this returned `interior_value` -- the zero-GRADIENT ghost, a DIFFERENT
            # boundary condition silently substituted for the one requested. It was not a rare
            # fallback either: `GhostBuffer.update()` never supplied the argument, so on that path
            # it was the ONLY branch that ever ran, and every EXTRAPOLATION_LINEAR request served
            # through it got a Neumann-0 wall. The caller now supplies it, from the same cells
            # `pad_array_with_ghosts` uses.
            raise ValueError(
                "LinearExtrapolationCalculator requires second_interior_value: the ghost is "
                "2*u_0 - u_1 and u_1 is not optional. Returning u_0 instead would impose the "
                "zero-gradient condition, which is a different boundary condition (#2059)."
            )
        # Vectorized: works for both scalar and array
        return 2.0 * interior_value - second_interior_value

    def __repr__(self) -> str:
        return "LinearExtrapolationCalculator()"


class QuadraticExtrapolationCalculator:
    """
    Calculator for quadratic extrapolation boundary conditions.

    Uses zero third derivative (d^3 u/dx^3 = 0) at boundary.
    Ghost = 3*u_0 - 3*u_1 + u_2

    Suitable for LQG-type problems with quadratic value functions.
    Supports vectorized operations for efficient array processing.
    """

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        second_interior_value: T | None = None,
        third_interior_value: T | None = None,
        **kwargs,
    ) -> T:
        """
        Compute ghost value via quadratic extrapolation (vectorized).

        Args:
            interior_value: Value at point adjacent to boundary (u_0)
            dx: Grid spacing (not used)
            side: Boundary side (not used)
            second_interior_value: Value at second interior point (u_1)
            third_interior_value: Value at third interior point (u_2)

        Returns:
            Ghost value = 3*u_0 - 3*u_1 + u_2
        """
        if second_interior_value is None or third_interior_value is None:
            # #2059: this degraded quadratic -> linear -> edge extension without telling anyone,
            # so a caller asking for EXTRAPOLATION_QUADRATIC could receive any of three different
            # boundary conditions depending on how many arguments happened to arrive. The
            # `pad_array_with_ghosts` path already refuses when the grid cannot carry the stencil,
            # with the reason "Refuse rather than silently dropping to a lower order"; this now
            # agrees with it.
            raise ValueError(
                "QuadraticExtrapolationCalculator requires second_interior_value and "
                "third_interior_value: the ghost is 3*u_0 - 3*u_1 + u_2. Silently dropping to "
                "linear or to edge extension imposes a different boundary condition (#2059)."
            )
        # Vectorized: works for both scalar and array
        return 3.0 * interior_value - 3.0 * second_interior_value + third_interior_value

    def __repr__(self) -> str:
        return "QuadraticExtrapolationCalculator()"


class ZeroFluxCalculator:
    """
    Calculator for zero total flux (J*n = 0) boundary conditions.

    For advection-diffusion equations, this ensures the total flux
    J = v*rho - D*grad(rho) vanishes at the boundary, preserving mass conservation.

    Formula: u_ghost = (2D + v*dx) / (2D - v*dx) * u_interior

    Physical meaning: No mass/probability crosses the boundary.
    Use cases:
    - Fokker-Planck density with impermeable walls
    - Any advection-diffusion equation requiring mass conservation

    **For zero gradient (du/dn = 0), use ZeroGradientCalculator instead.**

    Supports vectorized operations for efficient array processing.
    """

    def __init__(
        self,
        drift_velocity: float = 0.0,
        diffusion_coeff: float = 1.0,
        grid_type: GridType = GridType.CELL_CENTERED,
    ):
        """
        Initialize FP no-flux calculator.

        Args:
            drift_velocity: Normal component of drift (positive = outward)
            diffusion_coeff: Diffusion coefficient D = sigma^2/2
            grid_type: Grid type
        """
        self._drift = drift_velocity
        self._diffusion = diffusion_coeff
        self._grid_type = grid_type

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        drift_velocity: float | None = None,
        **kwargs,
    ) -> T:
        """
        Compute ghost value for FP no-flux BC (vectorized).

        Args:
            interior_value: Density at interior point(s)
            dx: Grid spacing
            side: Boundary side ('min' or 'max')
            drift_velocity: Override drift velocity (optional)

        Returns:
            Ghost value(s) ensuring zero total flux J*n = 0
        """
        outward_sign = 1.0 if side == "max" else -1.0
        v = drift_velocity if drift_velocity is not None else self._drift
        # Vectorized formula: works for both scalar and array
        return ghost_cell_fp_no_flux(
            interior_value,
            v,
            self._diffusion,
            dx,
            outward_sign,
            grid_type=self._grid_type,  # keyword, for the reason given in RobinCalculator (#2063)
        )

    def __repr__(self) -> str:
        return f"ZeroFluxCalculator(drift={self._drift}, D={self._diffusion})"


# Backward compatibility alias (with deprecation warning)
class FPNoFluxCalculator(ZeroFluxCalculator):
    """
    Deprecated alias for ZeroFluxCalculator.

    .. deprecated:: 0.16.11
        Use :class:`ZeroFluxCalculator` instead for J*n = 0 (mass conservation).
    """

    @deprecated(
        since="v0.16.11",
        replacement="Use ZeroFluxCalculator instead for J*n = 0 (mass conservation).",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# =============================================================================
# LinearConstraint: Bridge between Ghost Cells and Matrix Assembly
# =============================================================================
#
# The "Tier-Based Coefficient Folding" pattern from bc_architecture_analysis.md
#
# For EXPLICIT schemes: Ghost cells are filled with computed values
#   u_ghost = Calculator.compute(u_inner, dx)
#
# For IMPLICIT schemes: Ghost node relationships become matrix coefficients
#   u_ghost = sum(weights[k] * u[inner+k]) + bias
#
# This dataclass expresses the linear relationship for matrix folding:
#   Tier 1 (State/Dirichlet): weights={}, bias=value
#   Tier 2 (Gradient/Neumann): weights={0: 1.0}, bias=dx*grad
#   Tier 3 (Flux/Robin): weights={0: alpha}, bias=0
#   Tier 4 (Artificial/Extrapolation): weights={0: 2.0, 1: -1.0}, bias=0
# =============================================================================


@dataclass
class LinearConstraint:
    """
    Linear constraint expressing ghost cell as function of interior values.

    For matrix assembly, when a stencil accesses ghost index j, the assembler:
    1. Adds weight * w to A[i, inner+k] for each (k, w) in weights
    2. Subtracts weight * bias from b[i]

    This is the "Coefficient Folding" pattern from the 2+4 BC architecture.

    Attributes:
        weights: Mapping from relative offset to weight. Offset 0 = boundary cell,
                 offset 1 = one cell inward, etc.
        bias: Constant term (for Dirichlet values or gradient offsets)

    Examples:
        # Tier 1: Dirichlet u=g -> u_ghost = g (constant)
        LinearConstraint(weights={}, bias=g)

        # Tier 2: Neumann du/dn=0 -> u_ghost = u_inner
        LinearConstraint(weights={0: 1.0}, bias=0.0)

        # Tier 2: Neumann du/dn=g -> u_ghost = u_inner + dx*g
        LinearConstraint(weights={0: 1.0}, bias=dx * g)

        # Tier 3: Robin (FP no-flux) -> u_ghost = alpha * u_inner
        LinearConstraint(weights={0: alpha}, bias=0.0)

        # Tier 4: Linear extrapolation -> u_ghost = 2*u[0] - u[1]
        LinearConstraint(weights={0: 2.0, 1: -1.0}, bias=0.0)
    """

    weights: dict[int, float]
    bias: float = 0.0


__all__ = [
    # Topology implementations
    "PeriodicTopology",
    "BoundedTopology",
    # Calculator implementations (physics-based naming)
    "DirichletCalculator",
    "NeumannCalculator",
    "RobinCalculator",
    "ZeroGradientCalculator",  # du/dn = 0 (edge extension)
    "ZeroFluxCalculator",  # J*n = 0 (mass conservation)
    "LinearExtrapolationCalculator",
    "QuadraticExtrapolationCalculator",
    # Backward compatibility aliases
    "NoFluxCalculator",  # -> ZeroGradientCalculator
    "FPNoFluxCalculator",  # -> ZeroFluxCalculator
    # Matrix assembly support (Tier-Based Coefficient Folding)
    "LinearConstraint",
]

"""
Core Level Set Evolution Infrastructure.

This module implements the fundamental level set evolution equation:
    ∂φ/∂t + V|∇φ| = 0

where:
- φ(t,x): Level set function (signed distance function)
- V(x): Normal velocity field
- |∇φ|: Gradient magnitude

The zero level set {x : φ(x,t) = 0} represents the evolving interface.

Key Components:
- LevelSetFunction: Container for φ, normals, curvature, interface mask
- LevelSetEvolver: Godunov upwind scheme with CFL-adaptive substepping

Numerical Scheme:
    Godunov upwind for |∇φ| ensures monotonicity and stability
    CFL condition: max(|V|)·dt/h < 1 (enforced via adaptive substepping)

References:
- Osher & Sethian (1988): Fronts propagating with curvature-dependent speed
- Osher & Fedkiw (2003): Level Set Methods and Dynamic Implicit Surfaces, Chapter 6
- Sethian (1999): Level Set Methods and Fast Marching Methods
- Shu & Osher (1988): Efficient implementation of essentially non-oscillatory shock-capturing schemes

Created: 2026-01-18 (Issue #592 Milestone 3.1.1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid
    from mfgarchon.geometry.implicit.implicit_domain import ImplicitDomain

# Module logger
logger = get_logger(__name__)


class LevelSetFunction:
    """
    Container for level set function φ and derived quantities.

    Represents a snapshot of the level set at a single time:
        φ(x): Signed distance function
        n(x) = ∇φ/|∇φ|: Outward unit normal
        κ(x) = ∇·n: Mean curvature

    The zero level set {x : φ(x) = 0} represents the interface/boundary.

    Attributes:
        phi: Level set function values on grid, shape (Nx, Ny, ...) or (Nx, Ny, ..., Nz)
        geometry: Base implicit domain providing grid structure
        is_signed_distance: Whether |∇φ| ≈ 1 (true SDF property)

    Example:
        >>> # Circle with radius 0.3 centered at (0.5, 0.5)
        >>> phi = np.linalg.norm(X - center, axis=0) - radius
        >>> ls = LevelSetFunction(phi, geometry, is_signed_distance=True)
        >>> interface = ls.interface_mask(width=2*dx)  # Points near φ = 0
        >>> normals = ls.get_normal()  # Unit normal field
    """

    def __init__(
        self,
        phi: NDArray[np.float64],
        geometry: ImplicitDomain | TensorProductGrid,
        is_signed_distance: bool = False,
    ):
        """
        Initialize level set function.

        Args:
            phi: Level set values on grid, shape matching geometry
            geometry: Implicit domain or grid providing spatial structure
            is_signed_distance: Whether φ is a true signed distance function (|∇φ| = 1)

        Raises:
            ValueError: If phi shape doesn't match geometry
        """
        self.phi = phi.copy()  # Defensive copy
        self.geometry = geometry
        self.is_signed_distance = is_signed_distance

        # Validate shape compatibility
        expected_shape = self._get_field_shape()
        if phi.shape != expected_shape:
            raise ValueError(f"phi shape {phi.shape} doesn't match geometry field shape {expected_shape}")

    def _get_field_shape(self) -> tuple[int, ...]:
        """Extract field shape from geometry."""
        from mfgarchon.geometry.base import CartesianGrid

        if isinstance(self.geometry, CartesianGrid):
            return self.geometry.get_grid_shape()
        # ImplicitDomain - infer from phi
        return self.phi.shape

    def _get_grid_coordinates(self) -> list[NDArray[np.float64]]:
        """
        Extract 1D coordinate arrays from geometry.

        Returns list of 1D arrays, one per spatial dimension. For a 2D grid
        with shape (Nx, Ny), returns [x_coords, y_coords].

        Raises:
            TypeError: If geometry is not a TensorProductGrid (requires per-axis coordinates).
        """
        from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

        if isinstance(self.geometry, TensorProductGrid):
            return self.geometry.coordinates
        raise TypeError(f"Subcell extraction requires TensorProductGrid, got {type(self.geometry).__name__}.")

    @property
    def dimension(self) -> int:
        """Spatial dimension of the level set."""
        return len(self.phi.shape)

    def interface_mask(self, width: float = 0.05) -> NDArray[np.bool_]:
        """
        Compute boolean mask for interface region |φ| < width.

        This identifies the "narrow band" around the zero level set where
        reinitialization and curvature computations are most relevant.

        Args:
            width: Band width (default: 0.05)
                For width = ε, returns mask where |φ| < ε

        Returns:
            Boolean array, shape matching phi, True where |φ| < width

        Example:
            >>> # Get interface points with 3·dx band
            >>> mask = ls.interface_mask(width=3*dx)
            >>> interface_phi = phi[mask]  # Values near zero level set
        """
        return np.abs(self.phi) < width

    def get_normal(self) -> NDArray[np.float64]:
        """
        Compute unit normal field: n = ∇φ / |∇φ|.

        The normal points outward (from negative to positive φ).

        Returns:
            Normal field, shape (dimension, Nx, Ny, ...) where
            normal[d, i, j, ...] is the d-th component at grid point (i,j,...)

        Raises:
            AttributeError: If geometry doesn't support gradients

        Example:
            >>> normals = ls.get_normal()
            >>> # For 2D: normals[0] is n_x, normals[1] is n_y
            >>> n_x, n_y = normals
        """
        # Use geometry's gradient operator (Issue #595 infrastructure)
        grad_ops = self.geometry.get_gradient_operator()

        # Compute gradient components
        grad_phi = np.array([grad_op(self.phi) for grad_op in grad_ops])

        # Compute magnitude: |∇φ|
        # Add small epsilon to avoid division by zero
        grad_mag = np.linalg.norm(grad_phi, axis=0) + 1e-10

        # Normalize: n = ∇φ / |∇φ|
        normal = grad_phi / grad_mag

        return normal

    def get_curvature(self) -> NDArray[np.float64]:
        """
        Compute mean curvature: κ = ∇·(∇φ/|∇φ|) = ∇·n.

        For a circle of radius R, analytical curvature is κ = 1/R.
        Positive curvature indicates convex (outward-bulging) interface.

        Returns:
            Curvature field, shape matching phi

        Raises:
            AttributeError: If geometry doesn't support divergence

        Note:
            Curvature computation requires both gradient and divergence operators.
            For best accuracy, φ should be a signed distance function (|∇φ| = 1).

        Example:
            >>> kappa = ls.get_curvature()
            >>> # For sphere: kappa ≈ 1/R on interface
        """
        # Import here to avoid circular dependency
        from mfgarchon.geometry.level_set.curvature import compute_curvature

        return compute_curvature(self.phi, self.geometry)

    def get_interface_location_subcell(self) -> float | NDArray[np.float64]:
        """
        Extract interface location with subcell precision via linear interpolation.

        For a level set φ(x), the zero crossing occurs between grid points where
        φ changes sign. Linear interpolation gives O(dx^2) accuracy vs O(dx) for
        simple argmin(|φ|).

        Mathematical Formula:
            If φ[i] < 0 and φ[i+1] > 0 along axis d, then:
            x_interface[d] = x[i] + |φ[i]| / (|φ[i]| + |φ[i+1]|) * dx

        For nD, this is applied independently along each axis. Edge crossings
        from all axes are concatenated into a single point cloud.

        Returns
        -------
        interface_location : float | NDArray
            For 1D: Scalar x-coordinate of first zero crossing.
            For nD: Array of shape (M, ndim) with interface point coordinates.

        Raises
        ------
        ValueError
            If no zero crossing found (all φ same sign).
        TypeError
            If geometry is not a TensorProductGrid.

        Notes
        -----
        **Accuracy**: O(dx^2) for smooth level sets (vs O(dx) for argmin).

        Example
        -------
        >>> # 1D
        >>> x_interface = ls.get_interface_location_subcell()  # float
        >>> # 2D circle
        >>> points = ls.get_interface_location_subcell()  # (M, 2) array
        """
        if self.dimension == 1:
            return self._get_interface_1d_subcell()
        else:
            return self._get_interface_nd_subcell()

    def _get_interface_1d_subcell(self) -> float:
        """
        1D subcell interface extraction via linear interpolation.

        Returns
        -------
        x_interface : float
            x-coordinate of zero crossing with O(dx²) accuracy
        """
        phi = self.phi

        # Get grid coordinates
        x = self._get_grid_coordinates()[0]

        # Find zero crossings (sign changes)
        sign_changes = np.diff(np.sign(phi))
        crossing_indices = np.where(sign_changes != 0)[0]

        if len(crossing_indices) == 0:
            raise ValueError(
                "No zero crossing found in level set. "
                f"phi range: [{phi.min():.3f}, {phi.max():.3f}]. "
                "All values have same sign."
            )

        # Use first crossing (primary interface)
        i = crossing_indices[0]

        # Linear interpolation between phi[i] and phi[i+1]
        # Zero crossing at: x = x[i] + t*(x[i+1] - x[i])
        # where t = |phi[i]| / (|phi[i]| + |phi[i+1]|)

        phi_left = phi[i]
        phi_right = phi[i + 1]

        # Interpolation parameter (0 to 1)
        t = abs(phi_left) / (abs(phi_left) + abs(phi_right))

        # Interface location
        x_interface = x[i] + t * (x[i + 1] - x[i])

        return float(x_interface)

    def _get_interface_nd_subcell(self) -> NDArray[np.float64]:
        """
        nD subcell interface extraction via vectorized edge interpolation.

        For each axis d, finds edges where phi changes sign, then linearly
        interpolates the zero crossing along that axis. Points from all axes
        are concatenated into a single (M, ndim) array.

        Returns
        -------
        points : NDArray, shape (M, ndim)
            Interface point coordinates with O(dx^2) accuracy.

        Raises
        ------
        ValueError
            If no zero crossing found along any axis.
        """
        phi = self.phi
        coords = self._get_grid_coordinates()
        ndim = phi.ndim
        all_points: list[NDArray[np.float64]] = []

        for d in range(ndim):
            # Sign changes along axis d → shape with axis d reduced by 1
            sign_diff = np.diff(np.sign(phi), axis=d)
            crossings = np.argwhere(sign_diff != 0)  # (M, ndim) index array

            if len(crossings) == 0:
                continue

            # Left/right indices along axis d
            idx_left = list(crossings.T)  # list of ndim index arrays
            idx_right = list(crossings.T)
            idx_right[d] = idx_right[d] + 1  # shift right along crossing axis

            phi_left = phi[tuple(idx_left)]
            phi_right = phi[tuple(idx_right)]

            # Interpolation parameter with division-by-zero guard
            abs_left = np.abs(phi_left)
            abs_right = np.abs(phi_right)
            denom = abs_left + abs_right
            t = np.where(denom > 0, abs_left / denom, 0.5)

            # Build physical coordinates: (M, ndim)
            pts = np.empty((len(crossings), ndim), dtype=np.float64)
            for ax in range(ndim):
                if ax == d:
                    x_lo = coords[d][idx_left[d]]
                    x_hi = coords[d][idx_right[d]]
                    pts[:, ax] = x_lo + t * (x_hi - x_lo)
                else:
                    pts[:, ax] = coords[ax][crossings[:, ax]]

            all_points.append(pts)

        if len(all_points) == 0:
            raise ValueError(
                "No zero crossing found in level set. "
                f"phi range: [{phi.min():.3f}, {phi.max():.3f}]. "
                "All values have same sign."
            )

        return np.concatenate(all_points, axis=0)

    def __repr__(self) -> str:
        """String representation for debugging."""
        phi_min, phi_max = self.phi.min(), self.phi.max()
        interface_count = np.sum(self.interface_mask())

        return (
            f"LevelSetFunction(\n"
            f"  dimension={self.dimension},\n"
            f"  shape={self.phi.shape},\n"
            f"  range=[{phi_min:.3f}, {phi_max:.3f}],\n"
            f"  is_SDF={self.is_signed_distance},\n"
            f"  interface_points={interface_count}\n"
            f")"
        )


class LevelSetEvolver:
    """
    Evolve level set function via ∂φ/∂t + V|∇φ| = 0.

    Implements Godunov upwind or WENO5 scheme for |∇φ|, paired with
    explicit Euler or TVD-RK3 time integration. Uses CFL-adaptive
    substepping to ensure stability.

    Numerical Scheme:
        Spatial: Godunov upwind O(dx) or HJ-WENO5 O(dx^5)
        Temporal: Forward Euler O(dt) or TVD-RK3/SSP-RK3 O(dt^3)
        CFL: max(|V|)*dt/h < 1 (adaptive substeps if violated)

    For best accuracy, pair WENO5 with TVD-RK3 to avoid temporal error
    dominating the spatial discretization.

    Attributes:
        geometry: Grid structure (TensorProductGrid)
        scheme: Spatial scheme ("upwind" or "weno5")
        time_integrator: Temporal scheme ("euler" or "tvd_rk3")
        cfl_max: Maximum allowed CFL number (default: 0.9)

    Example:
        >>> # 1st-order upwind + Euler (default)
        >>> evolver = LevelSetEvolver(grid, scheme="upwind")
        >>> phi_new = evolver.evolve_step(phi0, velocity=0.5, dt=0.1)
        >>>
        >>> # 5th-order WENO5 + 3rd-order TVD-RK3 (high accuracy)
        >>> evolver = LevelSetEvolver(grid, scheme="weno5", time_integrator="tvd_rk3")
        >>> phi_new = evolver.evolve_step(phi0, velocity=0.5, dt=0.1)
    """

    def __init__(
        self,
        geometry: TensorProductGrid,
        scheme: str = "upwind",
        cfl_max: float = 0.9,
        time_integrator: str = "euler",
    ):
        """
        Initialize level set evolver.

        Args:
            geometry: Grid structure (must support gradient operators)
            scheme: Spatial scheme ("upwind" or "weno5")
                - "upwind": 1st-order Godunov upwind (default)
                - "weno5": 5th-order HJ-WENO Godunov (Jiang & Peng 2000)
            cfl_max: Maximum CFL number for stability (default: 0.9)
                CFL = max(|V|)*dt/h. If exceeded, automatic substepping is used.
            time_integrator: Temporal scheme ("euler" or "tvd_rk3")
                - "euler": 1st-order forward Euler (default)
                - "tvd_rk3": 3rd-order TVD/SSP Runge-Kutta (Shu & Osher 1988)

        Raises:
            ValueError: If scheme or time_integrator not supported
            AttributeError: If geometry doesn't support required operators
        """
        if scheme not in ("upwind", "weno5"):
            raise ValueError(
                f"Scheme must be 'upwind' or 'weno5', got '{scheme}'. See Issue #605 Phase 2 for WENO5 support."
            )
        if time_integrator not in ("euler", "tvd_rk3"):
            raise ValueError(f"time_integrator must be 'euler' or 'tvd_rk3', got '{time_integrator}'.")

        self.geometry = geometry
        self.scheme = scheme
        self.time_integrator = time_integrator
        self.cfl_max = cfl_max

        # Extract grid spacing (for CFL computation)
        # TensorProductGrid stores spacing per dimension
        if hasattr(geometry, "spacing"):
            self.spacing = geometry.spacing
        else:
            # Fallback: compute from bounds and Nx
            bounds = geometry.get_bounding_box()
            Nx_points = geometry.Nx_points
            self.spacing = [(b[1] - b[0]) / (n - 1) for b, n in zip(bounds, Nx_points, strict=False)]

        self.h_min = min(self.spacing)  # Minimum spacing (for CFL)

        logger.debug(
            f"LevelSetEvolver initialized: {geometry.dimension}D, scheme={scheme}, "
            f"time_integrator={time_integrator}, spacing={self.spacing}, CFL_max={cfl_max}"
        )

    def _compute_godunov_gradient_magnitude(
        self,
        phi: NDArray[np.float64],
        velocity: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        First-order Godunov upwind gradient magnitude |∇φ|, flux selected by the
        VELOCITY sign (Osher-Sethian / Rouy-Tourin).

        Per axis, with backward/forward differences D⁻, D⁺:
            V > 0: |∇φ|² += max(D⁻, 0)² + min(D⁺, 0)²
            V < 0: |∇φ|² += max(D⁺, 0)² + min(D⁻, 0)²

        This is monotone and entropy-satisfying for Hamilton-Jacobi equations.

        Issue #1602: the previous implementation selected the one-sided stencil
        from sign(∇φ) (via the shared ``gradient_upwind`` operator), which equals
        the correct Godunov magnitude only for V > 0 and gives the downwind
        (wrong-viscosity) update for V < 0 -- a shrinking domain then persists
        past its true extinction time. In the interior this is the first-order
        analogue of the weno5 path (:meth:`_compute_weno5_gradient_magnitude`),
        which already selects on the velocity sign.

        Boundary handling: ``gradient_backward``/``gradient_forward`` wrap
        periodically (np.roll), so at a domain edge the one-sided difference that
        references the far boundary is spurious. Under the no-flux/Neumann
        convention the ghost node equals the boundary value, so that difference
        is zero -- we zero those wrapped edge entries. This reproduces the
        BC-aware boundary of the previous ``get_gradient_operator`` path exactly
        (convention-agreement verified on phi=x: interior 1, low/high edge 0/1)
        and avoids a phantom boundary interface on non-symmetric fields
        (#1602 review).

        Args:
            phi: Level set function, shape (Nx, Ny, ...)
            velocity: Velocity field, same shape as phi.

        Returns:
            Gradient magnitude |∇φ|, same shape as phi

        Reference:
            Osher & Fedkiw (2003), Chapter 6.1 - Godunov's Method
        """
        from mfgarchon.operators.stencils.finite_difference import (
            gradient_backward,
            gradient_forward,
        )

        ndim = phi.ndim
        grad_mag_sq = np.zeros_like(phi)
        pos_mask = velocity > 0

        for d in range(ndim):
            h = float(self.spacing[d])
            d_minus = gradient_backward(phi, axis=d, h=h)
            d_plus = gradient_forward(phi, axis=d, h=h)

            # No-flux (Neumann-ghost) boundary: the backward difference at the low
            # edge and the forward difference at the high edge each reference a
            # periodic-wrapped ghost node; under no-flux the ghost equals the
            # boundary value, so the difference is zero. Without this the wrap
            # injects a spurious O(1/h) edge gradient (#1602 review).
            low_edge = tuple(0 if k == d else slice(None) for k in range(ndim))
            high_edge = tuple(-1 if k == d else slice(None) for k in range(ndim))
            d_minus[low_edge] = 0.0
            d_plus[high_edge] = 0.0

            contrib = np.zeros_like(phi)
            contrib[pos_mask] = np.maximum(d_minus[pos_mask], 0.0) ** 2 + np.minimum(d_plus[pos_mask], 0.0) ** 2
            contrib[~pos_mask] = np.maximum(d_plus[~pos_mask], 0.0) ** 2 + np.minimum(d_minus[~pos_mask], 0.0) ** 2

            grad_mag_sq += contrib

        return np.sqrt(grad_mag_sq)

    def _compute_weno5_gradient_magnitude(
        self,
        phi: NDArray[np.float64],
        velocity: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute gradient magnitude |nabla phi| using HJ-WENO5 Godunov scheme.

        Implements the Jiang & Peng (2000) HJ-WENO flux selection:
            For each dimension d, compute D- (left) and D+ (right) via WENO5.
            Godunov assembly based on velocity sign:
                V > 0: |nabla phi|^2 += max(D-,0)^2 + min(D+,0)^2
                V < 0: |nabla phi|^2 += max(D+,0)^2 + min(D-,0)^2

        Args:
            phi: Level set function, shape (Nx, ...).
            velocity: Velocity field, same shape as phi.

        Returns:
            Gradient magnitude |nabla phi|, same shape as phi.

        References:
            Jiang & Peng (2000): Weighted ENO schemes for Hamilton-Jacobi equations.
        """
        from mfgarchon.operators.reconstruction.weno import compute_weno5_derivative_nd

        ndim = phi.ndim
        spacings = [float(s) for s in self.spacing]

        grad_mag_sq = np.zeros_like(phi)

        for d in range(ndim):
            d_minus = compute_weno5_derivative_nd(phi, spacings, axis=d, bias="left")
            d_plus = compute_weno5_derivative_nd(phi, spacings, axis=d, bias="right")

            # Godunov flux selection per point
            # V > 0: information from left → max(D-,0)^2 + min(D+,0)^2
            # V < 0: information from right → max(D+,0)^2 + min(D-,0)^2
            pos_mask = velocity > 0

            contrib = np.zeros_like(phi)
            contrib[pos_mask] = np.maximum(d_minus[pos_mask], 0.0) ** 2 + np.minimum(d_plus[pos_mask], 0.0) ** 2
            contrib[~pos_mask] = np.maximum(d_plus[~pos_mask], 0.0) ** 2 + np.minimum(d_minus[~pos_mask], 0.0) ** 2

            grad_mag_sq += contrib

        return np.sqrt(grad_mag_sq)

    def _compute_rhs(
        self,
        phi: NDArray[np.float64],
        V_array: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute spatial operator L(phi) = -V * |nabla phi|.

        This is the right-hand side of dphi/dt = L(phi).

        Args:
            phi: Level set function, shape (Nx, ...).
            V_array: Velocity field, same shape as phi.

        Returns:
            L(phi) = -V * |nabla phi|, same shape as phi.
        """
        if self.scheme == "weno5":
            grad_mag = self._compute_weno5_gradient_magnitude(phi, V_array)
        else:
            grad_mag = self._compute_godunov_gradient_magnitude(phi, V_array)
        return -V_array * grad_mag

    def _step_tvd_rk3(
        self,
        phi: NDArray[np.float64],
        V_array: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """
        One TVD-RK3 (SSP-RK3) substep: 3rd-order time integration.

        Shu-Osher (1988) three-stage SSP Runge-Kutta:
            phi1 = phi_n + dt * L(phi_n)
            phi2 = 3/4 * phi_n + 1/4 * phi1 + 1/4 * dt * L(phi1)
            phi_{n+1} = 1/3 * phi_n + 2/3 * phi2 + 2/3 * dt * L(phi2)

        This method preserves total variation (TVD) under the same CFL
        constraint as forward Euler, while achieving 3rd-order accuracy.

        Args:
            phi: Level set function at time n, shape (Nx, ...).
            V_array: Velocity field, same shape as phi.
            dt: Substep size (already CFL-safe).

        Returns:
            Level set function at time n+1.
        """
        # Stage 1: forward Euler
        L_n = self._compute_rhs(phi, V_array)
        phi1 = phi + dt * L_n

        # Stage 2: convex combination
        L_1 = self._compute_rhs(phi1, V_array)
        phi2 = 0.75 * phi + 0.25 * phi1 + 0.25 * dt * L_1

        # Stage 3: convex combination
        L_2 = self._compute_rhs(phi2, V_array)
        phi_new = (1.0 / 3.0) * phi + (2.0 / 3.0) * phi2 + (2.0 / 3.0) * dt * L_2

        return phi_new

    def evolve_step(
        self,
        phi: NDArray[np.float64],
        velocity: float | NDArray[np.float64] | Callable[[NDArray], NDArray],
        dt: float,
    ) -> NDArray[np.float64]:
        """
        Evolve level set one time step: φⁿ⁺¹ = φⁿ - dt·V·|∇φ|.

        Uses CFL-adaptive substepping to ensure stability even for large dt.

        Args:
            phi: Current level set function, shape (Nx, Ny, ...)
            velocity: Normal velocity field V
                - float: Constant velocity (e.g., V = 1.0)
                - NDArray: Spatially-varying velocity, shape matching phi
                - Callable: Function V(X) returning velocity at mesh points X
            dt: Time step size (will be subdivided if CFL > cfl_max)

        Returns:
            Updated level set function φⁿ⁺¹, same shape as input

        Raises:
            ValueError: If phi shape doesn't match geometry

        Example:
            >>> # Constant velocity
            >>> phi_new = evolver.evolve_step(phi, velocity=1.0, dt=0.1)
            >>>
            >>> # Spatially-varying velocity
            >>> V = np.sin(X[0]) * np.cos(X[1])  # V(x,y)
            >>> phi_new = evolver.evolve_step(phi, velocity=V, dt=0.1)
        """
        # Convert velocity to array
        if callable(velocity):
            # Get mesh coordinates
            if hasattr(self.geometry, "meshgrid"):
                X = np.array(self.geometry.meshgrid())
            else:
                raise ValueError("Geometry must support meshgrid() for callable velocity")
            V_array = velocity(X)
        elif isinstance(velocity, (int, float)):
            V_array = np.full_like(phi, float(velocity))
        else:
            V_array = velocity

        # Validate shape
        if V_array.shape != phi.shape:
            raise ValueError(f"Velocity shape {V_array.shape} doesn't match phi shape {phi.shape}")

        # CFL check and adaptive substepping
        V_max = np.max(np.abs(V_array))
        cfl = V_max * dt / self.h_min

        if cfl > self.cfl_max:
            # Need substepping
            n_substeps = int(np.ceil(cfl / self.cfl_max))
            dt_sub = dt / n_substeps
            logger.debug(f"CFL={cfl:.2f} > {self.cfl_max}, using {n_substeps} substeps (dt_sub={dt_sub:.2e})")
        else:
            n_substeps = 1
            dt_sub = dt

        # Evolve via substeps
        phi_current = phi.copy()
        for _step in range(n_substeps):
            if self.time_integrator == "tvd_rk3":
                phi_current = self._step_tvd_rk3(phi_current, V_array, dt_sub)
            else:
                # Forward Euler: φⁿ⁺¹ = φⁿ + dt·L(φⁿ)
                phi_current = phi_current + dt_sub * self._compute_rhs(phi_current, V_array)

        return phi_current

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"LevelSetEvolver(\n"
            f"  dimension={self.geometry.dimension},\n"
            f"  scheme='{self.scheme}',\n"
            f"  time_integrator='{self.time_integrator}',\n"
            f"  spacing={self.spacing},\n"
            f"  CFL_max={self.cfl_max}\n"
            f")"
        )


if __name__ == "__main__":
    """Smoke test for LevelSetFunction and LevelSetEvolver."""
    print("Testing Level Set Core...")

    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

    # Test 1D: Circle translation with constant velocity
    print("\n[Test 1: 1D Circle Translation]")
    print("Problem: Translate circle interface with constant velocity V = 1.0")

    # Create 1D grid
    Nx = 100
    grid_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[Nx], boundary_conditions=no_flux_bc(dimension=1))
    x = grid_1d.coordinates[0]
    dx = grid_1d.spacing[0]

    print(f"  Grid: {Nx} points, dx = {dx:.4f}")

    # Initial level set: interface at x = 0.5
    phi0_1d = x - 0.5
    ls = LevelSetFunction(phi0_1d, grid_1d, is_signed_distance=True)

    print("  Initial interface at x = 0.5")
    print(f"  Level set: {ls}")

    # Get interface mask
    interface_mask = ls.interface_mask(width=2 * dx)
    print(f"  Interface points (|φ| < {2 * dx:.4f}): {np.sum(interface_mask)}")

    # Evolve with constant velocity V = 1.0
    evolver = LevelSetEvolver(grid_1d, scheme="upwind", cfl_max=0.9)
    print(f"\n  Evolver: {evolver}")

    V = 1.0  # Constant velocity
    dt = 0.1
    phi1_1d = evolver.evolve_step(phi0_1d, velocity=V, dt=dt)

    # Check interface moved to x = 0.5 + V·dt = 0.6
    x_expected = 0.5 + V * dt
    # Find zero crossing (where phi changes sign)
    zero_idx = np.where(np.diff(np.sign(phi1_1d)))[0]
    if len(zero_idx) > 0:
        x_interface = x[zero_idx[0]]
        error = np.abs(x_interface - x_expected)
        print(f"\n  Expected interface: x = {x_expected:.3f}")
        print(f"  Computed interface: x = {x_interface:.3f}")
        print(f"  Error: {error:.4f} ({error / dx:.2f} grid points)")
        assert error < 2 * dx, f"Interface error {error:.3f} > {2 * dx:.3f}"
        print("  ✓ Interface translation test passed!")
    else:
        print("  WARNING: No zero crossing found (interface may have left domain)")

    # Test 2D: Circle expansion
    print("\n[Test 2: 2D Circle - Normal Field]")
    print("Problem: Compute normal field for circular level set")

    # Create 2D grid
    Nx, Ny = 50, 50
    grid_2d = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[Nx, Ny], boundary_conditions=no_flux_bc(dimension=2)
    )
    X, Y = grid_2d.meshgrid()
    dx2d = grid_2d.spacing[0]

    print(f"  Grid: {Nx}×{Ny}, dx = {dx2d:.4f}")

    # Circle: φ = ||x - c|| - R
    center = np.array([0.5, 0.5])
    radius = 0.3
    phi0_2d = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

    ls_2d = LevelSetFunction(phi0_2d, grid_2d, is_signed_distance=True)
    print(f"  Circle: center={center}, radius={radius}")
    print(f"  Level set: {ls_2d}")

    # Compute normal field
    normals = ls_2d.get_normal()
    print(f"  Normal field shape: {normals.shape} (2, {Nx}, {Ny})")

    # Check normal magnitude = 1 (only on interface where it's meaningful)
    interface_2d = ls_2d.interface_mask(width=2 * dx2d)
    normal_mag = np.linalg.norm(normals, axis=0)
    normal_mag_interface = normal_mag[interface_2d]

    print(f"  Normal magnitude (full): min={normal_mag.min():.6f}, max={normal_mag.max():.6f}")
    print(f"  Normal magnitude (interface): min={normal_mag_interface.min():.6f}, max={normal_mag_interface.max():.6f}")
    assert np.allclose(normal_mag_interface, 1.0, atol=1e-6), "Normal not unit length on interface"
    print("  ✓ Unit normal test passed!")

    # Check normal direction on interface (should point radially outward)
    n_x_interface = normals[0][interface_2d]
    n_y_interface = normals[1][interface_2d]

    # Expected normal: (x - c) / ||x - c||
    X_interface = X[interface_2d]
    Y_interface = Y[interface_2d]
    n_x_expected = (X_interface - center[0]) / radius
    n_y_expected = (Y_interface - center[1]) / radius

    error_nx = np.max(np.abs(n_x_interface - n_x_expected))
    error_ny = np.max(np.abs(n_y_interface - n_y_expected))
    print(f"  Normal direction error: n_x={error_nx:.3e}, n_y={error_ny:.3e}")
    # Relaxed tolerance (0.15) for coarse grid - normal direction is approximate
    assert error_nx < 0.15, f"Normal direction error (x) too large: {error_nx:.3e}"
    assert error_ny < 0.15, f"Normal direction error (y) too large: {error_ny:.3e}"
    print("  ✓ Normal direction test passed!")

    print("\n✅ All Level Set Core tests passed!")

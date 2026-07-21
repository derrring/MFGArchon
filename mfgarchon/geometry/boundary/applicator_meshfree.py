"""
Meshfree boundary condition applicator for particle and collocation methods.

This module provides BC application for meshfree discretizations:
- Particle methods (Lagrangian particle FP solver)
- Collocation methods (RBF, GFDM)
- High-dimensional implicit domains (d >= 4)

BC Types Supported:
- Reflecting: Particles bounce off boundary (project back to interior)
- Absorbing: Particles at boundary are removed
- Periodic: Particles wrap around (hyperrectangle domains only)
- Penalty: Add penalty term to field values at boundary (collocation)
- Projection: Project field values at boundary nodes

Architecture:
    Domain provides geometry via GeometryProtocol (boundary detection, projection)
    MeshfreeApplicator applies BC based on domain's boundary info

Created: 2025-11-27
Part of: Unified Boundary Condition Architecture
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .conditions import BoundaryConditions

from mfgarchon.geometry.protocol import GeometryProtocol

# Import base class for inheritance
from .applicator_base import BaseMeshfreeApplicator
from .types import BCType


def _robin_alpha_beta(boundary_conditions) -> tuple[float, float]:
    """Read Robin (alpha, beta) from the BCSegment that carries them (Issue #1558).

    alpha/beta live on ``BCSegment``, not on ``BoundaryConditions`` -- the previous
    ``getattr(boundary_conditions, "beta", 0.0)`` always read the 0.0 default, silently collapsing
    every Robin BC to pure Dirichlet (field path: hard ``u = g/alpha``; particle path: absorbing).
    Read them from the segment whose type is ROBIN (for a uniform Robin BC this is the sole segment).
    """
    robin_seg = next((s for s in boundary_conditions.segments if s.bc_type == BCType.ROBIN), None)
    if robin_seg is None:
        raise ValueError("Robin BC resolved but no BCSegment carries the Robin alpha/beta coefficients.")
    return float(robin_seg.alpha), float(robin_seg.beta)


class MeshfreeApplicator(BaseMeshfreeApplicator):
    """
    Boundary condition applicator for meshfree methods.

    Works with any geometry implementing GeometryProtocol, using the geometry's
    boundary methods to detect and handle boundary conditions.

    Supports two modes:
    1. Particle BC: Apply BC to particle positions (reflecting, absorbing, periodic)
    2. Field BC: Apply BC to field values at collocation points (penalty, projection)

    Examples:
        >>> from mfgarchon.geometry import Hyperrectangle
        >>> domain = Hyperrectangle([[0, 1], [0, 1]])
        >>> applicator = MeshfreeApplicator(domain)
        >>>
        >>> # Particle BC
        >>> particles = np.array([[0.5, 0.5], [1.2, 0.3]])  # One outside
        >>> particles_updated = applicator.apply_particle_bc(particles, "reflecting")
        >>> assert np.all(domain.contains(particles_updated))
        >>>
        >>> # Field BC at collocation points
        >>> u = np.random.rand(100)  # Field at 100 collocation points
        >>> points = domain.sample_interior(100)
        >>> u_with_bc = applicator.apply_field_bc(u, points, bc_type="dirichlet", bc_value=0.0)
    """

    def __init__(self, geometry: GeometryProtocol):
        """
        Initialize meshfree BC applicator.

        Args:
            geometry: Geometry implementing GeometryProtocol with boundary methods
        """
        if not isinstance(geometry, GeometryProtocol):
            raise TypeError(f"Geometry must implement GeometryProtocol, got {type(geometry).__name__}")

        super().__init__(dimension=geometry.dimension)
        self.geometry = geometry

    # =========================================================================
    # Particle Boundary Conditions
    # =========================================================================

    def apply_particle_bc(
        self,
        particles: NDArray[np.floating],
        bc_type: Literal["reflecting", "absorbing", "periodic"] = "reflecting",
    ) -> NDArray[np.floating]:
        """
        Apply boundary conditions to particles.

        Args:
            particles: Particle positions - shape (N, d)
            bc_type: Boundary condition type
                - "reflecting": Project particles back into domain
                - "absorbing": Remove particles outside domain
                - "periodic": Wrap particles (hyperrectangle only)

        Returns:
            Updated particle positions (may have fewer particles for "absorbing")

        Examples:
            >>> particles = np.array([[0.5, 0.5], [1.2, 0.3]])
            >>> updated = applicator.apply_particle_bc(particles, "reflecting")
        """
        particles = np.atleast_2d(particles)

        if bc_type == "reflecting":
            return self._apply_reflecting_bc(particles)

        elif bc_type == "absorbing":
            return self._apply_absorbing_bc(particles)

        elif bc_type == "periodic":
            return self._apply_periodic_bc(particles)

        else:
            raise ValueError(f"Unknown particle BC type: {bc_type}")

    def _apply_reflecting_bc(
        self,
        particles: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Reflect particles outside domain back to interior.

        For bounded domains (hyperrectangles), uses modular fold reflection
        which correctly handles particles that travel multiple domain widths.
        Uses canonical implementation from geometry.boundary.corner (Issue #521).

        At corners, all dimensions are processed simultaneously, producing
        diagonal reflection (equivalent to 'average' corner strategy).

        For other domains, falls back to projection to interior.
        """
        from .corner import reflect_positions

        bounds = self.geometry.get_bounds()

        if bounds is not None:
            # For bounded domains, use modular fold reflection
            min_coords, max_coords = bounds
            bounds_list = list(zip(min_coords, max_coords, strict=True))
            return reflect_positions(particles, bounds_list)

        # Fallback for unbounded/complex domains: project to interior
        return self.geometry.project_to_interior(particles)

    def _apply_absorbing_bc(
        self,
        particles: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Remove particles outside domain."""
        on_boundary = self.geometry.is_on_boundary(particles)

        bounds = self.geometry.get_bounds()
        if bounds is None:
            # Unbounded domain - keep all particles
            return particles

        min_coords, max_coords = bounds
        inside = np.ones(len(particles), dtype=bool)

        for d in range(self.dimension):
            inside &= (particles[:, d] >= min_coords[d]) & (particles[:, d] <= max_coords[d])

        # For absorbing BC, remove particles at boundary too
        inside &= ~on_boundary

        return particles[inside]

    def _apply_periodic_bc(
        self,
        particles: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Wrap particles around domain (periodic BC).

        Uses canonical implementation from geometry.boundary.corner (Issue #521).
        """
        # Issue #711: Use canonical periodic.py (parallel to enforcement.py)
        from .periodic import wrap_positions

        bounds = self.geometry.get_bounds()
        if bounds is None:
            raise ValueError("Periodic BC requires bounded domain")

        min_coords, max_coords = bounds
        bounds_list = list(zip(min_coords, max_coords, strict=True))
        return wrap_positions(particles, bounds_list)

    # =========================================================================
    # Field Boundary Conditions (for collocation methods)
    # =========================================================================

    def apply_field_bc(
        self,
        field: NDArray[np.floating],
        points: NDArray[np.floating],
        bc_type: Literal["dirichlet", "neumann", "penalty"] = "dirichlet",
        bc_value: float | Callable[[NDArray], float] = 0.0,
        penalty_weight: float = 1e6,
    ) -> NDArray[np.floating]:
        """
        Apply boundary conditions to field values at collocation points.

        Args:
            field: Field values at collocation points - shape (N,)
            points: Collocation point positions - shape (N, d)
            bc_type: Boundary condition type
                - "dirichlet": Set field to bc_value at boundary
                - "neumann": Set normal derivative to bc_value (requires additional ops)
                - "penalty": Add penalty term pushing field toward bc_value
            bc_value: Boundary value (float or callable(points) -> float)
            penalty_weight: Weight for penalty method

        Returns:
            Updated field values with BC applied

        Examples:
            >>> u = np.random.rand(100)
            >>> points = domain.sample_interior(100)
            >>> u_bc = applicator.apply_field_bc(u, points, "dirichlet", bc_value=0.0)
        """
        field = field.copy()
        points = np.atleast_2d(points)

        # Find boundary points
        on_boundary = self.geometry.is_on_boundary(points)

        if not np.any(on_boundary):
            return field

        # Get boundary value
        if callable(bc_value):
            boundary_points = points[on_boundary]
            boundary_values = np.array([bc_value(p) for p in boundary_points])
        else:
            boundary_values = bc_value

        if bc_type == "dirichlet":
            field[on_boundary] = boundary_values

        elif bc_type == "penalty":
            # Soft enforcement via penalty
            field[on_boundary] = (field[on_boundary] + penalty_weight * boundary_values) / (1 + penalty_weight)

        elif bc_type == "neumann":
            # Neumann BC requires derivative info - not directly applicable to field values
            # This would need integration with the solver's derivative operators
            raise NotImplementedError(
                "Neumann BC for collocation requires solver-specific implementation. "
                "Use the solver's derivative operators to enforce Neumann conditions."
            )

        else:
            raise ValueError(f"Unknown field BC type: {bc_type}")

        return field

    # =========================================================================
    # Boundary Information Queries
    # =========================================================================

    def get_boundary_mask(
        self,
        points: NDArray[np.floating],
        tolerance: float = 1e-10,
    ) -> NDArray[np.bool_]:
        """
        Get boolean mask of boundary points.

        Args:
            points: Query points - shape (N, d)
            tolerance: Distance tolerance for boundary detection

        Returns:
            Boolean array - True for boundary points
        """
        return self.geometry.is_on_boundary(points, tolerance=tolerance)

    def get_boundary_normals(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Get outward normal vectors at points.

        Only meaningful for points near the boundary.

        Args:
            points: Query points - shape (N, d)

        Returns:
            Normal vectors - shape (N, d)
        """
        return self.geometry.get_boundary_normal(points)

    def get_boundary_regions(self) -> dict[str, dict]:
        """
        Get named boundary regions for mixed BC specification.

        Returns:
            Dictionary mapping region names to region info
        """
        return self.geometry.get_boundary_regions()

    # =========================================================================
    # Unified API (Issue #636 Phase 2)
    # =========================================================================

    def apply(
        self,
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions,
        points: NDArray[np.floating],
        time: float = 0.0,
    ) -> NDArray[np.floating]:
        """
        Apply boundary conditions to field values using unified BoundaryConditions.

        This is the unified API that accepts BoundaryConditions objects,
        mapping BCType to internal methods automatically.

        Args:
            field: Field values at collocation points - shape (N,)
            boundary_conditions: BC specification (unified BoundaryConditions)
            points: Collocation point positions - shape (N, d)
            time: Current time for time-dependent BC values

        Returns:
            Updated field values with BC applied

        BC Type Mapping:
            - BCType.DIRICHLET → Direct value assignment at boundary
            - BCType.NEUMANN → Not directly applicable (raises error)
            - BCType.ROBIN → Penalty method with α, β weights
            - BCType.PERIODIC → Not applicable for field values
            - BCType.NO_FLUX → Zero gradient (extrapolation)

        Examples:
            >>> from mfgarchon.geometry.boundary import dirichlet_bc
            >>> bc = dirichlet_bc(dimension=2, value=0.0)
            >>> u_with_bc = applicator.apply(u, bc, points)
        """
        field = field.copy()
        points = np.atleast_2d(points)

        # Find boundary points
        on_boundary = self.geometry.is_on_boundary(points)

        if not np.any(on_boundary):
            return field

        # Get BC type (uniform BC for now; fails loud if default_bc unset, Issue #1100)
        bc_type = boundary_conditions._resolve_default_bc("MeshfreeBoundaryApplicator.apply")

        if bc_type == BCType.DIRICHLET:
            # Get Dirichlet value
            bc_value = boundary_conditions.default_value
            if bc_value is None:
                bc_value = 0.0
            field[on_boundary] = bc_value

        elif bc_type == BCType.NEUMANN:
            # Neumann BC requires derivative operators
            raise NotImplementedError(
                "Neumann BC for meshfree methods requires solver-specific derivative operators. "
                "Use the solver's infrastructure to enforce Neumann conditions."
            )

        elif bc_type == BCType.ROBIN:
            # Robin BC: α*u + β*du/dn = g
            # For meshfree without derivative info, use penalty approximation
            alpha, beta = _robin_alpha_beta(boundary_conditions)
            bc_value = boundary_conditions.default_value
            if bc_value is None:
                bc_value = 0.0

            if np.isclose(beta, 0.0):
                # Pure Dirichlet: u = g/α
                field[on_boundary] = bc_value / alpha if not np.isclose(alpha, 0.0) else bc_value
            else:
                # Use penalty method as approximation
                penalty_weight = abs(alpha / beta) if not np.isclose(beta, 0.0) else 1e6
                field[on_boundary] = (field[on_boundary] + penalty_weight * bc_value) / (1 + penalty_weight)

        elif bc_type == BCType.NO_FLUX:
            # Zero flux: du/dn = 0
            # For meshfree, this means boundary values should match nearby interior
            # We approximate by keeping current values (no modification)
            pass

        elif bc_type == BCType.PERIODIC:
            # Periodic BC doesn't apply directly to field values at points
            # Would need to handle during interpolation/evaluation
            raise NotImplementedError(
                "Periodic BC for field values requires special handling during interpolation. "
                "Use apply_particles() for particle-based periodic BC."
            )

        else:
            raise ValueError(f"Unsupported BC type for meshfree: {bc_type}")

        return field

    def apply_particles(
        self,
        particles: NDArray[np.floating],
        boundary_conditions: BoundaryConditions,
    ) -> NDArray[np.floating]:
        """
        Apply boundary conditions to particle positions using unified BoundaryConditions.

        This is the unified API that accepts BoundaryConditions objects,
        mapping BCType to particle BC behavior automatically.

        Args:
            particles: Particle positions - shape (N, d)
            boundary_conditions: BC specification (unified BoundaryConditions)

        Returns:
            Updated particle positions

        BC Type Mapping:
            - BCType.NEUMANN (zero-flux) → Reflecting BC
            - BCType.NO_FLUX → Reflecting BC
            - BCType.DIRICHLET → Absorbing BC (particles removed at boundary)
            - BCType.PERIODIC → Wrap-around BC
            - BCType.ROBIN with β >> α → Reflecting BC
            - BCType.ROBIN with α >> β → Absorbing BC

        Examples:
            >>> from mfgarchon.geometry.boundary import neumann_bc
            >>> bc = neumann_bc(dimension=2)  # Zero-flux = reflecting
            >>> particles_updated = applicator.apply_particles(particles, bc)
        """
        # Fails loud if default_bc unset (Issue #1100)
        bc_type = boundary_conditions._resolve_default_bc("MeshfreeBoundaryApplicator.apply_particles")

        if bc_type == BCType.NEUMANN:
            # Zero-flux Neumann → reflecting BC for particles
            return self.apply_particle_bc(particles, "reflecting")

        elif bc_type == BCType.NO_FLUX:
            # Explicit no-flux → reflecting BC
            return self.apply_particle_bc(particles, "reflecting")

        elif bc_type == BCType.DIRICHLET:
            # Dirichlet → absorbing BC (particles "exit" at boundary)
            return self.apply_particle_bc(particles, "absorbing")

        elif bc_type == BCType.PERIODIC:
            # Periodic → wrap-around
            return self.apply_particle_bc(particles, "periodic")

        elif bc_type == BCType.ROBIN:
            # Robin: behavior depends on α/β ratio
            alpha, beta = _robin_alpha_beta(boundary_conditions)

            if np.isclose(beta, 0.0):
                # Pure Dirichlet-like → absorbing
                return self.apply_particle_bc(particles, "absorbing")
            elif np.isclose(alpha, 0.0):
                # Pure Neumann-like → reflecting
                return self.apply_particle_bc(particles, "reflecting")
            else:
                # Mixed Robin → default to reflecting (mass conservation)
                return self.apply_particle_bc(particles, "reflecting")

        else:
            raise ValueError(f"Unsupported BC type for particles: {bc_type}")


class SDFParticleBCHandler:
    """
    SDF-based particle boundary condition handler for complex geometry.

    Uses signed distance functions to detect boundary crossings and perform
    physics-based reflection with proper normal computation.

    Unlike rectangular domain handlers that only check axis-aligned bounds,
    this handler works with any geometry defined by an SDF function.

    The SDF convention is: negative inside, zero on boundary, positive outside.

    Examples:
        >>> from mfgarchon.utils.numerical import sdf_sphere
        >>> def circle_sdf(points):
        ...     return sdf_sphere(points, center=[0, 0], radius=1.0)
        >>> handler = SDFParticleBCHandler(circle_sdf, dimension=2)
        >>>
        >>> # Particles that crossed boundary get reflected
        >>> X_old = np.array([[0.5, 0], [0.9, 0]])
        >>> X_new = np.array([[0.6, 0], [1.2, 0]])  # Second one crossed
        >>> X_reflected = handler.apply_bc(X_old, X_new)
    """

    def __init__(
        self,
        sdf: Callable[[NDArray], NDArray],
        dimension: int,
        max_bisection_iterations: int = 20,
        bisection_tolerance: float = 1e-8,
        epsilon: float = 1e-5,
    ):
        """
        Initialize SDF-based particle BC handler.

        Args:
            sdf: Signed distance function. sdf(points) -> distances where
                negative = inside, zero = boundary, positive = outside
            dimension: Spatial dimension
            max_bisection_iterations: Max iterations for boundary point bisection
            bisection_tolerance: Tolerance for bisection convergence
            epsilon: Finite difference step for gradient computation
        """
        self.sdf = sdf
        self.dimension = dimension
        self.max_bisection_iterations = max_bisection_iterations
        self.bisection_tolerance = bisection_tolerance
        self.epsilon = epsilon

    def apply_bc(
        self,
        X_old: NDArray[np.floating],
        X_new: NDArray[np.floating],
        velocities: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
        """
        Apply boundary conditions to particles that crossed the boundary.

        For particles that moved from inside (sdf < 0) to outside (sdf > 0),
        finds the exact boundary crossing point and reflects them back.

        Args:
            X_old: Previous particle positions, shape (N, d)
            X_new: New particle positions (possibly outside), shape (N, d)
            velocities: Optional particle velocities for elastic reflection

        Returns:
            Tuple of (X_corrected, velocities_corrected)
            - X_corrected: Positions with boundary crossers reflected back
            - velocities_corrected: Velocities with normal component reversed (if provided)
        """
        X_old = np.atleast_2d(X_old)
        X_new = np.atleast_2d(X_new).copy()

        if velocities is not None:
            velocities = np.atleast_2d(velocities).copy()

        # Evaluate SDF at old and new positions
        sdf_old = self.sdf(X_old)
        sdf_new = self.sdf(X_new)

        # Detect boundary crossings: was inside (< 0), now outside (> 0)
        crossed = (sdf_old < 0) & (sdf_new > 0)
        crossed_indices = np.where(crossed)[0]

        if len(crossed_indices) == 0:
            return X_new, velocities

        # Process each crossing particle
        for i in crossed_indices:
            # Find boundary intersection point via bisection
            X_boundary = self._find_boundary_point(X_old[i], X_new[i])

            # Compute outward normal at boundary point
            normal = self._compute_normal(X_boundary)

            # Compute penetration vector (from boundary to current position)
            penetration = X_new[i] - X_boundary

            # Reflect: X_reflected = X_boundary + (penetration - 2*(penetration·n)*n)
            # This reflects the component of penetration normal to the boundary
            normal_component = np.dot(penetration, normal)
            X_new[i] = X_boundary + penetration - 2 * normal_component * normal

            # If still outside after reflection, project to boundary
            # (handles grazing angles and numerical issues)
            if self.sdf(X_new[i].reshape(1, -1))[0] > 0:
                X_new[i] = X_boundary

            # Reflect velocity if provided (elastic collision)
            if velocities is not None:
                v_normal = np.dot(velocities[i], normal)
                velocities[i] = velocities[i] - 2 * v_normal * normal

        return X_new, velocities

    def _find_boundary_point(
        self,
        p_inside: NDArray[np.floating],
        p_outside: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Find boundary crossing point using bisection.

        Args:
            p_inside: Point inside domain (sdf < 0)
            p_outside: Point outside domain (sdf > 0)

        Returns:
            Point on boundary (sdf ≈ 0) between p_inside and p_outside
        """
        a, b = 0.0, 1.0  # t parameter: p(t) = p_inside + t * (p_outside - p_inside)
        direction = p_outside - p_inside

        for _ in range(self.max_bisection_iterations):
            t_mid = (a + b) / 2
            p_mid = p_inside + t_mid * direction
            sdf_mid = self.sdf(p_mid.reshape(1, -1))[0]

            if abs(sdf_mid) < self.bisection_tolerance:
                return p_mid

            if sdf_mid < 0:
                a = t_mid  # Still inside, move toward outside
            else:
                b = t_mid  # Outside, move toward inside

        # Return best estimate
        t_final = (a + b) / 2
        return p_inside + t_final * direction

    def _compute_normal(
        self,
        point: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute outward unit normal at a point using finite differences on SDF.

        Args:
            point: Point at which to compute normal, shape (d,)

        Returns:
            Unit outward normal vector, shape (d,)
        """
        point = point.reshape(1, -1)
        grad = np.zeros(self.dimension)

        for i in range(self.dimension):
            p_plus = point.copy()
            p_minus = point.copy()
            p_plus[0, i] += self.epsilon
            p_minus[0, i] -= self.epsilon

            grad[i] = (self.sdf(p_plus)[0] - self.sdf(p_minus)[0]) / (2 * self.epsilon)

        # Normalize
        norm = np.linalg.norm(grad)
        if norm <= 1e-12:
            # Fail loud (Issue #1558): a vanishing SDF gradient has no well-defined outward
            # normal. Fabricating [1, 0, ...] reflects the particle along a geometry-independent
            # direction -- a silent-wrong that corrupts particle stepping. Mirror the #1047 raise
            # in project_to_domain rather than manufacturing a normal.
            raise RuntimeError(
                f"SDFParticleBCHandler: vanishing SDF gradient (|grad| = {norm:.3e}) at point "
                f"{point.ravel()}; the outward normal is undefined there. This indicates a "
                f"degenerate geometry (locally flat / saddle SDF, or a point on the medial axis). "
                f"Check geometry.signed_distance for correctness near this point."
            )
        return grad / norm

    def contains(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.bool_]:
        """
        Check if points are inside the domain (sdf < 0).

        Args:
            points: Points to check, shape (N, d)

        Returns:
            Boolean array, True if inside domain
        """
        return self.sdf(np.atleast_2d(points)) < 0


class ParticleReflector:
    """
    Optimized particle reflector for Lagrangian FP solvers.

    Handles reflection of particles that cross domain boundaries during
    time integration. Supports both simple projection and physics-based
    elastic reflection.

    Examples:
        >>> reflector = ParticleReflector(domain)
        >>> particles_new = reflector.reflect(particles, velocities, dt)
    """

    def __init__(
        self,
        geometry: GeometryProtocol,
        reflection_type: Literal["simple", "elastic"] = "simple",
    ):
        """
        Initialize particle reflector.

        Args:
            geometry: Domain geometry with boundary methods
            reflection_type: Type of reflection
                - "simple": Project to boundary (no velocity change)
                - "elastic": Reflect velocity at boundary normal
        """
        self.geometry = geometry
        self.reflection_type = reflection_type

    def reflect(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating] | None = None,
        dt: float = 1.0,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
        """
        Reflect particles at domain boundary.

        Args:
            positions: Particle positions after time step - shape (N, d)
            velocities: Particle velocities - shape (N, d), required for elastic
            dt: Time step (for elastic reflection distance calculation)

        Returns:
            Tuple of (updated_positions, updated_velocities)
        """
        positions = np.atleast_2d(positions)

        if self.reflection_type == "simple":
            # Simple projection back to interior
            new_positions = self.geometry.project_to_interior(positions)
            return new_positions, velocities

        elif self.reflection_type == "elastic":
            if velocities is None:
                raise ValueError("Elastic reflection requires velocities")

            velocities = np.atleast_2d(velocities)
            return self._elastic_reflect(positions, velocities, dt)

        else:
            raise ValueError(f"Unknown reflection type: {self.reflection_type}")

    def _elastic_reflect(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
        dt: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Elastic reflection with velocity reversal at boundary normal."""
        new_positions = positions.copy()
        new_velocities = velocities.copy()

        # Find particles at boundary
        on_boundary = self.geometry.is_on_boundary(positions)

        if not np.any(on_boundary):
            return new_positions, new_velocities

        # Project to interior
        new_positions[on_boundary] = self.geometry.project_to_interior(positions[on_boundary])

        # Reflect velocity: v_new = v - 2(v · n)n
        normals = self.geometry.get_boundary_normal(positions[on_boundary])
        v_normal = np.sum(velocities[on_boundary] * normals, axis=1, keepdims=True)
        new_velocities[on_boundary] = velocities[on_boundary] - 2 * v_normal * normals

        return new_positions, new_velocities


__all__ = [
    "MeshfreeApplicator",
    "ParticleReflector",
    "SDFParticleBCHandler",
]

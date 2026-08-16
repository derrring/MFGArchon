"""
Segment-aware boundary condition applicator for particle methods.

This module provides BC application for Lagrangian particle solvers using
BoundaryConditions segment matching. Unlike MeshfreeApplicator which uses
geometry-based boundary detection, this applicator queries BCSegments to
determine the appropriate action at each boundary point.

BC Type Interpretation for Particles -- owned by `particle_action_for_bc_type` in this
module, and shared with `MeshfreeApplicator.apply_particles`, which used to write it out
again and had drifted from it on four of BCType's eight members (#1956):

- DIRICHLET: Absorb particle (remove from simulation) - exits/sinks
- NEUMANN: Reflect particle (elastic bounce)
- REFLECTING / NO_FLUX: Reflect particle (elastic bounce)
- PERIODIC: Wrap particle to opposite boundary
- ROBIN: by coefficient -- `beta = 0` is Dirichlet, so absorb; otherwise reflect
- EXTRAPOLATION_LINEAR / _QUADRATIC: refused, they are field-truncation rules

Architecture:
    BoundaryConditions provides segment matching via get_bc_at_point()
    particle_action_for_bc_type interprets BCType for particle dynamics
    ParticleApplicator executes the action, per matched segment

Created: 2025-01-03
Part of: Unified Boundary Condition Architecture (Issue #536)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .types import BCType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .conditions import BoundaryConditions

ParticleAction = Literal["reflecting", "absorbing", "periodic"]


def particle_action_for_bc_type(
    bc_type: BCType,
    alpha: float | None = None,
    beta: float | None = None,
) -> ParticleAction:
    """Map a ``BCType`` to what a particle does when it reaches the boundary.

    One owner for a mapping that used to be written twice -- here and in
    ``MeshfreeApplicator.apply_particles`` -- and had drifted on **four of BCType's
    eight members**, measured on the same five particles and the same domain:

    ===========================  ===================  ==================
    BCType                       MeshfreeApplicator   ParticleApplicator
    ===========================  ===================  ==================
    ``DIRICHLET``                absorbing            absorbing
    ``NEUMANN`` / ``NO_FLUX``    reflecting           reflecting
    ``PERIODIC``                 periodic             periodic
    ``REFLECTING``               **ValueError**       reflecting
    ``ROBIN`` (alpha=1, beta=0)  **absorbing**        **reflecting**
    ``EXTRAPOLATION_*``          ValueError           **reflecting**
    ===========================  ===================  ==================

    The `ROBIN` row is the one that mattered: the *same* specification made one path
    build an impermeable wall and the other an absorbing one. Each divergence is
    resolved toward whichever path was right, so this function is not either previous
    behaviour:

    - `REFLECTING` reflects. It is `BCType`'s own documented particle spelling of an
      impermeable wall; refusing it was the defect.
    - `ROBIN` dispatches on the coefficients. With `beta = 0` the condition reads
      `alpha*u = g`, which is Dirichlet, so the particle is absorbed. With `alpha = 0`
      it reads `beta*du/dn = g`, a flux condition, so the particle reflects. A genuinely
      mixed Robin reflects, which conserves mass and is the conservative reading.
    - `EXTRAPOLATION_LINEAR` / `EXTRAPOLATION_QUADRATIC` raise. They are not boundary
      conditions on a particle at all -- they are a statement about how to continue a
      *field* past a truncated domain, and carry no boundary datum. Silently reflecting
      them, which is what the segment-aware path's ``else`` branch did, answers a
      question that was never asked.

    Args:
        bc_type: The condition to interpret.
        alpha: Robin coefficient on ``u``. Required when ``bc_type`` is ``ROBIN``.
        beta: Robin coefficient on ``du/dn``. Required when ``bc_type`` is ``ROBIN``.

    Returns:
        The particle action, in the vocabulary ``MeshfreeApplicator.apply_particle_bc``
        already accepted.

    Raises:
        ValueError: for a type with no particle interpretation, naming it.
    """
    if bc_type == BCType.DIRICHLET:
        return "absorbing"

    if bc_type in (BCType.NEUMANN, BCType.NO_FLUX, BCType.REFLECTING):
        return "reflecting"

    if bc_type == BCType.PERIODIC:
        return "periodic"

    if bc_type == BCType.ROBIN:
        if alpha is None or beta is None:
            raise ValueError(
                "particle_action_for_bc_type: BCType.ROBIN needs alpha and beta; "
                "they live on the BCSegment, not on BoundaryConditions."
            )
        if np.isclose(beta, 0.0):
            return "absorbing"
        return "reflecting"

    raise ValueError(
        f"particle_action_for_bc_type: {bc_type} has no particle interpretation. "
        "EXTRAPOLATION_LINEAR and EXTRAPOLATION_QUADRATIC describe how to continue a "
        "field past a truncated domain and say nothing about a particle that reaches it."
    )


class ParticleApplicator:
    """
    Segment-aware boundary condition applicator for particle methods.

    This applicator queries BoundaryConditions to determine the BC type at each
    boundary particle's location, then applies the appropriate action:

    - DIRICHLET: Absorb particle (remove from simulation)
    - NEUMANN: Reflect particle (elastic bounce)
    - REFLECTING / NO_FLUX: Reflect particle (elastic bounce)
    - PERIODIC: Wrap particle to opposite boundary

    Unlike MeshfreeApplicator which uses geometry-based BC, this class uses
    the BoundaryConditions segment matching to support mixed BC (e.g., exits
    on some walls, reflecting on others).

    Examples:
        >>> from mfgarchon.geometry.boundary import mixed_bc, BCSegment, BCType
        >>> bc = mixed_bc(dimension=2, segments=[
        ...     BCSegment("exit", BCType.DIRICHLET, value=0, boundary="right",
        ...               region={"y": (4.0, 6.0)}),
        ...     BCSegment("walls", BCType.REFLECTING, boundary="all", priority=-1),
        ... ])
        >>> applicator = ParticleApplicator()
        >>> remaining, absorbed, exits = applicator.apply(particles, bc, bounds)
    """

    def __init__(self) -> None:
        """Initialize particle applicator."""
        self._boundary_tolerance = 1e-10

    def apply(
        self,
        particles: NDArray[np.floating],
        bc: BoundaryConditions,
        bounds: list[tuple[float, float]],
    ) -> tuple[NDArray[np.floating], NDArray[np.bool_], NDArray[np.floating]]:
        """
        Apply segment-aware boundary conditions to particles.

        For each particle that has crossed or is at the boundary:
        1. Query bc.get_bc_at_point() to determine which BC segment applies
        2. Apply the appropriate action based on bc_type

        Args:
            particles: Particle positions, shape (N, d)
            bc: BoundaryConditions with segment definitions (must have domain_bounds set)
            bounds: Domain bounds per dimension [(xmin, xmax), ...] - used if bc.domain_bounds is None

        Returns:
            Tuple of:
            - remaining_particles: Particles not absorbed, shape (M, d) where M <= N
            - absorbed_mask: Boolean mask of absorbed particles, shape (N,)
            - exit_positions: Positions where particles were absorbed, shape (K, d)
        """

        particles = np.atleast_2d(particles)
        n_particles = len(particles)
        dimension = particles.shape[1]

        if n_particles == 0:
            return particles, np.array([], dtype=bool), np.array([]).reshape(0, dimension)

        # Convert bounds to arrays for vectorized operations
        bounds_arr = np.array(bounds)  # Shape: (d, 2)
        domain_min = bounds_arr[:, 0]
        domain_max = bounds_arr[:, 1]
        domain_size = domain_max - domain_min

        # Ensure BC has domain_bounds for segment matching
        if bc.domain_bounds is None:
            bc.domain_bounds = bounds_arr

        # Find particles at or beyond boundary
        at_min = particles <= domain_min + self._boundary_tolerance
        at_max = particles >= domain_max - self._boundary_tolerance
        at_boundary = np.any(at_min | at_max, axis=1)

        # Initialize output arrays
        absorbed_mask = np.zeros(n_particles, dtype=bool)
        exit_positions_list = []
        result_particles = particles.copy()

        # Process boundary particles
        boundary_indices = np.where(at_boundary)[0]

        for idx in boundary_indices:
            particle = result_particles[idx]

            # Determine which boundary the particle is at
            boundary_id = self._get_boundary_id(particle, domain_min, domain_max, dimension)

            # Query BC at this point (uses bc.domain_bounds internally)
            segment = bc.get_bc_at_point(
                particle,
                boundary_id=boundary_id,
            )

            if segment is None:
                # No matching segment - use default reflecting
                result_particles[idx] = self._reflect_particle(particle, domain_min, domain_max, domain_size)
                continue

            # alpha/beta come off the matched segment, so a mixed BC with a different
            # Robin coefficient per wall is read per wall. The uniform path's
            # `_robin_alpha_beta` cannot do that -- it takes the first Robin segment in
            # the whole specification, whichever face is being processed.
            action = particle_action_for_bc_type(segment.bc_type, segment.alpha, segment.beta)

            if action == "absorbing":
                absorbed_mask[idx] = True
                exit_positions_list.append(particle.copy())
            elif action == "reflecting":
                result_particles[idx] = self._reflect_particle(particle, domain_min, domain_max, domain_size)
            else:
                result_particles[idx] = self._wrap_particle(particle, domain_min, domain_size)

        # Build exit positions array
        if exit_positions_list:
            exit_positions = np.array(exit_positions_list)
        else:
            exit_positions = np.array([]).reshape(0, dimension)

        # Filter out absorbed particles
        remaining_particles = result_particles[~absorbed_mask]

        return remaining_particles, absorbed_mask, exit_positions

    def apply_with_flux_limits(
        self,
        particles: NDArray[np.floating],
        bc: BoundaryConditions,
        bounds: list[tuple[float, float]],
        flux_limits: dict[str, float],
    ) -> tuple[NDArray[np.floating], NDArray[np.bool_], NDArray[np.floating], dict[str, int]]:
        """
        Apply segment-aware BC with flux-limited absorption at DIRICHLET exits.

        When exit capacity is exceeded, particles are REFLECTED (queue) instead of
        absorbed. This creates physical congestion at exits.

        Args:
            particles: Particle positions, shape (N, d)
            bc: BoundaryConditions with segment definitions
            bounds: Domain bounds per dimension [(xmin, xmax), ...]
            flux_limits: Dict mapping segment name to max particles absorbed this step
                         e.g., {"exit_A": 10, "exit_B": 15}

        Returns:
            Tuple of:
            - remaining_particles: Particles not absorbed, shape (M, d)
            - absorbed_mask: Boolean mask of absorbed particles, shape (N,)
            - exit_positions: Positions where absorbed, shape (K, d)
            - absorbed_per_segment: Dict mapping segment name to absorbed count
        """

        particles = np.atleast_2d(particles)
        n_particles = len(particles)
        dimension = particles.shape[1]

        if n_particles == 0:
            return particles, np.array([], dtype=bool), np.array([]).reshape(0, dimension), {}

        bounds_arr = np.array(bounds)
        domain_min = bounds_arr[:, 0]
        domain_max = bounds_arr[:, 1]
        domain_size = domain_max - domain_min

        if bc.domain_bounds is None:
            bc.domain_bounds = bounds_arr

        # Find boundary particles
        at_min = particles <= domain_min + self._boundary_tolerance
        at_max = particles >= domain_max - self._boundary_tolerance
        at_boundary = np.any(at_min | at_max, axis=1)

        absorbed_mask = np.zeros(n_particles, dtype=bool)
        exit_positions_list = []
        result_particles = particles.copy()

        # Track absorption per segment for flux limiting
        absorbed_count: dict[str, int] = dict.fromkeys(flux_limits, 0)

        # Process boundary particles
        boundary_indices = np.where(at_boundary)[0]

        for idx in boundary_indices:
            particle = result_particles[idx]
            boundary_id = self._get_boundary_id(particle, domain_min, domain_max, dimension)

            segment = bc.get_bc_at_point(particle, boundary_id=boundary_id)

            if segment is None:
                result_particles[idx] = self._reflect_particle(particle, domain_min, domain_max, domain_size)
                continue

            action = particle_action_for_bc_type(segment.bc_type, segment.alpha, segment.beta)

            if action == "absorbing":
                seg_name = segment.name

                # Check flux capacity
                if seg_name in flux_limits:
                    capacity = flux_limits[seg_name]
                    current = absorbed_count.get(seg_name, 0)

                    if current < capacity:
                        # Capacity available - absorb
                        absorbed_mask[idx] = True
                        exit_positions_list.append(particle.copy())
                        absorbed_count[seg_name] = current + 1
                    else:
                        # Capacity full - REFLECT (queue at exit)
                        result_particles[idx] = self._reflect_particle(particle, domain_min, domain_max, domain_size)
                else:
                    # No flux limit for this segment - absorb immediately
                    absorbed_mask[idx] = True
                    exit_positions_list.append(particle.copy())

            elif action == "reflecting":
                result_particles[idx] = self._reflect_particle(particle, domain_min, domain_max, domain_size)

            else:
                result_particles[idx] = self._wrap_particle(particle, domain_min, domain_size)

        if exit_positions_list:
            exit_positions = np.array(exit_positions_list)
        else:
            exit_positions = np.array([]).reshape(0, dimension)

        remaining_particles = result_particles[~absorbed_mask]

        return remaining_particles, absorbed_mask, exit_positions, absorbed_count

    def _get_boundary_id(
        self,
        particle: NDArray[np.floating],
        domain_min: NDArray[np.floating],
        domain_max: NDArray[np.floating],
        dimension: int,
    ) -> str | None:
        """
        Determine which boundary a particle is at.

        Returns boundary ID using BCSegment convention:
        - 1D: "left", "right" (directional for 1D)
        - 2D: "x_min", "x_max", "y_min", "y_max" (axis-based for 2D+)
        - 3D: "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"
        - nD: "dim0_min", "dim0_max", etc. for d >= 3

        Note: Different conventions for 1D vs 2D+ to match common usage patterns.
        """
        # Handle 1D special case (uses directional naming)
        if dimension == 1:
            if particle[0] <= domain_min[0] + self._boundary_tolerance:
                return "left"
            if particle[0] >= domain_max[0] - self._boundary_tolerance:
                return "right"
            return None

        # 2D+ uses axis-based naming (x_min, y_max, etc.)
        axis_names = ["x", "y", "z"] + [f"dim{i}" for i in range(3, dimension)]

        for d in range(dimension):
            axis_name = axis_names[d] if d < len(axis_names) else f"dim{d}"
            if particle[d] <= domain_min[d] + self._boundary_tolerance:
                return f"{axis_name}_min"
            if particle[d] >= domain_max[d] - self._boundary_tolerance:
                return f"{axis_name}_max"

        return None

    def _reflect_particle(
        self,
        particle: NDArray[np.floating],
        domain_min: NDArray[np.floating],
        domain_max: NDArray[np.floating],
        domain_size: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Reflect particle back into domain using modular fold reflection.

        Handles particles that may have traveled multiple domain widths.
        Uses canonical implementation from geometry.boundary.corner (Issue #521).

        At corners, all dimensions are processed simultaneously, producing
        diagonal reflection (equivalent to 'average' corner strategy).
        """
        from .corner import reflect_positions

        # Build bounds from domain_min and domain_max
        bounds = list(zip(domain_min, domain_max, strict=True))
        return reflect_positions(particle, bounds)

    def _wrap_particle(
        self,
        particle: NDArray[np.floating],
        domain_min: NDArray[np.floating],
        domain_size: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Wrap particle to opposite boundary (periodic BC).

        Uses canonical implementation from geometry.boundary.corner (Issue #521).
        """
        from .corner import wrap_positions

        # Build bounds from domain_min and domain_size
        bounds = [(domain_min[d], domain_min[d] + domain_size[d]) for d in range(len(domain_min))]
        return wrap_positions(particle, bounds)


__all__ = ["ParticleApplicator"]

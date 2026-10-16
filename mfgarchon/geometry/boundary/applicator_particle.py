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
    particle_action_at is the one owner of what a particle does at a boundary point: the segment
    BoundaryConditions.segment_for_point gives there, at a corner by corner_strategy and elsewhere
    unmatched by default_bc, read through particle_action_for_bc_type
    ParticleApplicator executes the action

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
    from .types import BCSegment

ParticleAction = Literal["reflecting", "absorbing", "periodic"]


def particle_action_for_bc_type(
    bc_type: BCType,
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
    - `ROBIN` dispatches on `beta`, and on `beta` alone. With `beta = 0` the condition
      reads `alpha*u = g`, which is Dirichlet, so the particle is absorbed. Every other
      `beta` reflects.

      **This is two cases, not three, and the vocabulary is why.** A genuinely mixed
      Robin -- `alpha` and `beta` both nonzero -- is a *partially absorbing* wall: the
      particle counterpart of `alpha*m + beta*d_n m = g` is a diffusion that reflects with
      a probability set by `alpha/beta` (Feller's elastic boundary). `ParticleAction` has
      three members and none of them is that, so this function returns `"reflecting"`,
      which conserves mass and is the conservative reading of a wall it cannot express.
      ~~"With `alpha = 0` it reads `beta*du/dn = g` ... A genuinely mixed Robin reflects"~~
      was written as though those were separate branches [CORRECTED 2026-08-16, #1960]:
      they share one, so the `alpha = 0` case was undiscriminated and `test_robin_
      dispatches_on_its_coefficients` had two rows passing through the same `return`.

      `alpha` is therefore **not a parameter of this function**. It was required and never
      read. It comes back the day `ParticleAction` grows a partially-absorbing member,
      because that member's probability is exactly what `alpha/beta` sets.
    - `EXTRAPOLATION_LINEAR` / `EXTRAPOLATION_QUADRATIC` raise. They are not boundary
      conditions on a particle at all -- they are a statement about how to continue a
      *field* past a truncated domain, and carry no boundary datum. Silently reflecting
      them, which is what the segment-aware path's ``else`` branch did, answers a
      question that was never asked.

    Args:
        bc_type: The condition to interpret.
        beta: Robin coefficient on ``du/dn``. Required when ``bc_type`` is ``ROBIN``, and
            the only coefficient this mapping reads -- see the ROBIN note above.

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
        if beta is None:
            raise ValueError(
                "particle_action_for_bc_type: BCType.ROBIN needs beta; it lives on the "
                "BCSegment, not on BoundaryConditions. A default would be the #1558 failure "
                "a third time -- beta=0 is Dirichlet, so an unspecified Robin wall would "
                "silently become absorbing."
            )
        if np.isclose(beta, 0.0):
            return "absorbing"
        return "reflecting"

    raise ValueError(
        f"particle_action_for_bc_type: {bc_type} has no particle interpretation. "
        "EXTRAPOLATION_LINEAR and EXTRAPOLATION_QUADRATIC describe how to continue a "
        "field past a truncated domain and say nothing about a particle that reaches it."
    )


def segment_particle_action(segment: BCSegment) -> ParticleAction:
    """What a particle does at a wall that ``segment`` governs: `particle_action_for_bc_type` on its type and, for a
    Robin segment, its ``beta`` -- a classification by the sign of beta, which reads no number."""
    return particle_action_for_bc_type(segment.bc_type, segment.beta)


def particle_action_at(
    bc: BoundaryConditions,
    point: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
    geometry: object = None,
) -> tuple[ParticleAction, BCSegment]:
    """What a particle at or beyond the box ``bounds`` does there, and the segment that says so (#2490 B1d): the one
    owner of that decision for the particle family.

    ``point`` is clamped onto the box, and the segment is `BoundaryConditions.segment_for_point`'s at the clamp: at a
    corner, ``corner_strategy`` chooses among the faces' segments in the owner's corner order, and where no segment
    governs, ``default_bc`` does, refused by name where it is unset or a bare ROBIN.

    The clamp stands in for the point where the particle's path crossed the wall, which the per-axis applicator does
    not have. A particle beyond two faces therefore reads their corner, even where its path crossed only one of them.
    That is the rule until #2733 passes the exact contact point here.
    """
    box = np.asarray(bounds, dtype=float)
    clamped = np.clip(np.asarray(point, dtype=float), box[:, 0], box[:, 1])
    segment = bc.segment_for_point(clamped, geometry=geometry)
    return segment_particle_action(segment), segment


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

        For each particle that has crossed or is at the boundary, `particle_action_at` decides the action -- absorb,
        reflect or wrap -- and this applies it.

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

            action, _ = particle_action_at(bc, particle, bounds_arr)

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
            action, segment = particle_action_at(bc, particle, bounds_arr)

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

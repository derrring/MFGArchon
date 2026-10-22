"""
Dimension-agnostic boundary condition handling for particle-based FP solvers (Issue #635).

This module provides unified functions that work for any dimension:
- Topology detection from boundary conditions
- Boundary enforcement (periodic wrap, reflecting bounce)
- The contact loop: a step's first contacts with the particles' boundary, box and obstacles together (#2733)
- Segment-aware BC checking

All functions accept both 1D and nD inputs with consistent APIs.

Usage:
    from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import (
        get_topology_per_dimension,
        apply_boundary_conditions,
        reflect_at_first_contacts,
        needs_segment_aware_bc,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.geometry.boundary.applicator_particle import CONTACT_EPS
from mfgarchon.geometry.boundary.conditions import BoundaryConditions, periodic_faces
from mfgarchon.geometry.boundary.types import BCType

if TYPE_CHECKING:
    from mfgarchon.geometry.boundary.applicator_particle import ParticleAction
    from mfgarchon.geometry.boundary.types import BCSegment
    from mfgarchon.geometry.implicit import ImplicitDomain

# =============================================================================
# Topology Detection (unified)
# =============================================================================


def get_topology_per_dimension(
    boundary_conditions: BoundaryConditions | str | None,
    dimension: int,
    geometry: Any = None,
) -> list[str]:
    """
    Get grid topology for each dimension from boundary conditions (dimension-agnostic).

    This determines the INDEXING STRATEGY for particles, not the physical BC:
    - "periodic": Space wraps around (particles use modular arithmetic)
    - "bounded": Space has walls (particles reflect at boundaries)

    Note: This is about topology (how space connects), not physics (what values
    are prescribed). For particles, all non-periodic boundaries are treated as
    reflecting walls, regardless of whether the underlying BC is Dirichlet,
    Neumann, Robin, or no-flux.

    Parameters
    ----------
    boundary_conditions : BoundaryConditions or None
        Boundary condition specification. A torus's is ``periodic_bc``, which the problem's BC owner
        answers for a geometry periodic on every axis (#2530, row B9).
    dimension : int
        Number of spatial dimensions
    geometry : the solver's grid, on which the face owner decides a region or sdf_region placement

    Returns
    -------
    topologies : list[str]
        Topology per dimension: ["periodic", "bounded", ...]

    Examples
    --------
    >>> from mfgarchon.geometry.boundary import periodic_bc, neumann_bc
    >>> get_topology_per_dimension(periodic_bc(1), 1)  # ["periodic"]
    >>> get_topology_per_dimension(neumann_bc(2), 2)   # ["bounded", "bounded"]
    """
    # Default to bounded (reflecting walls) for all dimensions
    topologies = ["bounded"] * dimension

    if boundary_conditions is None:
        return topologies

    bc = boundary_conditions
    if not isinstance(bc, BoundaryConditions):
        raise TypeError(
            "get_topology_per_dimension takes a BoundaryConditions or None, not "
            f"{type(bc).__name__} ({bc!r}): read as bounded walls, it would solve a boundary nobody declared "
            "(#2634, review 2)."
        )

    # One condition everywhere: read with no placement, so no geometry is needed.
    uniform = bc.uniform_segment
    if uniform is not None:
        return ["periodic" if uniform.bc_type == BCType.PERIODIC else "bounded"] * dimension

    # Read through the face owner with periodicity as the equivalence (`periodic_faces`, #2512 row B1): the topology
    # asks only whether an axis wraps, so a segment covering part of a face is accepted where it agrees with the rest of
    # the face on that, and refused where it does not. Periodic topology needs both ends of an axis periodic.
    periodic = periodic_faces(bc, dimension, geometry)
    return ["periodic" if periodic[2 * d] and periodic[2 * d + 1] else "bounded" for d in range(dimension)]


def needs_segment_aware_bc(boundary_conditions: BoundaryConditions | None) -> bool:
    """
    Check if boundary conditions require segment-aware handling.

    The fast path applies one action on every wall, reflect or wrap. It is taken where
    `BoundaryConditions.uniform_segment` names one condition everywhere and a particle does not absorb under it, placed
    segments included; and, where the walls carry several conditions, where no segment is placed by more than a face
    label and every condition in play reflects a particle by `particle_action_for_bc_type`: each segment's, and
    ``default_bc``'s whenever it is set (#2490).

    Returns True if:
    - one condition covers the boundary (`BoundaryConditions.uniform_segment`) and it absorbs
    - the conditions differ and a segment is placed by more than a face label (`BoundaryConditions.placed_segments`),
      even where every condition reflects
    - the conditions differ and one of them does not reflect: an exit, or a periodic condition, including PERIODIC
      declared face by face with no ``default_bc``
    - no global segment covers the boundary and ``default_bc`` is set to a condition that does not reflect, even where
      the face segments leave it no wall: two REFLECTING faces beside a DIRICHLET default route per point
    - the conditions differ and ``default_bc`` is ROBIN, which carries no coefficients (#2042)

    Returns False where `uniform_segment` names one reflecting or periodic condition (a global segment, or
    ``default_bc`` with no segment differing from it), or where no segment is placed and every condition in play
    reflects: a REFLECTING segment beside a NO_FLUX default reflects on both.

    Parameters
    ----------
    boundary_conditions : BoundaryConditions or None
        Boundary condition specification.

    Returns
    -------
    bool
        True if segment-aware handling is needed

    Examples
    --------
    >>> needs_segment_aware_bc(periodic_bc(1))  # False (every wall wraps)
    >>> needs_segment_aware_bc(mixed_bc_with_exit)  # True (an exit absorbs, the walls reflect)
    """
    if boundary_conditions is None:
        return False

    bc = boundary_conditions
    if isinstance(bc, str):
        raise ValueError(f"needs_segment_aware_bc: a BC is a BoundaryConditions, not the string {bc!r}")

    from mfgarchon.geometry.boundary.applicator_particle import particle_action_for_bc_type, segment_particle_action

    def action(segment: BCSegment) -> ParticleAction:
        return segment_particle_action(segment, "FPParticleSolver's routing")

    # One condition everywhere, default_bc counted wherever no global segment covers the boundary: the fast path wraps
    # or reflects it, and its topology read answers a uniform BC without reading placements.
    uniform = bc.uniform_segment
    if uniform is not None:
        return action(uniform) == "absorbing"
    # Several conditions: the fast path reflects every particle. A segment placed by more than a face label routes per
    # point, where the BC's owner reads the placement; the fast path reads whole faces (#2739 review 1).
    if bc.placed_segments():
        return True
    # default_bc counts whenever it is set, even where the face segments leave it no wall; per point the owner decides
    # which condition governs. Read by the segments alone, a reflecting segment beside default_bc=DIRICHLET took the
    # reflecting fast path, and the walls the default makes absorbing reflected every particle (#2490).
    if bc.default_bc == BCType.ROBIN:
        return True  # a bare BCType carries no beta to read an action from; per point the owner refuses it (#2042)
    actions = [action(segment) for segment in bc.segments]
    if bc.default_bc is not None:
        actions.append(particle_action_for_bc_type(bc.default_bc))
    return any(a != "reflecting" for a in actions)


# =============================================================================
# Boundary Enforcement (unified)
# =============================================================================


def apply_boundary_conditions(
    particles: np.ndarray,
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    topology: str | list[str] = "bounded",
) -> np.ndarray:
    """
    Apply boundary handling per dimension based on topology (dimension-agnostic).

    This function delegates to the canonical implementations in
    mfgarchon.utils.numerical.particle.boundary (Issue #521).

    At corners, all dimensions are processed simultaneously (not sequentially),
    producing diagonal reflection. This is equivalent to 'average' corner
    strategy for position-based reflection.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions.
        - 1D: shape (num_particles,) or (num_particles, 1)
        - nD: shape (num_particles, dimension)
    bounds : list of tuple or tuple of tuple
        Bounds per dimension [(xmin, xmax), (ymin, ymax), ...]
    topology : str or list[str]
        Grid topology: "periodic" (wrap) or "bounded" (reflect).
        Can be a single string (same for all dims) or per-dimension list.

    Returns
    -------
    particles : np.ndarray
        Updated particle positions (may be modified in place for efficiency)

    Examples
    --------
    >>> particles_1d = np.array([1.5, -0.2, 0.5])  # Some outside [0, 1]
    >>> apply_boundary_conditions(particles_1d, [(0, 1)], "bounded")
    >>> # particles_1d now reflected into [0, 1]

    Note
    ----
    See Issue #521 for corner handling architecture. Position-based reflection
    uses fold reflection which implicitly produces diagonal corner reflection.
    """
    from mfgarchon.geometry.boundary.corner import reflect_positions, wrap_positions

    particles = np.asarray(particles)
    bounds_list = list(bounds)

    # Handle 1D case: convert to 2D for uniform processing
    is_1d_flat = particles.ndim == 1
    if is_1d_flat:
        particles = particles[:, np.newaxis]

    dimension = particles.shape[1] if particles.ndim > 1 else 1

    # Handle per-dimension topologies
    if isinstance(topology, str):
        topologies = [topology] * dimension
    else:
        topologies = list(topology)

    # Check if all dimensions have same topology (common case)
    all_periodic = all(t == "periodic" for t in topologies)
    all_bounded = all(t != "periodic" for t in topologies)

    if all_periodic:
        # Fast path: all periodic
        result = wrap_positions(particles, bounds_list)
    elif all_bounded:
        # Fast path: all bounded (reflecting)
        result = reflect_positions(particles, bounds_list)
    else:
        # Mixed topology: process dimension by dimension
        result = particles.copy()
        for d in range(dimension):
            xmin, xmax = bounds_list[d]
            Lx = xmax - xmin

            if Lx < 1e-14:
                continue  # Skip degenerate dimension

            dim_topology = topologies[d] if d < len(topologies) else "bounded"

            if dim_topology == "periodic":
                # Periodic: wrap around
                result[:, d] = xmin + (result[:, d] - xmin) % Lx
            else:
                # Bounded: fold reflection
                shifted = result[:, d] - xmin
                period = 2 * Lx
                pos_in_period = shifted % period
                in_second_half = pos_in_period > Lx
                pos_in_period[in_second_half] = period - pos_in_period[in_second_half]
                result[:, d] = xmin + pos_in_period

    # Convert back to 1D if input was 1D
    if is_1d_flat:
        return result.ravel()

    return result


# =============================================================================
# The contact loop (#2733)
# =============================================================================

#: D2: two contact normals with 1 + n1.n2 at or below this are anti-parallel, the faces of a zero-width corridor.
ANTI_PARALLEL = 1e-12
#: D5: a step reflected more often than this returns to its start.
MAX_REFLECTIONS = 20
#: D6: an end that fails `contains` moves along the last contact's inward normal by the first of these, relative to
#: the box's diagonal, doubling up to the second.
OFFSET_LADDER = (1e-12, 1e-6)


@dataclass(frozen=True)
class SteppedParticles:
    """One step's end for every particle (#2733): where each lands, or, for an absorbed one, where it met the exit; and
    the step's counts, which ``FPParticleSolver.obstacle_placement_history`` records."""

    positions: np.ndarray
    absorbed: np.ndarray
    counts: dict[str, int]


def reflect_at_first_contacts(
    starts: np.ndarray,
    ends: np.ndarray,
    domain: ImplicitDomain,
    box: np.ndarray,
    bc: BoundaryConditions,
    reader: str = "FPParticleSolver",
) -> SteppedParticles:
    """End a particle step at its contacts with the particles' boundary, in path order (#2733 D1-D6).

    ``starts`` are the positions before the step, inside ``domain`` by induction, and ``ends`` the raw Euler-Maruyama
    ends; ``box`` is the box the step is taken in, of shape (d, 2). One search, `ImplicitDomain.segment_intervals`,
    finds the first point where the path leaves ``domain``, over the obstacles and the box walls together.

    - **The action at a contact.** A contact on the box's boundary reads the boundary owner there
      (`segment_at_box_contact`); every other contact reflects (D2's box rule). An absorbing contact ends the particle
      there. A reflecting one reflects the rest of the step specularly in the outward normal at the contact (D3) and
      re-traces it.
    - **Ties (D4).** A contact within ``CONTACT_EPS`` of the previous one along the path is a tie. Two reflecting
      faces of a convex corner reflect in turn. Two whose normals are anti-parallel bound a zero-width corridor: the
      rest of the step loses its normal component, once, and is re-traced along the corridor (D2). Past the
      corridor's end that is a convention, not the gap's limit: the particle stays on the corridor's line. Where the
      step has no tangential part, a 1-D corridor or an isolated point of the domain, the rule has no answer and the
      step is refused by name.
    - **The cap (D5).** A step reflected more than ``MAX_REFLECTIONS`` times returns to its start, and is counted.
    - **The offset (D6).** A landing that fails ``domain.contains`` after a contact moves along that contact's inward
      normal, by the ladder ``OFFSET_LADDER``; one that still fails returns to its start, and is counted.
    - **The skip.** A particle whose segment provably reaches no surface is not searched: on a convex domain one whose
      end is inside, elsewhere one with ``|b - a| < |phi(a)| - eps``. ``|phi|`` is a lower bound on the distance to
      the boundary for every domain built from `Hyperrectangle` and `Hypersphere` by the CSG nodes.
    """
    from mfgarchon.geometry.boundary.applicator_particle import (
        on_the_box,
        segment_at_box_contact,
        segment_particle_action,
    )

    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float).reshape(starts.shape)
    n, dimension = starts.shape
    box = np.asarray(box, dtype=float).reshape(dimension, 2)
    eps = CONTACT_EPS * float(np.linalg.norm(box[:, 1] - box[:, 0]))
    counts = dict.fromkeys(("searched", "contacts", "reflections", "corridors", "offsets", "fallbacks", "absorbed"), 0)
    positions = ends.copy()
    absorbed = np.zeros(n, dtype=bool)
    if _convex(domain):
        # A start is inside by induction, so a segment whose end is inside a convex domain meets no surface. A searched
        # leg's start is checked where it is searched (`_first_exits`).
        rows = np.flatnonzero(~np.atleast_1d(np.asarray(domain.contains(ends), dtype=bool)))
    else:
        phi = np.atleast_1d(np.asarray(domain.signed_distance(starts), dtype=float))
        outside = phi > eps
        if outside.any():
            raise RuntimeError(
                f"{reader}: internal error -- a particle step starts outside the particles' domain, at "
                f"{starts[outside][0].tolist()} (phi {phi[outside][0]:.3e}). Every recorded particle is inside, so "
                "every step starts inside (#2733)."
            )
        length = np.linalg.norm(ends - starts, axis=1)
        rows = np.flatnonzero((length > 0) & ~_reaches_no_surface(domain, ends, phi, length, eps))
    counts["searched"] = int(rows.size)
    if rows.size == 0:
        return SteppedParticles(positions, absorbed, counts)

    owner = on_the_box(bc, box)
    uniform = owner.uniform_segment
    uniform_action = None if uniform is None else segment_particle_action(uniform, reader)
    # Each leg runs from ``position`` to ``end``. A reflection moves the end, e' = e - 2 ((e - p) . n) n, rather than
    # the remaining vector, so a coordinate the normal does not touch is carried exactly and a box face's coordinate
    # rounds once, as the fold's 2 h - e does: on the box the loop is the fold, bitwise.
    position, end = starts[rows].copy(), ends[rows].copy()
    total = np.linalg.norm(end - position, axis=1)
    previous_normal = np.full(
        (rows.size, dimension), np.nan
    )  # the last contact's normal, unless a corridor followed it
    last_normal = np.full((rows.size, dimension), np.nan)  # the last contact's normal, for D6
    reflections = np.zeros(rows.size, dtype=np.int64)
    live = np.ones(rows.size, dtype=bool)
    while live.any():
        legs = np.flatnonzero(live)
        a, b = position[legs], end[legs]
        leg_length = np.linalg.norm(b - a, axis=1)
        idle = leg_length == 0.0
        positions[rows[legs[idle]]] = b[idle]
        live[legs[idle]] = False
        legs, a, b, leg_length = legs[~idle], a[~idle], b[~idle], leg_length[~idle]
        if legs.size == 0:
            break
        inside, t, point, normal = _first_exits(domain, a, b, eps / leg_length)
        if not inside.all():
            raise RuntimeError(
                f"{reader}: internal error -- a particle step starts outside the particles' domain, at "
                f"{a[~inside][0].tolist()}. Every recorded particle is inside, so every step starts inside (#2733)."
            )
        clear = np.isnan(t)
        positions[rows[legs[clear]]] = b[clear]
        live[legs[clear]] = False
        hit = np.flatnonzero(~clear)
        i, p, nk, e = legs[hit], point[hit], normal[hit], b[hit]
        counts["contacts"] += int(hit.size)
        if uniform_action is not None:
            # One condition everywhere: with a particle domain it reflects, an absorbing one being refused at
            # construction off the box (`FPParticleSolver._refuse_absorbing_off_the_box`); without one every contact is
            # on the box. Either way it is the action at every contact.
            action = np.full(hit.size, uniform_action, dtype=object)
        else:
            action = np.empty(hit.size, dtype=object)
            for j in range(hit.size):
                _, segment = segment_at_box_contact(owner, p[j], eps)
                action[j] = "reflecting" if segment is None else segment_particle_action(segment, reader)
        unknown = (action != "absorbing") & (action != "reflecting")
        if unknown.any():
            raise RuntimeError(
                f"{reader}: internal error -- a {action[unknown][0]} contact at {p[unknown][0].tolist()} reached the "
                "contact loop; a BC that declares a periodic face takes the wrap, never this loop (#2733, #2704)."
            )
        absorb = action == "absorbing"
        positions[rows[i[absorb]]] = p[absorb]
        absorbed[rows[i[absorb]]] = True
        counts["absorbed"] += int(absorb.sum())
        live[i[absorb]] = False
        reflect = ~absorb
        # D4 and D2: a contact at the leg's start, against an anti-parallel last normal, is a corridor's tie.
        tie = (
            reflect
            & np.isfinite(previous_normal[i]).all(axis=1)
            & (t[hit] * leg_length[hit] <= eps)
            & (1.0 + np.sum(nk * previous_normal[i], axis=1) <= ANTI_PARALLEL)
        )
        if tie.any():
            tied = np.flatnonzero(tie)
            corridor_end = e[tied] - np.sum((e[tied] - p[tied]) * nk[tied], axis=1)[:, None] * nk[tied]
            trapped = np.linalg.norm(corridor_end - p[tied], axis=1) <= CONTACT_EPS * total[i[tied]]
            if trapped.any():
                j = tied[np.flatnonzero(trapped)[0]]
                raise ValueError(
                    f"{reader}: a particle at {p[j].tolist()} lies between two faces of the domain whose outward "
                    f"normals {previous_normal[i[j]].tolist()} and {nk[j].tolist()} are opposite, with no direction "
                    "along them, so its step has no part the two faces leave it (#2733). That is an isolated point of "
                    "the domain, and particles were most likely started on it: in 1-D a grid node on a face of the box "
                    "flush with an obstacle, such as x = 1 in DifferenceDomain([0, 1], [0.8, 1]), or the crack between "
                    "two touching obstacles. Give no initial mass to that node, or place initial_particles off it."
                )
            counts["corridors"] += int(tied.size)
            position[i[tied]], end[i[tied]] = p[tied], corridor_end
            previous_normal[i[tied]] = np.nan
            last_normal[i[tied]] = nk[tied]
        bounce = np.flatnonzero(reflect & ~tie)
        reflections[i[bounce]] += 1
        capped = reflections[i[bounce]] > MAX_REFLECTIONS
        positions[rows[i[bounce[capped]]]] = starts[rows[i[bounce[capped]]]]
        counts["fallbacks"] += int(capped.sum())
        live[i[bounce[capped]]] = False
        bounce = bounce[~capped]
        counts["reflections"] += int(bounce.size)
        position[i[bounce]] = p[bounce]
        end[i[bounce]] = e[bounce] - 2.0 * np.sum((e[bounce] - p[bounce]) * nk[bounce], axis=1)[:, None] * nk[bounce]
        previous_normal[i[bounce]] = nk[bounce]
        last_normal[i[bounce]] = nk[bounce]

    _offset_landings(domain, starts, positions, absorbed, rows, last_normal, box, counts, reader)
    return SteppedParticles(positions, absorbed, counts)


def _reaches_no_surface(
    domain: ImplicitDomain, b: np.ndarray, phi: np.ndarray, length: np.ndarray, eps: float
) -> np.ndarray:
    """Which segments a -> b provably reach no surface of a domain that is not convex (#2733), given ``phi`` = phi(a),
    a start inside: one shorter than ``|phi(a)| - eps``, since ``|phi|`` bounds the distance to the boundary from below
    for the library's leaves and CSG nodes. Never a sample: a skip by sampling or by a distance heuristic is #2707's
    tunnelling again. (A convex domain's skip, an end inside, is read in `reflect_at_first_contacts`.)"""
    return length < np.abs(phi) - eps


def _convex(domain: ImplicitDomain) -> bool:
    """Whether ``domain`` is convex by construction: a `Hyperrectangle`, a `Hypersphere`, or an intersection of convex
    domains. A difference is an intersection with a complement, which is not convex."""
    from mfgarchon.geometry.implicit import Hyperrectangle, Hypersphere, IntersectionDomain

    if isinstance(domain, (Hyperrectangle, Hypersphere)):
        return True
    if type(domain) is IntersectionDomain:
        return all(_convex(child) for child in domain.domains)
    return False


def _first_exits(
    domain: ImplicitDomain, a: np.ndarray, b: np.ndarray, eps_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each leg's first exit from ``domain``, read from `ImplicitDomain.segment_intervals` as `ImplicitDomain.first_contact`
    reads it, with one difference: a start within ``eps_t`` of its first interval is inside it. A leg starts where the
    last one met a surface, which rounding puts on either side of it. Returns whether each start is inside, the exit's
    t (NaN where the leg leaves nowhere before its end), the point, and the domain's outward unit normal there. A box
    leaf's contact is put on its face exactly, the leaf's own form."""
    from mfgarchon.geometry.implicit import Hyperrectangle

    found = domain.segment_intervals(a, b)
    lo, hi = found.t[:, 0, 0], found.t[:, 0, 1]
    inside = lo <= eps_t
    exits = inside & (hi < 1.0)
    rows = np.arange(len(a))
    t = np.where(exits, hi, np.nan)
    leaf = np.where(exits, found.leaf[rows, 0, 1], -1)
    face = np.where(exits, found.leaf_face[rows, 0, 1], -1)
    point = a + np.where(exits, hi, 0.0)[:, None] * (b - a)
    normal = np.full_like(a, np.nan)
    for index in np.unique(leaf[exits]):
        at = leaf == index
        surface = found.leaves[index]
        if isinstance(surface, Hyperrectangle):
            axis, side = face[at] // 2, face[at] % 2
            point[np.flatnonzero(at), axis] = np.asarray(surface.bounds, dtype=float)[axis, side]
        normal[at] = surface._face_normal(point[at], face[at])
    normal[exits & found.complemented[rows, 0, 1]] *= -1.0
    return inside, t, point, normal


def _offset_landings(
    domain: ImplicitDomain,
    starts: np.ndarray,
    positions: np.ndarray,
    absorbed: np.ndarray,
    rows: np.ndarray,
    last_normal: np.ndarray,
    box: np.ndarray,
    counts: dict[str, int],
    reader: str,
) -> None:
    """D6: a searched landing that fails ``domain.contains`` after a contact moves along that contact's inward normal,
    from ``OFFSET_LADDER[0]`` to ``OFFSET_LADDER[1]`` times the box's diagonal, doubling, until it is inside; past that,
    or with no contact to read a normal from, it returns to its start. A landing already inside is left where it is."""
    landed = rows[~absorbed[rows]]
    if landed.size == 0:
        return
    out = ~np.atleast_1d(np.asarray(domain.contains(positions[landed]), dtype=bool))
    if not out.any():
        return
    diagonal = float(np.linalg.norm(box[:, 1] - box[:, 0]))
    where = {row: k for k, row in enumerate(rows)}
    for row in landed[out]:
        inward = -last_normal[where[row]]
        placed = False
        if np.isfinite(inward).all():
            size = OFFSET_LADDER[0] * diagonal
            while size <= OFFSET_LADDER[1] * diagonal:
                moved = positions[row] + size * inward
                if bool(np.atleast_1d(domain.contains(moved[None, :]))[0]):
                    positions[row] = moved
                    counts["offsets"] += 1
                    placed = True
                    break
                size *= 2.0
        if not placed:
            positions[row] = starts[row]
            counts["fallbacks"] += 1


# =============================================================================
# Coordinate/Grid Utilities (unified)
# =============================================================================


def create_coordinate_arrays(
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    grid_shape: tuple[int, ...],
) -> list[np.ndarray]:
    """
    Create coordinate arrays from bounds and grid shape (dimension-agnostic).

    Parameters
    ----------
    bounds : list or tuple of (min, max) tuples
        Domain bounds per dimension
    grid_shape : tuple[int, ...]
        Number of grid points per dimension

    Returns
    -------
    coordinates : list[np.ndarray]
        1D coordinate arrays for each dimension

    Examples
    --------
    >>> coords = create_coordinate_arrays([(0, 1), (0, 2)], (11, 21))
    >>> coords[0]  # array([0.0, 0.1, ..., 1.0])
    >>> coords[1]  # array([0.0, 0.1, ..., 2.0])
    """
    coordinates = []
    for d, (bmin, bmax) in enumerate(bounds):
        n_points = grid_shape[d]
        coords = np.linspace(bmin, bmax, n_points)
        coordinates.append(coords)
    return coordinates


def compute_spacings(
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    grid_shape: tuple[int, ...],
) -> list[float]:
    """
    Compute grid spacings from bounds and shape (dimension-agnostic).

    Parameters
    ----------
    bounds : list or tuple of (min, max) tuples
        Domain bounds per dimension
    grid_shape : tuple[int, ...]
        Number of grid points per dimension

    Returns
    -------
    spacings : list[float]
        Grid spacing per dimension [dx, dy, ...]

    Examples
    --------
    >>> compute_spacings([(0, 1), (0, 2)], (11, 21))
    [0.1, 0.1]
    """
    spacings = []
    for d, (bmin, bmax) in enumerate(bounds):
        n_points = grid_shape[d]
        if n_points > 1:
            dx = (bmax - bmin) / (n_points - 1)
        else:
            dx = bmax - bmin  # Single point
        spacings.append(dx)
    return spacings


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing unified fp_particle_bc functions...")

    # Test 1: Topology detection
    # We can't easily test with real BoundaryConditions without imports
    # So test the None case and basic logic
    topo_none = get_topology_per_dimension(None, 2)
    assert topo_none == ["bounded", "bounded"], f"Expected bounded, got {topo_none}"
    print("  get_topology_per_dimension (None BC): OK")

    # Test 2: needs_segment_aware_bc
    result = needs_segment_aware_bc(None)
    assert result is False, "None BC should not need segment-aware"
    print("  needs_segment_aware_bc (None BC): OK")

    # Test 3: Boundary conditions - bounded (reflecting)
    # Test 1D
    particles_1d = np.array([-0.1, 0.5, 1.1, 1.5, 2.5])
    result_1d = apply_boundary_conditions(particles_1d, [(0.0, 1.0)], "bounded")
    assert result_1d.shape == (5,), f"Expected (5,), got {result_1d.shape}"
    assert np.all(result_1d >= 0), f"Negative values: {result_1d}"
    assert np.all(result_1d <= 1), f"Values > 1: {result_1d}"
    print("  apply_boundary_conditions (1D bounded): OK")

    # Test 2D
    particles_2d = np.array(
        [
            [-0.1, 0.5],  # x out of bounds (left)
            [0.5, -0.2],  # y out of bounds (bottom)
            [1.1, 0.5],  # x out of bounds (right)
            [0.5, 1.3],  # y out of bounds (top)
            [0.5, 0.5],  # inside
        ]
    )
    result_2d = apply_boundary_conditions(particles_2d, [(0.0, 1.0), (0.0, 1.0)], "bounded")
    assert result_2d.shape == (5, 2), f"Expected (5, 2), got {result_2d.shape}"
    assert np.all(result_2d >= 0), f"Negative values: {result_2d}"
    assert np.all(result_2d <= 1), f"Values > 1: {result_2d}"
    print("  apply_boundary_conditions (2D bounded): OK")

    # Test 4: Boundary conditions - periodic
    particles_periodic = np.array([-0.1, 0.5, 1.1, 1.5])
    result_periodic = apply_boundary_conditions(particles_periodic, [(0.0, 1.0)], "periodic")
    assert np.all(result_periodic >= 0), f"Negative values: {result_periodic}"
    assert np.all(result_periodic <= 1), f"Values > 1: {result_periodic}"
    # -0.1 should wrap to 0.9, 1.1 should wrap to 0.1
    assert np.abs(result_periodic[0] - 0.9) < 1e-10, f"Expected 0.9, got {result_periodic[0]}"
    assert np.abs(result_periodic[2] - 0.1) < 1e-10, f"Expected 0.1, got {result_periodic[2]}"
    print("  apply_boundary_conditions (1D periodic): OK")

    # Test 5: Mixed topology (x periodic, y bounded)
    particles_mixed = np.array(
        [
            [-0.1, -0.1],  # Both out
            [1.1, 1.1],  # Both out
        ]
    )
    result_mixed = apply_boundary_conditions(particles_mixed, [(0.0, 1.0), (0.0, 1.0)], ["periodic", "bounded"])
    # x should wrap, y should reflect
    assert np.abs(result_mixed[0, 0] - 0.9) < 1e-10, f"x periodic: expected 0.9, got {result_mixed[0, 0]}"
    assert np.abs(result_mixed[0, 1] - 0.1) < 1e-10, f"y bounded: expected 0.1, got {result_mixed[0, 1]}"
    print("  apply_boundary_conditions (mixed periodic/bounded): OK")

    # Test 6: Coordinate array creation
    coords = create_coordinate_arrays([(0.0, 1.0), (0.0, 2.0)], (11, 21))
    assert len(coords) == 2
    assert len(coords[0]) == 11
    assert len(coords[1]) == 21
    assert np.abs(coords[0][-1] - 1.0) < 1e-10
    assert np.abs(coords[1][-1] - 2.0) < 1e-10
    print("  create_coordinate_arrays: OK")

    # Test 7: Spacing computation
    spacings = compute_spacings([(0.0, 1.0), (0.0, 2.0)], (11, 21))
    assert len(spacings) == 2
    assert np.abs(spacings[0] - 0.1) < 1e-10
    assert np.abs(spacings[1] - 0.1) < 1e-10
    print("  compute_spacings: OK")

    print("\nAll smoke tests passed!")

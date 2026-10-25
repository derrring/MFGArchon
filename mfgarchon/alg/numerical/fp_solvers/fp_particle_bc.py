"""
Dimension-agnostic boundary condition handling for particle-based FP solvers (Issue #635).

This module provides unified functions that work for any dimension:
- Topology detection from boundary conditions
- The contact loop: a step's first contacts with the particles' boundary, box and obstacles together (#2733), which
  ends every CPU step, a periodic outer boundary's included (#2704)
- Segment-aware BC checking

All functions accept both 1D and nD inputs with consistent APIs.

Usage:
    from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import (
        get_topology_per_dimension,
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
# The contact loop (#2733)
# =============================================================================

#: D2: two contact normals with 1 + n1.n2 at or below this are anti-parallel, the faces of a zero-width corridor.
ANTI_PARALLEL = 1e-12
#: D5: a step reflected more often than this returns to its start.
MAX_REFLECTIONS = 20
#: A step that wraps more often than this is refused (#2704): the loop searches once per period crossed, and a step of
#: about 1e17 periods never ends, since its shift by one period rounds away.
MAX_WRAPS = 1000
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
    ends; ``box`` is the box the step is taken in, of shape (d, 2). One search, `ImplicitDomain.first_contact`, finds
    the first point where the path leaves ``domain``, over the obstacles and the box walls together.

    - **The action at a contact.** A contact on the box's boundary reads the boundary owner there
      (`segment_at_box_contact`); every other contact reflects (D2's box rule). An absorbing contact ends the particle
      there. A reflecting one reflects the rest of the step specularly in the outward normal at the contact (D3) and
      re-traces it. A periodic one goes on with the rest of the step from its image on the opposite face (#2704), and
      a step that wraps more than ``MAX_WRAPS`` times is refused by name.
    - **A corner with a periodic face** (`periodic_corner_actions`): the owner's corner order decides only whether the
      particle is absorbed; otherwise each face acts on its own axis, a periodic one wrapping and any other reflecting.
    - **Ties (D4).** A contact within ``CONTACT_EPS`` of the previous one along the path is a tie. Two reflecting
      faces of a convex corner reflect in turn. Two whose normals are anti-parallel bound a zero-width corridor: the
      rest of the step loses its normal component, once, and is re-traced along the corridor (D2). Past the
      corridor's end that is a convention, not the gap's limit: the particle stays on the corridor's line. Where the
      step has no tangential part, a 1-D corridor or an isolated point of the domain, the rule has no answer and the
      step is refused by name.
    - **The cap (D5).** A step reflected more than ``MAX_REFLECTIONS`` times returns to its start, and is counted.
    - **The offset (D6).** A point the loop computed that fails ``domain.contains`` moves along the last contact's
      inward normal, by the ladder ``OFFSET_LADDER``: a landing, and a restart, the start of a leg re-traced from a
      contact that rounding put outside. One the ladder cannot place returns to its start, and is counted.
    - **The skip.** A particle whose segment provably reaches no surface is not searched: on a convex domain one whose
      end is inside, elsewhere one with ``|b - a| < |phi(a)| - eps``. ``|phi|`` is a lower bound on the distance to
      the boundary for every domain built from `Hyperrectangle` and `Hypersphere` by the CSG nodes.
    """
    from mfgarchon.geometry.boundary.applicator_particle import (
        on_the_box,
        periodic_corner_actions,
        segment_at_box_contact,
        segment_particle_action,
    )
    from mfgarchon.geometry.boundary.bc_utils import declares_periodic
    from mfgarchon.geometry.implicit import Hyperrectangle

    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float).reshape(starts.shape)
    n, dimension = starts.shape
    box = np.asarray(box, dtype=float).reshape(dimension, 2)
    eps = CONTACT_EPS * float(np.linalg.norm(box[:, 1] - box[:, 0]))
    counts = dict.fromkeys(
        ("searched", "contacts", "reflections", "wraps", "corridors", "offsets", "fallbacks", "absorbed"), 0
    )
    positions = ends.copy()
    absorbed = np.zeros(n, dtype=bool)
    if _convex(domain):
        # A start is inside by induction, so a segment whose end is inside a convex domain meets no surface. A searched
        # leg's start is checked where it is searched (`first_contact`'s ``inside_at_start``).
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
    any_periodic = declares_periodic(owner)
    # The particles' domain is the step's box itself, so every contact is on a face of it.
    box_only = type(domain) is Hyperrectangle and np.array_equal(np.asarray(domain.bounds, dtype=float), box)
    # Each leg runs from ``position`` to ``end``. A reflection moves the end, e' = e - 2 ((e - p) . n) n, rather than
    # the remaining vector, so a coordinate the normal does not touch is carried exactly and a box face's coordinate
    # is e - 2 (e - h): 2 h - e correctly rounded where e - h is exact, and within a few ulps of the larger of e and h
    # otherwise. The fold computes it as xmin + (2L - (x - xmin) % 2L), so on the box, short of the cap, the two agree
    # to rounding, and bitwise on the high faces of a box at the origin.
    position, end = starts[rows].copy(), ends[rows].copy()
    total = np.linalg.norm(end - position, axis=1)
    previous_normal = np.full(
        (rows.size, dimension), np.nan
    )  # the last contact's normal, unless a corridor followed it
    last_normal = np.full((rows.size, dimension), np.nan)  # the last contact's normal, for D6
    reflections = np.zeros(rows.size, dtype=np.int64)
    wrapped = np.zeros(rows.size, dtype=np.int64)
    restarted = np.zeros(rows.size, dtype=bool)  # moved inside by D6 since the last search that found it inside
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
        found = domain.first_contact(a, b)
        outside = ~found.inside_at_start
        if outside.any():
            # A restart, a leg from the last contact, that rounding put outside: D6 at the restart. A first leg's start is
            # a recorded particle's, inside by induction.
            first = outside & ~np.isfinite(last_normal[legs]).all(axis=1)
            if first.any():
                raise RuntimeError(
                    f"{reader}: internal error -- a particle step starts outside the particles' domain, at "
                    f"{a[first][0].tolist()}. Every recorded particle is inside, so every step starts inside (#2733)."
                )
            _offset_restarts(domain, legs[outside], position, last_normal, restarted, box, live, counts)
            failed = legs[outside][~live[legs[outside]]]
            positions[rows[failed]] = starts[rows[failed]]
            inside = ~outside
            legs, a, b, leg_length = legs[inside], a[inside], b[inside], leg_length[inside]
            t, point, normal = found.t[inside], found.point[inside], found.normal[inside]
        else:
            t, point, normal = found.t, found.point, found.normal
        restarted[legs] = False
        clear = np.isnan(t)
        positions[rows[legs[clear]]] = b[clear]
        live[legs[clear]] = False
        hit = np.flatnonzero(~clear)
        i, p, nk, e = legs[hit], point[hit], normal[hit], b[hit]
        counts["contacts"] += int(hit.size)
        corner_faces: dict[int, list[tuple[int, int, bool]]] = {}
        if uniform_action is not None and (uniform_action != "periodic" or box_only):
            # One condition everywhere, the action at every contact: reflecting, on the box or off it; absorbing, where
            # a particle domain is refused at construction (`FPParticleSolver._refuse_absorbing_off_the_box`), so
            # every contact is on the box; periodic only on the box itself, where a corner's faces wrap in turn. A
            # torus with a particle domain is read per contact below, so a contact on an obstacle reflects (D2).
            action = np.full(hit.size, uniform_action, dtype=object)
        else:
            action = np.empty(hit.size, dtype=object)
            for j in range(hit.size):
                corner = periodic_corner_actions(owner, p[j], eps, reader) if any_periodic else None
                if corner is None:
                    _, segment = segment_at_box_contact(owner, p[j], eps)
                    action[j] = "reflecting" if segment is None else segment_particle_action(segment, reader)
                elif corner[0]:
                    action[j] = "absorbing"
                else:
                    action[j], corner_faces[j] = "corner", corner[1]
        absorb = action == "absorbing"
        positions[rows[i[absorb]]] = p[absorb]
        absorbed[rows[i[absorb]]] = True
        counts["absorbed"] += int(absorb.sum())
        live[i[absorb]] = False
        # A periodic face (#2704): the rest of the step goes on from the contact's image on the opposite face, the path
        # in the periodic cover, so the faces it meets after the wrap act in path order. The start is set on that
        # face's bound and the end moved by the box's span along the axis.
        wrap = np.flatnonzero(action == "periodic")
        if wrap.size:
            axis = np.argmax(np.abs(nk[wrap]), axis=1)
            high = nk[wrap, axis] > 0
            span = box[axis, 1] - box[axis, 0]
            rows_w = np.arange(wrap.size)
            start, stop = p[wrap].copy(), e[wrap].copy()
            start[rows_w, axis] = np.where(high, box[axis, 0], box[axis, 1])
            stop[rows_w, axis] += np.where(high, -span, span)
            position[i[wrap]], end[i[wrap]] = start, stop
            previous_normal[i[wrap]] = np.nan
            last_normal[i[wrap]] = -nk[wrap]
            wrapped[i[wrap]] += 1
            counts["wraps"] += int(wrap.size)
        # The tie rule at a corner with a periodic face, not absorbed there: each face the rest of the step leaves
        # through acts on its own axis at once. A face it moves along or away from is not crossed, so it takes no
        # action: reflecting an axis already moving inward would send it out, and a step from a corner would bounce
        # between corners until the wrap cap.
        for j, faces in corner_faces.items():
            k = i[j]
            start, stop, outward = p[j].copy(), e[j].copy(), np.zeros(dimension)
            leg = e[j] - p[j]
            reflected = False
            for axis_j, side, wraps in faces:
                if not (leg[axis_j] > 0.0 if side == 1 else leg[axis_j] < 0.0):
                    continue
                if wraps:
                    start[axis_j] = box[axis_j, 1 - side]
                    stop[axis_j] += (box[axis_j, 1] - box[axis_j, 0]) * (-1.0 if side == 1 else 1.0)
                    outward[axis_j] = 1.0 if side == 0 else -1.0
                    wrapped[k] += 1
                    counts["wraps"] += 1
                else:
                    stop[axis_j] = 2.0 * box[axis_j, side] - stop[axis_j]
                    outward[axis_j] = 1.0 if side == 1 else -1.0
                    reflected = True
            if not outward.any():
                raise RuntimeError(
                    f"{reader}: internal error -- a contact at the corner {p[j].tolist()} whose step leaves through "
                    "none of its faces, so nothing would move it and the loop would not end (#2808)."
                )
            position[k], end[k] = start, stop
            previous_normal[k] = np.nan
            last_normal[k] = outward / np.linalg.norm(outward)
            if reflected:
                reflections[k] += 1
                counts["reflections"] += 1
        over = np.flatnonzero(wrapped > MAX_WRAPS)
        if over.size:
            raise ValueError(
                f"{reader}: a particle's step from {starts[rows[over[0]]].tolist()} to {ends[rows[over[0]]].tolist()} "
                f"wrapped more than {MAX_WRAPS} times through periodic faces (#2704). The loop searches once per "
                "period crossed, and a step of about 1e17 periods never ends, since its shift by one period rounds "
                "away. A step this long means the time step is too large for the drift and the noise: take more time "
                "steps."
            )
        unknown = (action != "absorbing") & (action != "reflecting") & (action != "periodic") & (action != "corner")
        if unknown.any():
            raise RuntimeError(
                f"{reader}: internal error -- a {action[unknown][0]} contact at {p[unknown][0].tolist()} reached the "
                "contact loop, which acts only by absorbing, reflecting or wrapping (#2733, #2704)."
            )
        reflect = action == "reflecting"
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


def _offset_restarts(
    domain: ImplicitDomain,
    legs: np.ndarray,
    position: np.ndarray,
    last_normal: np.ndarray,
    restarted: np.ndarray,
    box: np.ndarray,
    live: np.ndarray,
    counts: dict[str, int],
) -> None:
    """D6 at a restart: a leg from the last contact whose start rounding put outside the domain, as a contact with a
    curved surface does about half the time, moves along that contact's inward normal by `_inward_offsets` and is
    searched again. One moved once already, or that the ladder cannot place, ends its step (D5): the caller returns it
    to its start."""
    diagonal = float(np.linalg.norm(box[:, 1] - box[:, 0]))
    again, fresh = legs[restarted[legs]], legs[~restarted[legs]]
    moved, placed = _inward_offsets(domain, position[fresh], -last_normal[fresh], diagonal)
    position[fresh[placed]] = moved[placed]
    restarted[fresh[placed]] = True
    counts["offsets"] += int(placed.sum())
    failed = np.concatenate([again, fresh[~placed]])
    live[failed] = False
    counts["fallbacks"] += int(failed.size)


def _inward_offsets(
    domain: ImplicitDomain, points: np.ndarray, inward: np.ndarray, diagonal: float
) -> tuple[np.ndarray, np.ndarray]:
    """D6: each point moved along its ``inward`` direction by ``OFFSET_LADDER[0]`` to ``OFFSET_LADDER[1]`` times
    ``diagonal``, doubling, to the first place ``domain.contains``. Returns the moved points and which were placed; one
    past the ladder, or with no direction to move along, is not."""
    moved = np.array(points, dtype=float)
    placed = np.zeros(len(moved), dtype=bool)
    todo = np.flatnonzero(np.isfinite(inward).all(axis=1))
    size = OFFSET_LADDER[0] * diagonal
    while todo.size and size <= OFFSET_LADDER[1] * diagonal:
        candidate = moved[todo] + size * inward[todo]
        inside = np.atleast_1d(np.asarray(domain.contains(candidate), dtype=bool))
        moved[todo[inside]] = candidate[inside]
        placed[todo[inside]] = True
        todo = todo[~inside]
        size *= 2.0
    return moved, placed


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
    """D6 at a landing: a searched landing that fails ``domain.contains`` after a contact moves along that contact's
    inward normal by `_inward_offsets`; past the ladder, or with no contact to read a normal from, it returns to its
    start. A landing already inside is left where it is."""
    landed = rows[~absorbed[rows]]
    if landed.size == 0:
        return
    out = ~np.atleast_1d(np.asarray(domain.contains(positions[landed]), dtype=bool))
    if not out.any():
        return
    diagonal = float(np.linalg.norm(box[:, 1] - box[:, 0]))
    out_rows = landed[out]
    where = np.searchsorted(rows, out_rows)
    moved, placed = _inward_offsets(domain, positions[out_rows], -last_normal[where], diagonal)
    positions[out_rows] = np.where(placed[:, None], moved, starts[out_rows])
    counts["offsets"] += int(placed.sum())
    counts["fallbacks"] += int((~placed).sum())


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

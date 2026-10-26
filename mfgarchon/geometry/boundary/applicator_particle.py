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
    ParticleApplicator executes the action by where a step ended, through each rule's owner
    (CONVENTIONS.md § *Particles at a wall*: the end-position rule)

Created: 2025-01-03
Part of: Unified Boundary Condition Architecture (Issue #536)
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from mfgarchon.utils.deprecation import deprecated

from .corner.position import absorb_at_walls, reflect_positions, walls_reached
from .periodic import wrap_positions
from .tolerances import ONWALL_TOL, SDF_BOUNDARY_TOL
from .types import BCSegment, BCType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.geometry.implicit import ImplicitDomain

    from .conditions import BoundaryConditions

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


def segment_particle_action(segment: BCSegment, where: str) -> ParticleAction:
    """What a particle does at ``segment``, for a reader that routes or compares on it and writes no row: its type's
    action (`particle_action_for_bc_type`), a Robin segment's read on its ``beta`` as stored, the owner's form for
    such readers (`segment_robin_coefficients`). `needs_segment_aware_bc` routes on it and `same_particle_action`
    compares on it; ``where`` names the reader in a refusal."""
    from .types import segment_robin_coefficients

    beta = (
        segment_robin_coefficients(segment, where=where, as_stored=True)[1] if segment.bc_type == BCType.ROBIN else None
    )
    return particle_action_for_bc_type(segment.bc_type, beta)


def same_particle_action(first: BCSegment, second: BCSegment) -> bool:
    """The particle reader's sameness of two segments, for the boundary owner's whole-BC check (#2490 B1e-2): a particle
    cannot tell them apart. It reads the action (`segment_particle_action`) and never a segment's value. A segment carrying a ``flux_capacity`` is the same as no other: the flux-limited path
    (`ParticleApplicator.apply_with_flux_limits`) counts a capped exit's absorptions under its own name, so an outranked
    one would have its limit dropped."""
    if first.flux_capacity is not None or second.flux_capacity is not None:
        return False
    where = "the particle reader's whole-BC check"
    return segment_particle_action(first, where) == segment_particle_action(second, where)


def _with_box(bc: BoundaryConditions, bounds: NDArray[np.floating]) -> BoundaryConditions:
    """``bc`` as declared, the box `ParticleApplicator.refuse_segments_it_would_drop` explains a refusal on: itself when
    it carries ``domain_bounds``, where a region it places past the solver's box lies, else a copy that carries
    ``bounds``. Every particle query reads on the step's box instead (`on_the_box`, #2733). The copy keeps the BC's
    geometry binding, so a placement only the grid can decide, a marked ``region_name``, still reads the grid. Never a
    write to ``bc`` (#2738): it can be the problem's own BC."""
    if bc.domain_bounds is not None:
        return bc
    return replace(bc, segments=list(bc.segments), domain_bounds=bounds)


def on_the_box(bc: BoundaryConditions, box: NDArray[np.floating]) -> BoundaryConditions:
    """``bc`` read on ``box``, the box a particle step was taken in, whatever ``domain_bounds`` it carries: the copy the
    contact loop and its face check ask the owner through (`segment_at_box_contact`). A crossing is read in the box it
    was crossed in, so a BC whose own ``domain_bounds`` differ from the step's box is no longer read on two boxes, the
    face from one and the point from the other (#2733, the folded #2767 review 4 P1). Never a write to ``bc`` (#2738); the
    copy keeps its geometry binding, so a marked ``region_name`` still reads the grid."""
    return replace(bc, segments=list(bc.segments), domain_bounds=np.asarray(box, dtype=float))


def segment_at_box_contact(
    bc_on_box: BoundaryConditions, point: NDArray[np.floating], tolerance: float
) -> tuple[str | None, BCSegment | None]:
    """The face a contact at ``point`` is read on and the segment the owner governs it by, or (None, None) where the
    point lies on no face of the box ``bc_on_box`` is read on (`on_the_box`): there every contact reflects, #2733 D2's
    box rule. On one face it is that face, named by the owner (`BoundaryFace.to_string`), and its segment
    (`BoundaryConditions.face_segments_at_point`); at a corner or
    an edge no one face, and the owner resolves the faces by its corner rule (`BoundaryConditions.segment_for_point`,
    D4), so the loop never restates that rule. The contact loop and `ParticleApplicator.refuse_segments_it_would_drop`
    reach the owner only through this, so the check reads every face cell as the loop reads a contact there."""
    readings = bc_on_box.face_segments_at_point(point, tolerance=tolerance)
    if not readings:
        return None, None
    if len(readings) == 1:
        return readings[0][0].to_string(), readings[0][1]
    return None, bc_on_box.get_bc_at_point(point, tolerance=tolerance)


def periodic_corner_actions(
    bc_on_box: BoundaryConditions, point: NDArray[np.floating], tolerance: float, reader: str
) -> tuple[bool, list[tuple[int, int, bool]]] | None:
    """The tie rule at a corner of the box one of whose faces is periodic (CONVENTIONS.md § *Particles at a wall*):
    ``None`` where ``point`` is on fewer than two faces, or on no periodic one, and `segment_at_box_contact` decides.
    Otherwise ``(absorbed, [(axis, side, wraps), ...])``: the owner's corner order (`BoundaryConditions.segment_for_point`)
    decides only whether the particle is absorbed, the one bit a corner cannot split per axis, and every face at the
    corner comes with whether its segment wraps, ``side`` 0 for the min face. The faces are listed where it absorbs too,
    for a reader whose exit can turn a particle back (`ParticleApplicator.apply_with_flux_limits`). If it is not
    absorbed, the caller acts on the faces the step crosses there, or the end has reached: a periodic face wraps its
    axis, and any other returns the particle along it."""
    readings = bc_on_box.face_segments_at_point(point, tolerance=tolerance)
    if len(readings) < 2:
        return None
    wraps = [segment_particle_action(segment, reader) == "periodic" for _, segment, _ in readings]
    if not any(wraps):
        return None
    corner = bc_on_box.get_bc_at_point(point, tolerance=tolerance)
    return segment_particle_action(corner, reader) == "absorbing", [
        (face.axis, 0 if face.side == "min" else 1, wrap) for (face, _, _), wrap in zip(readings, wraps, strict=True)
    ]


#: #2733 D4's ε, relative to the box's diagonal: two contacts this close along a path are one tie, a leg's start this
#: close to its first interval is inside it, and a face's trace is read this far inside the box. One value for the
#: contact loop (`reflect_at_first_contacts`) and the B1e-2 check that reads where it can cross.
CONTACT_EPS = 1e-12

#: B1e-2 with a particles' domain (#2733 PR 2): half the shortest piece of the domain's trace on a box face, relative to
#: the box's diagonal, that counts as one particles can cross (`ParticleApplicator.refuse_segments_it_would_drop`).
CROSSABLE_HALF_WIDTH = 1e-4


def _absorbs_somewhere(bc: BoundaryConditions, reader: str) -> bool:
    """Whether a condition of ``bc`` absorbs a particle: a segment's, or ``default_bc``'s."""
    if any(segment_particle_action(segment, reader) == "absorbing" for segment in bc.segments):
        return True
    return (
        bc.default_bc is not None
        and bc.default_bc != BCType.ROBIN
        and particle_action_for_bc_type(bc.default_bc) == "absorbing"
    )


def _refuse_exits_no_particle_crosses(
    cells: list[tuple[int, int, NDArray[np.float64], NDArray[np.float64]]],
    readings: list[tuple[NDArray[np.float64], str | None, BCSegment | None]],
    domain: ImplicitDomain,
    box: NDArray[np.float64],
    diagonal: float,
    reader: str,
) -> None:
    """Refuse, by name, an absorbing segment that governs face cells of ``box`` and none a particle can cross (#2733 PR
    2): one where the particles' ``domain`` has no trace on the face within the cell (`ImplicitDomain.face_trace`, read
    ``CONTACT_EPS`` of the diagonal inside) at least ``2 * CROSSABLE_HALF_WIDTH`` of the diagonal long. Every contact
    near such a cell reflects, so the exit would be read nowhere: a ``boundary="x_max"`` exit on a disc in its bounding
    box, or an exit on the corridor an obstacle flush with a wall leaves. A segment that governs no cell at all is the
    owner's check's (`refuse_segments_governing_no_point`)."""
    shift, piece = CONTACT_EPS * diagonal, 2.0 * CROSSABLE_HALF_WIDTH * diagonal
    governed: dict[int, tuple[BCSegment, list[str], list[bool]]] = {}
    for (axis, side, cell, _), (_, face, segment) in zip(cells, readings, strict=True):
        if segment is None or segment_particle_action(segment, reader) != "absorbing":
            continue
        entry = governed.setdefault(id(segment), (segment, [], []))
        entry[1].append(face or "a corner")
        entry[2].append(domain.face_trace(axis, side, box, shift, cell) >= piece)
    unread = [(segment, sorted(set(faces))) for segment, faces, crossed in governed.values() if not any(crossed)]
    if unread:
        named = "; ".join(f"{segment.name!r} on {', '.join(faces)}" for segment, faces in unread)
        raise ValueError(
            f"{reader}: the absorbing segment(s) {named} govern only face cells no particle can cross: the particles' "
            f"domain ({type(domain).__name__}) has no boundary there on the box's face, so every contact near them "
            "reflects and the exit would be read nowhere (#2733). A cell counts where the domain's trace on the face, "
            f"read {CONTACT_EPS:g} of the box's diagonal inside, has a piece at least {2 * CROSSABLE_HALF_WIDTH:g} of "
            "the diagonal long (ImplicitDomain.face_trace). Place the exit on a part of a face the domain's boundary "
            "covers; an exit on a curved boundary or an obstacle's surface is #2789."
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

    It reads a particle by where its step ended, not by its path: at a corner one of whose faces is periodic, the
    corner's segment decides only whether the particle is absorbed, and each face it reached acts on its own axis
    (`apply`; #2550, #2803's shape). The particle solver reads the path instead (`reflect_at_first_contacts`).

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
        self._boundary_tolerance = ONWALL_TOL

    def apply(
        self,
        particles: NDArray[np.floating],
        bc: BoundaryConditions,
        bounds: list[tuple[float, float]],
    ) -> tuple[NDArray[np.floating], NDArray[np.bool_], NDArray[np.floating]]:
        """
        Apply segment-aware boundary conditions to particles, read by where each step ended (the end-position rule,
        CONVENTIONS.md § *Particles at a wall*).

        A particle at or past a face of ``bounds``, within ``ONWALL_TOL`` (`walls_reached`), is read at its nearest
        point of the box (`_segment_at`): on one face that face's segment, at a corner the owner's corner order. That
        segment decides whether it is absorbed, through the absorbing wall's owner (`absorb_at_walls`): removed, never
        moved, its end recorded. If it is not, each face it is at or past acts on its own axis by its own segment
        (`_act_per_face`): a periodic face wraps the axis, any other folds it. A face the nearest point is read on and
        the particle has not reached takes no action on its axis, though at a corner it takes part in the corner order.

        Args:
            particles: Particle positions, shape (N, d). Read, never written.
            bc: BoundaryConditions with segment definitions. Read, never written (#2738).
            bounds: Domain bounds per dimension [(xmin, xmax), ...] - the box every query is read on (`on_the_box`)

        Returns:
            Tuple of:
            - remaining_particles: Particles not absorbed, shape (M, d) where M <= N
            - absorbed_mask: Boolean mask of absorbed particles, shape (N,)
            - exit_positions: Positions where particles were absorbed, their ends, shape (K, d)
        """
        remaining, absorbed_mask, exit_positions, _ = self._apply(particles, bc, bounds, {})
        return remaining, absorbed_mask, exit_positions

    @deprecated(
        since="v0.22.0",
        replacement="ParticleApplicator.apply, which is this method with no limits and returns its first three values; "
        "a capacity-limited exit has no replacement (#2575).",
        reason="no FP solver honours a capacity-limited exit: each refuses a segment carrying flux_capacity (#2575), "
        "and the solver method that called this one had no caller and is deleted (#2550)",
    )
    def apply_with_flux_limits(
        self,
        particles: NDArray[np.floating],
        bc: BoundaryConditions,
        bounds: list[tuple[float, float]],
        flux_limits: dict[str, float],
    ) -> tuple[NDArray[np.floating], NDArray[np.bool_], NDArray[np.floating], dict[str, int]]:
        """
        Apply segment-aware BC with flux-limited absorption at absorbing exits. Deprecated (#2550): with no limits it is
        `apply`, and a capacity-limited exit has no replacement, since no FP solver honours one (#2575).

        `apply`'s end-position rule, with one difference: a segment named in ``flux_limits`` absorbs, in particle
        order, only while its count this call is below its limit. Past it the particle is not absorbed and the faces it
        reached act as in `apply`, so at that exit it is folded back (a queue at the exit).

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
        return self._apply(particles, bc, bounds, flux_limits)

    def _apply(
        self,
        particles: NDArray[np.floating],
        bc: BoundaryConditions,
        bounds: list[tuple[float, float]],
        flux_limits: dict[str, float],
    ) -> tuple[NDArray[np.floating], NDArray[np.bool_], NDArray[np.floating], dict[str, int]]:
        """`apply`'s end-position rule, with ``flux_limits`` read in particle order (`apply_with_flux_limits`): its
        three values, and the count absorbed per limited segment."""
        particles = np.atleast_2d(particles)
        n_particles, dimension = particles.shape
        if n_particles == 0:
            return particles, np.array([], dtype=bool), np.array([]).reshape(0, dimension), {}

        box = np.array(bounds, dtype=float)
        bc = on_the_box(bc, box)
        reached = walls_reached(particles, box, self._boundary_tolerance)
        leaves = np.zeros(n_particles, dtype=bool)
        result_particles = particles.copy()
        absorbed_count: dict[str, int] = dict.fromkeys(flux_limits, 0)

        for idx in np.flatnonzero((reached >= 0).any(axis=1)):
            segment = self._segment_at(bc, particles[idx], box)
            # The matched segment's own beta classifies a Robin wall, so a mixed BC with a different Robin coefficient
            # per wall is read per wall; the meshfree applicator's uniform path reads the BC's one condition and refuses
            # a mixed BC (`applicator_meshfree.the_one_condition`). No matching segment: reflect.
            action = "reflecting" if segment is None else segment_particle_action(segment, "ParticleApplicator.apply")
            if action == "absorbing":
                name = segment.name
                if name not in flux_limits:
                    leaves[idx] = True
                    continue
                if absorbed_count[name] < flux_limits[name]:
                    leaves[idx] = True
                    absorbed_count[name] += 1
                    continue
                # Capacity full: turned back at the exit, by the faces it reached.
            result_particles[idx] = self._act_per_face(bc, particles[idx], box, reached[idx], action)

        absorbed_mask, exit_positions = absorb_at_walls(particles, box, self._boundary_tolerance, leaves)
        return result_particles[~absorbed_mask], absorbed_mask, exit_positions, absorbed_count

    def refuse_segments_it_would_drop(
        self,
        bc: BoundaryConditions,
        bounds: list[tuple[float, float]],
        reader: str = "ParticleApplicator",
        domain: ImplicitDomain | None = None,
    ) -> None:
        """Refuse, by name, a segment of ``bc`` that governs no part of the faces of ``bounds`` a particle crosses
        (#2490 B1e-2), so its condition would be dropped: the boundary owner's point-family check
        (`BoundaryConditions.refuse_segments_governing_no_point`) over this reader's own points, with its sameness,
        `same_particle_action`.

        The contact loop reads a particle where its step crosses the box, on the face it crosses, through
        `segment_at_box_contact` on `on_the_box`'s copy (#2733). The check reads, through the same function, one point
        in each cell of an arrangement of each face (`BoundaryConditions.face_cells`), so a segment that governs a set
        of positive width on a face governs one of them, except where a placement read from a signed distance is
        sampled on a lattice. A set of no width -- a corner, an edge, a range that is one coordinate on the face -- is
        not read: a crossing meets it with probability zero, so such a segment is refused here. Each cell is read
        within ``CONTACT_EPS`` of the box's diagonal, the tolerance the loop reads a contact at.

        With the particles' ``domain``, an obstacle domain or a curved one, an absorbing segment that governs face
        cells and none a particle can cross is refused first, naming its faces and the domain
        (`_refuse_exits_no_particle_crosses`). A cell counts where the domain's trace on the face within it
        (`ImplicitDomain.face_trace`, read ``CONTACT_EPS`` of the box's diagonal inside) has a piece at least
        ``2 * CROSSABLE_HALF_WIDTH`` of the diagonal long. Elsewhere -- a disc's bounding box away from its tangent
        points, the corridor of an obstacle flush with a wall -- no particle reaches the cell and every contact near it
        reflects. A reflecting segment there is left admitted: it is what the loop does anyway. A tangency of a ball of
        radius ``r`` counts as crossable for ``r >= h^2 / (2 s)``, 5000 diagonals; a crossable piece shorter than ``2 h``
        is not counted."""
        box = np.array(bounds, dtype=float)
        # Read on the step's box, as the loop reads; explained on the BC's own box, where a region it places past the
        # solver's box lies (#2782 review 4's reason (b)).
        declared = _with_box(bc, box)
        own = None if bc.domain_bounds is None else np.asarray(bc.domain_bounds, dtype=float).reshape(box.shape)
        bc = on_the_box(bc, box)
        cells, lattice = bc.face_cells(box)
        diagonal = float(np.linalg.norm(box[:, 1] - box[:, 0]))
        readings = [(point, *segment_at_box_contact(bc, point, CONTACT_EPS * diagonal)) for _, _, _, point in cells]
        if domain is not None and _absorbs_somewhere(bc, reader):
            _refuse_exits_no_particle_crosses(cells, readings, domain, box, diagonal, reader)
        notes = []
        if lattice is not None:
            notes.append(
                f"An sdf_region, or a normal_direction read from domain_sdf, is read on a lattice of {lattice} equal parts "
                f"along each axis of a face (FACE_LATTICE_CELLS), so a set one holds that contains no box of side "
                f"1/{lattice} of the face's extent is missed: widen it, or place it by region"
            )
        if own is not None and np.abs(own - box).max() > CONTACT_EPS * diagonal:
            notes.append(
                f"The BC's domain_bounds {own.tolist()} are not the solver's box {box.tolist()}: a particle crosses the "
                "solver's box and is read where it crosses it (#2733), so a placement on a face of the BC's own box that "
                "is not on the solver's is read nowhere. Give the BC the solver's box, and place the exit on that box's "
                "face"
            )
        bc.refuse_segments_governing_no_point(
            readings,
            reader,
            f"the {len(cells)} points it reads the faces of the box {box.tolist()} on, one in each cell of an "
            "arrangement of each face",
            equivalent=same_particle_action,
            note=". ".join(notes),
            explain=lambda segment: declared.why_no_face_cell(segment, box),
        )

    def _segment_at(
        self, bc_on_box: BoundaryConditions, particle: NDArray[np.floating], box: NDArray[np.floating]
    ) -> BCSegment | None:
        """The segment that governs a boundary particle: read at its nearest point of ``box`` through the contact
        loop's reader (`segment_at_box_contact`), at the owner's default tolerance (``SDF_BOUNDARY_TOL``), the one the
        first-face read asked the owner at. On one face that is the face's segment there, and a placement holds the
        point within that tolerance; within it of a second face too, or past both, the owner's corner rule decides
        (#2512 row B1, stage B1d), where the first face found, x before y, used to. The applicator's own band,
        ``ONWALL_TOL``, only decides which particles are at the boundary (`walls_reached`)."""
        point = np.clip(particle, box[:, 0], box[:, 1])
        return segment_at_box_contact(bc_on_box, point, SDF_BOUNDARY_TOL)[1]

    def _act_per_face(
        self,
        bc_on_box: BoundaryConditions,
        end: NDArray[np.floating],
        box: NDArray[np.floating],
        reached: NDArray[np.int8],
        action: str,
    ) -> NDArray[np.floating]:
        """``end`` returned to the box by each face it is at or past (``reached``, its row of `walls_reached`), each on
        its own axis by its own segment: a periodic face wraps the axis (`periodic.wrap_positions`), any other folds it
        (`reflect_positions`). At a corner one of whose faces is periodic each face's segment is the owner's reading
        there (`periodic_corner_actions`). Elsewhere each face takes the kind of ``action``, the action of the segment
        `_segment_at` read: on one face that is the face's own, and at a corner none of whose faces wraps every face
        folds."""
        point = np.clip(end, box[:, 0], box[:, 1])
        corner = periodic_corner_actions(bc_on_box, point, SDF_BOUNDARY_TOL, "ParticleApplicator.apply")
        wraps = None if corner is None else {(axis, side): wrap for axis, side, wrap in corner[1]}
        out = end.copy()
        for axis in np.flatnonzero(reached >= 0):
            # A reached face is within ONWALL_TOL of its plane, so the corner read it at SDF_BOUNDARY_TOL (> ONWALL_TOL).
            face = (int(axis), int(reached[axis]))
            if (action == "periodic") if wraps is None else wraps[face]:
                out = wrap_positions(out, box, periodic_dims=(face[0],))
            else:
                out[axis] = reflect_positions(out[axis : axis + 1], box[axis : axis + 1])[0]
        return out


__all__ = ["ParticleApplicator", "same_particle_action", "segment_particle_action"]

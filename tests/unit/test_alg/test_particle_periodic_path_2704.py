"""The particle solver reads a step under a periodic boundary condition by its path, not by where it ended (#2704,
#2803).

`FPParticleSolver._finish_step` ends every CPU step at its first contacts with the particles' boundary
(`reflect_at_first_contacts`); a contact on a periodic face goes on with the rest of the step from its image on the
opposite face. Before, a BC declaring a periodic face took `ParticleApplicator.apply`, which reads the end: at a corner
it gave one action for every axis, so a channel corner wrapped through the wall or reflected across the seam (#2803),
and an exit the path crossed was read by the corner order at the end (#2704 comment 6335147485).

**The oracle.** On a box a specular reflection flips one axis's velocity and a wrap keeps it, so each axis's motion
along a straight step is independent of the others'. A step that crosses at most one face per axis therefore ends, per
axis, at its raw coordinate, folded or wrapped where it crossed a wall or a periodic face; and it is absorbed at the
earliest crossing of an exit, where each other axis has its coordinate at that time. A crossing of two faces at one
time, a tie, follows the tie rule (CONVENTIONS.md § *Particles at a wall*): the owner's corner order decides only
whether the particle is absorbed, and otherwise a periodic face wraps its axis and any other face reflects it.

**The grid**, every row through `_finish_step`: the periodic axis (x, y) x the other axis's faces (walls, an exit at the
corner the step crosses with a wall opposite, or periodic) x the face crossed first (the periodic one, the other, both
at once) x the corner crossed (all four, so both signs of the step on each axis). Then the priority row, a periodic axis
at priority 1 beside an exit, on all four corners and orders; a 3-D edge (a wall on y beside an exit on z, #2800's U3),
and ties at 3-D edges and vertices with a periodic face; steps that start at a corner or an edge and leave through one
face, steps toward a face within the tolerance that they cross later or never, a rest shorter than the band, faces met
just inside and just outside the band, and a particle domain whose face lies a hair inside the box's; the torus on long
steps against the periodic owner, and the refusal of a step past ``MAX_WRAPS``; the exit strip of 6335147485 at solve
level; and an obstacle met after a wrap, an obstacle beside a torus however the torus is spelt, the refusal of one that
touches a face of the periodic axis, flush, past it or on the other face, and in 3-D of any (#2807).

**Rivals**, each a copy of the package at this commit with one replacement, run with this file: listed in the PR.

Held fixed: ``corner_strategy="priority"``; constant values; one face crossed at most per axis in the grid, longer
steps only on the torus; the 1-D GPU path, which wraps or folds by topology alone; `ParticleApplicator.apply`, public
and no longer on the solver's path (#2550).
"""

from __future__ import annotations

import itertools
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import MAX_WRAPS
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, periodic_bc
from mfgarchon.geometry.boundary.periodic import wrap_positions
from mfgarchon.geometry.implicit import DifferenceDomain, Hyperrectangle, Hypersphere

P, D, NF = BCType.PERIODIC, BCType.DIRICHLET, BCType.NO_FLUX
AXES = "xy"
CORNERS = list(itertools.product((0, 1), repeat=2))  # the side crossed on x, on y: 0 the min face, 1 the max
ORDERS = ("periodic first", "other first", "tie")


def _solver(bc: BoundaryConditions, d: int = 2, **extra) -> FPParticleSolver:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[11] * d, boundary_conditions=bc)
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.2)
    conditions = Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FPParticleSolver(
            MFGProblem(model=model, domain=grid, conditions=conditions, Nt=2), num_particles=10, seed=0, **extra
        )


def _finish(solver: FPParticleSolver, previous, raw, d: int = 2):
    """Every particle's end and whether it was absorbed, in the input's order."""
    previous, raw = np.atleast_2d(np.asarray(previous, float)), np.atleast_2d(np.asarray(raw, float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kept, absorbed, _ = solver._finish_step(raw, previous, [(0.0, 1.0)] * d, solver._needs_segment_aware_bc())
    absorbed = np.zeros(len(raw), dtype=bool) if absorbed is None else absorbed
    ends = np.full(raw.shape, np.nan)
    ends[~absorbed] = kept
    return ends, absorbed


def _face(axis: int, side: int) -> str:
    return f"{AXES[axis]}_{('min', 'max')[side]}"


def _bc(periodic_axis: int, other: str, corner: tuple[int, int], priority: int = 0) -> BoundaryConditions:
    """The periodic axis's two faces; on the other axis two walls, an exit at ``corner``'s side with a wall opposite, or
    two more periodic faces."""
    p, o = periodic_axis, 1 - periodic_axis
    segments = [BCSegment(name=_face(p, s), bc_type=P, boundary=_face(p, s), priority=priority) for s in (0, 1)]
    for s in (0, 1):
        kind = {"walls": NF, "periodic": P}.get(other, D if s == corner[o] else NF)
        extra = {"value": 0.0} if kind is D else {}
        segments.append(BCSegment(name=_face(o, s), bc_type=kind, boundary=_face(o, s), **extra))
    return BoundaryConditions(dimension=2, segments=segments)


def _step(periodic_axis: int, corner: tuple[int, int], order: str):
    """A straight step outward past ``corner``: 0.04 on each axis from 0.01 or 0.02 inside, so the periodic face is met
    at t = 0.25 and the other at 0.5, or the reverse, or both at 0.5."""
    inside = {"periodic first": (0.01, 0.02), "other first": (0.02, 0.01), "tie": (0.02, 0.02)}[order]
    depth = np.empty(2)
    depth[periodic_axis], depth[1 - periodic_axis] = inside
    outward = np.array([1.0 if s == 1 else -1.0 for s in corner])
    bound = np.array(corner, dtype=float)
    return bound - outward * depth, bound - outward * depth + outward * 0.04


def _expected(start, end, actions, tie_absorbs: bool):
    """The per-axis closed form: ``actions[axis]`` is what the crossed face of that axis does ('wrap', 'reflect' or
    'absorb'), for a step crossing one face per axis. Returns (end or None, absorbed)."""
    times = {}
    for axis in range(len(start)):
        bound = 1.0 if end[axis] > 1.0 else 0.0 if end[axis] < 0.0 else None
        if bound is not None:
            times[axis] = (bound - start[axis]) / (end[axis] - start[axis])
    tie = len(times) > 1 and np.isclose(min(times.values()), max(times.values()))
    if tie:
        if tie_absorbs:
            return None, True
        actions = {axis: ("wrap" if actions[axis] == "wrap" else "reflect") for axis in actions}
    elif any(actions[axis] == "absorb" for axis in times):
        return None, True
    out = np.array(end, dtype=float)
    for axis in times:
        bound = 1.0 if end[axis] > 1.0 else 0.0
        out[axis] = end[axis] - np.sign(end[axis] - 0.5) if actions[axis] == "wrap" else 2.0 * bound - end[axis]
    return out, False


def _grid_rows():
    for p, other, order, corner in itertools.product((0, 1), ("walls", "exit", "periodic"), ORDERS, CORNERS):
        yield pytest.param(
            p, other, order, corner, id=f"periodic_{AXES[p]}-{other}-{order}-corner_{corner[0]}{corner[1]}"
        )


@pytest.mark.parametrize(("periodic_axis", "other", "order", "corner"), list(_grid_rows()))
def test_a_step_under_a_periodic_bc_ends_where_its_path_takes_it(periodic_axis, other, order, corner):
    start, end = _step(periodic_axis, corner, order)
    ends, absorbed = _finish(_solver(_bc(periodic_axis, other, corner)), start, end)
    o = 1 - periodic_axis
    actions = {periodic_axis: "wrap", o: {"walls": "reflect", "exit": "absorb", "periodic": "wrap"}[other]}
    # At a tie the corner order decides absorption: at one priority DIRICHLET comes first, so the exit absorbs.
    expected, expected_absorbed = _expected(start, end, actions, tie_absorbs=other == "exit")
    assert absorbed.tolist() == [expected_absorbed], (start.tolist(), end.tolist())
    if not expected_absorbed:
        np.testing.assert_allclose(ends[0], expected, atol=1e-12)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("corner", CORNERS, ids=[f"corner_{x}{y}" for x, y in CORNERS])
def test_a_periodic_face_that_outranks_an_exit_decides_only_a_tie(order, corner):
    """y periodic at priority 1 beside an exit on x. The path absorbs wherever it crosses the exit; the priority
    reaches only a tie, where the corner order picks the periodic face, so the particle is not absorbed: it wraps in y
    and the exit's face reflects it in x."""
    start, end = _step(1, corner, order)
    ends, absorbed = _finish(_solver(_bc(1, "exit", corner, priority=1)), start, end)
    expected, expected_absorbed = _expected(start, end, {1: "wrap", 0: "absorb"}, tie_absorbs=False)
    assert absorbed.tolist() == [expected_absorbed], (start.tolist(), end.tolist())
    if not expected_absorbed:
        np.testing.assert_allclose(ends[0], expected, atol=1e-12)


def test_a_three_dimensional_edge_takes_its_wall_then_its_exit():
    """#2800's U3: x periodic, an exit on z_max, NO_FLUX elsewhere. The step meets y_max's wall at t = 0.25 and the exit
    at t = 0.5, where it is absorbed. Main absorbs it too since #2800; it stays as the 3-D control for the path, with no
    periodic face at its edge. The ties at a 3-D corner with a periodic face are the next test."""
    bc = BoundaryConditions(
        dimension=3,
        segments=[
            BCSegment(name="px0", bc_type=P, boundary="x_min"),
            BCSegment(name="px1", bc_type=P, boundary="x_max"),
            BCSegment(name="exit", bc_type=D, value=0.0, boundary="z_max"),
        ],
        default_bc=NF,
    )
    _, absorbed = _finish(_solver(bc, d=3), [[0.5, 0.99, 0.98]], [[0.5, 1.03, 1.02]], d=3)
    assert absorbed.tolist() == [True]


def _on(face: str, kind: BCType, priority: int = 0) -> BCSegment:
    return BCSegment(name=face, bc_type=kind, boundary=face, priority=priority, **({"value": 0.0} if kind is D else {}))


#: A 3-D step 0.02 inside each face it crosses and 0.04 long on each, so it meets them all at t = 0.5: a tie. NO_FLUX
#: elsewhere. The expected end is the tie rule's, by hand: absorbed, or each crossed face's own action on its axis.
_TIES_3D = {
    "edge: x periodic, y wall": ([_on("x_min", P), _on("x_max", P)], [0.98, 0.98, 0.5], [0.02, 0.98, 0.5]),
    "corner: x periodic, y and z walls": ([_on("x_min", P), _on("x_max", P)], [0.98, 0.98, 0.98], [0.02, 0.98, 0.98]),
    "corner: x and y periodic, z wall": (
        [_on(f, P) for f in ("x_min", "x_max", "y_min", "y_max")],
        [0.98, 0.02, 0.98],
        [0.02, 0.98, 0.98],
    ),
    "corner: x periodic, an exit on z": ([_on("x_min", P), _on("x_max", P), _on("z_max", D)], [0.98] * 3, None),
    "corner: x periodic at priority 1, an exit on z": (
        [_on("x_min", P, 1), _on("x_max", P, 1), _on("z_max", D)],
        [0.98] * 3,
        [0.02, 0.98, 0.98],
    ),
}


@pytest.mark.parametrize("row", list(_TIES_3D))
def test_a_tie_at_a_three_dimensional_corner_with_a_periodic_face_takes_the_tie_rule(row):
    """The corner order decides only absorption: an exit at one priority absorbs (DIRICHLET first), a periodic face at
    priority 1 does not. Otherwise every face at the corner acts, on its own axis, each of the three at a vertex."""
    segments, start, expected = _TIES_3D[row]
    end = [s + (0.04 if s > 0.5 else -0.04) if s != 0.5 else s for s in start]
    solver = _solver(BoundaryConditions(dimension=3, segments=segments, default_bc=NF), d=3)
    ends, absorbed = _finish(solver, [start], [end], d=3)
    assert absorbed.tolist() == [expected is None]
    if expected is not None:
        np.testing.assert_allclose(ends[0], expected, atol=1e-12)


def test_a_torus_wraps_a_long_step_as_the_periodic_owner_does():
    """Every face periodic: the loop wraps at each contact, so a step of up to 30 periods ends where
    `periodic.wrap_positions` puts its raw end. A wrap is not a reflection, so the loop's cap on reflections per step
    does not reach it."""
    rng = np.random.default_rng(0)
    start = rng.uniform(0.0, 1.0, (40, 2))
    raw = start + rng.uniform(-30.0, 30.0, (40, 2))
    ends, absorbed = _finish(
        _solver(BoundaryConditions(dimension=2, segments=[BCSegment(name="all", bc_type=P)])), start, raw
    )
    assert not absorbed.any()
    np.testing.assert_allclose(ends, wrap_positions(raw, [(0.0, 1.0), (0.0, 1.0)]), atol=1e-9)


def test_a_step_that_wraps_more_than_max_wraps_times_is_refused_by_name():
    """The loop searches once per period a step crosses, and a step of about 1e17 periods never ends, since its shift
    by one period rounds away. So a step of ``MAX_WRAPS`` wraps ends where the owner puts it, and one more is
    refused."""
    torus = _solver(BoundaryConditions(dimension=2, segments=[BCSegment(name="all", bc_type=P)]))
    start = [[0.5, 0.5]]
    ends, _ = _finish(torus, start, [[0.8 + MAX_WRAPS, 0.5]])
    np.testing.assert_allclose(ends, [[0.8, 0.5]], atol=1e-9)
    with pytest.raises(ValueError, match=rf"wrapped more than {MAX_WRAPS} times through periodic faces"):
        _finish(torus, start, [[1.8 + MAX_WRAPS, 0.5]])


def test_an_exit_strip_beside_periodic_faces_absorbs_what_its_path_crosses():
    """#2704 comment 6335147485, row 2: an exit strip on x_max, ``region`` y in [0.42, 0.48], and periodic y faces; 200
    particles driven across it. The solver absorbs exactly the particles whose raw step crosses x = 1 inside the strip;
    reading the end, it absorbed 188 where the crossing reading gave 196."""
    steps = []
    solver_cls = FPParticleSolver

    class Spy(solver_cls):
        def _finish_step(self, raw, previous, bounds, use_segment_aware_bc):
            out = super()._finish_step(raw, previous, bounds, use_segment_aware_bc)
            steps.append((np.array(previous, float).reshape(raw.shape), np.array(raw, float)))
            return out

    bc = BoundaryConditions(
        segments=[
            BCSegment(name="exit", bc_type=D, value=0.0, region={"x": (1.0, 1.0), "y": (0.42, 0.48)}),
            BCSegment(name="py0", bc_type=P, boundary="y_min"),
            BCSegment(name="py1", bc_type=P, boundary="y_max"),
        ],
        dimension=2,
        default_bc=NF,
    )
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
    )
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.05
    )
    conditions = Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = Spy(
            MFGProblem(model=model, domain=grid, conditions=conditions, Nt=5),
            num_particles=200,
            seed=0,
            boundary_conditions=bc,
        )
        solver.solve_fp_system(
            initial_particles=np.random.default_rng(0).uniform([0.8, 0.43], [0.95, 0.47], (200, 2)),
            drift_field=lambda t, x, m: np.broadcast_to(np.array([3.0, 0.0]), np.shape(x)),
            drift_needs_density=False,
            show_progress=False,
        )
    crossing = 0
    for a, b in steps:
        crosses = b[:, 0] > 1.0
        y = a[:, 1] + (1.0 - a[:, 0]) / np.where(crosses, b[:, 0] - a[:, 0], 1.0) * (b[:, 1] - a[:, 1])
        crossing += int(np.sum(crosses & (y >= 0.42) & (y <= 0.48)))
    assert crossing > 150
    assert solver.total_absorbed == crossing


def _channel(**extra):
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="px0", bc_type=P, boundary="x_min"),
            BCSegment(name="px1", bc_type=P, boundary="x_max"),
            BCSegment(name="w0", bc_type=NF, boundary="y_min"),
            BCSegment(name="w1", bc_type=NF, boundary="y_max"),
        ],
    )
    return bc


def test_a_step_that_wraps_meets_an_obstacle_on_its_image():
    """#2704's capability: periodic x, walls on y, an obstacle [0.1, 0.3] x [0.4, 0.6]. A step from (0.95, 0.5) to
    (1.15, 0.5) wraps at x = 1, goes on from (0, 0.5) and meets the obstacle's face x = 0.1, which reflects it to
    (0.05, 0.5). The chord from its start is no path: read by its end it lands in the obstacle, at (0.15, 0.5)."""
    domain = DifferenceDomain(
        Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), Hyperrectangle(np.array([[0.1, 0.3], [0.4, 0.6]]))
    )
    solver = _solver(_channel(), implicit_domain=domain)
    ends, absorbed = _finish(solver, [[0.95, 0.5], [0.97, 0.9]], [[1.15, 0.5], [1.03, 0.9]])
    assert not absorbed.any()
    np.testing.assert_allclose(ends, [[0.05, 0.5], [0.03, 0.9]], atol=1e-12)


_TORI = {
    "periodic_bc": lambda d: periodic_bc(dimension=d),
    "one global segment": lambda d: BoundaryConditions(dimension=d, segments=[BCSegment(name="all", bc_type=P)]),
    "default_bc": lambda d: BoundaryConditions(dimension=d, segments=[], default_bc=P),
    "a segment per face": lambda d: BoundaryConditions(
        dimension=d, segments=[_on(f"{a}_{s}", P) for a in AXES[:d] for s in ("min", "max")]
    ),
}


@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("spelling", list(_TORI))
def test_a_torus_reflects_a_step_off_an_obstacle_however_it_is_spelt(spelling, d):
    """D2's box rule under a torus: a contact on an obstacle, clear of every face, reflects. A torus spelt as one
    condition has one action for every contact on the box; read at the obstacle too, it wrapped along the obstacle's
    normal until the wrap cap refused a step 0.15 long."""
    domain = DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0]] * d)), Hyperrectangle(np.array([[0.4, 0.6]] * d)))
    if d == 1:
        start, raw, expected = [[0.3]], [[0.45]], [[0.35]]
    else:
        start, raw, expected = [[0.3, 0.5], [0.5, 0.3]], [[0.45, 0.5], [0.5, 0.45]], [[0.35, 0.5], [0.5, 0.35]]
    ends, absorbed = _finish(_solver(_TORI[spelling](d), d=d, implicit_domain=domain), start, raw, d=d)
    assert not absorbed.any()
    np.testing.assert_allclose(ends, expected, atol=1e-12)


def _obstacle_2d():
    return DifferenceDomain(
        Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), Hyperrectangle(np.array([[0.4, 0.6], [0.4, 0.6]]))
    )


#: A step that starts at a corner or an edge, or meets the box near one, and leaves through one face only. Only a face
#: the rest of the step leaves through is crossed, so only it acts; whether the particle is absorbed is the corner
#: order's at the point. Each expected end by hand.
_AT_A_CORNER = {
    "channel, from corner (0, 0), out through x_min": (
        lambda: _channel(),
        None,
        2,
        [0.0, 0.0],
        [-0.01, 0.02],
        [0.99, 0.02],
    ),
    "channel, from corner (1, 1), out through the wall y_max": (
        lambda: _channel(),
        None,
        2,
        [1.0, 1.0],
        [0.98, 1.01],
        [0.98, 0.99],
    ),
    "channel, 1e-13 off corner (0, 0)": (lambda: _channel(), None, 2, [1e-13, 1e-13], [-0.01, 0.02], [0.99, 0.02]),
    "torus by face, from corner (1, 1), out through y_max": (
        lambda: _TORI["a segment per face"](2),
        None,
        2,
        [1.0, 1.0],
        [0.98, 1.01],
        [0.98, 0.01],
    ),
    "torus beside an obstacle, from corner (0, 0)": (
        lambda: periodic_bc(dimension=2),
        _obstacle_2d,
        2,
        [0.0, 0.0],
        [-0.01, 0.02],
        [0.99, 0.02],
    ),
    "3-D, x periodic, from the edge (0, 0, z), out through x_min": (
        lambda: BoundaryConditions(dimension=3, segments=[_on("x_min", P), _on("x_max", P)], default_bc=NF),
        None,
        3,
        [0.0, 0.0, 0.5],
        [-0.01, 0.02, 0.5],
        [0.99, 0.02, 0.5],
    ),
    "an exit at the corner: the particle on it is absorbed at the point": (
        lambda: BoundaryConditions(
            dimension=2, segments=[_on("x_min", P), _on("x_max", P), _on("y_min", D)], default_bc=NF
        ),
        None,
        2,
        [0.0, 0.0],
        [-0.01, 0.02],
        None,
    ),
    "6e-4 from corner (1, 1), past the tie band: a face contact on x_max, then clear of the exit": (
        lambda: BoundaryConditions(
            dimension=2, segments=[_on("x_min", P), _on("x_max", P), _on("y_max", D)], default_bc=NF
        ),
        None,
        2,
        [0.99, 0.999],
        [1.01, 0.9998],
        [0.01, 0.9998],
    ),
    # #2808 review 3: a face within the tolerance that the step moves toward but never reaches, or reaches only far
    # along the path, is not crossed at the contact.
    "x periodic at priority 1 beside an exit on y_max: toward y_max, never reaching it": (
        lambda: BoundaryConditions(
            dimension=2, segments=[_on("x_min", P, 1), _on("x_max", P, 1), _on("y_max", D)], default_bc=NF
        ),
        None,
        2,
        [0.01, 1.0 - 1e-13],
        [-0.01, 1.0 - 5e-14],
        [0.99, 1.0 - 5e-14],
    ),
    "y_max periodic on one side only (#2713), an exit on y_min: toward y_max, never reaching it": (
        lambda: BoundaryConditions(dimension=2, segments=[_on("y_max", P), _on("y_min", D)], default_bc=NF),
        None,
        2,
        [0.99, 1.0 - 1e-13],
        [1.01, 1.0 - 5e-14],
        [0.99, 1.0 - 5e-14],
    ),
    "x periodic at priority 1 beside an exit on y_max: y_max crossed far along the path, after the wrap": (
        lambda: BoundaryConditions(
            dimension=2, segments=[_on("x_min", P, 1), _on("x_max", P, 1), _on("y_max", D)], default_bc=NF
        ),
        None,
        2,
        [0.01, 1.0 - 1e-13],
        [-0.5, 1.0 + 1e-13],
        None,
    ),
    # #2808 review 4: the rest of the step shorter than the band, the band's width on both sides, and a particle domain
    # whose face lies a hair inside the box's.
    "y_max periodic on one side only, an exit on y_min: a rest shorter than the band that stops short of y_max": (
        lambda: BoundaryConditions(dimension=2, segments=[_on("y_max", P), _on("y_min", D)], default_bc=NF),
        None,
        2,
        [1.0 - 1e-13, 1.0 - 2e-13],
        [1.0 + 1e-13, 1.0 - 1e-13],
        [1.0 - 1e-13, 1.0 - 1e-13],
    ),
    "y_max periodic on one side only, an exit on y_min: x_min reached 5e-13 along the path after y_max, a tie": (
        lambda: BoundaryConditions(dimension=2, segments=[_on("y_max", P), _on("y_min", D)], default_bc=NF),
        None,
        2,
        [5e-13, 1.0],
        [-0.02, 1.0 + 1e-13],
        [0.02, 1e-13],
    ),
    "x periodic at priority 1 beside an exit on y_max: y_max reached 1e-10 along the path, past the band": (
        lambda: BoundaryConditions(
            dimension=2, segments=[_on("x_min", P, 1), _on("x_max", P, 1), _on("y_max", D)], default_bc=NF
        ),
        None,
        2,
        [0.01, 1.0 - 1e-13 - 1e-5],
        [-0.5, 1.0 - 1e-13 - 1e-5 + 0.51e-3],
        None,
    ),
    "channel, a particle domain whose x_max lies 1e-13 inside the box's: the face met is the one crossed": (
        lambda: _channel(),
        lambda: DifferenceDomain(
            Hyperrectangle(np.array([[0.0, 1.0 - 1e-13], [0.0, 1.0]])),
            Hyperrectangle(np.array([[0.4, 0.6], [0.4, 0.6]])),
        ),
        2,
        [1.0 - 1e-13 - 1e-9, 1.0 - 2e-12 - 1e-7],
        [1.0 - 1e-13 + 1e-9, 1.0 - 2e-12 + 1e-7],
        [1e-9 - 1e-13, 1.0 - 1e-7 + 2e-12],
    ),
}


@pytest.mark.parametrize("row", list(_AT_A_CORNER))
def test_a_step_at_a_corner_acts_only_by_the_faces_it_leaves_through(row):
    """#2808 review 2: a step from a corner that leaves through one face bounced between corners until the wrap cap,
    since every face within the contact tolerance acted, the one it moves away from included. Lattice or point-source
    initial particles reach a corner exactly."""
    bc, domain, d, start, raw, expected = _AT_A_CORNER[row]
    extra = {} if domain is None else {"implicit_domain": domain()}
    ends, absorbed = _finish(_solver(bc(), d=d, **extra), [start], [raw], d=d)
    assert absorbed.tolist() == [expected is None]
    if expected is not None:
        np.testing.assert_allclose(ends[0], expected, atol=1e-12)


def _drift_x(t, x, m):
    return np.broadcast_to(np.array([2.0, 0.0]), np.shape(x))


_TOUCHING = {  # an obstacle in the channel, and the face of the periodic axis x it meets
    "reaching past x_max": ([[0.8, 1.1], [0.4, 0.6]], "max"),
    "flush with x_max": ([[0.8, 1.0], [0.4, 0.6]], "max"),
    "touching x_min only": ([[-0.1, 0.2], [0.4, 0.6]], "min"),
}


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("obstacle", list(_TOUCHING))
def test_an_obstacle_touching_a_periodic_face_is_refused_by_name(obstacle):
    """An obstacle touching either face of x, the periodic axis, flush or past it: a wrapped step would go on from an
    image the obstacle may hold, which the loop does not place, so the solve is refused, naming the face."""
    bounds, side = _TOUCHING[obstacle]
    domain = DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), Hyperrectangle(np.array(bounds)))
    with pytest.raises(NotImplementedError, match=f"meets the {side} face of axis 0, an axis with a periodic face"):
        _solver(_channel(), implicit_domain=domain).solve_fp_system(
            initial_particles=np.random.default_rng(0).uniform([0.3, 0.05], [0.7, 0.3], (10, 2)),
            drift_field=_drift_x,
            drift_needs_density=False,
            show_progress=False,
        )


@pytest.mark.filterwarnings("ignore")
def test_an_obstacle_clear_of_the_periodic_faces_solves_with_wraps():
    """The capability #2688 refused: an obstacle clear of the faces of x under the channel solves, and particles
    wrap."""
    box = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))
    clear = DifferenceDomain(box, Hyperrectangle(np.array([[0.4, 0.6], [0.4, 0.6]])))
    particles = np.random.default_rng(0).uniform([0.7, 0.05], [0.95, 0.95], (10, 2))
    drift = _drift_x
    solver = _solver(_channel(), implicit_domain=clear)
    solver.solve_fp_system(
        initial_particles=particles, drift_field=drift, drift_needs_density=False, show_progress=False
    )
    assert sum(step["wraps"] for step in solver.obstacle_placement_history) > 0


@pytest.mark.filterwarnings("ignore")
def test_a_particle_domain_under_a_periodic_face_is_refused_in_three_dimensions():
    """In d >= 3 `face_trace` returns the longest piece over midpoint lines, and a ball touching x_max of the unit cube
    leaves one whole, so the 2-D test cannot see the touch and any particle domain under a periodic face is refused
    (#2807). When `face_trace` sees it, the first assertion fails and the refusal can narrow to a touching obstacle."""
    cube = np.array([[0.0, 1.0]] * 3)
    touching = DifferenceDomain(Hyperrectangle(cube), Hypersphere(center=[1.0, 0.5, 0.5], radius=0.2))
    assert touching.face_trace(0, 1, cube, 1e-9) == pytest.approx(1.0)
    bc = BoundaryConditions(
        dimension=3,
        segments=[
            BCSegment(name="px0", bc_type=P, boundary="x_min"),
            BCSegment(name="px1", bc_type=P, boundary="x_max"),
        ],
        default_bc=NF,
    )
    solver = _solver(bc, d=3, implicit_domain=touching)
    with pytest.raises(NotImplementedError, match=r"in 3-D with a periodic condition on the outer boundary.*#2807"):
        solver.solve_fp_system(
            initial_particles=np.random.default_rng(0).uniform(0.1, 0.5, (10, 3)),
            drift_field=lambda t, x, m: np.broadcast_to(np.array([2.0, 0.0, 0.0]), np.shape(x)),
            drift_needs_density=False,
            show_progress=False,
        )

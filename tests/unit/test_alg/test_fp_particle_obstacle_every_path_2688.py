"""Every forward-step path of `FPParticleSolver` records its particles inside an obstacle domain, or refuses (#2688).

Three CPU paths step particles -- the 1-D grid drift, the n-D grid drift, and the callable drift -- and each crossed
with the outer boundary condition it can take, uniform or segment-aware (absorbing), and with `preserve_indices` and
the caller's `initial_particles` where those exist. Each ends its step in `_finish_step`: a step that meets the
domain's boundary before the box's wall lands where `ImplicitDomain.place_step` places it (#2707); the outer boundary
condition acts on the rest, then the placement on its reflected survivors, by their paths. Before, the 1-D grid path
never placed (1990 of 2000 particles inside the obstacle at the peak, i2688/prod/product_base.log) and the
n-D paths placed BEFORE the wall reflection, which put particles back inside an obstacle that meets the wall (13438
particle-times). The GPU path, a periodic outer boundary and initial mass inside an obstacle are refused by name.

What reddens this file: any path whose step stops calling the placement, or places only before the boundary condition
(its cell); the placement indexing ``previous`` without the absorbed mask (the mask test); a step placed by a boundary
other than the first it meets (the first-boundary rows); the initial sampler dropping ``navigable=`` (the cells'
t = 0, whose jitter at the box's edge nodes leaves the domain).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, periodic_bc
from mfgarchon.geometry.implicit import DifferenceDomain, Hyperrectangle, Hypersphere, UnionDomain

_NT, _N = 15, 600

# 2-D: the obstacle meets the wall x = 1, so a wall reflection can land in it. 1-D: an interval cut from the middle.
_DOMAIN = {
    2: DifferenceDomain(
        Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), Hyperrectangle(np.array([[0.9, 1.05], [0.2, 0.8]]))
    ),
    1: DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0]])), Hyperrectangle(np.array([[0.4, 0.6]]))),
}


def _absorbing_at(d, wall="x_max"):
    boundaries = ["x_min", "x_max", "y_min", "y_max"][: 2 * d]
    return BoundaryConditions(
        segments=[
            BCSegment(name=b, bc_type=BCType.DIRICHLET if b == wall else BCType.NO_FLUX, value=0.0, boundary=b)
            for b in boundaries
        ],
        dimension=d,
    )


def _problem(d, bc, bounds=None):
    n = 21 if d == 2 else 101
    bounds = bounds or [(0.0, 1.0)] * d
    grid = TensorProductGrid(bounds=bounds, Nx_points=[n] * d, boundary_conditions=bc)
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        volatility=0.1 if d == 2 else 0.05,
    )
    volume = float(np.prod([hi - lo for lo, hi in bounds]))
    conditions = Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0 / volume, T=0.3)
    return MFGProblem(model=model, domain=grid, conditions=conditions, Nt=_NT), n


def _m_initial(d, n, centres):
    """Gaussian bumps, zero at the nodes inside the obstacle unless a bump is put there on purpose."""
    xs = np.linspace(0.0, 1.0, n)
    pts = xs.reshape(-1, 1) if d == 1 else np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1).reshape(-1, 2)
    m = sum(np.exp(-np.sum((pts - np.asarray(c)) ** 2, axis=1) / 0.002) for c in centres)
    if all(_DOMAIN[d].contains(np.atleast_2d(c))[0] for c in centres):
        m = m * _DOMAIN[d].contains(pts)
    m = m.reshape((n,) * d)
    return m / (m.sum() * (xs[1] - xs[0]) ** d)


def _drift(d, n, path):
    """Toward the obstacle and, in 2-D, toward the wall it meets: (1, -0.6); +2 in 1-D."""
    if path == "callable":
        if d == 1:
            return lambda t, x, m: 2.0 + 0.0 * np.asarray(x, dtype=float)
        return lambda t, x, m: np.column_stack([np.full(len(x), 1.0), np.full(len(x), -0.6)])
    xs = np.linspace(0.0, 1.0, n)
    if d == 1:
        return np.tile(-2.0 * xs, (_NT + 1, 1))  # the 1-D grid path differentiates U: U = -2x is drift +2
    drift = np.zeros((_NT + 1, n, n, 2))
    drift[..., 0], drift[..., 1] = 1.0, -0.6
    return drift


def _solve(d, bc, path, centres=None, initial_particles=None, **solver_kwargs):
    problem, n = _problem(d, bc)
    if centres is None:
        # An absorbing BC gets a second bump that the drift carries to the exit, so absorption and placement both act.
        absorbing = isinstance(bc, BoundaryConditions) and any(s.bc_type == BCType.DIRICHLET for s in bc.segments)
        centres = (
            ([(0.93, 0.88)] + ([(0.5, 0.5)] if absorbing else [])) if d == 2 else [0.3] + ([0.7] if absorbing else [])
        )
    solver = FPParticleSolver(
        problem, num_particles=_N, implicit_domain=_DOMAIN[d], seed=2688, boundary_conditions=bc, **solver_kwargs
    )
    kwargs = {"drift_field": _drift(d, n, path), "show_progress": False}
    if path == "grid" and d == 2:
        kwargs["drift_is_precomputed"] = True
    if path == "callable":
        kwargs["drift_needs_density"] = False
    if initial_particles is not None:
        kwargs["initial_particles"] = initial_particles
    else:
        kwargs["M_initial"] = _m_initial(d, n, centres)
    solver.solve_fp_system(**kwargs)
    return solver


def _navigable_start():
    return np.random.default_rng(2688).uniform([0.6, 0.6], [0.85, 0.95], (_N, 2))


_CELLS = {
    "1-D grid, reflecting": (1, "reflecting", "grid", {}),
    "1-D grid, absorbing": (1, "absorbing", "grid", {}),
    "n-D grid, reflecting": (2, "reflecting", "grid", {}),
    "n-D grid, absorbing": (2, "absorbing", "grid", {}),
    "callable 1-D, reflecting": (1, "reflecting", "callable", {}),
    "callable 1-D, absorbing": (1, "absorbing", "callable", {}),
    "callable 2-D, reflecting": (2, "reflecting", "callable", {}),
    "callable 2-D, absorbing": (2, "absorbing", "callable", {}),
    "callable 2-D, absorbing, preserve_indices": (2, "absorbing", "callable", {"preserve_indices": True}),
    "callable 2-D, reflecting, initial_particles": (2, "reflecting", "callable", {"initial": True}),
}


@pytest.mark.parametrize("cell", list(_CELLS))
def test_no_step_path_records_a_particle_outside_the_domain(cell):
    """Every recorded particle, at every time, is navigable; and the cell exercises what it names -- the placement
    acted, and an absorbing cell absorbed."""
    d, wall, path, options = _CELLS[cell]
    bc = _absorbing_at(d) if wall == "absorbing" else no_flux_bc(dimension=d)
    kwargs = {}
    if options.get("preserve_indices"):
        kwargs = {"preserve_indices": True, "density_mode": "query_only"}
    solver = _solve(
        d, bc, path, initial_particles=_navigable_start() if options.get("initial") or kwargs else None, **kwargs
    )

    outside = []
    for particles in solver.M_particles_trajectory:
        p = np.asarray(particles, dtype=float).reshape(-1, d)
        p = p[np.all(np.isfinite(p), axis=1)]  # preserve_indices marks the absorbed with NaN
        outside.append(int((~np.atleast_1d(_DOMAIN[d].contains(p))).sum()))
    assert sum(outside) == 0, f"{cell}: particles recorded outside the domain, per time: {outside}"
    placed = sum(sum(h.values()) for h in solver.obstacle_placement_history)
    assert placed > 0, f"{cell}: the placement never acted, so this cell does not test it"
    if wall == "absorbing":
        assert solver.total_absorbed > 0, (
            f"{cell}: nothing was absorbed, so this cell does not test the absorbing branch"
        )


def test_the_placement_reads_each_survivors_own_previous_position():
    """The boundary condition absorbs, so the survivors are a subset, and a reflected survivor is placed by its own
    path, through the absorbed mask. Particle 0 passes above the obstacle and leaves at the absorbing wall. Particle 1
    reflects off the wall y = 0 at (0.95167, 0), and its reflected leg ends at (0.96, 0.25) inside the obstacle, which
    crosses the absorbing wall: it is placed on the bottom face its leg entered, y = 0.2. Read from particle 0's start,
    its path would enter through the face x = 0.9, and it would land at (0.9, 0.25). (#2707: a step that stays in the
    box no longer reaches this indexing, since it is placed before the boundary condition.)"""
    problem, _ = _problem(2, _absorbing_at(2))
    solver = FPParticleSolver(
        problem, num_particles=2, implicit_domain=_DOMAIN[2], seed=0, boundary_conditions=_absorbing_at(2)
    )
    previous = np.array([[0.85, 0.85], [0.95, 0.05]])
    raw = np.array([[1.2, 0.95], [0.96, -0.25]])
    recorded, absorbed, n_absorbed = solver._finish_step(raw, previous, [(0.0, 1.0), (0.0, 1.0)], True)
    assert n_absorbed == 1, n_absorbed
    assert absorbed.tolist() == [True, False], absorbed
    np.testing.assert_allclose(recorded, [[0.96, 0.2 - 1e-10]], rtol=0, atol=1e-12)


_BOX = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))
# Review 1's barrier: A and B meet at a concave corner beside C (#2732).
_A = Hyperrectangle(np.array([[0.40, 0.70], [-0.1, 0.45]]))
_B = Hyperrectangle(np.array([[0.40, 0.45], [0.40, 1.1]]))
_C = Hyperrectangle(np.array([[0.55, 0.70], [0.44, 0.65]]))
# Review 2's two rooms: a union as the domain, not connected (#2732).
_ROOMS = UnionDomain(
    [Hyperrectangle(np.array([[0.0, 0.5], [0.0, 1.0]])), Hyperrectangle(np.array([[0.52, 1.0], [0.0, 1.0]]))]
)
_FIRST_BOUNDARY = {
    # name: (domain, the absorbing wall or None, previous, raw, recorded or None if absorbed)
    "a convex arc crossed: the projection, which the path reaches": (
        DifferenceDomain(_BOX, Hypersphere(center=np.array([0.5, 0.5]), radius=0.2)),
        None,
        [0.2, 0.5],
        [0.35, 0.52],
        np.array([0.5, 0.5]) + (0.2 + 1e-10) * np.array([-0.15, 0.02]) / np.hypot(0.15, 0.02),
    ),
    "sliding along a disc from its face: the projection": (
        DifferenceDomain(_BOX, Hypersphere(center=np.array([0.5, 0.5]), radius=0.2)),
        None,
        [0.3 - 1e-10, 0.5],
        [0.305, 0.51],
        np.array([0.5, 0.5]) + (0.2 + 1e-10) * np.array([-0.195, 0.01]) / np.hypot(0.195, 0.01),
    ),
    "through a thin obstacle into a second: the first one's near face": (
        DifferenceDomain(
            _BOX,
            UnionDomain(
                [
                    Hyperrectangle(np.array([[0.5, 0.52], [0.2, 0.8]])),
                    Hyperrectangle(np.array([[0.6, 0.9], [0.2, 0.8]])),
                ]
            ),
        ),
        None,
        [0.4, 0.5],
        [0.65, 0.5],
        [0.5 - 1e-10, 0.5],
    ),
    "1-D, past the middle: the near face, not the far one": (
        DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0]])), Hyperrectangle(np.array([[0.4, 0.6]]))),
        None,
        [0.3],
        [0.55],
        [0.4 - 1e-10],
    ),
    "entered by one face of a rectangle, ending nearer another: the face it entered": (
        _DOMAIN[2],
        None,
        [0.85, 0.77],
        [0.94, 0.77],
        [0.9 - 1e-10, 0.77],
    ),
    "a thin slab jumped over: the near face": (
        DifferenceDomain(_BOX, Hyperrectangle(np.array([[0.5, 0.52], [0.2, 0.8]]))),
        None,
        [0.4, 0.5],
        [0.6, 0.5],
        [0.5 - 1e-10, 0.5],
    ),
    "an obstacle before an absorbing wall: placed, not absorbed": (
        DifferenceDomain(_BOX, Hyperrectangle(np.array([[0.9, 0.95], [0.2, 0.8]]))),
        "x_max",
        [0.85, 0.5],
        [1.1, 0.5],
        [0.9 - 1e-10, 0.5],
    ),
    "an obstacle flush with an absorbing wall, met first: its corner with the wall, not absorbed": (
        DifferenceDomain(_BOX, Hyperrectangle(np.array([[0.9, 1.0], [0.2, 0.8]]))),
        "x_max",
        [0.9547534651728411, 0.8562972391037148],
        [1.0275983976791534, 0.7639797123595877],
        [1.0 - 1e-10, 0.8 + 1e-10],
    ),
    "a slab the whole step's scan passes over, before an absorbing wall: absorbed (#2710)": (
        DifferenceDomain(_BOX, Hyperrectangle(np.array([[0.92, 0.935], [0.2, 0.8]]))),
        "x_max",
        [0.5, 0.5],
        [1.5, 0.5],
        None,
    ),
    "an obstacle crossing an absorbing wall, entered by its inner face, the end beyond the wall: that face": (
        _DOMAIN[2],
        "x_max",
        [0.85, 0.75],
        [1.2, 0.84],
        [0.9 - 1e-10, 0.84],
    ),
    "an obstacle flush with the wall x = 0, the end beyond the wall: the face entered, not the wall": (
        DifferenceDomain(_BOX, Hyperrectangle(np.array([[0.0, 0.1], [0.2, 0.8]]))),
        None,
        [0.12, 0.5],
        [-0.03, 0.5],
        [0.1 + 1e-10, 0.5],
    ),
    "a union's concave corner, components A, B, C: the face entered": (
        DifferenceDomain(_BOX, UnionDomain([_A, _B, _C])),
        None,
        [0.35, 0.40],
        [0.56, 0.447],
        [0.4 - 1e-10, 0.447],
    ),
    "a union's concave corner, components C, A, B: the face entered": (
        DifferenceDomain(_BOX, UnionDomain([_C, _A, _B])),
        None,
        [0.35, 0.40],
        [0.56, 0.447],
        [0.4 - 1e-10, 0.447],
    ),
    "a reflected step that starts on the wall, into an obstacle: the face its reflected leg entered": (
        DifferenceDomain(_BOX, Hyperrectangle(np.array([[0.05, 0.15], [0.2, 0.8]]))),
        None,
        [0.0, 0.5],
        [-0.13, 0.52],
        [0.05 - 1e-10, 0.52],
    ),
    "a reflected leg clipping the corner of an obstacle flush with the next wall: the face it clipped": (
        DifferenceDomain(_BOX, Hyperrectangle(np.array([[0.5, 0.8], [0.9, 1.0]]))),
        None,
        [0.97, 0.75],
        [1.22355, 1.025],
        [0.8 + 1e-10, 0.975],
    ),
    "a union domain, a step from the first interval toward the second: the face of the one it left": (
        UnionDomain([Hyperrectangle(np.array([[0.0, 0.4]])), Hyperrectangle(np.array([[0.6, 1.0]]))]),
        None,
        [0.3],
        [0.55],
        [0.4 - 1e-10],
    ),
    "two rooms, a step toward the other: the face of the room it left": (
        _ROOMS,
        None,
        [0.49, 0.5],
        [0.515, 0.5],
        [0.5 - 1e-10, 0.5],
    ),
    "a ring's hole stepped out of: the hole's face, not the ring's outer one": (
        DifferenceDomain(
            _BOX,
            DifferenceDomain(
                Hypersphere(center=np.array([0.5, 0.5]), radius=0.3),
                Hypersphere(center=np.array([0.5, 0.5]), radius=0.25),
            ),
        ),
        None,
        [0.7, 0.5],
        [0.79, 0.5],
        [0.75 - 1e-10, 0.5],
    ),
    "a reflected leg across a thin slab, ending inside the domain: the face it entered": (
        DifferenceDomain(_BOX, Hyperrectangle(np.array([[0.85, 0.87], [0.2, 0.8]]))),
        None,
        [0.95, 0.5],
        [1.2, 0.5],
        [0.87 + 1e-10, 0.5],
    ),
    "the wall first, then a reflected leg past an obstacle's middle: the near face": (
        DifferenceDomain(_BOX, Hyperrectangle(np.array([[0.85, 0.95], [0.2, 0.8]]))),
        None,
        [0.97, 0.5],
        [1.12, 0.5],
        [0.95 + 1e-10, 0.5],
    ),
    "an absorbing wall met first, its hit rounding past the wall: absorbed": (
        DifferenceDomain(
            Hyperrectangle(np.array([[0.1, 0.7], [-0.3, 1.9]])), Hyperrectangle(np.array([[0.5, 0.55], [0.5, 1.0]]))
        ),
        "x_min",
        [0.3681010452004989, 1.629651317640937],
        [-0.2879055071371419, 1.593352815202474],
        None,
        [(0.1, 0.7), (-0.3, 1.9)],
    ),
}


_ARMS = DifferenceDomain(
    _BOX,
    UnionDomain(
        [
            Hyperrectangle(np.array([[0.8, 1.05], [0.2, 0.45]])),
            Hyperrectangle(np.array([[0.9, 1.05], [0.45, 0.9]])),
            Hyperrectangle(np.array([[0.3, 0.32], [0.1, 0.9]])),
        ]
    ),
)


@pytest.mark.parametrize(
    "case", ["toward the seam", "beyond the wall at the seam"], ids=["toward the seam", "beyond the wall at the seam"]
)
def test_a_step_into_two_touching_components_lands_at_their_concave_corner(case):
    """#2732 review 1: the arms [0.8, 1.05] x [0.2, 0.45] and [0.9, 1.05] x [0.45, 0.9] touch along y = 0.45, where
    the union's phi is 0 inside the closed obstacle. Each step enters the lower arm and lands at the arms' concave
    corner (0.9, 0.45), to the 1e-10 of each face, with a blended normal at the upper arm's own corner (#2698):
    - toward the seam: the projection stayed on the seam, where phi's gradient is 0, and read as placed. A degenerate
      normal now counts as not placed;
    - beyond the wall at the seam: the projection is the wall's point of the seam, (1 - 1e-10, 0.45), phi = 0, which
      the normals' test alone kept. The end is outside the arm the path entered, so rule (a) refuses it."""
    previous, end = {
        "toward the seam": ([0.7828459091496743, 0.5135627060726424], [0.9249879579933417, 0.39948193165556767]),
        "beyond the wall at the seam": (
            [0.8153611900522864, 0.45842686791332643],
            [1.1074949068892654, 0.41584568028921276],
        ),
    }[case]
    landed = _ARMS.place_step(np.array([previous]), np.array([end]))
    np.testing.assert_allclose(landed, [[0.9, 0.45]], rtol=0, atol=1e-9)
    assert float(np.atleast_1d(_ARMS.signed_distance(landed))[0]) < 0, landed


def test_a_step_through_a_lens_lands_by_the_disc_it_entered():
    """#2732 review 1: a step enters the lens of the discs centred (0.55, 0.5) and (0.4, 0.5), radius 0.1, through
    the right disc and ends in the left one. The end's projection is on the left disc's far side, which the normals'
    test alone kept; the end is outside the disc the path entered, so rule (a) refuses it, and the face map places it
    on the tangent half-space at the entry, in closed form: the end minus its depth past that plane along the plane's
    normal, then 1e-10 out."""
    centre, radius = np.array([0.55, 0.5]), 0.1
    lens = DifferenceDomain(
        _BOX,
        UnionDomain([Hypersphere(center=np.array([0.4, 0.5]), radius=0.1), Hypersphere(center=centre, radius=radius)]),
    )
    previous = np.array([0.7592528057332772, 0.39503817891560256])
    end = np.array([0.3695070973761362, 0.4293527663288494])
    direction = end - previous
    # The entry: the segment's first point at distance r from the right disc's centre.
    a, b, c = (
        direction @ direction,
        2 * direction @ (previous - centre),
        (previous - centre) @ (previous - centre) - radius**2,
    )
    entry = previous + (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a) * direction
    inward = (centre - entry) / radius  # the domain's outward normal there points into the disc
    expected = end - ((end - entry) @ inward + 1e-10) * inward
    np.testing.assert_allclose(lens.place_step(previous[None, :], end[None, :]), [expected], rtol=0, atol=1e-9)


def test_a_step_from_outside_the_domain_is_refused_on_every_branch():
    """#2732 review 1: `place_step` refuses a ``previous`` outside the domain on a step that leaves it, and only the
    face map's branch did. The step from (0.5, 0.31) to (0.5, 0.33), both inside the disc obstacle, has an end the
    projection places, and it is refused. The control, from (0.5, 0.25) inside the domain to the same end, lands at the
    disc's face 1e-10 out."""
    disc = DifferenceDomain(_BOX, Hypersphere(center=np.array([0.5, 0.5]), radius=0.2))
    end = np.array([[0.5, 0.33]])
    with pytest.raises(ValueError, match=r"previous position \[0\.5 +0\.31\] is not inside the domain"):
        disc.place_step(np.array([[0.5, 0.31]]), end)
    np.testing.assert_allclose(disc.place_step(np.array([[0.5, 0.25]]), end), [[0.5, 0.3 - 1e-10]], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("previous", "end"),
    [([0.45, 0.05], [0.6, -0.05]), ([0.49, 5e-7], [0.53, 2e-7])],
    ids=["past the corner", "crossing within 1e-6 of the corner"],
)
def test_a_step_past_a_rooms_corner_lands_in_that_room_at_its_nearest_point(previous, end):
    """#2732 review 2: a step leaves the left room of `_ROOMS` beside its corner (0.5, 0). Its nearest point in that
    room is the clamp of the end to the room: the corner, and (0.5, 2e-7). The projection onto the domain put the
    first in the other room, at (0.6, 1e-10); the second raised in the face map, whose crossing within 1e-6 of the
    corner read the corner's blended normal. Each lands within 1e-9 of the clamp, inside the room it left."""
    landed = _ROOMS.place_step(np.array([previous]), np.array([end]))
    clamp = np.clip(end, [0.0, 0.0], [0.5, 1.0])
    np.testing.assert_allclose(landed, [clamp], rtol=0, atol=1e-9)
    assert landed[0, 0] < 0.5, landed
    assert float(np.atleast_1d(_ROOMS.signed_distance(landed))[0]) < 0, landed


@pytest.mark.parametrize(
    ("case", "navigable"),
    [
        ("the point where an obstacle meets the wall it is flush with", False),
        ("the seam between two touching obstacles", False),
        ("the seam between two touching rooms", True),
        ("a corner of the box", True),
    ],
)
def test_a_particle_starts_only_in_the_closure_of_the_domains_interior(case, navigable):
    """#2732 review 2: `ImplicitDomain.navigable`, the regularized set, is where a particle may start. The first two
    have phi = 0, so `contains` holds them, and no interior point is next to them: a step from one names no face it
    crossed, and the face map raised on it. They are refused at t = 0, by name. The last two are the permission half:
    phi = 0 with interior next to them, so they start; they started under `contains` too, and pass there by design."""
    box = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))
    domain, point = {
        "the point where an obstacle meets the wall it is flush with": (
            DifferenceDomain(box, Hyperrectangle(np.array([[0.0, 0.1], [0.2, 0.8]]))),
            [0.0, 0.5],
        ),
        "the seam between two touching obstacles": (
            DifferenceDomain(
                box,
                UnionDomain(
                    [
                        Hyperrectangle(np.array([[0.3, 0.5], [0.3, 0.7]])),
                        Hyperrectangle(np.array([[0.5, 0.7], [0.3, 0.7]])),
                    ]
                ),
            ),
            [0.5, 0.5],
        ),
        "the seam between two touching rooms": (
            UnionDomain(
                [Hyperrectangle(np.array([[0.0, 0.5], [0.0, 1.0]])), Hyperrectangle(np.array([[0.5, 1.0], [0.0, 1.0]]))]
            ),
            [0.5, 0.5],
        ),
        "a corner of the box": (DifferenceDomain(box, Hyperrectangle(np.array([[0.4, 0.6], [0.4, 0.6]]))), [0.0, 0.0]),
    }[case]
    assert bool(np.atleast_1d(domain.contains(np.array([point])))[0])
    problem, _ = _problem(2, no_flux_bc(dimension=2))
    solver = FPParticleSolver(problem, num_particles=_N, implicit_domain=domain, seed=2688)
    start = np.random.default_rng(2688).uniform([0.05, 0.85], [0.25, 0.95], (_N, 2))
    start[:3] = point
    kwargs = {"initial_particles": start, "drift_field": _drift(2, 21, "callable"), "show_progress": False}
    if navigable:
        solver.solve_fp_system(drift_needs_density=False, **kwargs)
        return
    with pytest.raises(ValueError, match=r"3 of 600 are not where a particle can start.*ImplicitDomain\.navigable"):
        solver.solve_fp_system(drift_needs_density=False, **kwargs)


@pytest.mark.parametrize("case", list(_FIRST_BOUNDARY))
def test_the_first_boundary_a_step_meets_decides(case):
    """#2707: the step from ``previous`` to ``raw`` lands by the first boundary it meets, in closed form.

    - The arc: the end is inside the disc, short of its centre; the segment from where the step entered to the end
      stays in the disc, and the normals there and at the radial projection agree, so the projection stands.
    - Sliding along a disc: the step starts on the disc's face and ends 0.005 inside it; the projection stands, the
      radial one. A rule that asked the straight path from the start to the projection to stay outside the disc
      rejected it, since that chord dips into the disc, and placed the step on the tangent line instead.
    - Through a thin obstacle into a second: the end is in the second obstacle, whose normal at the projection agrees
      with the first's at the entry, but the segment from the entry to the end leaves the first obstacle; the first
      one's near face, x = 0.5, places it.
    - 1-D: the end 0.55 is past the middle of [0.4, 0.6], so its projection is 0.6, whose normal is opposite the
      entry's; the face it entered, 0.4, places it.
    - A rectangle's corner: the step enters [0.9, 1.05] x [0.2, 0.8] through its left face and ends at (0.94, 0.77),
      nearer the top face, where the projection is. The normals at the entry and at the projection are orthogonal, so
      the left face places it (the base recorded the top face's (0.94, 0.8 + 1e-10)).
    - The slab: the end is inside the domain, past a slab of width 0.02; it lands on the face it met, x = 0.5.
    - Before an absorbing wall: the step meets the obstacle at x = 0.9 before the wall; it is placed there and not
      absorbed.
    - Flush with an absorbing wall: the step meets the obstacle's top face before the wall and lands at their corner,
      1e-10 from the wall. The boundary condition does not act on it: its applicator absorbs within 1e-10 of a
      Dirichlet wall (`ParticleApplicator`).
    - A slab before an absorbing wall, thinner than the whole step's scan step: the scan of the step from 0.5 to 1.5
      samples x = 0.906 and 0.9375 and passes the slab [0.92, 0.935] over (#2710), so the step reaches the wall and
      is absorbed. The solver's search and `place_step`'s are one: a search of the segment cut at the wall alone
      (step 0.5 / 32) saw the slab, and `place_step`, which did not, placed the step at the wall, unabsorbed.
    - Review 1 of #2732, three far-side landings the first version of the rule kept, each now on the face entered:
      - an obstacle crossing the absorbing wall x = 1, entered through its left face, the end beyond the wall: the end
        was outside the domain all the way, through the obstacle and past the wall, and the wall's normal agreed with
        the face's, so it landed at the wall, unabsorbed. The rule now reads the leaf the path entered, the obstacle,
        and the end is outside it;
      - an obstacle flush with x = 0, where phi's central difference at the seam is exactly 0: the projection stayed
        on the wall;
      - a union's concave corner, where the corner owner's tie between A and C went to the first listed: the rule
        reads the entered leaf's own normal, in either order.
    - Review 2 of #2732, three landings on a surface the path never met, where the domain is the leaf's inside:
      the projection onto the domain reached the second interval of a union, the other room, and the ring's outer
      face 0.8 from its hole. The leaf's far side holds those, so the end was on it and the normals agreed; the
      projection is now the leaf's own.
    - A reflected leg across a thin slab that ends inside the domain beyond it: placed by the face it entered. The
      reflected set is every survivor whose raw step left the box; read as the survivors that ended outside the
      domain, it misses this leg, and the step is recorded at its end, 0.8, past the slab (#2732 review 2).
    - A reflected step that starts exactly on the wall x = 0: its reflected leg is its chord, and it is placed though
      the wall hit at its start lies outside the step.
    - A reflected leg that clips the corner of an obstacle flush with the top wall just before it hits that wall: the
      clip is shorter than the leg's scan step, and only the sample 1e-9 of the leg before its end sees it (#2706).
    - The wall first: the step hits x = 1 and its reflected leg ends at 0.88, past the middle 0.9 of [0.85, 0.95]; the
      projection is on the far face 0.85, whose normal is opposite the entry's, so the face the leg entered, 0.95,
      places it.
    - The absorbing wall x = 0.1 of the box (0.1, 0.7) x (-0.3, 1.9), met first: the step's first crossing, bisected,
      lies before the wall's hit as computed, by rounding, so compared with the wall itself it reads as meeting the
      domain's boundary first and would be placed at the wall. The cut 1e-9 of the box short of the wall leaves it to
      the absorbing condition (#1064). With no margin, 1,656 of 100,000 such steps through that wall read this way,
      and none with it (`i2707/pins/probe_row6c.py`); on [0, 1]^2 none do (`probe_margin3.py`)."""
    domain, wall, previous, raw, expected, *box = _FIRST_BOUNDARY[case]
    d = len(previous)
    bounds = box[0] if box else [(0.0, 1.0)] * d
    bc = no_flux_bc(dimension=d) if wall is None else _absorbing_at(d, wall)
    problem, _ = _problem(d, bc, bounds)
    solver = FPParticleSolver(problem, num_particles=1, implicit_domain=domain, seed=0, boundary_conditions=bc)
    recorded, _, n_absorbed = solver._finish_step(
        np.array([raw], dtype=float), np.array([previous], dtype=float), bounds, wall is not None
    )
    if expected is None:
        assert n_absorbed == 1, recorded
        assert recorded.shape == (0, d), recorded
        return
    assert n_absorbed == 0, n_absorbed
    np.testing.assert_allclose(recorded, [expected], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "case",
    [
        "periodic, n-D grid",
        "periodic, callable",
        "initial_particles inside the obstacle",
        "M_initial inside the obstacle, n-D",
        "M_initial inside the obstacle, 1-D nodes",
        "an implicit_domain box short of the grid's",
        "an implicit_domain box beyond the grid's",
        "a periodic face on one side only",
        "GPU path",
    ],
)
def test_what_the_placement_cannot_do_is_refused(case):
    """A periodic face (a wrapped step is not a segment, #2704), initial mass inside an obstacle (moving it changes
    the caller's measure), an implicit_domain whose box is not the grid's, and the GPU path (its particles never meet
    the placement) are refused."""
    if case.startswith("periodic"):
        with pytest.raises(
            NotImplementedError, match=r"periodic condition on the outer boundary.*not supported.*#2704"
        ):
            _solve(2, periodic_bc(dimension=2), "grid" if "grid" in case else "callable")
    elif case == "a periodic face on one side only":
        # #2706 review 2: periodic on x_min alone reads as a bounded topology, yet the applicator wraps through it.
        one_sided = BoundaryConditions(
            segments=[
                BCSegment(name=b, bc_type=BCType.PERIODIC if b == "x_min" else BCType.NO_FLUX, value=0.0, boundary=b)
                for b in ("x_min", "x_max", "y_min", "y_max")
            ],
            dimension=2,
        )
        with pytest.raises(NotImplementedError, match=r"periodic condition on the outer boundary.*not supported"):
            _solve(2, one_sided, "callable", initial_particles=_navigable_start())
    elif case.startswith("initial_particles"):
        start = _navigable_start()
        start[:3] = [0.95, 0.5]
        with pytest.raises(
            ValueError, match=r"initial_particles, 3 of 600 are not where a particle can start \(inside an obstacle"
        ):
            _solve(2, no_flux_bc(dimension=2), "callable", initial_particles=start)
    elif case == "M_initial inside the obstacle, n-D":
        with pytest.raises(
            ValueError, match=r"cell\(s\) carrying mass have no navigable point.*Leaving it out would drop"
        ):
            _solve(2, no_flux_bc(dimension=2), "grid", centres=[(0.97, 0.5)])
        # The message's remedy, run as written: the density zeroed at the nodes inside the obstacles.
        problem, n = _problem(2, no_flux_bc(dimension=2))
        coordinates = [np.linspace(0.0, 1.0, n)] * 2
        density = _m_initial(2, n, [(0.97, 0.5)])
        nodes = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape(-1, len(coordinates))
        density = density * _DOMAIN[2].contains(nodes).reshape(density.shape)
        solver = FPParticleSolver(problem, num_particles=_N, implicit_domain=_DOMAIN[2], seed=2688)
        solver.solve_fp_system(
            M_initial=density, drift_field=_drift(2, n, "grid"), drift_is_precomputed=True, show_progress=False
        )
        assert _DOMAIN[2].contains(np.asarray(solver.M_particles_trajectory[0], dtype=float).reshape(-1, 2)).all()
    elif case == "M_initial inside the obstacle, 1-D nodes":
        with pytest.raises(
            ValueError,
            match=r"grid node\(s\) carrying m_initial's mass are not where a particle can start \(inside an obstacle.*Leaving it out would drop",
        ):
            _solve(1, no_flux_bc(dimension=1), "grid", centres=[0.5])
    elif case == "an implicit_domain box short of the grid's":
        # #2706 review 1, B1: the BC reflects at the grid's box, and the placement leaves to it whatever is outside
        # the domain's box, so a particle between the two would be recorded outside the domain.
        short = DifferenceDomain(
            Hyperrectangle(np.array([[0.0, 0.8], [0.0, 1.0]])), Hyperrectangle(np.array([[0.3, 0.4], [0.3, 0.4]]))
        )
        problem, _ = _problem(2, no_flux_bc(dimension=2))
        solver = FPParticleSolver(problem, num_particles=_N, implicit_domain=short, seed=0)
        with pytest.raises(ValueError, match=r"is not the grid's box"):
            solver.solve_fp_system(
                initial_particles=_navigable_start() * [0.9, 1.0],
                drift_field=_drift(2, 21, "callable"),
                drift_needs_density=False,
                show_progress=False,
            )
    elif case == "an implicit_domain box beyond the grid's":
        # #2706 review 2: the BC reflects at the grid's box first, and a domain whose box reaches past it lets the
        # projection put a particle on the obstacle's far face x = 1.05, beyond the wall. The advice -- the same
        # obstacle cut from the grid's box -- then records nothing outside the grid's box or the domain.
        obstacle = Hyperrectangle(np.array([[0.9, 1.05], [0.2, 0.8]]))
        wide = DifferenceDomain(Hyperrectangle(np.array([[-0.5, 1.5], [-0.5, 1.5]])), obstacle)
        problem, _ = _problem(2, no_flux_bc(dimension=2))
        drift = {"drift_field": lambda t, x, m: np.column_stack([np.full(len(x), 8.0), np.zeros(len(x))])}
        drift.update(drift_needs_density=False, show_progress=False)
        solver = FPParticleSolver(problem, num_particles=_N, implicit_domain=wide, seed=0)
        with pytest.raises(ValueError, match=r"is not the grid's box"):
            solver.solve_fp_system(initial_particles=_navigable_start(), **drift)
        advised = DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), obstacle)
        solver = FPParticleSolver(problem, num_particles=_N, implicit_domain=advised, seed=0)
        solver.solve_fp_system(initial_particles=_navigable_start(), **drift)
        recorded = np.concatenate([np.asarray(P, dtype=float).reshape(-1, 2) for P in solver.M_particles_trajectory])
        outside_the_box = np.any((recorded < 0.0) | (recorded > 1.0), axis=1)
        assert not outside_the_box.any(), recorded[outside_the_box][:3]
        assert advised.contains(recorded).all()
    else:
        pytest.importorskip("torch")
        problem, n = _problem(1, no_flux_bc(dimension=1))
        solver = FPParticleSolver(problem, num_particles=2000, implicit_domain=_DOMAIN[1], seed=2688, backend="torch")
        with pytest.raises(NotImplementedError, match=r"GPU path does not place particles"):
            solver.solve_fp_system(
                M_initial=_m_initial(1, n, [0.3]), drift_field=_drift(1, n, "grid"), show_progress=False
            )
        assert solver.current_strategy.name != "cpu", (
            "the selector sent this solve to the CPU path, so it tests nothing"
        )


def test_a_step_that_absorbs_every_particle_records_none():
    """Every particle of the step leaves at the absorbing wall, so the placement gets no survivors: shape (0, 2)."""
    problem, _ = _problem(2, _absorbing_at(2))
    solver = FPParticleSolver(
        problem, num_particles=2, implicit_domain=_DOMAIN[2], seed=0, boundary_conditions=_absorbing_at(2)
    )
    recorded, _absorbed, n_absorbed = solver._finish_step(
        np.array([[1.2, 0.95], [1.3, 0.1]]), np.array([[0.85, 0.95], [0.85, 0.1]]), [(0.0, 1.0)] * 2, True
    )
    assert recorded.shape == (0, 2), recorded.shape
    assert n_absorbed == 2, n_absorbed


def test_a_reflected_step_is_placed_by_the_face_its_path_met():
    """#2706 review 1, M3: a reflected step's path is broken at the wall, and the face it met is on that path.

    From (0.985, 0.95) the raw step to (1.015, 0.7) hits the wall at (1.0, 0.825) and its reflected leg enters O
    through O's top face at (0.997, 0.8). The chord from the start to the reflected end, (0.985, 0.7), crosses the
    small obstacle Q's top face instead, a face the particle never approached; placed by the chord it landed at
    (0.985, 0.82). Placed by the path, it is projected onto O's top face: (0.985, 0.8 + 1e-10)."""
    obstacles = UnionDomain(
        [Hyperrectangle(np.array([[0.9, 1.05], [0.2, 0.8]])), Hyperrectangle(np.array([[0.984, 0.99], [0.81, 0.82]]))]
    )
    domain = DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), obstacles)
    problem, _ = _problem(2, no_flux_bc(dimension=2))
    solver = FPParticleSolver(problem, num_particles=1, implicit_domain=domain, seed=0)
    recorded, _, _ = solver._finish_step(np.array([[1.015, 0.7]]), np.array([[0.985, 0.95]]), [(0.0, 1.0)] * 2, False)
    np.testing.assert_allclose(recorded, [[0.985, 0.8 + 1e-10]], rtol=0, atol=1e-12)


def test_a_reflected_step_that_clips_a_flush_obstacles_corner_is_placed_on_its_top_face():
    """#2706: the obstacle [0.9, 1.0] x [0.2, 0.8] is flush with the wall x = 1. This step clips the obstacle's corner
    just before it hits the wall at (1.0, 0.79896), which lies ON the obstacle's face, phi = 0. That excursion is
    shorter than a scan step, so it was missed, and the placement raised. Found by `i2688/m3/reflected_faces_code.py`
    on pop's flush rectangle. The first boundary the step meets is the top face, at x = 0.9992, before the wall
    (#2707), so the top face and then the wall place it at their corner, (1 - 1e-10, 0.8 + 1e-10). Before #2707 the
    wall reflected it first, to (2 - raw_x, 0.8 + 1e-10)."""
    flush = DifferenceDomain(
        Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), Hyperrectangle(np.array([[0.9, 1.0], [0.2, 0.8]]))
    )
    problem, _ = _problem(2, no_flux_bc(dimension=2))
    solver = FPParticleSolver(problem, num_particles=1, implicit_domain=flush, seed=0)
    raw = np.array([[1.0275983976791534, 0.7639797123595877]])
    recorded, _, _ = solver._finish_step(
        raw, np.array([[0.9547534651728411, 0.8562972391037148]]), [(0.0, 1.0)] * 2, False
    )
    np.testing.assert_allclose(recorded, [[1.0 - 1e-10, 0.8 + 1e-10]], rtol=0, atol=1e-12)


@pytest.mark.parametrize("path", ["grid", "callable"])
def test_every_placement_in_a_solve_reads_its_own_particles_start(path):
    """#2706 review 1, S1: in a solve, each particle that ends a step in the obstacle is placed from its own start.

    The drift is linear, 8 * ((1, 0.5) - x), so each step's end is known from its start up to noise of order 1e-7,
    and particles enter the crossing rectangle through its left, top and bottom faces. Each recorded position is
    checked against the placement owner's answer (`ImplicitDomain.place_step`) from that particle's own start. A start
    read from another particle names another face and moves the placement by O(0.1)."""
    nt, rate, target = 5, 8.0, np.array([1.0, 0.5])
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[21, 21], boundary_conditions=no_flux_bc(dimension=2))
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=1e-6
        ),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.5),
        Nt=nt,
    )
    solver = FPParticleSolver(
        problem, num_particles=_N, implicit_domain=_DOMAIN[2], seed=2706, boundary_conditions=no_flux_bc(dimension=2)
    )
    if path == "callable":
        rng = np.random.default_rng(2706)
        starts = np.vstack(
            [
                rng.uniform([0.86, 0.3], [0.89, 0.7], (_N // 3, 2)),
                rng.uniform([0.95, 0.85], [0.99, 0.95], (_N // 3, 2)),
                rng.uniform([0.95, 0.05], [0.99, 0.15], (_N - 2 * (_N // 3), 2)),
            ]
        )
        solver.solve_fp_system(
            initial_particles=starts,
            drift_field=lambda t, x, m: rate * (target - np.asarray(x)),
            drift_needs_density=False,
            show_progress=False,
        )
    else:
        xs = np.linspace(0.0, 1.0, 21)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        drift = np.zeros((nt + 1, 21, 21, 2))
        drift[..., 0], drift[..., 1] = rate * (1.0 - X), rate * (0.5 - Y)
        solver.solve_fp_system(
            M_initial=_m_initial(2, 21, [(0.85, 0.5), (0.95, 0.9), (0.95, 0.1)]),
            drift_field=drift,
            drift_is_precomputed=True,
            show_progress=False,
        )
    traj = [np.asarray(P, dtype=float).reshape(-1, 2) for P in solver.M_particles_trajectory]
    dt = problem.T / nt
    report: dict[str, int] = {}
    for t in range(len(traj) - 1):
        start = traj[t]
        expected = _DOMAIN[2].place_step(start, start + rate * (target - start) * dt, report=report)
        np.testing.assert_allclose(traj[t + 1], expected, rtol=0, atol=1e-4, err_msg=f"step {t}")
    assert report.get("face_of_entry", 0) > _N, f"the face map must act for many particles here: {report}"


def test_a_single_particle_steps_in_two_dimensions():
    """#2691: `reflect_positions`, `absorb_positions` and `wrap_positions` returned shape (d,) for a (1, d) row, and the
    step's placement then read a 2-D point as d points of dimension 1. A (1, d) row now comes back as (1, d)."""
    problem, _ = _problem(2, no_flux_bc(dimension=2))
    solver = FPParticleSolver(problem, num_particles=1, implicit_domain=_DOMAIN[2], seed=0, density_mode="query_only")
    solver.solve_fp_system(
        initial_particles=np.array([[0.85, 0.5]]),
        drift_field=_drift(2, 21, "callable"),
        drift_needs_density=False,
        show_progress=False,
    )
    for particles in solver.M_particles_trajectory:
        p = np.asarray(particles, dtype=float)
        assert p.reshape(-1, 2).shape == (1, 2), p.shape
        assert _DOMAIN[2].contains(p.reshape(-1, 2)).all(), p


def _interval_minus(lo, hi):
    return DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0]])), Hyperrectangle(np.array([[lo, hi]])))


_XS = np.linspace(0.0, 1.0, 11)  # spacing 0.1: the node 0.3 owns the cell [0.25, 0.35]


def test_a_jitter_in_an_obstacle_is_redrawn_within_its_cell():
    """t = 0 of #2688: the node 0.3 is navigable and its cell meets the obstacle at 0.32, so a jitter landing past it
    is redrawn -- within the same cell, so the cell keeps its mass: every sample lies in [0.25, 0.32)."""
    from mfgarchon.utils.numerical.particle import sample_from_density

    density = np.zeros(11)
    density[3] = 1.0
    x = sample_from_density(density, [_XS], 2000, seed=0, navigable=_interval_minus(0.32, 0.6).contains)[:, 0]
    assert np.all((x >= 0.25) & (x < 0.32)), (x.min(), x.max())
    assert np.ptp(x) > 0.05, "the redraw must spread over the cell's navigable part"


def test_a_cell_with_its_node_inside_but_a_navigable_part_is_sampled_there():
    """The node 0.3 is inside the obstacle [0.28, 0.6], but its cell's probe lattice holds 0.26, which is not: the
    cell has a navigable part, so its mass is sampled there rather than refused."""
    from mfgarchon.utils.numerical.particle import sample_from_density

    density = np.zeros(11)
    density[3] = 1.0
    x = sample_from_density(density, [_XS], 2000, seed=0, navigable=_interval_minus(0.28, 0.6).contains)[:, 0]
    assert np.all((x >= 0.25) & (x < 0.28)), (x.min(), x.max())


@pytest.mark.parametrize(("share", "refused"), [(1e-9, False), (1e-6, True)])
@pytest.mark.parametrize("route", ["sample_from_density", "1-D grid nodes"])
def test_mass_inside_an_obstacle_is_left_out_up_to_the_gates_threshold(route, share, refused):
    """#2706: mass inside the obstacle, at ``share`` of the total, goes to `mass_fabrication_gate.leave_out_or_raise`
    by both routes: the sampler's blocked cells (the cell of the node 0.6 lies wholly inside [0.45, 0.75]) and the
    1-D grid path's nodes (the node 0.5, inside [0.4, 0.6]). Above the gate's threshold (1e-8) the solve is refused,
    naming the share; at or below it the mass is left out and nothing is placed there."""
    from mfgarchon.utils.numerical.particle import sample_from_density

    message = r"Leaving it out would drop a fraction 1\.000e-06 of the total mass \(threshold 1e-08\)"
    if route == "1-D grid nodes":
        problem, n = _problem(1, no_flux_bc(dimension=1))
        m0 = _m_initial(1, n, [0.3])
        m0[50] = share / (1.0 - share) * m0.sum()  # the node 0.5 holds `share` of the total
        solver = FPParticleSolver(problem, num_particles=2000, implicit_domain=_DOMAIN[1], seed=2688)
        kwargs = {"M_initial": m0, "drift_field": _drift(1, n, "grid"), "show_progress": False}
        if refused:
            with pytest.raises(ValueError, match=message):
                solver.solve_fp_system(**kwargs)
            return
        solver.solve_fp_system(**kwargs)
        start = np.asarray(solver.M_particles_trajectory[0], dtype=float).reshape(-1, 1)
        assert _DOMAIN[1].contains(start).all()
        return

    domain = _interval_minus(0.45, 0.75)
    density = np.zeros(11)
    density[2], density[6] = 1.0 - share, share
    if refused:
        with pytest.raises(ValueError, match=message):
            sample_from_density(density, [_XS], 2000, seed=0, navigable=domain.contains)
        return
    x = sample_from_density(density, [_XS], 2000, seed=0, navigable=domain.contains)
    assert domain.contains(x).all()


@pytest.mark.parametrize("route", ["the gate", "the sampler's cells", "the 1-D grid's nodes"])
def test_the_gate_leaves_out_the_blocked_entries_and_renormalises_the_rest(route):
    """#2706 review 2: what the two leave-out routes hand to the draw. Pinned at the owner and at each caller's use of
    what the owner returns (review 3, S1), because no draw can see it: numpy's choice divides its cdf by the last entry
    itself, and Generator's accepts a deficit of 1.4e-8, above the gate's 1e-8; a cell holding 1e-8 of the mass is not
    drawn once in 200,000 draws (i2688/r2fix/t0/choice_renorm.py, numpy 2.5.3). The blocked entries are 0, the rest
    keep their ratios and sum to 1."""
    from mfgarchon.utils.numerical.mass_fabrication_gate import leave_out_or_raise
    from mfgarchon.utils.numerical.particle.sampling import _leave_out_cells_with_no_navigable_point

    if route == "the gate":
        probs = np.array([0.5, 0.3, 0.2 - 4e-9, 4e-9])
        blocked = np.array([False, False, False, True])
        kept = leave_out_or_raise(probs, blocked, context="test", why="", remedy="")
    elif route == "the sampler's cells":
        probs = np.zeros(11)  # the cell of the node 0.6 lies wholly inside the obstacle [0.45, 0.75]
        probs[2], probs[6] = 1.0 - 4e-9, 4e-9
        blocked = np.arange(11) == 6
        kept = _leave_out_cells_with_no_navigable_point(probs, [_XS], _interval_minus(0.45, 0.75).contains)
    else:
        problem, n = _problem(1, no_flux_bc(dimension=1))
        solver = FPParticleSolver(problem, num_particles=10, implicit_domain=_DOMAIN[1], seed=0)
        probs = np.zeros(n)  # the node 0.5 is inside the obstacle (0.4, 0.6)
        probs[30], probs[50] = 1.0 - 4e-9, 4e-9
        blocked = np.arange(n) == 50
        kept = solver._leave_out_nodes_inside_an_obstacle(np.linspace(0.0, 1.0, n), probs)
    assert (kept[blocked] == 0.0).all()
    assert kept.sum() == pytest.approx(1.0, abs=1e-15)
    unblocked = ~blocked & (probs > 0)
    np.testing.assert_allclose(kept[unblocked] / kept[unblocked][0], probs[unblocked] / probs[unblocked][0], rtol=1e-15)


def test_a_reflected_step_off_two_walls_is_placed_by_the_face_its_folded_path_met():
    """#2706 review 2: a step that hits the wall x = 1 and then y = 1. From (0.55, 0.57) to raw (1.15, 1.01): it hits
    x = 1 at (1, 0.90) and y = 1 at x = 1.13636, folded to 0.86364; it ends at the fold (0.85, 0.99), inside the
    obstacle O = [0.70, 0.95] x [0.93, 1.02], which meets the top wall. O's nearest face to that end is beyond the
    wall, so the face its path met places it. That is the second leg's: from (1, 0.90) toward (0.86364, 1) it enters
    O through the right face x = 0.95, at y = 0.93667; placed at (0.95 + 1e-10, 0.99). The hits taken in the other
    order, or the chord, enter through the bottom face (0.85, 0.93 - 1e-10); unfolded hits name the wall x = 1."""
    domain = DifferenceDomain(
        Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), Hyperrectangle(np.array([[0.70, 0.95], [0.93, 1.02]]))
    )
    problem, _ = _problem(2, no_flux_bc(dimension=2))
    solver = FPParticleSolver(problem, num_particles=1, implicit_domain=domain, seed=0)
    raw = np.array([[1.15, 1.01]])
    recorded, _, _ = solver._finish_step(raw, np.array([[0.55, 0.57]]), [(0.0, 1.0)] * 2, False)
    np.testing.assert_allclose(recorded, [[0.95 + 1e-10, 2.0 - raw[0, 1]]], rtol=0, atol=1e-12)


def test_two_faces_meeting_at_a_tiny_angle_still_place_at_their_apex():
    """#2706 review 1, S3: a point projected onto two faces through the origin with outward normals (cos a, +-sin a),
    half-angle a = 3e-8. The nearest point of their intersection is the apex, the origin. The pair's Gram determinant
    is below the solver's 1e-14, so the single-face projection stands in, 3e-8 from the apex, and it satisfies the
    other face to within a slack relative to the problem's scale. With an absolute 1e-15 slack it was refused (None),
    and the particle raised."""
    from mfgarchon.geometry.implicit.implicit_domain import _project_onto_half_spaces

    a = 3e-8
    normals = np.array([[np.cos(a), np.sin(a)], [np.cos(a), -np.sin(a)]])
    placed = _project_onto_half_spaces(np.array([1.0, 0.0]), np.zeros((2, 2)), normals, 0.0)
    assert placed is not None
    assert np.linalg.norm(placed) <= 2 * a, placed

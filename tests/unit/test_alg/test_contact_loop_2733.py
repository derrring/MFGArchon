"""#2733 PR 2: the particle FP ends every step at its first contacts with the particles' boundary
(`reflect_at_first_contacts`), and refuses what that loop cannot read. Admission: oracles. Each row is a closed form,
an exact law, or a measured band; none is the code's output read back.

- The loop: specular reflection at each contact in path order, a box contact reading the boundary owner and every
  other contact reflecting (D2's box rule); ties, the zero-width corridor's rule, the cap and the offset (D4-D6).
- The refusals: an absorbing condition the owner can place off the box (`_refuse_absorbing_off_the_box`); a box-face
  exit no particle can cross (B1e-2's check on the particles' domain, `ImplicitDomain.face_trace`); a step left with
  no direction along a corridor.
- The oracles: #2707's table, the ring of B1, the box fold bitwise, the method-of-images law at a flat wall, the Gibbs
  density with a reflecting obstacle, the 3-D room's shadow, and the exact skip.

What reddens this file: a contact read off the box, a wrong normal, a reflection of the remaining vector instead of
the end (the bitwise fold), a corridor rule that keeps the normal part or traps a 1-D particle, a skip that is not
exact, a refusal widened or narrowed, an owner read around `segment_at_box_contact`.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import reflect_at_first_contacts
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc
from mfgarchon.geometry.boundary.corner import reflect_positions
from mfgarchon.geometry.implicit import DifferenceDomain, Hyperrectangle, Hypersphere, IntersectionDomain, UnionDomain

D, NF = BCType.DIRICHLET, BCType.NO_FLUX
_BOX = np.array([[0.0, 1.0], [0.0, 1.0]])


def _square():
    return Hyperrectangle(_BOX)


def _model(sigma=0.05):
    return Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=sigma
    )


def _problem(domain, sigma=0.05, T=0.5, nt=20):
    return MFGProblem(
        model=_model(sigma),
        domain=domain,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=T),
        Nt=nt,
    )


def _grid(d=2, n=11):
    return TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[n] * d, boundary_conditions=no_flux_bc(dimension=d))


def _exit_bc(d=2, **placement):
    name = "right" if d == 1 else "x_max"
    return BoundaryConditions(
        segments=[
            BCSegment(name="exit", bc_type=D, value=0.0, boundary=name, **placement),
            BCSegment(name="walls", bc_type=NF),
        ],
        dimension=d,
    )


def _step(domain, a, b, bc=None, box=_BOX):
    return reflect_at_first_contacts(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        domain,
        box,
        no_flux_bc(dimension=box.shape[0]) if bc is None else bc,
    )


# ---- The refusal of an absorbing condition the owner places off the box (ruled 2026-10-22) -------------------------

_DISC = Hypersphere(center=[0.5, 0.5], radius=0.4)
_OFF_BOX = r"an absorbing condition is placed where it reaches the particles' boundary off the grid's box"


def _disc_bc(segment):
    return BoundaryConditions(
        segments=[segment], dimension=2, domain_bounds=_BOX, default_bc=NF, domain_sdf=_DISC.signed_distance
    )


@pytest.mark.parametrize(
    "segment",
    [
        BCSegment(name="exit", bc_type=D, value=0.0, normal_direction=np.array([1.0, 0.0]), normal_tolerance=0.5),
        BCSegment(name="exit", bc_type=D, value=0.0, sdf_region=lambda x: float(0.75 - np.asarray(x)[0])),
    ],
    ids=["normal_direction under domain_sdf", "sdf_region"],
)
def test_an_exit_on_a_discs_arc_is_refused_whatever_the_drift(segment):
    """At main the arc exit absorbed 400 of 400 toward (1.2, 0.5), through the box reader at the tangent point, and 0
    of 400 toward (0.95, 1.05) with 252 particles on the arc inside the exit (#2733's folded finding). Construction knows
    no drift, so both placements are refused; reading the owner on the arc is #2789."""
    with pytest.raises(ValueError, match=_OFF_BOX + r".*#2789"):
        FPParticleSolver(_problem(_DISC), num_particles=50, seed=1, boundary_conditions=_disc_bc(segment))


@pytest.mark.parametrize(
    ("label", "geometry", "implicit", "bc"),
    [
        ("a box exit on x_max", _grid(), None, _exit_bc()),
        (
            "an obstacle with an exit by boundary=",
            _grid(),
            DifferenceDomain(_square(), Hypersphere(center=[0.5, 0.5], radius=0.2)),
            _exit_bc(),
        ),
        (
            "an all-reflecting BC on the disc",
            _DISC,
            None,
            _disc_bc(BCSegment(name="wall", bc_type=NF, normal_direction=np.array([1.0, 0.0]), normal_tolerance=0.5)),
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_what_the_owner_answers_on_the_box_alone_is_admitted(label, geometry, implicit, bc):
    """The control: a segment placed on one box face by name is never answered off the box (`BCSegment.matches_point`
    returns False with no face), and a BC that absorbs nowhere places nothing. Each builds and steps."""
    kw = {} if implicit is None else {"implicit_domain": implicit}
    solver = FPParticleSolver(_problem(geometry, T=0.1, nt=2), num_particles=50, seed=1, boundary_conditions=bc, **kw)
    start = np.c_[np.full(50, 0.15), np.linspace(0.45, 0.55, 50)]
    solver.solve_fp_system(
        initial_particles=start, drift_field=lambda t, x, m: np.zeros_like(np.asarray(x)), drift_needs_density=False
    )


@pytest.mark.parametrize(
    "bc",
    [
        dirichlet_bc(dimension=2, value=0.0),
        BoundaryConditions(segments=[], dimension=2, default_bc=D, domain_bounds=_BOX),
    ],
    ids=["dirichlet_bc everywhere", "default_bc=DIRICHLET"],
)
def test_absorbing_everywhere_beside_an_obstacle_is_refused_and_its_advice_runs(bc):
    """Ruled 2026-10-22, the over-refusal kept: at main this BC ran, absorbed at the box and kept every particle out
    of the obstacle, 305 of 400 absorbed. The owner answers it on the obstacle's surface, where the loop reflects, so it
    is refused, naming the spelling and the remedy; the remedy, one exit per box face and a reflecting default, runs."""
    obstacle = DifferenceDomain(_square(), Hypersphere(center=[0.6, 0.5], radius=0.15))
    with pytest.raises(ValueError, match=_OFF_BOX + r".*dirichlet_bc\(\.\.\.\) or default_bc=DIRICHLET.*#2789"):
        FPParticleSolver(_problem(_grid()), num_particles=50, seed=1, boundary_conditions=bc, implicit_domain=obstacle)
    faces = ["x_min", "x_max", "y_min", "y_max"]
    advised = BoundaryConditions(
        segments=[BCSegment(name=f, bc_type=D, value=0.0, boundary=f) for f in faces], dimension=2, default_bc=NF
    )
    solver = FPParticleSolver(
        _problem(_grid()), num_particles=400, seed=1, boundary_conditions=advised, implicit_domain=obstacle
    )
    start = np.c_[np.full(400, 0.2), np.random.default_rng(3).uniform(0.4, 0.6, 400)]
    solver.solve_fp_system(
        initial_particles=start,
        drift_field=lambda t, x, m: 3.0 * (np.array([1.3, 0.5]) - np.asarray(x)),
        drift_needs_density=False,
        show_progress=False,
    )
    assert solver.total_absorbed > 0


# ---- B1e-2 on the particles' domain: an exit no particle can cross is refused (ruled 2026-10-22) --------------------

_FLUSH = DifferenceDomain(_square(), Hyperrectangle(np.array([[0.8, 1.0], [0.4, 0.6]])))
_NOT_CROSSED = r"govern only face cells no particle can cross: the particles' domain \(\w+\) has no boundary there"


@pytest.mark.parametrize(
    ("label", "d", "geometry", "implicit", "placement", "admitted"),
    [
        ("a disc with boundary=x_max", 2, _DISC, None, {}, False),
        ("the box with boundary=x_max", 2, _grid(), None, {}, True),
        ("a flush obstacle, an exit on its corridor", 2, _grid(), _FLUSH, {"region": {"y": (0.4, 0.6)}}, False),
        ("a flush obstacle, a whole-face exit", 2, _grid(), _FLUSH, {}, True),
        (
            "a sphere of radius 100 tangent to x_max, the domain inside it",
            2,
            _grid(),
            IntersectionDomain([_square(), Hypersphere(center=[1.0 - 100.0, 0.5], radius=100.0)]),
            {},
            False,
        ),
        (
            "1-D: x = 1, an isolated point",
            1,
            _grid(1, 21),
            DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0]])), Hyperrectangle(np.array([[0.8, 1.0]]))),
            {},
            False,
        ),
        (
            "3-D: a room, a whole-face exit",
            3,
            _grid(3),
            DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0]] * 3)), Hyperrectangle(np.array([[0.4, 0.6]] * 3))),
            {},
            True,
        ),
        (
            "3-D: an exit on a flush obstacle's footprint",
            3,
            _grid(3),
            DifferenceDomain(
                Hyperrectangle(np.array([[0.0, 1.0]] * 3)),
                Hyperrectangle(np.array([[0.8, 1.0], [0.4, 0.6], [0.4, 0.6]])),
            ),
            {"region": {"y": (0.4, 0.6), "z": (0.4, 0.6)}},
            False,
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_box_face_exit_is_read_only_where_particles_can_cross_it(label, d, geometry, implicit, placement, admitted):
    """A face cell counts where the domain's trace on it (`ImplicitDomain.face_trace`, read 1e-12 of the diagonal
    inside) has a piece at least 2e-4 of the diagonal long: exact in 1-D and 2-D, a midpoint-line rule above. At main a
    `boundary="x_max"` exit on the disc absorbed 400 of 400 on axis, at the box reader's tangent point; the loop reflects
    at the arc first and absorbed none, so the head refuses it. The residuals: a ball of radius 5000 diagonals or more,
    tangent with the domain inside, is admitted and unread; a crossable piece shorter than 2e-4 of the diagonal is
    refused. The radius-100 sphere's trace is 3.4e-5 of the face, below 2.8e-4."""
    kw = {} if implicit is None else {"implicit_domain": implicit}
    build = lambda: FPParticleSolver(  # noqa: E731
        _problem(geometry, T=0.1, nt=2), num_particles=20, seed=0, boundary_conditions=_exit_bc(d, **placement), **kw
    )
    if admitted:
        build()
    else:
        with pytest.raises(ValueError, match=r"'exit' on (x_max|right).*" + _NOT_CROSSED + r".*#2789"):
            build()


#: The boundary owner's point readers: what a function calls to ask which segment governs a point.
_OWNER_POINT_READERS = {"get_bc_at_point", "face_segments_at_point", "segment_for_point", "segment_for_face"}


def _calls(function) -> dict[str, list[int]]:
    """The names ``function``'s body calls, as an attribute or bare, with their lines."""
    import ast
    import inspect
    import textwrap

    called: dict[str, list[int]] = {}
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(function)))):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            called.setdefault(name, []).append(node.lineno)
    return called


def test_the_face_check_reads_each_cell_as_the_loop_reads_a_contact():
    """B1e-2's check and the loop reach the owner only through `segment_at_box_contact`. On a face cell the first-axis
    rule already names the face the point lies on, so no behavioural row tells a direct owner read from it: the pin is
    structural, as #2788's FP-FDM row. The control is the function itself, whose own owner reads the walk must see."""
    from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import reflect_at_first_contacts as loop
    from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator, segment_at_box_contact

    control = _calls(segment_at_box_contact)
    assert {"face_segments_at_point", "get_bc_at_point"} <= control.keys(), f"the walk sees no owner read: {control}"
    for reader in (ParticleApplicator.refuse_segments_it_would_drop, loop):
        called = _calls(reader)
        assert "segment_at_box_contact" in called, f"{reader.__qualname__} no longer reads through the contact reader"
        direct = {name: lines for name, lines in called.items() if name in _OWNER_POINT_READERS}
        assert not direct, f"{reader.__qualname__} reads the owner at a point around the contact reader: {direct}"


# ---- The corridor a flush obstacle leaves (amended 2026-10-22) ------------------------------------------------------

_CRACK = DifferenceDomain(
    _square(),
    UnionDomain(
        [Hyperrectangle(np.array([[0.25, 0.5], [0.25, 0.75]])), Hyperrectangle(np.array([[0.5, 0.75], [0.25, 0.75]]))]
    ),
)
_TANGENT = DifferenceDomain(_square(), Hypersphere(center=[0.75, 0.5], radius=0.25))


@pytest.mark.parametrize(
    ("domain", "a", "step", "landing"),
    [
        (_FLUSH, (1.0, 0.5), (-0.05, 0.03), (1.0, 0.53)),
        (_TANGENT, (1.0, 0.5), (-0.0625, 0.03125), (1.0, 0.53125)),
        (_CRACK, (0.5, 0.5), (0.0625, 0.03125), (0.5, 0.53125)),
    ],
    ids=["1a, the flush obstacle's corridor", "2, the tangent disc", "3a, the crack between two obstacles"],
)
def test_a_zero_width_corridor_keeps_the_gaps_limit(domain, a, step, landing):
    """The gap's limit, while the path stays in the corridor: the step meets one face at t = 0, is reflected, meets the
    opposite face at t = 0 of the re-trace, and the tie of anti-parallel normals removes the normal part once; the
    tangential rest runs along the corridor. 2 contacts, 1 reflection, 1 corridor, no fallback."""
    stepped = _step(domain, [a], [np.add(a, step)])
    np.testing.assert_allclose(stepped.positions, [landing], rtol=0, atol=1e-12)
    assert (stepped.counts["contacts"], stepped.counts["reflections"], stepped.counts["corridors"]) == (2, 1, 1)
    assert stepped.counts["fallbacks"] == 0


@pytest.mark.parametrize(
    ("domain", "a", "step", "landing"),
    [(_FLUSH, (1.0, 0.5), (-0.05, 0.15), (1.0, 0.65)), (_CRACK, (0.5, 0.5), (0.0625, 0.375), (0.5, 0.875))],
    ids=["1b, past the flush obstacle's corner", "3b, past the crack's end"],
)
def test_past_a_corridors_end_the_particle_stays_on_its_line_by_convention(domain, a, step, landing):
    """Ruled 2026-10-22: a convention, not the gap's limit. Past the corridor's end no single limit exists -- in 1b
    either bounce parity lands at (0.98333, 0.65), only because the wall folds one; in 3b the parity of infinitely many
    bounces decides between (0.5 -+ 0.02083, 0.875) -- and the particle stays on the corridor's line. It differs from
    either parity by at most the step's normal part times the step left, 0.0167 in 1b."""
    stepped = _step(domain, [a], [np.add(a, step)])
    np.testing.assert_allclose(stepped.positions, [landing], rtol=0, atol=1e-12)
    assert stepped.counts["corridors"] == 1


@pytest.mark.parametrize("step", [(-0.05, 0.03), (0.05, 0.03)], ids=["the obstacle's face first", "the box face first"])
def test_an_absorbing_face_at_a_corridor_absorbs_whichever_face_is_met_first(step):
    """1c and 1d: with x_max an exit, the obstacle-box tie is decided by the box face's action (D2), so a start on the
    corridor is absorbed at (1, 0.5) in either order. (1, 0.5) lies on one box face; no corner rule enters."""
    stepped = _step(_FLUSH, [(1.0, 0.5)], [np.add((1.0, 0.5), step)], bc=_exit_bc())
    assert stepped.absorbed.tolist() == [True]
    np.testing.assert_allclose(stepped.positions, [[1.0, 0.5]], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("domain", "point"),
    [
        (DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0]])), Hyperrectangle(np.array([[0.8, 1.0]]))), 1.0),
        (
            DifferenceDomain(
                Hyperrectangle(np.array([[0.0, 1.0]])),
                UnionDomain([Hyperrectangle(np.array([[0.3, 0.5]])), Hyperrectangle(np.array([[0.5, 0.7]]))]),
            ),
            0.5,
        ),
    ],
    ids=["x = 1 beside a flush obstacle", "the crack at 0.5"],
)
@pytest.mark.parametrize("sign", [-1.0, 1.0], ids=["inward", "outward"])
def test_a_step_with_no_direction_along_a_corridor_is_refused(domain, point, sign):
    """Ruled 2026-10-22: in 1-D a corridor is an isolated point of the domain, with no tangential direction, so the
    corridor rule leaves no step; it is refused by name, keeping main's loud outcome (`project_to_interior` raised). The
    1-D grid path puts about 5% of its particles on such a node when it carries mass; with that node's mass zeroed the
    same solves run."""
    with pytest.raises(ValueError, match=r"an isolated point of the domain.*Give no initial mass to that node"):
        _step(domain, [[point]], [[point + sign * 0.02]], box=np.array([[0.0, 1.0]]))


# ---- Oracles ----------------------------------------------------------------------------------------------------------


def test_a_step_across_a_ring_reflects_at_its_outer_circle():
    """B1, 2-D: (0.9, 0.5) -> (0.5005, 0.53) meets the ring's outer circle, radius 0.3, at its smaller root, and is
    reflected there in the domain's outward normal, which points into the ring. The reflected end, (1.0978, 0.5450), lies
    past the wall x = 1, which reflects it once more: it lands at (2 - 1.0978, 0.5450). The projection landed a quarter
    of the ring away; a285d3be at (0.799, 0.537)."""
    ring = DifferenceDomain(_square(), DifferenceDomain(Hypersphere([0.5, 0.5], 0.3), Hypersphere([0.5, 0.5], 0.1)))
    a, b, c = np.array([0.9, 0.5]), np.array([0.5005, 0.53]), np.array([0.5, 0.5])
    d, f = b - a, a - c
    t = float(np.min(np.roots([d @ d, 2 * f @ d, f @ f - 0.3**2])))
    p = a + t * d
    n = -(p - c) / np.linalg.norm(p - c)
    reflected = b - 2.0 * ((b - p) @ n) * n
    assert reflected[0] > 1.0
    stepped = _step(ring, [a], [b])
    np.testing.assert_allclose(stepped.positions, [[2.0 - reflected[0], reflected[1]]], rtol=0, atol=1e-12)
    assert stepped.counts["reflections"] == 2


def test_a_1d_step_across_a_ring_reflects_at_the_ring_and_then_the_wall():
    """B1, 1-D: [0, 1] minus ([0.2, 0.8] minus [0.4, 0.6]); 0.1 -> 0.5 is reflected at 0.2, then at 0, and lands at
    0.1. The face map raised; a285d3be placed it at 0.2."""
    ring = DifferenceDomain(
        Hyperrectangle(np.array([[0.0, 1.0]])),
        DifferenceDomain(Hyperrectangle(np.array([[0.2, 0.8]])), Hyperrectangle(np.array([[0.4, 0.6]]))),
    )
    stepped = _step(ring, [[0.1]], [[0.5]], box=np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(stepped.positions, [[0.1]], rtol=0, atol=1e-12)
    assert stepped.counts["reflections"] == 2


def test_the_loop_on_the_box_is_the_fold_bitwise():
    """On the box alone the loop is today's fold: reflecting the leg's end, e' = e - 2 ((e - p) . n) n, carries a
    coordinate the normal does not touch exactly and rounds a face's once, as 2 h - e does. 20000 steps from near the
    (1, 1) corner, many reflected in both faces."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0.85, 1.0, (20000, 2))
    b = a + rng.normal(0.0, 0.1, (20000, 2))
    stepped = _step(_square(), a, b)
    assert stepped.counts["reflections"] > 5000
    assert np.array_equal(stepped.positions, reflect_positions(b.copy(), [(0.0, 1.0), (0.0, 1.0)]))


def test_the_skip_changes_no_landing(monkeypatch):
    """The exact skip, |b - a| < |phi(a)| - eps on a domain that is not convex: on a stress population beside two
    obstacles, the loop with it lands every particle where searching every particle does, bitwise."""
    from mfgarchon.alg.numerical.fp_solvers import fp_particle_bc

    domain = DifferenceDomain(
        _square(),
        UnionDomain([Hypersphere(center=[0.3, 0.5], radius=0.15), Hyperrectangle(np.array([[0.6, 0.8], [0.2, 0.9]]))]),
    )
    rng = np.random.default_rng(5)
    a = rng.uniform(0.0, 1.0, (20000, 2))
    a = a[np.atleast_1d(domain.contains(a))][:8000]
    b = a + rng.normal(0.0, 0.08, a.shape)
    skipped = _step(domain, a, b)
    monkeypatch.setattr(
        fp_particle_bc, "_reaches_no_surface", lambda domain, b, phi, length, eps: np.zeros(len(b), bool)
    )
    searched = _step(domain, a, b)
    assert skipped.counts["searched"] < searched.counts["searched"] == len(a)
    assert skipped.counts["contacts"] == searched.counts["contacts"] > 1000
    assert np.array_equal(skipped.positions, searched.positions)


def test_a_flat_reflecting_wall_gives_the_method_of_images_law():
    """One step from x0 = 0.05 by N(0, s^2), s = 0.1, at the reflecting wall x = 0 of [0, 1]: specular reflection
    lands at |x0 + xi|, whose law is the method of images', F(x) = Phi((x - x0) / s) - Phi((-x - x0) / s) (Kerr et al.
    2008, eq. 3.9), at any dt. 200000 landings against it: the Kolmogorov-Smirnov distance within 1.63 / sqrt(N)."""
    from scipy.stats import norm

    x0, s, n = 0.05, 0.1, 200000
    xi = np.random.default_rng(7).normal(0.0, s, n)
    stepped = _step(
        Hyperrectangle(np.array([[0.0, 1.0]])), np.full((n, 1), x0), (x0 + xi)[:, None], box=np.array([[0.0, 1.0]])
    )
    landed = np.sort(stepped.positions[:, 0])
    law = norm.cdf((landed - x0) / s) - norm.cdf((-landed - x0) / s)
    distance = max(np.max(np.arange(1, n + 1) / n - law), np.max(law - np.arange(n) / n))
    assert distance < 1.63 / np.sqrt(n), distance


def test_a_reflecting_obstacle_keeps_the_gibbs_density():
    """dX = -a e_x dt + sigma dW on [0, 1]^2 minus a disc of radius 0.2 at the centre, reflecting everywhere, keeps
    rho ∝ exp(-2 a x / sigma^2): started from rho, the x-marginal after T = 2 matches rho's, exp(-2 a x / sigma^2) times
    the domain's width at x, in every one of 10 bins within 4 binomial standard deviations. Measured over 4 seeds:
    within 2.4. At dt = 0.05 the reflected Euler scheme's own boundary error shows, 15.7 deviations at the wall x = 0,
    the same on the box alone at main and at the head; dt here is 0.005, where it is 1.7."""
    a, sigma, n, nt, T = 0.5, 0.5, 40000, 400, 2.0
    k = 2.0 * a / sigma**2
    domain = DifferenceDomain(_square(), Hypersphere(center=[0.5, 0.5], radius=0.2))
    edges = np.linspace(0.0, 1.0, 11)
    xs = np.linspace(0.0, 1.0, 20001)
    density = np.exp(-k * xs) * (1.0 - 2.0 * np.sqrt(np.clip(0.04 - (xs - 0.5) ** 2, 0.0, None)))
    cdf = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2.0 * np.diff(xs))])
    expected = np.diff(np.interp(edges, xs, cdf / cdf[-1]))
    rng = np.random.default_rng(101)
    starts = []
    while len(starts) < n:
        c = rng.uniform(0.0, 1.0, (4 * n, 2))
        keep = (rng.uniform(0.0, 1.0, 4 * n) < np.exp(-k * c[:, 0])) & np.atleast_1d(domain.contains(c))
        starts.extend(c[keep])
    solver = FPParticleSolver(
        _problem(_grid(2, 21), sigma=sigma, T=T, nt=nt),
        num_particles=n,
        seed=1,
        implicit_domain=domain,
        density_mode="query_only",
    )
    solver.solve_fp_system(
        initial_particles=np.asarray(starts[:n]),
        drift_field=lambda t, x, m: np.broadcast_to(np.array([-a, 0.0]), np.shape(x)),
        drift_needs_density=False,
        show_progress=False,
    )
    final = np.asarray(solver.M_particles_trajectory[-1], dtype=float).reshape(-1, 2)
    got = np.histogram(final[:, 0], bins=edges)[0] / len(final)
    z = (got - expected) * n / np.sqrt(n * expected * (1.0 - expected))
    assert np.all(np.abs(z) < 4.0), np.round(z, 1)
    assert sum(h["fallbacks"] for h in solver.obstacle_placement_history) == 0
    assert sum(h["contacts"] for h in solver.obstacle_placement_history) > 0


def test_no_step_crosses_an_obstacle():
    """#2707's table: 500 particles, T 0.5, Nt 20, sigma 0.05, on the keyword route. The peak count on the obstacle's far
    side is 0 in every case, the control included. At main: 500, 500, 0 and 9. The last two rows are #2733's thin
    feature, the folded #2710: slabs of width 1e-3 and 1e-4, under 1/32 of the drift's step 0.075. At main: 500 and
    500."""
    drift_1d = lambda t, x, m: 3.0 * (1.0 - np.asarray(x))  # noqa: E731
    drift_2d = lambda t, x, m: np.broadcast_to(np.array([3.0, 0.0]), np.shape(x))  # noqa: E731
    cases = [
        (1, [[0.15, 0.25]], drift_1d, [0.02], [0.1], 0.25),
        (2, [[0.48, 0.52], [-0.1, 1.1]], drift_2d, [0.05, 0.05], [0.4, 0.95], 0.52),
        (1, [[0.15, 0.45]], drift_1d, [0.02], [0.1], 0.45),
        (2, [[0.4, 0.6], [-0.1, 1.1]], drift_2d, [0.05, 0.05], [0.35, 0.95], 0.6),
        (2, [[0.5, 0.501], [-0.1, 1.1]], drift_2d, [0.05, 0.05], [0.4, 0.95], 0.501),
        (2, [[0.5, 0.5001], [-0.1, 1.1]], drift_2d, [0.05, 0.05], [0.4, 0.95], 0.5001),
    ]
    for d, obstacle, drift, lo, hi, far in cases:
        domain = DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0]] * d)), Hyperrectangle(np.array(obstacle)))
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)] * d, Nx_points=[41] if d == 1 else [21, 21], boundary_conditions=no_flux_bc(dimension=d)
        )
        solver = FPParticleSolver(
            _problem(grid), num_particles=500, seed=1, implicit_domain=domain, density_mode="query_only"
        )
        start = np.random.default_rng(1).uniform(lo, hi, (500, d))
        solver.solve_fp_system(
            initial_particles=start, drift_field=drift, drift_needs_density=False, show_progress=False
        )
        peak = max(
            int(np.sum(np.asarray(p, dtype=float).reshape(-1, d)[:, 0] > far)) for p in solver.M_particles_trajectory
        )
        assert peak == 0, (obstacle, peak)
        assert sum(h["fallbacks"] for h in solver.obstacle_placement_history) == 0
        assert sum(h["contacts"] for h in solver.obstacle_placement_history) > 0


def test_a_3d_room_holds_behind_its_obstacle_the_particles_its_shadow_holds():
    """The unit cube minus [0.4, 0.6]^3, an exit on x_max, 2000 particles from x = 0.2 with y, z uniform in [0.3, 0.7],
    drift 3 ((1.3, 0.5, 0.5) - x), sigma 0.05, T 0.5. Noise-free, x(t) = 1.3 - 1.1 e^{-3t} meets the face x = 0.4 when
    y - 0.5 and z - 0.5 have shrunk by 9/11, so the 740 starts with both within 0.1 * 11/9 of 0.5 are held behind it and
    the other 1260 reach the exit (x(0.5) = 1.0546). Over 12 seeds the head absorbed 1250 to 1289; the band is 1260 +- 60.
    Main absorbed 1944 to 1961: the projection slid about 700 particles past the obstacle, #2707's class."""
    room = DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0]] * 3)), Hyperrectangle(np.array([[0.4, 0.6]] * 3)))
    solver = FPParticleSolver(
        _problem(_grid(3)),
        num_particles=2000,
        seed=1,
        density_mode="query_only",
        boundary_conditions=_exit_bc(3),
        implicit_domain=room,
    )
    start = np.c_[np.full(2000, 0.2), np.random.default_rng(3).uniform(0.3, 0.7, (2000, 2))]
    solver.solve_fp_system(
        initial_particles=start,
        drift_field=lambda t, x, m: 3.0 * (np.array([1.3, 0.5, 0.5]) - np.asarray(x)),
        drift_needs_density=False,
        show_progress=False,
    )
    assert abs(solver.total_absorbed - 1260) <= 60, solver.total_absorbed


# ---- #2732's catalogue, as properties -------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(1, 9))
def test_a1_a_flush_obstacle_at_a_corner_lets_every_step_run(seed):
    """A1: [0, 1]^2 minus [0, 0.1] x [0.2, 0.8], 600 particles near the corner (0, 0.2), drift (-1, 1). The face map
    raised on every seed (0 of 8 ran); every step runs and records its particles inside."""
    domain = DifferenceDomain(_square(), Hyperrectangle(np.array([[0.0, 0.1], [0.2, 0.8]])))
    solver = FPParticleSolver(
        _problem(_grid(2, 21), T=0.3, nt=15),
        num_particles=600,
        seed=seed,
        implicit_domain=domain,
        density_mode="query_only",
    )
    start = np.random.default_rng(seed).uniform([0.1, 0.1], [0.2, 0.19], (600, 2))
    solver.solve_fp_system(
        initial_particles=start,
        drift_field=lambda t, x, m: np.broadcast_to(np.array([-1.0, 1.0]), np.shape(x)),
        drift_needs_density=False,
        show_progress=False,
    )
    for p in solver.M_particles_trajectory:
        assert np.all(np.atleast_1d(domain.contains(np.asarray(p, dtype=float).reshape(-1, 2))))


def test_b2_two_rooms_sharing_a_wall_land_a_step_the_same_in_either_order():
    """B2: the union [0, 0.5] x [0, 1] with [0.3, 1] x [0, 1]; (0.3, 0.01) -> (0.6, -0.01) is reflected at the floor
    and lands at (0.6, 0.01) whichever room comes first. The projection landed (0.5, 0.0) in one order."""
    rooms = [Hyperrectangle(np.array([[0.0, 0.5], [0.0, 1.0]])), Hyperrectangle(np.array([[0.3, 1.0], [0.0, 1.0]]))]
    for order in (rooms, rooms[::-1]):
        stepped = _step(UnionDomain(order), [(0.3, 0.01)], [(0.6, -0.01)])
        np.testing.assert_allclose(stepped.positions, [[0.6, 0.01]], rtol=0, atol=1e-12)


def test_b3_steps_from_a_rooms_corner_point_all_land():
    """B3: 400 Gaussian steps, sigma 0.05, from the left room's corner point (0.5, 0) on the floor of the two rooms'
    union. The face map raised on 11 of them (the blended corner normal, #2698); every one lands inside."""
    rooms = UnionDomain(
        [Hyperrectangle(np.array([[0.0, 0.5], [0.0, 1.0]])), Hyperrectangle(np.array([[0.3, 1.0], [0.0, 1.0]]))]
    )
    a = np.tile([0.5, 0.0], (400, 1))
    stepped = _step(rooms, a, a + np.random.default_rng(3).normal(0.0, 0.05, (400, 2)))
    assert np.all(np.atleast_1d(rooms.contains(stepped.positions)))
    assert stepped.counts["fallbacks"] == 0


def test_m1_a_step_that_cuts_an_l_shapes_concave_corner_does_not_land_on_it():
    """M1: the L-shape [0, 1]^2 minus (0.5, 1] x (0.5, 1]; (0.4, 0.4) -> (0.6, 0.55) leaves through the quadrant's
    bottom face y = 0.5 at (0.5333, 0.5) and lands at (0.6, 2 * 0.5 - 0.55) = (0.6, 0.45). The projection put every such
    step on the corner point (0.5, 0.5)."""
    domain = DifferenceDomain(_square(), Hyperrectangle(np.array([[0.5, 1.0], [0.5, 1.0]])))
    stepped = _step(domain, [(0.4, 0.4)], [(0.6, 0.55)])
    np.testing.assert_allclose(stepped.positions, [[0.6, 0.45]], rtol=0, atol=1e-12)


# ---- What the loop holds every step to ------------------------------------------------------------------------------

_CHANNEL = DifferenceDomain(
    _square(),
    UnionDomain(
        [Hyperrectangle(np.array([[0.0, 1.0], [0.0, 0.49]])), Hyperrectangle(np.array([[0.0, 1.0], [0.51, 1.0]]))]
    ),
)


def test_a_step_reflected_more_often_than_the_cap_returns_to_its_start():
    """D5: in the channel 0.49 <= y <= 0.51, a step from (0.1, 0.5) rising 0.5 crosses the channel's width 25 times, past
    the cap of 20 reflections, and returns to its start, counted; one rising 0.38 is reflected 19 times and lands where
    the fold of its end in the channel puts it."""
    stepped = _step(_CHANNEL, [(0.1, 0.5), (0.1, 0.5)], [(0.2, 1.0), (0.2, 0.88)])
    np.testing.assert_allclose(stepped.positions[0], [0.1, 0.5], rtol=0, atol=0)
    assert stepped.counts["fallbacks"] == 1
    folded = reflect_positions(np.array([[0.2, 0.88]]), [(0.0, 1.0), (0.49, 0.51)])
    np.testing.assert_allclose(stepped.positions[1:], folded, rtol=0, atol=1e-12)


@pytest.mark.parametrize("path", ["callable drift", "a sampled initial density"])
def test_a_box_only_solve_starts_every_particle_in_the_box(path):
    """#2733 holds the box to #2688's rule: every step starts inside the particles' domain, the box when there is no
    other. At main a box-only 2-D solve recorded 367 of 4000 particles outside [0, 1]^2 at t = 0, a boundary node's
    jitter, and the first step folded them in; the sampler now redraws such a jitter within its cell, which leaves the
    t = 0 law as it was, and the caller's initial particles outside the box are refused by name."""
    grid = _grid(2, 21)
    solver = FPParticleSolver(_problem(grid, T=0.1, nt=2), num_particles=4000, seed=1)
    if path == "callable drift":
        outside = np.array([[0.5, 0.5]] * 3999 + [[1.01, 0.5]])
        with pytest.raises(ValueError, match=r"of the particles given as initial_particles, 1 of 4000 are outside"):
            solver.solve_fp_system(
                initial_particles=outside,
                drift_field=lambda t, x, m: np.zeros_like(np.asarray(x)),
                drift_needs_density=False,
            )
        return
    solver.solve_fp_system(
        M_initial=np.ones((21, 21)),
        drift_field=np.zeros((3, 21, 21, 2)),
        drift_is_precomputed=True,
        show_progress=False,
    )
    start = np.asarray(solver.M_particles_trajectory[0], dtype=float).reshape(-1, 2)
    assert np.all((start >= 0.0) & (start <= 1.0))


def test_a_landing_just_outside_the_domain_after_a_contact_is_offset_inward():
    """D6, on its own helper: no natural step reaches it -- over about 4.5 million random steps beside a disc, two discs
    and a disc touching a wall, every landing passed `contains` -- so it is pinned directly. A landing 1e-13 inside a
    box obstacle, after a contact on its face x = 0.9 with outward normal +e_x, is moved along -e_x by the ladder's
    first rung that puts it in the domain (from 1e-12 of the diagonal, doubling), and counted; one with no contact to read
    a normal from returns to its start."""
    from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import _offset_landings

    domain = DifferenceDomain(_square(), Hyperrectangle(np.array([[0.9, 1.0], [0.2, 0.8]])))
    starts = np.array([[0.8, 0.5], [0.8, 0.5]])
    positions = np.array([[0.9 + 1e-13, 0.5], [0.9 + 1e-13, 0.5]])
    normals = np.array([[1.0, 0.0], [np.nan, np.nan]])
    counts = {"offsets": 0, "fallbacks": 0}
    _offset_landings(domain, starts, positions, np.zeros(2, bool), np.arange(2), normals, _BOX, counts, "test")
    assert np.atleast_1d(domain.contains(positions[:1]))[0]
    assert 0.9 - 1e-6 * np.sqrt(2.0) <= positions[0, 0] <= 0.9
    np.testing.assert_array_equal(positions[1], starts[1])
    assert counts == {"offsets": 1, "fallbacks": 1}

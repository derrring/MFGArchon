"""FEM reads the face owner (#2512 row B1, stage B1b; #2595).

Each boundary facet's segment is `BoundaryConditions.segments_for_facets`' answer, and a node that facets governed by
different segments share takes `segment_for_shared_node`. The plan and its approval: #2595 comment 6145845773.

- An external oracle: each spelling of "a no-flux wall at x = 0, an exit at x = 1" that the owner reads as that BC is
  solved by HJB-FEM to the exact solution of the backward heat equation (H frozen at p = 0).
- #2595's rows, one pin each, on the issue's 1-D reproducer.
- An agreement oracle: on box meshes, every facet's answer is the point owner's at the facet midpoint, and on a face
  the face owner answers for, it imposes that face's condition.
- A tag of the mesh is that tag; only a name that is no tag reads as the box face it aliases ((A')).
"""

from __future__ import annotations

import itertools

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.conditions import _same_condition
from mfgarchon.geometry.boundary.types import BoundaryFace

skfem = pytest.importorskip("skfem")

T, SIGMA, G = 0.5, 0.4, 0.7
NT = 10
_WALL = BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0, boundary="x_min")
_EXIT = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G, boundary="x_max")


def _bc(*segments: BCSegment, **kwargs) -> BoundaryConditions:
    kwargs.setdefault("default_bc", BCType.NO_FLUX)
    return BoundaryConditions(dimension=1, segments=list(segments), **kwargs)


def _interval(bc: BoundaryConditions, n: int = 16):
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    geometry = Mesh1D(bounds=(0.0, 1.0), num_elements=n)
    geometry.generate_mesh()
    geometry.boundary_conditions = bc
    return geometry


def _hjb(geometry, nt: int = NT):
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    # Unit mass on the domain's own measure: Mesh1D's uniform-cell measure gives a constant c the mass c (n+1)/n.
    c = geometry.num_elements / (geometry.num_elements + 1) if isinstance(geometry, Mesh1D) else 1.0
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=SIGMA
    )
    conditions = Conditions(
        m_initial=lambda x: c + 0.0 * np.asarray(x, dtype=float)[..., 0], u_terminal=lambda x: 0.0, T=T
    )
    return HJBFEMSolver(MFGProblem(model=model, domain=geometry, conditions=conditions, Nt=nt))


def _u(bc: BoundaryConditions) -> np.ndarray:
    """#2595's reproducer: HJB-FEM, 16 elements on [0, 1], sigma = 0.4, T = 0.5, Nt = 10, M = 1, U(T) = 0."""
    solver = _hjb(_interval(bc))
    n = solver._disc.dof_coordinates.shape[0]
    return np.asarray(solver.solve_hjb_system(M_density=np.ones((NT + 1, n)), U_terminal=np.zeros(n)))


# --- the external oracle ---------------------------------------------------------------------------------------------


def _heat_exact(x: np.ndarray) -> np.ndarray:
    """u(0, x) for -u_t - D u_xx = 0, u(T) = 0, u_x(0) = 0, u(1) = G, with D = sigma^2 / 2: a quarter-wave cosine
    series, l_k = (k + 1/2) pi, coefficient -2 G sin(l_k) / l_k."""
    lam = (np.arange(4000) + 0.5) * np.pi
    return G + (-2.0 * G * np.sin(lam) / lam * np.exp(-(SIGMA**2 / 2) * lam**2 * T)) @ np.cos(np.outer(lam, x))


_WALL_AND_EXIT = {
    "named": _bc(_WALL, _EXIT),
    "row 1: a global exit, the wall at priority 0": _bc(
        _WALL, BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G)
    ),
    "row 1: a global exit, the wall at priority 1": _bc(
        BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0, boundary="x_min", priority=1),
        BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G),
    ),
    "row 3: the exit placed by region": _bc(
        _WALL,
        BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G, region={0: (0.9, 1.0)}),
        default_bc=BCType.DIRICHLET,
        default_value=G,
    ),
    "row 5a: the exit is the DIRICHLET default": _bc(_WALL, default_bc=BCType.DIRICHLET, default_value=G),
    "row 6: the exit named by the alias 'right'": _bc(
        _WALL, BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G, boundary="right")
    ),
    "row 11: the exit named 'all'": _bc(
        _WALL, BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G, boundary="all")
    ),
}


@pytest.mark.parametrize("spelling", list(_WALL_AND_EXIT))
def test_each_spelling_of_a_wall_and_an_exit_solves_to_the_exact_solution(spelling):
    """External oracle (class 2). Each BC here is, by the owner's reading, a no-flux wall at x = 0 and an exit at
    u = 0.7 at x = 1. With ``U_coupling_prev = 0`` the Picard step evaluates the quadratic H at p = 0, where it is 0, so
    HJB-FEM solves the backward heat equation, whose solution with that BC is `_heat_exact`. Measured at n = 32,
    Nt = 200, on the "named" spelling: max error 4.26e-4, falling with dt (1.86e-3 at Nt = 50, 1.07e-4 at n = 64,
    Nt = 800). Before #2595 each of the other spellings condensed the wall too and missed by 0.699 at x = 0."""
    nt = 200
    solver = _hjb(_interval(_WALL_AND_EXIT[spelling], n=32), nt=nt)
    x = solver._disc.dof_coordinates[:, 0]
    U = np.asarray(
        solver.solve_hjb_system(
            M_density=np.ones((nt + 1, x.size)), U_terminal=np.zeros(x.size), U_coupling_prev=np.zeros((nt + 1, x.size))
        )
    )[0]
    np.testing.assert_allclose(U, _heat_exact(x), atol=2e-3)


# --- #2595's rows ------------------------------------------------------------------------------------------------------


def test_row_5b_a_neumann_default_loads_the_facets_no_segment_holds():
    """Row 5b: a NEUMANN(1) default was never read, so the solve was bit-identical to a NO_FLUX default. It now loads
    x_min as a segment naming it does, entry for entry."""
    neumann = BCSegment(name="load", bc_type=BCType.NEUMANN, value=1.0, boundary="x_min")
    by_default = _u(_bc(_EXIT, default_bc=BCType.NEUMANN, default_value=1.0))
    np.testing.assert_array_equal(by_default, _u(_bc(_EXIT, neumann)))
    assert np.abs(by_default - _u(_bc(_EXIT))).max() > 0.1


@pytest.mark.parametrize(
    ("default", "refusal"),
    [
        ({"default_bc": BCType.ROBIN, "default_value": 0.3}, r"default_bc is ROBIN, but default_bc is a bare BCType"),
        (
            {"default_bc": None},
            r"no segment governs 1 boundary facet\(s\) of the mesh, and default_bc was not specified",
        ),
    ],
    ids=["5c-robin", "5d-unset"],
)
def test_rows_5c_and_5d_a_default_the_owner_refuses_is_refused(default, refusal):
    """Rows 5c and 5d: a ROBIN default, or none, on a facet no segment holds was solved as a wall. The owner refuses
    both by its existing names (`_default_segment`)."""
    with pytest.raises(ValueError, match=refusal):
        _u(_bc(_EXIT, **default))


@pytest.mark.parametrize("name", [0, ""])
def test_row_8_a_falsy_name_is_refused_naming_both_readings(name):
    """Row 8: a segment named 0 or "" was read as no name, the whole boundary, and condensed the wall. A name is a tag
    of the mesh or the box face it aliases, and these are neither."""
    with pytest.raises(ValueError, match=r"which this mesh does not resolve\. Read as a tag of the mesh, it is none"):
        _u(_bc(BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G, boundary=name)))


def _square_with_region_1(bc: BoundaryConditions):
    """The unit square, ``MeshTri.init_tensor`` 4 x 4, whose MeshData tags the x_max facets 1 (``region_1``) and the
    rest 2, as in #2595's row 9."""
    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    xs = np.linspace(0.0, 1.0, 5)
    mesh_data = skfem_to_meshdata(skfem.MeshTri.init_tensor(xs, xs))
    x = mesh_data.vertices[mesh_data.boundary_faces][..., 0]
    mesh_data.boundary_tags = np.where(np.isclose(x.min(axis=1), 1.0), 1, 2)
    geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    geometry.mesh_data = mesh_data
    geometry.boundary_conditions = bc
    return geometry


@pytest.mark.parametrize(("wall_priority", "dropped"), [(0, "wall"), (1, "exit")])
def test_row_9_of_two_segments_on_one_tag_the_one_governing_nothing_is_refused(wall_priority, dropped):
    """Row 9: ``[DIRICHLET(0.7) region_1, NO_FLUX x_max]`` on a mesh whose region_1 is the x_max facets condensed x_max
    at either priority. Both are placed, so priority, then declaration order, picks one; the other governs no facet and
    imposes a different condition, so it is refused by name, as two segments on one box face are (`_check_shadowed`).
    A segment imposing the same condition there changes nothing and is accepted."""
    exit_ = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G, boundary="region_1")
    wall = BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0, boundary="x_max", priority=wall_priority)
    with pytest.raises(
        ValueError, match=rf"govern no facet of the mesh, so their conditions would be dropped: '{dropped}'"
    ):
        _hjb(_square_with_region_1(BoundaryConditions(dimension=2, segments=[exit_, wall], default_bc=BCType.NO_FLUX)))
    same = BCSegment(name="same", bc_type=BCType.DIRICHLET, value=G, boundary="x_max", priority=wall_priority)
    _hjb(_square_with_region_1(BoundaryConditions(dimension=2, segments=[exit_, same], default_bc=BCType.NO_FLUX)))


def _disc(bc: BoundaryConditions):
    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    geometry = Mesh2D(domain_type="circle", bounds=(-1.0, 1.0, -1.0, 1.0))
    geometry.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_circle(3))
    geometry.boundary_conditions = bc
    return geometry


def test_row_10_a_wall_on_an_empty_tag_beside_a_global_exit_is_refused():
    """Row 10: on the disc the axis tags hold no facet, so ``[NO_FLUX "x_min", DIRICHLET(0.7) global]`` condensed the
    boundary at x < -0.9 to 0.7. The wall governs no facet and differs from the exit, so it is refused by name."""
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0, boundary="x_min"),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G),
        ],
        default_bc=BCType.NO_FLUX,
    )
    with pytest.raises(ValueError, match=r"segment 'wall' names the tag 'x_min' \(boundary='x_min'\), which holds no"):
        _hjb(_disc(bc))


# --- the facet entry against the point and face owners -----------------------------------------------------------------


def _box_mesh(dim: int):
    """A box mesh with its axis tags, as the FEM solvers build it: [0, 2] (x 4 elements) x [0, 1] (x 2) x [0, 3]."""
    from mfgarchon.alg.numerical.fem.mesh_adapter import _tag_axis_aligned_boundaries

    axes = [np.linspace(0.0, 2.0, 9), np.linspace(0.0, 1.0, 5), np.linspace(0.0, 3.0, 4)][:dim]
    mesh = {1: lambda: skfem.MeshLine(axes[0]), 2: lambda: skfem.MeshTri.init_tensor(*axes)}.get(
        dim, lambda: skfem.MeshHex.init_tensor(*axes)
    )()
    return _tag_axis_aligned_boundaries(mesh), np.array([[a[0], a[-1]] for a in axes])


_PLACEMENTS = {
    1: [{}, {"boundary": "x_min"}, {"boundary": "left"}, {"boundary": "right"}, {"boundary": "all"},
        {"region": {0: (1.0, 2.0)}}, {"normal_direction": [1.0]}],
    2: [{}, {"boundary": "x_min"}, {"boundary": "left"}, {"boundary": "y_max"}, {"boundary": "all"},
        {"region": {"x": (0.0, 1.0)}}, {"region": {"y": (0.3, 0.7)}}, {"boundary": "x_max", "region": {"y": (0.0, 0.5)}},
        {"normal_direction": [0.0, 1.0]}, {"sdf_region": lambda p: float(np.linalg.norm(p) - 0.9)},
        {"normal_direction": [1.0, 1.0], "normal_tolerance": 0.5}],
    3: [{}, {"boundary": "z_min"}, {"boundary": "front"}, {"boundary": "back"}, {"region": {"z": (0.0, 1.5)}},
        {"normal_direction": [0.0, 0.0, 1.0]}, {"normal_direction": [1.0, 1.0, 1.0], "normal_tolerance": 0.5}],
}  # fmt: skip
_CONDITIONS = [
    {"bc_type": BCType.DIRICHLET, "value": 0.7},
    {"bc_type": BCType.DIRICHLET, "value": 0.2},
    {"bc_type": BCType.NO_FLUX, "value": 0.0},
    {"bc_type": BCType.NEUMANN, "value": 1.0},
    {"bc_type": BCType.ROBIN, "value": 0.3, "alpha": 1.0, "beta": 1.0},
]
_DEFAULTS = [{"default_bc": BCType.NO_FLUX}, {"default_bc": BCType.DIRICHLET, "default_value": 0.5}, {}]


def _generated(dim: int):
    """Two-segment BCs over the placements, each pair once with two conditions cycled through, three priority pairs
    and three defaults."""
    placements = _PLACEMENTS[dim]
    for i, (a, b) in enumerate(itertools.product(range(len(placements)), repeat=2)):
        for priorities in ((0, 0), (0, 1), (1, 0)):
            segments = [
                BCSegment(name=f"s{k}", priority=priority, **_CONDITIONS[c], **placements[p])
                for k, (p, c, priority) in enumerate(zip((a, b), (i % 5, (3 * i + 1) % 5), priorities, strict=True))
            ]
            for default in _DEFAULTS:
                yield segments, default


def _same_answer(first: BCSegment, second: BCSegment) -> bool:
    """The same segment, or both the owner's default segment for the same ``default_bc``."""
    return first is second or (first.name == second.name == "default" and _same_condition(first, second))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_every_facet_takes_the_point_owners_answer_at_its_midpoint_and_the_face_owners_on_its_face(dim):
    """Agreement oracle (class 2, the B1a shape). Over generated two-segment BCs (placements by boundary, alias,
    "all", region, sdf_region and normal_direction; five conditions; three priority pairs; three defaults), wherever
    the facet entry answers:

    - each facet's segment is the point owner's (`segment_for_point`) at the facet's midpoint, everywhere; a midpoint
      lies on one face, so the corner rule does not enter;
    - on each face the face owner answers for, every facet's segment imposes the face's condition, and where one
      segment governs the face whole, it is that segment.

    The counts below are floors on what the run reaches, so a generator that stopped producing a kind of case would
    fail here rather than pass on less."""
    from mfgarchon.alg.numerical.fem.bc_adapter import _boundary_facets

    mesh, bounds = _box_mesh(dim)
    _, facets = _boundary_facets(mesh)
    answered = refused = split_faces = face_checks = 0
    for segments, default in _generated(dim):
        bc = BoundaryConditions(dimension=dim, segments=segments, domain_bounds=bounds, **default)
        try:
            answers = bc.segments_for_facets(facets)
        except ValueError:
            refused += 1
            continue
        answered += 1
        for midpoint, answer in zip(facets.midpoints, answers, strict=True):
            point_answer = bc.segment_for_point(midpoint)
            assert _same_answer(answer, point_answer), (segments, default, midpoint, answer, point_answer)
        for axis, side in itertools.product(range(dim), ("min", "max")):
            on_face = np.isclose(facets.midpoints[:, axis], bounds[axis, 0 if side == "min" else 1])
            try:
                governing = bc.segment_for_face(BoundaryFace(axis, side))
            except ValueError:
                continue
            face_answers = [answer for answer, here in zip(answers, on_face, strict=True) if here]
            assert all(_same_condition(answer, governing) for answer in face_answers), (segments, default, axis, side)
            if len({id(answer) for answer in face_answers}) > 1:
                split_faces += 1
            else:
                assert _same_answer(face_answers[0], governing), (segments, default, axis, side)
                face_checks += 1
    # The generator is deterministic. Measured at c0e728ca: (answered, refused, face checks, split faces) =
    # (258, 183, 516, 0) in 1-D, (624, 465, 1554, 66) in 2-D and (225, 216, 1074, 12) in 3-D. Each count must stay within
    # 5% of that, so an entry that refuses, or answers, a twentieth more often than measured fails here.
    measured = {1: (258, 183, 516, 0), 2: (624, 465, 1554, 66), 3: (225, 216, 1074, 12)}[dim]
    for name, count, expected in zip(
        ("answered", "refused", "face checks", "split faces"),
        (answered, refused, face_checks, split_faces),
        measured,
        strict=True,
    ):
        assert abs(count - expected) <= 0.05 * expected, (name, count, expected)


def test_facet_normals_point_outward_on_a_curved_boundary():
    """The normal a ``normal_direction`` is read against: on the disc it is the outward radial direction, to the
    polygon's resolution."""
    from mfgarchon.alg.numerical.fem.bc_adapter import _boundary_facets

    _, facets = _boundary_facets(skfem.MeshTri.init_circle(3))
    radial = facets.midpoints / np.linalg.norm(facets.midpoints, axis=1, keepdims=True)
    assert np.einsum("ij,ij->i", facets.normals, radial).min() > 0.99


# --- names, nodes and the whole-boundary verdict -----------------------------------------------------------------------


def _square_basis(extra_tags: dict | None = None):
    from mfgarchon.alg.numerical.fem.assembly import create_basis
    from mfgarchon.alg.numerical.fem.mesh_adapter import _tag_axis_aligned_boundaries

    xs = np.linspace(0.0, 1.0, 5)
    mesh = _tag_axis_aligned_boundaries(skfem.MeshTri.init_tensor(xs, xs))
    if extra_tags:
        mesh = mesh.with_boundaries(extra_tags)
    return create_basis(mesh, order=1)


def _dirichlet_x(basis, bc: BoundaryConditions) -> np.ndarray:
    from mfgarchon.alg.numerical.fem.bc_adapter import get_dirichlet_dofs_and_values

    dofs, _ = get_dirichlet_dofs_and_values(basis, bc)
    return basis.doflocs[0, dofs]


def test_a_mesh_tag_called_left_is_that_tag_and_an_alias_only_names_what_no_tag_does():
    """(A'), with the exact tag first. A gmsh group called "left" is that group: here the x_max facets, so a segment
    named "left" condenses x = 1. On the same square without that tag, "left" reads as the box face it aliases, x_min."""
    exit_ = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G, boundary="left")
    bc = BoundaryConditions(dimension=2, segments=[exit_], default_bc=BCType.NO_FLUX)
    np.testing.assert_allclose(_dirichlet_x(_square_basis({"left": lambda x: np.isclose(x[0], 1.0)}), bc), 1.0)
    np.testing.assert_allclose(_dirichlet_x(_square_basis(), bc), 0.0)


@pytest.mark.parametrize(
    ("x_min_priority", "y_min_priority", "order", "corner"),
    [(0, 0, "x_min first", 0.7), (0, 0, "y_min first", 0.2), (0, 1, "x_min first", 0.2), (1, 0, "y_min first", 0.7)],
)
def test_a_node_two_dirichlet_segments_share_takes_the_corner_rule(x_min_priority, y_min_priority, order, corner):
    """The node at (0, 0), on a DIRICHLET(0.7) x_min facet and a DIRICHLET(0.2) y_min facet, takes the grid's corner
    rule (`segment_for_corner`): priority, then declaration order. #1489's F4 refused it as a conflict on the
    condensation path and let the Newton path take the last segment declared (#2541); both now read one answer, the
    corner owner's, checked here against `segment_for_corner` itself."""
    from mfgarchon.alg.numerical.fem.bc_adapter import get_dirichlet_dofs_and_values

    left = BCSegment(name="L", bc_type=BCType.DIRICHLET, value=0.7, boundary="x_min", priority=x_min_priority)
    bottom = BCSegment(name="B", bc_type=BCType.DIRICHLET, value=0.2, boundary="y_min", priority=y_min_priority)
    segments = [left, bottom] if order == "x_min first" else [bottom, left]
    bc = BoundaryConditions(dimension=2, segments=segments, default_bc=BCType.NO_FLUX)
    basis = _square_basis()
    dofs, values = get_dirichlet_dofs_and_values(basis, bc)
    at_corner = values[np.linalg.norm(basis.doflocs[:, dofs], axis=0) < 1e-12]
    assert at_corner.tolist() == [corner]
    assert len(set(dofs.tolist())) == len(dofs)
    assert bc.segment_for_corner([BoundaryFace(0, "min"), BoundaryFace(1, "min")]).value == corner


def test_a_node_a_higher_priority_natural_segment_shares_is_left_free_and_a_blending_strategy_refuses_it():
    """The corner rule at a node a NO_FLUX x_min (priority 1) and a DIRICHLET y_min share: the wall's, so (0, 0) is not
    condensed while the rest of y_min is. Under ``corner_strategy="average"`` the two conditions differ at that node,
    and the owner refuses it (`_corner_choice`)."""
    from mfgarchon.alg.numerical.fem.bc_adapter import refuse_what_fem_cannot_impose

    wall = BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0, boundary="x_min", priority=1)
    bottom = BCSegment(name="B", bc_type=BCType.DIRICHLET, value=0.2, boundary="y_min")
    basis = _square_basis()
    x = _dirichlet_x(basis, BoundaryConditions(dimension=2, segments=[wall, bottom], default_bc=BCType.NO_FLUX))
    assert not np.isclose(x, 0.0).any()
    assert len(x) == 4  # y_min's five nodes less the corner
    averaged = BoundaryConditions(
        dimension=2, segments=[wall, bottom], default_bc=BCType.NO_FLUX, corner_strategy="average"
    )
    with pytest.raises(ValueError, match=r"corner_strategy='average' blends outward normals"):
        refuse_what_fem_cannot_impose(basis, averaged, "test")


def test_the_whole_boundary_verdict_reads_the_default():
    """`is_pure_neumann` read the segment types and never ``default_bc``: a BC whose only Dirichlet is the default was
    "pure Neumann", so the weak-form solvers skipped condensation. It reads the owner's answer per facet now."""
    from mfgarchon.alg.numerical.fem.bc_adapter import is_pure_neumann

    mesh = _square_basis().mesh
    wall = BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0, boundary="x_min")
    assert not is_pure_neumann(mesh, BoundaryConditions(dimension=2, segments=[wall], default_bc=BCType.DIRICHLET))
    assert is_pure_neumann(mesh, BoundaryConditions(dimension=2, segments=[wall], default_bc=BCType.NO_FLUX))


@pytest.mark.parametrize("name", ["x_min", "left"])
def test_2305_a_neumann_load_on_the_default_channel_or_an_alias_is_assembled_where_the_owner_places_it(name):
    """#2305: a NEUMANN(g) arriving only through ``default_bc``/``default_value`` reached no assembly; and "left" must be
    one face with "x_min" (#1939). The load is ``D g`` times the length it is placed on, by the partition of unity:
    the perimeter 4 for the default, the x_min wall 1 for a segment named either way, with the rest NO_FLUX."""
    from mfgarchon.alg.numerical.fem.bc_adapter import assemble_robin_terms

    basis = _square_basis()
    _, by_default = assemble_robin_terms(
        basis, BoundaryConditions(dimension=2, segments=[], default_bc=BCType.NEUMANN, default_value=5.0), 0.125,
        natural_bc="gradient",
    )  # fmt: skip
    assert by_default.sum() == pytest.approx(0.125 * 5.0 * 4.0, rel=1e-12)
    named = BoundaryConditions(
        dimension=2,
        segments=[BCSegment(name="L", bc_type=BCType.NEUMANN, value=5.0, boundary=name)],
        default_bc=BCType.NO_FLUX,
    )
    _, on_the_wall = assemble_robin_terms(basis, named, 0.125, natural_bc="gradient")
    assert on_the_wall.sum() == pytest.approx(0.125 * 5.0 * 1.0, rel=1e-12)
    assert np.allclose(basis.doflocs[0, np.flatnonzero(on_the_wall)], 0.0)


def _walls(**condition) -> list[BCSegment]:
    """Two priority-1 segments on x_max placed by region, y in (0, 0.3) and y in (0.45, 1): on the 5 x 5 square they
    govern the facets [0, 0.25] and [0.5, 1] and leave [0.25, 0.5] to whatever else is on x_max."""
    return [
        BCSegment(name="w1", boundary="x_max", region={"y": (0.0, 0.3)}, priority=1, **condition),
        BCSegment(name="w2", boundary="x_max", region={"y": (0.45, 1.0)}, priority=1, **condition),
    ]


_EXIT_ON_X_MAX = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G, boundary="x_max")
#: The refusal's advice: on only the facet it should govern, at a priority above the segments that took its nodes.
_ADVISED_EXIT = BCSegment(
    name="exit", bc_type=BCType.DIRICHLET, value=G, boundary="x_max", region={"y": (0.3, 0.45)}, priority=2
)


@pytest.mark.parametrize(
    "walls",
    [{"bc_type": BCType.NO_FLUX}, {"bc_type": BCType.DIRICHLET, "value": 0.0}],
    ids=["natural-walls", "dirichlet-walls"],
)
def test_a_dirichlet_segment_the_corner_rule_leaves_no_node_is_refused_and_the_advice_runs(walls):
    """Review 1 of #2643, B1. On the 5 x 5 square an exit on x_max governs the facet y in [0.25, 0.5] between two
    priority-1 walls placed by region; the corner rule gives both its nodes to the walls, so P1 FEM imposed no exit at
    all. It is refused, naming the walls. The advice runs whether the walls are natural or DIRICHLET(0) (review 2 of
    #2643, B1, found the earlier advice -- equal priority, declared after -- failing for DIRICHLET walls, where a tie
    goes to the segment declared first): the exit restricted to its facet at priority 2 holds both nodes at 0.7."""
    from mfgarchon.alg.numerical.fem.bc_adapter import get_dirichlet_dofs_and_values

    basis = _square_basis()
    refused = BoundaryConditions(dimension=2, segments=[_EXIT_ON_X_MAX, *_walls(**walls)], default_bc=BCType.NO_FLUX)
    with pytest.raises(ValueError, match=r"segment 'exit' governs 1 boundary facet\(s\) but holds none of their nodes"):
        get_dirichlet_dofs_and_values(basis, refused)
    advised = BoundaryConditions(dimension=2, segments=[*_walls(**walls), _ADVISED_EXIT], default_bc=BCType.NO_FLUX)
    dofs, values = get_dirichlet_dofs_and_values(basis, advised)
    on_exit = np.isclose(basis.doflocs[0, dofs], 1.0) & (basis.doflocs[1, dofs] > 0.2) & (basis.doflocs[1, dofs] < 0.55)
    np.testing.assert_allclose(basis.doflocs[1, dofs[on_exit]], [0.25, 0.5])
    np.testing.assert_allclose(values[on_exit], G)


def test_a_segment_one_of_whose_nodes_goes_to_the_same_condition_is_not_refused():
    """Review 2 of #2643, B2. When one of the segments that took the exit's nodes imposes the exit's own condition --
    w1 = DIRICHLET(0.7), w2 = NO_FLUX -- the exit's condition still holds at the node w1 took, so it is not refused:
    the system is the one an exit restricted to the facets it keeps would give."""
    from mfgarchon.alg.numerical.fem.bc_adapter import get_dirichlet_dofs_and_values

    basis = _square_basis()
    w1 = BCSegment(name="w1", bc_type=BCType.DIRICHLET, value=G, boundary="x_max", region={"y": (0.0, 0.3)}, priority=1)
    w2 = BCSegment(name="w2", bc_type=BCType.NO_FLUX, boundary="x_max", region={"y": (0.45, 1.0)}, priority=1)
    mixed = BoundaryConditions(dimension=2, segments=[_EXIT_ON_X_MAX, w1, w2], default_bc=BCType.NO_FLUX)
    restricted_exit = BCSegment(
        name="exit", bc_type=BCType.DIRICHLET, value=G, boundary="x_max", region={"y": (0.0, 0.45)}
    )
    restricted = BoundaryConditions(dimension=2, segments=[restricted_exit, w2], default_bc=BCType.NO_FLUX)
    dofs, values = get_dirichlet_dofs_and_values(basis, mixed)
    expected_dofs, expected_values = get_dirichlet_dofs_and_values(basis, restricted)
    assert sorted(zip(dofs.tolist(), values.tolist(), strict=True)) == sorted(
        zip(expected_dofs.tolist(), expected_values.tolist(), strict=True)
    )
    np.testing.assert_allclose(basis.doflocs[:, dofs].T[np.argsort(basis.doflocs[1, dofs])], [[1.0, 0.0], [1.0, 0.25]])


def test_an_exit_on_a_face_one_element_thick_between_higher_priority_walls_is_refused():
    """Review 1 of #2643, B1's second case: on a box one element thick in y, an exit on x_max at priority 0 between
    NO_FLUX y_min and y_max at priority 1 has both its nodes taken, and is refused. A same-condition neighbour is
    effect-neutral: a DIRICHLET(0.7) y_min and y_max take the nodes and impose the exit's condition there."""
    from mfgarchon.alg.numerical.fem.assembly import create_basis
    from mfgarchon.alg.numerical.fem.bc_adapter import get_dirichlet_dofs_and_values
    from mfgarchon.alg.numerical.fem.mesh_adapter import _tag_axis_aligned_boundaries

    basis = create_basis(
        _tag_axis_aligned_boundaries(skfem.MeshTri.init_tensor(np.linspace(0.0, 1.0, 5), np.array([0.0, 1.0]))),
        order=1,
    )
    exit_ = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G, boundary="x_max")

    def sides(**condition) -> list[BCSegment]:
        return [BCSegment(name=face, boundary=face, priority=1, **condition) for face in ("y_min", "y_max")]

    with pytest.raises(ValueError, match=r"segment 'exit' governs 1 boundary facet\(s\) but holds none of their nodes"):
        get_dirichlet_dofs_and_values(
            basis,
            BoundaryConditions(
                dimension=2, segments=[exit_, *sides(bc_type=BCType.NO_FLUX)], default_bc=BCType.NO_FLUX
            ),
        )
    same = BoundaryConditions(
        dimension=2, segments=[exit_, *sides(bc_type=BCType.DIRICHLET, value=G)], default_bc=BCType.NO_FLUX
    )
    dofs, values = get_dirichlet_dofs_and_values(basis, same)
    assert np.isclose(basis.doflocs[0, dofs], 1.0).sum() == 2
    np.testing.assert_allclose(values, G)


def test_a_region_name_is_the_mesh_tag_of_that_name():
    """Review 1 of #2643, B2(i). On a mesh, ``region_name`` is read as the tag of that name, with no alias reading: a
    DIRICHLET segment with ``region_name="exit_tag"`` condenses that tag's nodes (here x = 1) and no other, where main
    ignored the field and condensed the whole boundary. A name that is no tag is refused."""
    basis = _square_basis({"exit_tag": lambda x: np.isclose(x[0], 1.0)})
    by_tag = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G, region_name="exit_tag")
    np.testing.assert_allclose(
        _dirichlet_x(basis, BoundaryConditions(dimension=2, segments=[by_tag], default_bc=BCType.NO_FLUX)), 1.0
    )
    aliased = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=G, region_name="right")
    with pytest.raises(ValueError, match=r"names region_name='right'.*a region_name is read only as a tag"):
        _dirichlet_x(basis, BoundaryConditions(dimension=2, segments=[aliased], default_bc=BCType.NO_FLUX))


def test_a_tag_that_also_holds_interior_facets_is_refused():
    """Review 1 of #2643, B2(ii). A tag holding facets inside the domain -- here the line x = 1/2 -- is refused when a
    segment names it: a boundary condition governs boundary facets, and restricting the tag to them would drop the
    rest in silence. Main condensed the interior nodes."""
    from mfgarchon.alg.numerical.fem.assembly import create_basis
    from mfgarchon.alg.numerical.fem.mesh_adapter import _tag_axis_aligned_boundaries

    xs = np.linspace(0.0, 1.0, 5)
    mesh = _tag_axis_aligned_boundaries(skfem.MeshTri.init_tensor(xs, xs)).with_boundaries(
        {"mid": lambda x: np.isclose(x[0], 0.5)}, boundaries_only=False
    )
    basis = create_basis(mesh, order=1)
    named = BCSegment(name="cut", bc_type=BCType.DIRICHLET, value=G, boundary="mid")
    with pytest.raises(ValueError, match=r"the mesh tag 'mid', which also holds facets inside the domain"):
        _dirichlet_x(basis, BoundaryConditions(dimension=2, segments=[named], default_bc=BCType.NO_FLUX))

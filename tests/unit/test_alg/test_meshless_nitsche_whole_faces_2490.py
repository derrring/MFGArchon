"""The meshless Nitsche path places conditions on whole faces, by the face owner (#2490, #2512 row B1 stage B1c-2).

`nitsche._dirichlet_faces` asks `BoundaryConditions.segment_for_face` which segment governs each bounding-box face,
and imposes Dirichlet data on the faces a Dirichlet condition governs, ``default_bc`` included. The owner places a
face label and ``normal_direction`` on an axis-aligned face, and refuses a segment that governs no face and a face
nothing governs. A ``region`` or ``sdf_region`` placement that decides a Dirichlet condition is refused here, since the
owner decides it on the problem's grid, not on the faces Nitsche integrates (#2717, #2712). Before B1c-2 this path had
its own face reader, `_segment_faces`, which refused every restricted placement, and before
#2490 it applied such a segment to every face it named (a 1-D Dirichlet segment on ``region={0: (0.5, 1)}`` absorbed
0.6652, the both-walls figure, where x_max alone gives 0.8326).
"""

import pytest
from test_meshless_galerkin_declares_bcs import XS, _problem

import numpy as np

from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc

CLASSES = [MeshlessGalerkinHJBSolver, MeshlessGalerkinFPSolver]


def _bc(*segments, default=BCType.NO_FLUX, default_value=None):
    return BoundaryConditions(dimension=1, segments=list(segments), default_bc=default, default_value=default_value)


def _fp_mass_ratio(bc) -> float:
    solver = MeshlessGalerkinFPSolver(_problem(bc), XS, delta=0.35)
    n = solver.n_dof
    M = np.asarray(solver.solve_fp_system(np.ones(n), np.zeros((5, n))))
    x = XS[:, 0]
    return float(np.trapezoid(M[-1], x) / np.trapezoid(M[0], x))


def _dirichlet(**where):
    return BCSegment(name="d", bc_type=BCType.DIRICHLET, value=0.0, **where)


def test_normal_direction_on_an_axis_aligned_face_governs_that_face():
    """An axis-aligned face has one outward normal, so ``normal_direction`` names a whole face exactly in any
    dimension, and the owner gives it that face: the same solve as ``boundary="x_max"``."""
    assert _fp_mass_ratio(_bc(_dirichlet(normal_direction=np.array([1.0])))) == _fp_mass_ratio(
        _bc(_dirichlet(boundary="x_max"))
    )


def _bound_to_a_grid(bc, n, bounds=((0.0, 1.0), (0.0, 1.0))):
    """``bc`` as the problem's grid binds it, which is where the owner decides a ``region``."""
    from mfgarchon.geometry import TensorProductGrid

    grid = TensorProductGrid(bounds=list(bounds), Nx_points=[n, n], boundary_conditions=bc)
    return grid, grid.get_boundary_conditions()


@pytest.mark.parametrize(
    "case",
    [
        "1-D region",
        "1-D sdf_region",
        "1-D no-flux region before a Dirichlet face",
        "1-D no-flux region beside a Dirichlet default",
        "2-D region on 70% of x_max, a 3x3 grid",
        "2-D region on 70% of x_max, a 5x5 grid",
        "2-D region on 90% of x_max, a 5x5 grid",
        "2-D region beyond the cloud's face",
        "2-D region marked on a 3x3 grid, named by region_name",
        "2-D region marked on a 3x3 grid, named x_max",
    ],
)
def test_a_region_or_sdf_region_placement_is_refused_and_a_face_label_is_the_remedy(case):
    """#2717 review 1, B2: the owner decides a ``region`` or an ``sdf_region`` on the problem's grid, not on the box
    faces Nitsche integrates, so a Dirichlet condition placed that way is refused, as before B1c-2. On a 3x3 grid a
    region over 70% of x_max read as the whole face and imposed Dirichlet on all of it; a region over 90% did the same
    on 5x5; a region at x > 0.95 gave a cloud on [0.1, 0.9]^2 its x_max face; a region marked on a 3x3 grid over 70%
    of x_max and named by ``region_name`` read as the whole face too, and so did one named "x_max", since the grid's mark
    decides before the label (review 2). The refusal names #2712, and its remedy,
    a face label, assembles. A natural condition placed by a region is refused where it decides where a Dirichlet
    condition lies: before a Dirichlet segment in the face order, or beside a Dirichlet default."""
    from mfgarchon.alg.numerical.meshless_galerkin.discretization import discretization_from_cloud
    from mfgarchon.alg.numerical.meshless_galerkin.nitsche import (
        assemble_nitsche_terms,
        refuse_what_nitsche_cannot_place,
    )

    refusal = r"placed by region, sdf_region or a marked region_name.*#2712"
    if case.startswith("1-D"):
        wall = BCSegment(name="w", bc_type=BCType.NO_FLUX, region={0: (0.5, 1.0)}, priority=1)
        bc = {
            "1-D region": _bc(_dirichlet(region={0: (0.5, 1.0)})),
            "1-D sdf_region": _bc(_dirichlet(sdf_region=lambda x: np.asarray(x)[..., 0] - 0.5)),
            "1-D no-flux region before a Dirichlet face": _bc(wall, _dirichlet(boundary="x_max")),
            "1-D no-flux region beside a Dirichlet default": _bc(wall, default=BCType.DIRICHLET, default_value=0.0),
        }[case]
        with pytest.raises(NotImplementedError, match=refusal):
            MeshlessGalerkinFPSolver(_problem(bc), XS, delta=0.35)
        return
    marked = case.startswith("2-D region marked")
    if case == "2-D region beyond the cloud's face":
        side = np.linspace(0.1, 0.9, 9)
        segment, n, delta = _dirichlet(region={0: (0.95, 1.0)}), 11, 0.35
    elif marked:
        side = np.linspace(0.0, 1.0, 9)
        name = "x_max" if case.endswith("x_max") else "exit"
        segment, n, delta = _dirichlet(region_name=name), 3, 3.5 / 8
    else:
        side = np.linspace(0.0, 1.0, 9)
        span = (0.3, 1.0) if "70%" in case else (0.05, 0.95)
        segment, n, delta = _dirichlet(boundary="x_max", region={1: span}), 3 if "3x3" in case else 5, 3.5 / 8
    cloud = np.array([[x, y] for x in side for y in side])
    disc = discretization_from_cloud(cloud, delta=delta, degree=2, n_gauss=4)
    grid, bc = _bound_to_a_grid(BoundaryConditions(dimension=2, segments=[segment], default_bc=BCType.NO_FLUX), n)
    if marked:
        grid.mark_region(name, predicate=lambda x: (np.asarray(x)[..., 0] > 0.99) & (np.asarray(x)[..., 1] > 0.3))
    with pytest.raises(NotImplementedError, match=refusal):
        refuse_what_nitsche_cannot_place(bc, 2)
    face = BoundaryConditions(dimension=2, segments=[_dirichlet(boundary="x_max")], default_bc=BCType.NO_FLUX)
    refuse_what_nitsche_cannot_place(face, 2)
    block, _ = assemble_nitsche_terms(disc, face, 0.045, 20.0, 4)
    assert block is not None
    assert block.nnz > 0


def test_a_region_named_face_is_that_face_only():
    """``region_name="x_max"`` is the face x_max: the same solve as ``boundary="x_max"``."""
    named = _fp_mass_ratio(_bc(_dirichlet(region_name="x_max")))
    explicit = _fp_mass_ratio(_bc(_dirichlet(boundary="x_max")))
    both = _fp_mass_ratio(_bc(_dirichlet(boundary="x_min"), _dirichlet(boundary="x_max")))
    assert named == explicit
    assert abs(named - both) > 0.1, "x_max alone must not absorb like both walls"


@pytest.mark.parametrize("cls", CLASSES)
def test_boundary_all_covers_every_face(cls):
    """``boundary="all"`` governs every face, as the point resolver says (#1953). A NO_FLUX "all" segment
    leaves no face to a DIRICHLET default, so it constructs; #2523 refused it."""
    segment = BCSegment(name="nf", bc_type=BCType.NO_FLUX, boundary="all")
    cls(_problem(_bc(segment, default=BCType.DIRICHLET, default_value=0.0)), XS, delta=0.35)


def test_a_dirichlet_all_segment_is_imposed_on_every_face():
    assert _fp_mass_ratio(_bc(_dirichlet(boundary="all"))) == _fp_mass_ratio(dirichlet_bc(dimension=1, value=0.0))


def test_the_fp_absorbs_a_shared_dirichlet_because_the_translator_says_so():
    """The FP assembles its Nitsche data load from the BC it holds, which the weak-form base read
    through `fp_view_of_shared_bc`: a shared DIRICHLET(0.5) is an exit with value 0, so the FP solves
    bit-identically to DIRICHLET(0). No second site decides it (#2512, convention row 5)."""
    assert _fp_mass_ratio(dirichlet_bc(dimension=1, value=0.5)) == _fp_mass_ratio(dirichlet_bc(dimension=1, value=0.0))
    assert _fp_mass_ratio(dirichlet_bc(dimension=1, value=0.0)) < _fp_mass_ratio(no_flux_bc(dimension=1))


def _nf(**where):
    return BCSegment(name="nf", bc_type=BCType.NO_FLUX, **where)


def test_a_face_belongs_to_the_highest_priority_segment_that_names_it():
    """A priority -1 Dirichlet "all" under a priority 1 no-flux x_min is a wall on x_max only, as
    `get_bc_at_point` resolves it. Imposed on both faces it absorbed 0.6652 (main raised)."""
    layered = _bc(_dirichlet(boundary="all", priority=-1), _nf(boundary="x_min", priority=1))
    separated = _bc(_nf(boundary="x_min"), _dirichlet(boundary="x_max"))
    assert _fp_mass_ratio(layered) == pytest.approx(_fp_mass_ratio(separated), abs=1e-12)


@pytest.mark.parametrize(
    "segments",
    [
        pytest.param(lambda: (_nf(priority=1), _dirichlet(boundary="x_max")), id="a higher-priority global segment"),
        pytest.param(lambda: (_nf(boundary="x_max"), _dirichlet(boundary="x_max")), id="equal priority, no-flux first"),
        pytest.param(
            lambda: (_dirichlet(boundary="x_max"), _nf(boundary="x_max")), id="equal priority, Dirichlet first"
        ),
        pytest.param(
            lambda: (_dirichlet(priority=-1), _nf(boundary="x_min", priority=1), _nf(boundary="x_max", priority=1)),
            id="a global Dirichlet under higher-priority no-flux faces",
        ),
    ],
)
def test_a_segment_that_governs_no_face_is_refused(segments):
    """A segment another one outranks on every face it names governs nothing, and its condition would be dropped:
    the owner refuses it (#2512 row B1). Before B1c-2 this path solved each, as the higher-ranked segment alone."""
    with pytest.raises(ValueError, match="govern no face"):
        MeshlessGalerkinFPSolver(_problem(_bc(*segments())), XS, delta=0.35)


def test_the_hjb_takes_each_faces_value_from_its_own_segment():
    """A Dirichlet "all" (g = 0) under a Dirichlet x_max (g = 1): each face takes its own segment's value.
    Imposing both on x_max averaged them (u(1) 0.5067); main raised on the "all" segment."""
    layered = _bc(
        BCSegment(name="all", bc_type=BCType.DIRICHLET, value=0.0, boundary="all", priority=-1),
        BCSegment(name="right", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_max", priority=1),
    )
    separated = _bc(
        BCSegment(name="left", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min"),
        BCSegment(name="right", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_max"),
    )
    values = []
    for bc in (layered, separated):
        solver = MeshlessGalerkinHJBSolver(_problem(bc), XS, delta=0.35)
        n = solver.n_dof
        U = solver.solve_hjb_system(
            M_density=np.ones((5, n)), U_terminal=np.cos(np.pi * XS[:, 0]), U_coupling_prev=np.zeros((5, n))
        )
        values.append(np.asarray(U))
    np.testing.assert_allclose(values[0], values[1], rtol=0, atol=1e-12)


@pytest.mark.parametrize("cls", CLASSES)
def test_a_lower_priority_segment_cannot_take_a_face_and_is_not_read(cls):
    """Only segments up to the last Dirichlet one can claim a face from it, so a lower-priority segment
    this path cannot place does not make the BC refused."""
    cls(_problem(_bc(_dirichlet(boundary="x_max", priority=1), _nf(region={0: (0.0, 0.5)}))), XS, delta=0.35)


@pytest.mark.parametrize("zero", [np.int64(0), np.zeros(1), None], ids=["int64", "size1_array", "none"])
def test_a_zero_the_translator_accepts_is_a_zero_here(zero):
    """The FP holds the translated BC, which leaves a verifiably zero value as it is; the Nitsche data
    reader must agree with that owner (`_describe_bc_value`), as main's FP did by never reading it."""
    assert _fp_mass_ratio(
        _bc(_nf(boundary="x_min"), BCSegment(name="d", bc_type=BCType.DIRICHLET, value=zero, boundary="x_max"))
    ) == _fp_mass_ratio(_bc(_nf(boundary="x_min"), _dirichlet(boundary="x_max")))


@pytest.mark.parametrize("value", [np.int64(1), np.float32(1.0), np.ones(1)], ids=["int64", "float32", "size1_array"])
def test_any_real_scalar_is_a_dirichlet_value(value):
    """The HJB imposes the value; a NumPy scalar or a size-1 array is the number it holds."""

    def u(v):
        bc = _bc(_nf(boundary="x_min"), BCSegment(name="d", bc_type=BCType.DIRICHLET, value=v, boundary="x_max"))
        solver = MeshlessGalerkinHJBSolver(_problem(bc), XS, delta=0.35)
        n = solver.n_dof
        return np.asarray(
            solver.solve_hjb_system(
                M_density=np.ones((5, n)), U_terminal=np.cos(np.pi * XS[:, 0]), U_coupling_prev=np.zeros((5, n))
            )
        )

    np.testing.assert_array_equal(u(value), u(1.0))


@pytest.mark.parametrize(
    "value", [np.complex128(1j), np.array([1j]), np.array([1 + 5j])], ids=["scalar", "imag", "mixed"]
)
def test_a_complex_dirichlet_value_is_refused(value):
    """`float()` on a NumPy complex drops the imaginary part with only a ComplexWarning; the HJB would have
    imposed g = 0 for 1j and g = 1 for 1 + 5j (main raised)."""
    bc = _bc(_nf(boundary="x_min"), BCSegment(name="d", bc_type=BCType.DIRICHLET, value=value, boundary="x_max"))
    solver = MeshlessGalerkinHJBSolver(_problem(bc), XS, delta=0.35)
    n = solver.n_dof
    with pytest.raises(NotImplementedError, match="complex"):
        solver.solve_hjb_system(
            M_density=np.ones((5, n)), U_terminal=np.cos(np.pi * XS[:, 0]), U_coupling_prev=np.zeros((5, n))
        )


def test_a_segment_placed_on_no_face_is_refused_at_construction():
    """``boundary="inlet"`` names no face of a box, so the owner cannot tell which face it governs: refused when the
    solver is built, where the Nitsche terms are decided, not at the first solve."""
    bc = _bc(BCSegment(name="inlet", bc_type=BCType.NO_FLUX, boundary="inlet", priority=1), _dirichlet())
    with pytest.raises(ValueError, match="names no face"):
        MeshlessGalerkinFPSolver(_problem(bc), XS, delta=0.35)

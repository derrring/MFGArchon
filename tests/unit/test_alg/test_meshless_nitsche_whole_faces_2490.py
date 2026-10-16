"""The meshless Nitsche path places conditions on whole faces, by the face owner (#2490, #2512 row B1 stage B1c-2).

`nitsche._dirichlet_faces` asks `BoundaryConditions.segment_for_face` which segment governs each bounding-box face,
and imposes Dirichlet data on the faces a Dirichlet condition governs, ``default_bc`` included. The owner places
``region``, ``sdf_region``, ``normal_direction`` and ``region_name`` where they cover a whole face, and refuses a
partial cover, a segment that governs no face, a placement it cannot decide, and a face nothing governs. Before
B1c-2 this path had its own face reader, `_segment_faces`, which refused every restricted placement, and before
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


@pytest.mark.parametrize(
    ("restriction", "face"),
    [
        ({"region": {0: (0.5, 1.0)}}, "x_max"),
        ({"sdf_region": lambda x: np.asarray(x)[..., 0] - 0.5}, "x_min"),
        ({"normal_direction": np.array([1.0])}, "x_max"),
    ],
    ids=["region", "sdf_region", "normal_direction"],
)
def test_a_placement_that_covers_a_whole_face_governs_that_face(restriction, face):
    """In 1-D a face is one point, so each of these placements covers one face whole, and the owner gives it that
    face: the same solve as ``boundary=face``. This path refused all three before B1c-2."""
    assert _fp_mass_ratio(_bc(_dirichlet(**restriction))) == _fp_mass_ratio(_bc(_dirichlet(boundary=face)))


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

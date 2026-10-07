"""The meshless Nitsche path places conditions on whole faces only, and refuses what it cannot place (#2490).

`nitsche._segment_faces` gives the bounding-box faces a segment names; `_dirichlet_faces` gives each Dirichlet
segment the faces no earlier segment in the BC's priority order claims (ties by declaration order), as
`get_bc_at_point` reads them. The Nitsche assembly imposes Dirichlet data there, the pair's pure-Neumann
answer comes from the same placement, and the default-DIRICHLET guard counts coverage with `_segment_faces`.
Before, the assembly applied a segment whose placement it could not read to every face it named, or to every
face when it named none (a 1-D Dirichlet segment on ``region={0: (0.5, 1)}`` absorbed 0.6652, the both-walls
figure, where x_max alone gives 0.8326), and the guard read `BoundaryConditions._segment_covers`.
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


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "restriction",
    [
        {"region": {0: (0.5, 1.0)}},
        {"sdf_region": lambda x: np.asarray(x)[..., 0] - 0.5},
        {"normal_direction": np.array([1.0])},
    ],
    ids=["region", "sdf_region", "normal_direction"],
)
def test_a_dirichlet_segment_on_part_of_the_boundary_is_refused(cls, restriction):
    with pytest.raises(NotImplementedError, match="#2490") as exc:
        cls(_problem(_bc(_dirichlet(**restriction))), XS, delta=0.35)
    if "sdf_region" in restriction:
        # The withdrawn curved-domain route says so, and where it comes back.
        assert "withdrawn" in str(exc.value)
        assert "#1139" in str(exc.value)


@pytest.mark.parametrize("cls", CLASSES)
def test_a_restricted_segment_does_not_count_as_covering_its_face(cls):
    """NO_FLUX on part of the boundary with a DIRICHLET default: the point resolver gives x_max to the
    default, which Nitsche does not read. Main solved that wall as no-flux (FP mass ratio 1.0000)."""
    segment = BCSegment(name="nf", bc_type=BCType.NO_FLUX, region={0: (0.0, 0.5)})
    with pytest.raises(NotImplementedError, match="#2490"):
        cls(_problem(_bc(segment, default=BCType.DIRICHLET, default_value=0.5)), XS, delta=0.35)


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


def test_an_unscoped_segment_of_higher_priority_takes_every_face():
    """No-flux with no boundary at priority 1 over a Dirichlet x_max: no-flux everywhere (1.0000). Main
    imposed the Dirichlet too (0.8326, the x_max-wall figure)."""
    layered = _bc(_nf(priority=1), _dirichlet(boundary="x_max"))
    assert _fp_mass_ratio(layered) == pytest.approx(_fp_mass_ratio(no_flux_bc(dimension=1)), abs=1e-12)


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


def test_a_non_dirichlet_sdf_segment_is_refused_by_the_coverage_guard_not_as_the_curved_route():
    segment = BCSegment(name="nf", bc_type=BCType.NO_FLUX, sdf_region=lambda x: np.asarray(x)[..., 0] - 0.5)
    with pytest.raises(NotImplementedError, match="#2490") as exc:
        MeshlessGalerkinFPSolver(_problem(_bc(segment, default=BCType.DIRICHLET, default_value=0.0)), XS, delta=0.35)
    assert "withdrawn" not in str(exc.value)


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


def test_equal_priorities_go_by_declaration_order():
    """Ties are read in declaration order, as `get_bc_at_point` reads them: the first segment naming a
    face takes it."""
    no_flux_first = _bc(_nf(boundary="x_max"), _dirichlet(boundary="x_max"))
    dirichlet_first = _bc(_dirichlet(boundary="x_max"), _nf(boundary="x_max"))
    assert _fp_mass_ratio(no_flux_first) == pytest.approx(_fp_mass_ratio(no_flux_bc(dimension=1)), abs=1e-12)
    assert _fp_mass_ratio(dirichlet_first) == pytest.approx(
        _fp_mass_ratio(_bc(_nf(boundary="x_min"), _dirichlet(boundary="x_max"))), abs=1e-12
    )


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


def test_a_segment_refused_for_outranking_a_dirichlet_one_says_so():
    """A non-Dirichlet segment is read only because it can take a face from a Dirichlet one; the refusal
    names that reason, not just its unplaceable boundary name."""
    bc = _bc(BCSegment(name="inlet", bc_type=BCType.NO_FLUX, boundary="inlet", priority=1), _dirichlet())
    with pytest.raises(NotImplementedError, match="outranks a Dirichlet segment"):
        MeshlessGalerkinFPSolver(_problem(bc), XS, delta=0.35)

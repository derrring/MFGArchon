"""The meshless Nitsche path places conditions on whole faces only, and refuses what it cannot place (#2490).

One predicate, `nitsche._segment_faces`, says which bounding-box faces a segment governs; the Nitsche
assembly imposes Dirichlet data there and the default-DIRICHLET guard counts coverage with it. Before,
the assembly stretched a segment restricted to part of a face onto whole faces (a 1-D Dirichlet segment on
``region={0: (0.5, 1)}`` absorbed 0.6652, the both-walls figure, where x_max alone gives 0.8326), and the
guard read `BoundaryConditions._segment_covers`, which counts such a segment as covering its face.
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

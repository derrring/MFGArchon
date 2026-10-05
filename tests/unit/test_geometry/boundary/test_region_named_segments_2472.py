"""A region-named BC segment governs the faces its region covers, and no others (#2472).

Measured on main at `1c84a29c`, with the documented idiom (`mark_region(name, boundary=...)` +
`mixed_bc_from_regions`) on a 9 x 9 grid and u = 1:

- one region, an outlet on `x_max`: `is_uniform` ignored `region_name`, so the outlet's Dirichlet ghost
  was written on all four faces;
- two regions, outlet on `x_max` and inlet on `x_min`: `get_bc_type_at_boundary` answered DIRICHLET on
  every face (a segment with no `boundary` covered them all), while the ghosts were the mirror on every
  face. The ghost path's region lookup indexed the grid's flat region mask with a 2-D face index, raised
  IndexError, and swallowed it at debug level, so the outlet reached no face.

The solvers and the FDM operators read a BC without the geometry, so a whole-face region is resolved to
its faces where the geometry is in hand: when `mixed_bc_from_regions` builds the BC. Refusing it at read
time instead, the first version of this fix, broke every region-route solve, including zero-flux ones
that main solved bitwise equal to plain no-flux (review 1 of #2491). A region covering part of a face
keeps `region_name`, and the face-level readers resolve it through one resolver or refuse it.

Also pinned: a uniform BC's unused `default_bc` no longer makes `geometric_operations` report it as mixed
(#2472 item 2).
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, pad_array_with_ghosts
from mfgarchon.geometry.boundary.applicator_fdm import FDMApplicator
from mfgarchon.geometry.boundary.bc_utils import geometric_operations
from mfgarchon.geometry.boundary.conditions import mixed_bc_from_regions

N = 9
_UNIT = np.array([[0.0, 1.0], [0.0, 1.0]])


def _grid(points=(N, N)):
    d = len(points)
    return TensorProductGrid(
        bounds=[(0.0, 1.0)] * d, Nx_points=list(points), boundary_conditions=no_flux_bc(dimension=d)
    )


def _faces(padded):
    return {"x_min": padded[0, 1:-1], "x_max": padded[-1, 1:-1], "y_min": padded[1:-1, 0], "y_max": padded[1:-1, -1]}


_OUTLET = BCSegment(name="outlet", bc_type=BCType.DIRICHLET, value=0.0)
# A value the default does not share, so whether the inlet reaches x_min is visible.
_INLET = BCSegment(name="inlet", bc_type=BCType.DIRICHLET, value=2.0)
_WALL = BCSegment(name="wall", bc_type=BCType.NO_FLUX)


@pytest.mark.parametrize("with_inlet", [False, True], ids=["one_region", "two_regions"])
def test_a_whole_face_region_becomes_its_face(with_inlet):
    grid = _grid()
    grid.mark_region("outlet", boundary="x_max")
    config = {"outlet": _OUTLET, "default": _WALL}
    if with_inlet:
        grid.mark_region("inlet", boundary="x_min")
        config["inlet"] = _INLET
    bc = mixed_bc_from_regions(grid, config)

    assert "default" in config, "mixed_bc_from_regions removed the caller's default entry"
    assert {s.boundary for s in bc.segments} == ({"x_max", "x_min"} if with_inlet else {"x_max"})
    assert all(s.region_name is None for s in bc.segments)

    # The readers that have no geometry -- the solvers' and the FDM operators' route -- now agree.
    expected = {"x_max": -1.0, "x_min": 3.0 if with_inlet else 1.0, "y_min": 1.0, "y_max": 1.0}
    spacing = 1.0 / (N - 1)
    for padded in (
        pad_array_with_ghosts(np.ones((N, N)), bc, spacing=spacing),
        pad_array_with_ghosts(np.ones((N, N)), bc, geometry=grid),
    ):
        for face, ghost in _faces(padded).items():
            np.testing.assert_array_equal(ghost, expected[face], err_msg=face)
    assert bc.get_bc_type_at_boundary("y_min") is BCType.NO_FLUX


def test_a_zero_flux_region_route_solves_as_plain_no_flux():
    """The solve the first version of this fix broke: main gave it bitwise equal to plain no-flux."""
    nx, nt = 21, 10
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    grid.mark_region("outlet", boundary="x_max")
    region_bc = mixed_bc_from_regions(
        grid, {"outlet": BCSegment(name="outlet", bc_type=BCType.NEUMANN, value=0.0), "default": _WALL}
    )

    def solve(bc):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.disable(logging.WARNING)
            try:
                problem = MFGProblem(
                    model=Model(
                        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.3
                    ),
                    domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=bc),
                    conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: np.cos(np.pi * x), T=0.5),
                    Nt=nt,
                )
                x = np.linspace(0.0, 1.0, nx)
                return HJBFDMSolver(problem).solve_hjb_system(
                    np.ones((nt + 1, nx)), np.cos(np.pi * x), np.zeros((nt + 1, nx))
                )
            finally:
                logging.disable(logging.NOTSET)

    np.testing.assert_array_equal(solve(region_bc), solve(no_flux_bc(dimension=1)))


def _raw(region_name, bc_type=BCType.DIRICHLET):
    """A region-named segment built directly, not through mixed_bc_from_regions."""
    return BoundaryConditions(
        segments=[BCSegment(name="s", bc_type=bc_type, value=0.0, region_name=region_name)],
        dimension=2,
        default_bc=BCType.NO_FLUX,
        domain_bounds=_UNIT,
    )


def test_a_raw_region_segment_goes_through_the_shared_resolver():
    grid = _grid()
    grid.mark_region("outlet", boundary="x_max")
    bc = _raw("outlet")
    assert not bc.is_uniform

    ghosts = _faces(pad_array_with_ghosts(np.ones((N, N)), bc, geometry=grid))
    np.testing.assert_array_equal(ghosts["x_max"], -1.0)
    for face in ("x_min", "y_min", "y_max"):
        np.testing.assert_array_equal(ghosts[face], 1.0, err_msg=face)

    # The readers without a geometry refuse a name that is not a face label ...
    with pytest.raises(ValueError, match="known only to the geometry"):
        bc.get_bc_type_at_boundary("y_min")
    with pytest.raises(ValueError, match="known only to the geometry"):
        FDMApplicator(dimension=2).enforce_values(np.ones((N, N)), bc, spacing=(0.125, 0.125))

    # ... and resolve one that is (control).
    labelled = _raw("x_max")
    assert labelled.get_bc_type_at_boundary("x_max") is BCType.DIRICHLET
    assert labelled.get_bc_type_at_boundary("y_min") is BCType.NO_FLUX
    field = FDMApplicator(dimension=2).enforce_values(np.ones((N, N)), labelled, spacing=(0.125, 0.125))
    np.testing.assert_array_equal(field[-1, :], 0.0)
    np.testing.assert_array_equal(field[0, 1:-1], 1.0)


def test_a_region_covering_part_of_a_face_keeps_its_name_and_is_refused_face_level():
    grid = _grid()
    grid.mark_region("strip", predicate=lambda x: x[:, 0] > 0.6)
    bc = mixed_bc_from_regions(grid, {"strip": _OUTLET, "default": _WALL})
    assert bc.segments[0].region_name == "strip"
    with pytest.raises(ValueError, match="covers part of the"):
        pad_array_with_ghosts(np.ones((N, N)), bc, geometry=grid)


def test_a_region_covering_no_whole_face_is_refused():
    grid = _grid()
    grid.mark_region("centre", predicate=lambda x: np.all(np.abs(x - 0.5) < 0.2, axis=1))
    with pytest.raises(ValueError, match="covers no whole boundary face"):
        pad_array_with_ghosts(np.ones((N, N)), _raw("centre"), geometry=grid)


@pytest.mark.parametrize(
    ("points", "face"),
    [((9, 2), "y_min"), ((2, 9), "x_max"), ((2, 2), "x_max")],
    ids=["9x2_y_min", "2x9_x_max", "2x2_x_max"],
)
def test_a_whole_face_on_a_two_point_axis_is_not_mistaken_for_a_partial_cover(points, face):
    """On a two-point axis every point of a face is a corner shared with another face; those corners
    are not a partial cover of the neighbouring faces (review 1 of #2491, B2)."""
    grid = _grid(points)
    grid.mark_region("r", boundary=face)
    bc = mixed_bc_from_regions(grid, {"r": _OUTLET, "default": _WALL})
    assert [s.boundary for s in bc.segments] == [face]


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [(None, {"periodic"}), ("x_min", {"periodic", "reflect"})],
    ids=["uniform_default_unread", "mixed_default_read"],
)
def test_only_a_default_some_face_reads_counts_as_an_operation(boundary, expected):
    bc = BoundaryConditions(
        segments=[BCSegment(name="p", bc_type=BCType.PERIODIC, boundary=boundary)],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0]]),
    )
    assert geometric_operations(bc) == expected


def test_a_template_that_restricts_itself_is_refused_as_before():
    """The template must not carry its own face or range: before #2472 BCSegment refused that mix, and the
    face conversion must not turn it into a silently overwritten or stretched condition (review 2, N1)."""
    grid = _grid()
    grid.mark_region("outlet", boundary="x_max")
    for template in (
        BCSegment(name="o", bc_type=BCType.DIRICHLET, value=0.0, boundary="y_min"),
        BCSegment(name="o", bc_type=BCType.DIRICHLET, value=0.0, region={"y": (0.4, 0.6)}),
    ):
        with pytest.raises(ValueError, match="Cannot mix region specification methods"):
            mixed_bc_from_regions(grid, {"outlet": template, "default": _WALL})


def test_a_kept_region_may_not_be_named_like_a_face():
    """A partial region named "top" would be read as the whole y_max face by every reader without the
    geometry (review 2, N2)."""
    grid = _grid()
    grid.mark_region("top", predicate=lambda x: (x[:, 1] > 0.99) & (x[:, 0] < 0.4))
    with pytest.raises(ValueError, match="reads as a face"):
        mixed_bc_from_regions(grid, {"top": _OUTLET, "default": _WALL})


def test_a_region_spanning_two_faces_becomes_both():
    grid = _grid()
    mask = np.zeros((N, N), dtype=bool)
    mask[-1, :] = True  # x_max
    mask[:, -1] = True  # y_max
    grid.mark_region("exit", mask=mask.ravel())
    bc = mixed_bc_from_regions(grid, {"exit": _OUTLET, "default": _WALL})
    assert sorted((s.name, s.boundary) for s in bc.segments) == [("outlet[x_max]", "x_max"), ("outlet[y_max]", "y_max")]
    ghosts = _faces(pad_array_with_ghosts(np.ones((N, N)), bc, spacing=1.0 / (N - 1)))
    for face, expected in (("x_max", -1.0), ("y_max", -1.0), ("x_min", 1.0), ("y_min", 1.0)):
        np.testing.assert_array_equal(ghosts[face], expected, err_msg=face)

    capped = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, flux_capacity=1.0)
    with pytest.raises(ValueError, match="flux_capacity"):
        mixed_bc_from_regions(grid, {"exit": capped, "default": _WALL})


def test_a_zero_flux_region_covering_part_of_a_face_enforces_as_plain_no_flux():
    """The route InterpolationApplicator's zero-flux short-circuit serves now that whole-face regions
    become named faces."""
    from mfgarchon.geometry.boundary.applicator_interpolation import InterpolationApplicator

    grid = _grid()
    grid.mark_region("strip", predicate=lambda x: x[:, 0] > 0.6)
    region_bc = mixed_bc_from_regions(
        grid, {"strip": BCSegment(name="s", bc_type=BCType.NEUMANN, value=0.0), "default": _WALL}
    )
    assert region_bc.segments[0].region_name == "strip"
    rng = np.random.default_rng(2472)
    field = rng.standard_normal((N, N))
    applicator = InterpolationApplicator(dimension=2)
    np.testing.assert_array_equal(
        applicator.enforce_values(field.copy(), region_bc),
        applicator.enforce_values(field.copy(), no_flux_bc(dimension=2)),
    )


def test_the_only_region_on_every_face_is_uniform():
    """Splitting it per face made it mixed: FP-FVM refused it, and a periodic one ran through the per-face
    path and solved 0.135 away from periodic_bc under FDM_UPWIND (review 3, NB1)."""
    from mfgarchon.geometry.boundary import periodic_bc

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=periodic_bc(dimension=1))
    grid.mark_region("ends", predicate=lambda x: np.isclose(x[:, 0], 0.0) | np.isclose(x[:, 0], 1.0))
    bc = mixed_bc_from_regions(grid, {"ends": BCSegment(name="p", bc_type=BCType.PERIODIC)})
    assert bc.is_uniform
    assert [(s.boundary, s.region_name) for s in bc.segments] == [(None, None)]
    field = np.sin(2 * np.pi * np.linspace(0.0, 1.0, N)) + np.linspace(0.0, 0.3, N)
    np.testing.assert_array_equal(
        pad_array_with_ghosts(field, bc, spacing=1.0 / (N - 1)),
        pad_array_with_ghosts(field, periodic_bc(dimension=1), spacing=1.0 / (N - 1)),
    )


def test_a_template_naming_a_region_is_accepted_on_a_whole_face():
    """Before #2472 the config key replaced a template's own region_name; the face conversion must start
    from that validated segment, not refuse the template (review 3, NB2)."""
    grid = _grid()
    grid.mark_region("outlet", boundary="x_max")
    template = BCSegment(name="o", bc_type=BCType.DIRICHLET, value=0.0, region_name="other")
    bc = mixed_bc_from_regions(grid, {"outlet": template, "default": _WALL})
    assert [(s.boundary, s.region_name) for s in bc.segments] == [("x_max", None)]

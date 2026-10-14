"""GFDM classifies its boundary points through the point owner (#2512 row B1, stage B1c-1).

`HJBGFDMSolver._preclassify_boundary_points` asks `BoundaryConditions.segment_for_point` for every boundary collocation
point, with the problem's geometry, its ``BOUNDARY_TOL`` and its own equivalence, `_same_bc_row`: two segments are the
same when GFDM writes the same row for them. Before, it walked the segments itself, in its own priority order, on the
first face `identify_boundary_face` returned.

The row at a corner is written along the normal of R1 (#2512 plan, gap 2): the first face, in face order, whose
segment at the point writes the row the point's segment writes. The pins:

1. On `test_hjb_gfdm_bc_newton_residual`'s split x_max, the owner gives the x_max corners to ``bottom`` and ``top``,
   whose condition the x_max strips share, so R1 keeps the x_max normal GFDM wrote before. The face of the chosen
   segment itself (R2) would turn those rows to y.
2. A Neumann corner whose two faces carry different data: u = a x + b y meets every face's datum, so the corner row
   is exact only along the normal of the face whose datum it carries. The first face at the point would give |a - b|.
3. GFDM's equivalence decides the corner: NEUMANN(0) on a strip and NO_FLUX on the wall write the same row, so x_max
   answers the wall and the corner is the Dirichlet's; under the owner's default they differ and the corner is the
   strip's, a Neumann row where GFDM writes a Dirichlet one.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

NX = 9


def _solver(bc: BoundaryConditions) -> HJBGFDMSolver:
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[NX, NX], boundary_conditions=bc)
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)))
    conditions = Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0)
    problem = MFGProblem(model=model, domain=grid, conditions=conditions, Nt=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(
            problem,
            collocation_points=grid.get_spatial_grid(),
            delta=2.0 / (NX - 1),
            k_neighbors=12,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            monotonicity_scheme="none",
        )


def _index(solver: HJBGFDMSolver, point: list[float]) -> int:
    distance = np.linalg.norm(solver.collocation_points - np.asarray(point), axis=1)
    return int(np.argmin(distance))


def test_the_split_x_max_corners_keep_their_x_normal():
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(name="right_below", bc_type=BCType.NO_FLUX, boundary="x_max", region={"y": (0.0, 0.4)}),
            BCSegment(name="right_above", bc_type=BCType.NO_FLUX, boundary="x_max", region={"y": (0.6, 1.0)}),
            BCSegment(
                name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", region={"y": (0.4, 0.6)}, priority=1
            ),
        ],
    )
    solver = _solver(bc)
    for corner, name in (([1.0, 0.0], "bottom"), ([1.0, 1.0], "top")):
        i = _index(solver, corner)
        assert solver._bc_segment_per_point[i].name == name, corner
        assert solver._bc_normal_per_point[i] == pytest.approx([1.0, 0.0]), corner


def test_a_neumann_corner_row_is_exact_along_the_face_whose_datum_it_carries():
    a, b = 1.0, 3.0
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="x_min", bc_type=BCType.NEUMANN, value=-a, boundary="x_min"),
            BCSegment(name="x_max", bc_type=BCType.NEUMANN, value=a, boundary="x_max"),
            BCSegment(name="y_min", bc_type=BCType.NEUMANN, value=-b, boundary="y_min", priority=1),
            BCSegment(name="y_max", bc_type=BCType.NEUMANN, value=b, boundary="y_max"),
        ],
    )
    solver = _solver(bc)
    u = a * solver.collocation_points[:, 0] + b * solver.collocation_points[:, 1]
    i = _index(solver, [0.0, 0.0])
    assert solver._bc_segment_per_point[i].name == "y_min"  # priority 1 takes the corner
    bc_enum, segment, normal = solver._classify_boundary_point(i, 0, True, None, None)
    row, target = solver._bc_row_for_point(i, bc_enum, segment, normal, 2, solver.n_points, 0.0)
    assert abs(row @ u - target) < 1e-8 * abs(a - b)


def test_gfdm_judges_a_corner_by_the_row_it_writes():
    from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import _same_bc_row

    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="wall", bc_type=BCType.NO_FLUX),
            BCSegment(
                name="strip", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max", region={"y": (0.75, 1.0)}, priority=2
            ),
            BCSegment(name="D", bc_type=BCType.DIRICHLET, value=5.0, boundary="y_max", priority=1),
        ],
    )
    solver = _solver(bc)
    i = _index(solver, [1.0, 1.0])
    assert solver._bc_segment_per_point[i].name == "D"
    assert solver._classify_boundary_point(i, 0, True, None, None)[0] is BCType.DIRICHLET
    grid = solver.problem.geometry
    corner = solver.collocation_points[i]
    assert bc.segment_for_point(corner, geometry=grid, equivalent=_same_bc_row).name == "D"
    assert bc.segment_for_point(corner, geometry=grid).name == "strip"  # the owner's default, for contrast

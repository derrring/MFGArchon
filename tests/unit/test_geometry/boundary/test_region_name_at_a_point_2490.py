"""#2490 B1e-1: a ``region_name`` segment is read at a point in the order the face rule reads it -- the marking
geometry's mask, then a face label, then a refusal -- and a point the mask cannot be read at is refused, not read as
outside the region."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc


@pytest.mark.parametrize(
    "row", ["a face label, read by both readers", "a point off the marking grid", "a point between the nodes"]
)
def test_a_region_named_segment_is_read_at_a_point_as_on_its_face(row):
    """- ``region_name="x_max"`` with no geometry names the face x_max: the face reader gives it x_max, and the point
      reader required a geometry and refused it at a point of that face. Past the wall with no face, at (1.2, 0.5), it
      reads as ``boundary="x_max"`` does, where the clip to the BC's box once put that point on x_max.
    - A grid that marks "strip" (y > 0.4), read through a BC whose box reaches x = 2: the point (2, 0.5) lies on its
      x_max, a whole grid width off the grid. The point rule read the mask at the node it clamped the point to,
      (1, 0.5), so the strip governed a point the grid never marked. It is refused, and the message's remedy -- the
      BC's box equal to the grid's -- reads the strip at (1, 0.5).
    - Between the nodes the mask is read at the nearest node: (1, 0.45) lies 0.05 from the marked node y = 0.5 and 0.2
      from the unmarked y = 0.25, and reads the strip. The point rule truncated the index to the node below, 0.25."""
    if row == "a face label, read by both readers":
        bc = BoundaryConditions(
            segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, region_name="x_max")],
            dimension=2,
            default_bc=BCType.NO_FLUX,
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )
        assert bc.segment_for_face("x_max").name == "exit"
        assert bc.segment_for_point(np.array([1.0, 0.5])).name == "exit"
        assert bc.segment_for_point(np.array([0.0, 0.5])).name == "default"
        by_face = BoundaryConditions(
            segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max")],
            dimension=2,
            default_bc=BCType.NO_FLUX,
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )
        past = np.array([1.2, 0.5])
        assert bc.segment_for_point(past).name == by_face.segment_for_point(past).name == "default"
        return
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[5, 5], boundary_conditions=no_flux_bc(dimension=2)
    )
    grid.mark_region("strip", predicate=lambda x: x[:, 1] > 0.4)
    segments = [
        BCSegment(name="strip", bc_type=BCType.DIRICHLET, value=0.0, region_name="strip"),
        BCSegment(name="rest", bc_type=BCType.NO_FLUX, value=0.0),
    ]
    inside = BoundaryConditions(segments=segments, dimension=2, domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]))
    if row == "a point between the nodes":
        assert inside.segment_for_point(np.array([1.0, 0.45]), geometry=grid).name == "strip"
        assert inside.segment_for_point(np.array([1.0, 0.3]), geometry=grid).name == "rest"
        return
    wide = BoundaryConditions(segments=segments, dimension=2, domain_bounds=np.array([[0.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(
        ValueError, match=r"lies outside the grid that marks it.*domain_bounds equal to that grid's box"
    ):
        wide.segment_for_point(np.array([2.0, 0.5]), geometry=grid)
    assert inside.segment_for_point(np.array([1.0, 0.5]), geometry=grid).name == "strip"


@pytest.mark.parametrize(
    "row",
    [
        "a particle past a wall",
        "an implicit domain that marks it",
        "another dimension",
        "outside the grid's own box",
        "the walls of a cell-centred grid",
        "a descending axis",
    ],
)
def test_a_region_named_segment_is_read_or_refused_truly_off_the_grid(row):
    """- A grid that marks the whole x_max face as "exit", its own BC that exit beside walls: FP-particle steps a
      particle past a wall and reads it there. It is read at its nearest point of the BC's box, as main read it at the
      grid's clamped node: (1.2, 0.75) on x_max is the exit and (0.6, -0.3) on y_min a wall, and the solve absorbs.
      (#2744's review 1 measured the solve raising "lies off the grid" when such a point was refused.)
    - A Hypersphere marks "exit" but the owner reads a region by its mask on a grid's nodes: the refusal says so and
      names ``sdf_region``, where it said the Hypersphere defined no region of that name.
    - A point of three coordinates on a 2-D marking grid, which main read as outside the region, is refused for its
      dimension, not for coordinates the grid has.
    - A BC box that reaches past the marking grid's own box, here from x = -0.2 on a grid on [0, 1]: (-0.2, 0.5) is
      refused, where main read it at the clamped node. On this non-uniform grid (end cell 0.05) it is four end cells off.
      A point past the BC's box too, (-0.5, 0.5), is refused by the point the caller passed.
    - A cell-centred grid, nodes 0.05 ... 0.95 on the box [0, 1]: its walls lie half a cell past its end nodes, inside
      its box, and are read at the end nodes, as at main.
    - x nodes 1, 0.75, ..., 0 and "exit" at x > 0.5: the face rule reads the first node as x_min (#2490) and puts the
      exit on x_min, so a point reading by coordinate would put it on x_max. It is refused, by the point the caller
      passed: (1.2, 0.5), past the wall, is read at (1, 0.5)."""
    if row == "a particle past a wall":
        from mfgarchon import Conditions, MFGProblem, Model
        from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
        )
        grid.mark_region("exit", boundary="x_max")
        bc = BoundaryConditions(
            segments=[
                BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, region_name="exit"),
                BCSegment(name="walls", bc_type=BCType.NO_FLUX, value=0.0),
            ],
            dimension=2,
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )
        grid.set_boundary_conditions(bc)
        assert bc.get_bc_at_point(np.array([1.2, 0.75]), "x_max", geometry=grid).name == "exit"
        assert bc.get_bc_at_point(np.array([0.6, -0.3]), "y_min", geometry=grid).name == "walls"
        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.3
            ),
            domain=grid,
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.3),
            Nt=3,
        )
        solver = FPParticleSolver(problem, num_particles=400, seed=0)
        solver.solve_fp_system(M_initial=np.ones((11, 11)), show_progress=False)
        assert solver.total_absorbed > 0, "no particle reached the exit"
        return
    from mfgarchon.geometry import Hypersphere

    segment = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, region_name="exit")
    if row == "an implicit domain that marks it":
        disk = Hypersphere(center=[0.5, 0.5], radius=0.5)
        disk.mark_region("exit", predicate=lambda x: x[:, 0] > 0.9)
        with pytest.raises(ValueError, match=r"Hypersphere marks a region of that name.*sdf_region"):
            segment.matches_point(np.array([1.0, 0.5]), "x_max", np.array([[0.0, 1.0], [0.0, 1.0]]), geometry=disk)
        return
    if row == "another dimension":
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[5, 5], boundary_conditions=no_flux_bc(dimension=2)
        )
        grid.mark_region("exit", predicate=lambda x: x[:, 1] > 0.4)
        with pytest.raises(ValueError, match=r"has 3 coordinates where the TensorProductGrid that marks it has 2 axes"):
            segment.matches_point(np.array([1.0, 0.75, 0.0]), None, None, geometry=grid)
        return
    xs = {
        "outside the grid's own box": [0.0, 0.05, 0.1, 0.55, 1.0],
        "the walls of a cell-centred grid": list(np.linspace(0.05, 0.95, 10)),
        "a descending axis": [1.0, 0.75, 0.5, 0.25, 0.0],
    }[row]
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[len(xs), 3],
        spacing_type="custom",
        custom_coordinates=[np.array(xs), np.array([0.0, 0.5, 1.0])],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    grid.mark_region("exit", predicate=lambda x: x[:, 0] > 0.5)
    segments = [segment, BCSegment(name="rest", bc_type=BCType.NO_FLUX, value=0.0)]
    if row == "outside the grid's own box":
        wide = BoundaryConditions(segments=segments, dimension=2, domain_bounds=np.array([[-0.2, 1.0], [0.0, 1.0]]))
        with pytest.raises(
            ValueError, match=r"lies outside the grid that marks it, whose box on axis 0 is \[0.0, 1.0\]"
        ):
            wide.segment_for_point(np.array([-0.2, 0.5]), geometry=grid)
        with pytest.raises(ValueError, match=r"the point \[-0.5, 0.5\] lies outside the grid that marks it"):
            wide.segment_for_point(np.array([-0.5, 0.5]), geometry=grid)
        return
    bc = BoundaryConditions(segments=segments, dimension=2, domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]))
    if row == "a descending axis":
        with pytest.raises(
            ValueError, match=r"do not strictly ascend on axis 0, .*the point \[1.2, 0.5\].*first node as its min face"
        ):
            bc.segment_for_point(np.array([1.2, 0.5]), geometry=grid)
        return
    assert bc.segment_for_point(np.array([1.0, 0.5]), geometry=grid).name == "exit"
    assert bc.segment_for_point(np.array([0.0, 0.5]), geometry=grid).name == "rest"

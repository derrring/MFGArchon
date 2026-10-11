"""FEM has one facet rule for every segment type (#2595, rows 2 and 7).

`_find_segment_facets`, which places the ROBIN and NEUMANN(g) loads, kept a rule of its own after #2593 gave the
Dirichlet DOFs `_segment_boundary_facets`. It fell back to the whole boundary on a name the mesh does not tag: a
ROBIN(0.3) named "left" gave u(0, x_max) = 0.0565 against 0.0001 named "x_min", where the BC layer says x_max is
NO_FLUX (row 2). And it placed nothing on a tag that holds no facet: on the disc a ROBIN named "x_max" changed the
solution by exactly 0 (row 7). It now returns `_segment_boundary_facets`, so both raise as a Dirichlet segment's
name does, on every route a ROBIN or NEUMANN(g) segment reaches: HJB-FEM's shared BC and FP-FEM's own ROBIN.
FP-FEM refuses an explicit NEUMANN before any facet is placed.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc

NT = 10
_REACH = [("hjb-shared", "ROBIN"), ("hjb-shared", "NEUMANN"), ("fp-explicit", "ROBIN")]
_REACH_IDS = ["hjb-shared-ROBIN", "hjb-shared-NEUMANN", "fp-explicit-ROBIN"]


def _interval(bc: BoundaryConditions):
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    geometry = Mesh1D(bounds=(0.0, 1.0), num_elements=16)
    geometry.generate_mesh()
    geometry.boundary_conditions = bc
    return geometry


def _disc(tag_right: bool):
    """The unit disc (``MeshTri.init_circle(3)``): its axis tags, placed by bounding box, hold no facets. With
    ``tag_right`` the boundary faces with x > 0.9 at both ends carry MeshData tag 1, as ``region_1``."""

    def build(bc: BoundaryConditions):
        import skfem

        from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
        from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

        mesh_data = skfem_to_meshdata(skfem.MeshTri.init_circle(3))
        if tag_right:
            x = mesh_data.vertices[mesh_data.boundary_faces][..., 0]
            mesh_data.boundary_tags = np.where(x.min(axis=1) > 0.9, 1, 0)
        geometry = Mesh2D(domain_type="circle", bounds=(-1.0, 1.0, -1.0, 1.0))
        geometry.mesh_data = mesh_data
        geometry.boundary_conditions = bc
        return geometry

    return build


def _segment(kind: str, name):
    if kind == "ROBIN":
        return BCSegment(name="r", bc_type=BCType.ROBIN, value=0.3, alpha=1.0, beta=1.0, boundary=name)
    return BCSegment(name="n", bc_type=BCType.NEUMANN, value=1.0, boundary=name)


def _bc(dimension: int, *segments: BCSegment) -> BoundaryConditions:
    return BoundaryConditions(dimension=dimension, segments=list(segments), default_bc=BCType.NO_FLUX)


def _solve(route: str, domain, bc: BoundaryConditions):
    """HJB-FEM with ``bc`` as the shared BC (u(t, x), terminal 0, M = 1), or FP-FEM with ``bc`` as its own BC and a
    no-flux shared one (m(t, x), m0 = 1). Returns the DOF x-coordinates and the field at every time level."""
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    geometry = domain(bc if route == "hjb-shared" else no_flux_bc(dimension=bc.dimension))
    # Unit mass on the domain's own measure: Mesh1D's uniform-cell measure gives a constant c the mass c (n+1)/n.
    c = 16 / 17 if isinstance(geometry, Mesh1D) else 1.0
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.4)
    conditions = Conditions(
        m_initial=lambda x: c + 0.0 * np.asarray(x, dtype=float)[..., 0], u_terminal=lambda x: 0.0, T=0.5
    )
    problem = MFGProblem(model=model, domain=geometry, conditions=conditions, Nt=NT)
    if route == "hjb-shared":
        solver = HJBFEMSolver(problem)
        n = solver._disc.dof_coordinates.shape[0]
        field = solver.solve_hjb_system(M_density=np.ones((NT + 1, n)), U_terminal=np.zeros(n))
    else:
        solver = FPFEMSolver(problem, order=1, boundary_conditions=bc)
        n = solver._disc.dof_coordinates.shape[0]
        field = solver.solve_fp_system(M_initial=np.ones(n), potential_field=None)
    return solver._disc.dof_coordinates[:, 0], np.asarray(field)


@pytest.mark.parametrize(("route", "kind"), _REACH, ids=_REACH_IDS)
def test_a_load_named_by_an_alias_is_refused_and_one_named_by_its_tag_stays_on_its_face(route, kind):
    """Row 2. "left" is x_min to the BC layer and no tag of the mesh: it used to load the whole boundary, and now
    raises, naming the tag the alias means. Named "x_min" the load stays on x_min. Measured over every time level:
    it moves the solution at x_min by 5.6e-2 to 2.2e-1, and at x_max by at most 1.8e-4. A whole-boundary load moves
    x_max by 5.6e-2 or more, so 1e-3 separates the two."""
    with pytest.raises(
        ValueError, match=r"no such tagged boundary \(available: .*The BC layer reads 'left' as 'x_min'"
    ):
        _solve(route, _interval, _bc(1, _segment(kind, "left")))

    x, base = _solve(route, _interval, _bc(1))
    _, named = _solve(route, _interval, _bc(1, _segment(kind, "x_min")))
    moved = np.abs(named - base).max(axis=0)
    assert moved[np.isclose(x, 0.0)][0] > 0.04
    assert moved[np.isclose(x, 1.0)][0] < 1e-3


@pytest.mark.parametrize(("route", "kind"), _REACH, ids=_REACH_IDS)
def test_a_load_named_by_an_empty_tag_is_refused_and_the_tagged_one_is_placed(route, kind):
    """Row 7. On the disc "x_max" is a tag with no facet in it: the load used to be placed nowhere (the solution moved
    by exactly 0), and it now raises with the advice to tag the mesh. Tagged and named "region_1", the load is placed:
    it moves the solution by 6.0e-2 to 2.4e-1, measured."""
    with pytest.raises(ValueError, match=r"none of its tags holds a facet.*Tag the mesh before naming a face"):
        _solve(route, _disc(tag_right=False), _bc(2, _segment(kind, "x_max")))

    _, base = _solve(route, _disc(tag_right=True), _bc(2))
    _, tagged = _solve(route, _disc(tag_right=True), _bc(2, _segment(kind, "region_1")))
    assert np.abs(tagged - base).max() > 0.04

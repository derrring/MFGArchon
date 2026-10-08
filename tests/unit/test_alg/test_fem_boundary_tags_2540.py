"""A mesh's tagged boundary faces become ``region_<tag>`` on the facets they are, not on facet ids equal to
their positions (#2540).

``MeshData.boundary_faces`` lists faces by vertex; scikit-fem numbers facets on its own. Reading a face's
position as its facet id put ``region_1`` of a ``Mesh1D`` (the right wall, position 1) on facet 1, the
interior vertex x = h, so a Dirichlet value or an exit was imposed there and x = 1 was left free.
"""

from __future__ import annotations

import pytest
import skfem

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fem.mesh_adapter import meshdata_to_skfem, skfem_to_meshdata
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D


def _solve_with_exit_on(boundary: str):
    geo = Mesh1D(bounds=(0.0, 1.0), num_elements=10)
    geo.generate_mesh()
    geo.boundary_conditions = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="L", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="R", bc_type=BCType.DIRICHLET, value=1.0, boundary=boundary),
        ],
    )
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.8),
        domain=geo,
        conditions=Conditions(m_initial=lambda p: 1.0 / 1.1, u_terminal=lambda p: 0.0, T=0.5),
        Nt=5,
    )
    result = problem.solve()
    return np.asarray(result.U), np.asarray(result.M)


def test_a_region_tag_on_the_default_path_solves_as_the_wall_it_names():
    """``Mesh1D`` tags its right wall 1. Through ``problem.solve`` (FEM_P1 on a mesh), the exit on ``region_1``
    must be the exit on ``x_max``: U = 1 and m = 0 there. Measured before the fix, at this setup: U(0) = 0.1235
    and m(T) = 1.089 at x = 1, with U = 1 and m = 7e-9 at x = 0.1."""
    U_tag, M_tag = _solve_with_exit_on("region_1")
    U_wall, M_wall = _solve_with_exit_on("x_max")
    np.testing.assert_array_equal(U_tag, U_wall)
    np.testing.assert_array_equal(M_tag, M_wall)


_X, _Y = np.linspace(0.0, 1.0, 5), np.linspace(0.0, 0.6, 4)
_X3, _Y3, _Z3 = np.linspace(0.0, 1.0, 3), np.linspace(0.0, 0.8, 3), np.linspace(0.0, 0.6, 3)
_MESHES = {
    "triangle": lambda: skfem.MeshTri.init_tensor(_X, _Y),
    "quad": lambda: skfem.MeshQuad.init_tensor(_X, _Y),
    "tetrahedron": lambda: skfem.MeshTet.init_tensor(_X3, _Y3, _Z3),
    # scikit-fem stores hexahedron facets cyclically, not sorted like the others
    "hexahedron": lambda: skfem.MeshHex.init_tensor(_X3, _Y3, _Z3),
}


def _box_meshdata(element: str):
    """A box mesh's MeshData, boundary faces in scikit-fem's boundary order, each face's vertices reversed so
    that the match cannot rely on vertex order."""
    md = skfem_to_meshdata(_MESHES[element]())
    md.boundary_faces = md.boundary_faces[:, ::-1].copy()
    return md


def _walls(md) -> list[np.ndarray]:
    """For each axis wall (axis 0 low, axis 0 high, axis 1 low, ...), the boundary faces lying on it."""
    corners = md.vertices[md.boundary_faces]
    walls = []
    for axis in range(md.vertices.shape[1]):
        for bound in (md.vertices[:, axis].min(), md.vertices[:, axis].max()):
            walls.append(np.all(np.isclose(corners[..., axis], bound), axis=1))
    return walls


@pytest.mark.parametrize("element", sorted(_MESHES))
def test_tagged_faces_land_on_their_own_facets(element):
    """Every wall but the last gets its own tag, the last stays untagged (0), and one face is listed twice."""
    md = _box_meshdata(element)
    walls = _walls(md)
    faces = md.boundary_faces.copy()
    tags = np.zeros(len(faces), dtype=np.int64)
    for tag, on in enumerate(walls[:-1], start=1):
        tags[on] = tag
    twice = np.flatnonzero(tags == 1)[0]
    md.boundary_faces = np.vstack([faces, faces[twice]])
    md.boundary_tags = np.append(tags, 1)

    converted = meshdata_to_skfem(md)

    assert "region_0" not in converted.boundaries
    for tag, on in enumerate(walls[:-1], start=1):
        placed = [tuple(sorted(f)) for f in converted.facets[:, converted.boundaries[f"region_{tag}"]].T]
        assert len(placed) == len(set(placed)), f"region_{tag} lists a facet twice"
        assert set(placed) == {tuple(sorted(f)) for f in faces[on]}, f"region_{tag} is not its wall"


def test_untagged_faces_add_no_region_whatever_the_tag_count():
    """``Mesh2D`` hands over an all-zero placeholder sized by vertex count, not face count."""
    md = _box_meshdata("triangle")
    md.boundary_tags = np.zeros(len(md.vertices), dtype=np.int64)
    assert len(md.boundary_tags) != len(md.boundary_faces)

    converted = meshdata_to_skfem(md)

    assert not any(name.startswith("region_") for name in converted.boundaries)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("not_a_facet", "is not a facet of the mesh"),
        ("one_tag_short", "boundary tags for"),
        ("no_faces", "5 boundary tags for 0 boundary faces"),
    ],
)
def test_a_tag_that_cannot_be_placed_raises(case, message):
    md = _box_meshdata("triangle")
    md.boundary_tags = np.zeros(len(md.boundary_faces), dtype=np.int64)
    md.boundary_tags[0] = 2
    if case == "not_a_facet":
        md.boundary_faces[0] = [0, md.vertices.shape[0] - 1]  # opposite corners: no edge joins them
    elif case == "one_tag_short":
        md.boundary_tags = md.boundary_tags[:-1]
    else:
        md.boundary_faces = md.boundary_faces[:0]
        md.boundary_tags = np.full(5, 2, dtype=np.int64)
    with pytest.raises(ValueError, match=message):
        meshdata_to_skfem(md)

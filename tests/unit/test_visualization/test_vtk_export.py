"""
Unit tests for VTK export of FEM/unstructured-mesh MFG solutions (Issue #895).

Covers ``mfgarchon.visualization.vtk_export``:

- Round-trip: export a mesh + nodal fields to ``.vtu``, read back with
  ``meshio.read``, and assert points / cells / point_data survive.
- Time series: per-timestep ``.vtu`` + a ParaView ``.pvd`` collection, with each
  timestep round-tripping its own field.
- Fail-loud validation: wrong field length and unsupported element types raise.
- End-to-end smoke: solve a tiny FEM MFG problem and export the result.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import meshio
import pytest

import numpy as np

from mfgarchon.geometry.meshes.mesh_data import MeshData
from mfgarchon.visualization.vtk_export import export_mesh_solution_vtk, export_time_series_vtk


def _two_triangle_mesh() -> MeshData:
    """Unit square split into two triangles: 4 vertices, 2 elements, 2D."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    return MeshData(
        vertices=vertices,
        elements=elements,
        element_type="triangle",
        boundary_tags=np.zeros(0, dtype=np.int64),
        element_tags=np.zeros(2, dtype=np.int64),
        boundary_faces=np.zeros((0, 2), dtype=np.int64),
        dimension=2,
    )


def _single_tetra_mesh() -> MeshData:
    """A single reference tetrahedron: 4 vertices, 1 element, 3D."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return MeshData(
        vertices=vertices,
        elements=elements,
        element_type="tetrahedron",
        boundary_tags=np.zeros(0, dtype=np.int64),
        element_tags=np.zeros(1, dtype=np.int64),
        boundary_faces=np.zeros((0, 3), dtype=np.int64),
        dimension=3,
    )


@pytest.mark.unit
class TestSingleExportRoundTrip:
    """(a) export -> meshio.read -> points / cells / point_data match the originals."""

    def test_triangle_vtu_roundtrip(self, tmp_path):
        md = _two_triangle_mesh()
        fields = {"U": np.array([1.0, 2.0, 3.0, 4.0]), "M": np.array([0.1, 0.2, 0.3, 0.4])}

        out = export_mesh_solution_vtk(md, fields, tmp_path / "sol.vtu")
        assert out.exists()
        assert out.stat().st_size > 0

        back = meshio.read(out)

        # Points: VTK is 3D, so 2D coords are padded with a zero z-column. The (x, y)
        # block must match exactly; the padded column must be zeros.
        assert back.points.shape == (4, 3)
        np.testing.assert_array_equal(back.points[:, :2], md.vertices)
        np.testing.assert_array_equal(back.points[:, 2], np.zeros(4))

        # Cells: connectivity must survive exactly (integers).
        np.testing.assert_array_equal(back.cells_dict["triangle"], md.elements)

        # Point data: float fields round-trip (allclose).
        for name, values in fields.items():
            np.testing.assert_allclose(back.point_data[name], values)

    def test_tetra_vtu_roundtrip(self, tmp_path):
        md = _single_tetra_mesh()
        fields = {"U": np.array([1.0, -2.0, 0.5, 3.0])}

        out = export_mesh_solution_vtk(md, fields, tmp_path / "tet.vtu")
        back = meshio.read(out)

        np.testing.assert_array_equal(back.points, md.vertices)  # already 3D, no padding
        np.testing.assert_array_equal(back.cells_dict["tetra"], md.elements)
        np.testing.assert_allclose(back.point_data["U"], fields["U"])

    def test_accepts_skfem_mesh(self, tmp_path):
        """The mesh arg also accepts a skfem.Mesh (converted via the existing adapter)."""
        skfem = pytest.importorskip("skfem", reason="scikit-fem required")
        mesh = skfem.MeshTri.init_sqsymmetric().refined(1)
        n = mesh.p.shape[1]
        fields = {"U": np.arange(n, dtype=float)}

        out = export_mesh_solution_vtk(mesh, fields, tmp_path / "skfem.vtu")
        back = meshio.read(out)

        np.testing.assert_array_equal(back.points[:, :2], mesh.p.T)
        np.testing.assert_array_equal(back.cells_dict["triangle"], mesh.t.T)
        np.testing.assert_allclose(back.point_data["U"], fields["U"])


@pytest.mark.unit
class TestTimeSeries:
    """(b) time series: a .pvd indexing N per-timestep .vtu files that each round-trip."""

    def test_pvd_references_each_timestep_and_roundtrips(self, tmp_path):
        md = _two_triangle_mesh()
        n_t, n = 3, md.num_vertices
        rng = np.random.RandomState(0)
        U = rng.rand(n_t, n)
        M = rng.rand(n_t, n)
        times = np.array([0.0, 0.5, 1.0])

        pvd = export_time_series_vtk(md, {"U": U, "M": M}, times, tmp_path / "series.pvd")
        assert pvd.exists()

        # The .pvd must list exactly n_t datasets, each pointing at an existing .vtu,
        # with the correct timestep attribute.
        root = ET.parse(pvd).getroot()
        datasets = root.findall("./Collection/DataSet")
        assert len(datasets) == n_t

        for k, ds in enumerate(datasets):
            assert float(ds.attrib["timestep"]) == times[k]
            vtu = pvd.parent / ds.attrib["file"]
            assert vtu.exists()
            assert vtu.stat().st_size > 0

            back = meshio.read(vtu)
            # Each .vtu carries THAT timestep's slice of every field.
            np.testing.assert_allclose(back.point_data["U"], U[k])
            np.testing.assert_allclose(back.point_data["M"], M[k])
            np.testing.assert_array_equal(back.cells_dict["triangle"], md.elements)


@pytest.mark.unit
class TestFailLoud:
    """(c) clear, loud failures for malformed inputs (fail-fast convention)."""

    def test_field_length_mismatch_raises(self, tmp_path):
        md = _two_triangle_mesh()  # 4 vertices
        with pytest.raises(ValueError, match=r"length 3 but the mesh has 4 vertices"):
            export_mesh_solution_vtk(md, {"U": np.ones(3)}, tmp_path / "bad.vtu")

    def test_unsupported_element_type_raises(self, tmp_path):
        md = MeshData(
            vertices=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            elements=np.array([[0, 1, 2]], dtype=np.int64),
            element_type="polygon",  # not in the meshio cell map
            boundary_tags=np.zeros(0, dtype=np.int64),
            element_tags=np.zeros(1, dtype=np.int64),
            boundary_faces=np.zeros((0, 2), dtype=np.int64),
            dimension=2,
        )
        with pytest.raises(ValueError, match=r"Unsupported element_type 'polygon'.*Supported"):
            export_mesh_solution_vtk(md, {"U": np.ones(3)}, tmp_path / "bad.vtu")

    def test_non_mesh_argument_raises(self, tmp_path):
        with pytest.raises(TypeError, match=r"MeshData or skfem.Mesh"):
            export_mesh_solution_vtk("not a mesh", {"U": np.ones(3)}, tmp_path / "bad.vtu")

    def test_time_series_non_2d_field_raises(self, tmp_path):
        md = _two_triangle_mesh()
        with pytest.raises(ValueError, match=r"must be 2D"):
            export_time_series_vtk(md, {"U": np.ones(4)}, [0.0], tmp_path / "bad.pvd")

    def test_time_series_time_length_mismatch_raises(self, tmp_path):
        md = _two_triangle_mesh()
        U = np.ones((3, md.num_vertices))
        with pytest.raises(ValueError, match=r"3 timesteps but 2 times"):
            export_time_series_vtk(md, {"U": U}, [0.0, 1.0], tmp_path / "bad.pvd")


@pytest.mark.integration
def test_export_solved_fem_mfg_result(tmp_path):
    """(d) End-to-end: solve a tiny FEM MFG problem, export the (Nt+1, N) U/M, re-read."""
    skfem = pytest.importorskip("skfem", reason="scikit-fem required for FEM solve")

    from mfgarchon.alg.numerical.fem.assembly import assemble_mass
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    mesh = skfem.MeshTri.init_sqsymmetric().refined(1)
    geom = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    geom.mesh_data = skfem_to_meshdata(mesh)
    geom.boundary_conditions = no_flux_bc(dimension=2)
    components = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    problem = MFGProblem(
        geometry=geom,
        T=0.2,
        Nt=4,
        sigma=0.3,
        components=components,
        coupling_coefficient=0.5,
        boundary_conditions=no_flux_bc(dimension=2),
    )

    hjb, fp = HJBFEMSolver(problem), FPFEMSolver(problem)
    n_dof, n_t = fp._basis.N, problem.Nt
    mass = assemble_mass(fp._basis)
    m0 = np.exp(-10 * ((mesh.p[0] - 0.5) ** 2 + (mesh.p[1] - 0.5) ** 2))
    m0 /= (mass @ m0).sum()
    u_sol = np.zeros((n_t + 1, n_dof))
    m_sol = np.tile(m0, (n_t + 1, 1))
    for _ in range(3):  # a few Picard sweeps: finite, re-readable result (not convergence)
        u_new = np.asarray(hjb.solve_hjb_system(m_sol, np.zeros(n_dof), u_sol))
        m_new = np.asarray(fp.solve_fp_system(m0, u_new))
        u_sol = 0.5 * u_sol + 0.5 * u_new
        m_sol = 0.5 * m_sol + 0.5 * m_new
    assert np.all(np.isfinite(u_sol))
    assert np.all(np.isfinite(m_sol))

    # P1 FEM: n_dof == number of mesh vertices, so nodal fields export directly.
    assert n_dof == geom.mesh_data.num_vertices

    times = np.linspace(0.0, problem.T, n_t + 1)
    pvd = export_time_series_vtk(geom.mesh_data, {"U": u_sol, "M": m_sol}, times, tmp_path / "mfg.pvd")
    assert pvd.exists()
    assert pvd.stat().st_size > 0

    root = ET.parse(pvd).getroot()
    datasets = root.findall("./Collection/DataSet")
    assert len(datasets) == n_t + 1
    last = meshio.read(pvd.parent / datasets[-1].attrib["file"])
    assert last.points.shape[0] == n_dof
    np.testing.assert_allclose(last.point_data["M"], m_sol[-1])

    # Single-snapshot terminal export also re-reads cleanly.
    vtu = export_mesh_solution_vtk(geom.mesh_data, {"U": u_sol[-1], "M": m_sol[-1]}, tmp_path / "terminal.vtu")
    assert vtu.stat().st_size > 0
    assert meshio.read(vtu).points.shape[0] == n_dof

"""Integration tests for the FEM solver *classes* (not raw assembly).

Before these, no test instantiated ``HJBFEMSolver`` / ``FPFEMSolver`` — ``test_fem_coupled_mfg.py``
hand-rolls a driftless Picard loop on raw ``K``/``M`` and never touches the solver classes, so the
FEM solver path was unverified. These tests pin the three things the FEM-readiness survey found
broken and that this change fixes:

1. the factory can build a FEM pair (the ``_scheme_family``/duality wiring),
2. the FP advection operator is mass-conserving (the ``-C^T`` fix),
3. a pre-built mesh can be injected without gmsh (the ``generate_mesh`` memoization).

Coupled FEM *through ``FixedPointIterator``* is still NOT exercised here: the iterator requires a
``CartesianGrid`` geometry while the FEM solvers require an unstructured mesh (the last seam of the
coupled-FEM chain). The other two seams are now closed — the BC accessor returns a real
``BoundaryConditions`` (seam 1) and ``meshdata_to_skfem`` tags axis-aligned named walls so Dirichlet
segments resolve (seam 2, #607) — so coupled FEM is exercised via a manual Picard loop below.
"""

from __future__ import annotations

import pytest

import numpy as np

skfem = pytest.importorskip("skfem", reason="scikit-fem required for FEM tests")


def _fem_problem(refine: int = 2):
    """A gmsh-free 2D FEM MFG problem: a skfem built-in mesh injected as MeshData."""
    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    mesh = skfem.MeshTri.init_sqsymmetric().refined(refine)
    geom = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    geom.mesh_data = skfem_to_meshdata(mesh)
    # BC attaches to the GEOMETRY (the codebase pattern; MFGProblem(boundary_conditions=) is
    # dropped for mesh geometries). The geometry BC accessor then surfaces it to the solver.
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
        Nt=5,
        sigma=0.3,
        components=components,
        coupling_coefficient=0.5,
        boundary_conditions=no_flux_bc(dimension=2),
    )
    return problem, mesh


@pytest.mark.integration
class TestFEMSolverPath:
    def test_factory_creates_fem_pair_with_traits(self):
        """Issue #773 / #580: the documented factory entry point must build a FEM pair. It used to
        raise (HJB/FP inherited ``_scheme_family=GENERIC`` → duality VALIDATION_SKIPPED → factory
        raised). Both solvers now carry ``SchemeFamily.FEM`` and the pair validates as Type-A."""
        from mfgarchon.alg.base_solver import SchemeFamily
        from mfgarchon.factory.scheme_factory import NumericalScheme, create_paired_solvers

        problem, _ = _fem_problem()
        hjb, fp = create_paired_solvers(problem, NumericalScheme.FEM_P1)
        assert type(hjb).__name__ == "HJBFEMSolver"
        assert type(fp).__name__ == "FPFEMSolver"
        assert hjb._scheme_family is SchemeFamily.FEM
        assert fp._scheme_family is SchemeFamily.FEM

    @pytest.mark.parametrize("order", [1, 2])
    def test_fp_advection_is_mass_conserving(self, order):
        """The FP advection operator must have zero column sums (mass conservation). The raw
        convective form ``C[i,j] = ∫ φ_i (v·∇φ_j)`` does NOT (column sums ≈ ∫ v·∇φ_j); the
        operator is assembled as ``-C^T`` whose column sums vanish since ``Σ_i φ_i = 1``. A random
        value function gives a non-divergence-free drift, so this fails on the prior ``+C``.

        Parametrized over P1 and P2: the ``-C^T`` mass-conservation property is *order-agnostic*
        (partition of unity ``Σ_i φ_i = 1`` holds for any Lagrange order), so P2 must conserve
        mass exactly as P1 does — a regression guard for the P2 FEM path (#470)."""
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

        problem, _ = _fem_problem()
        fp = FPFEMSolver(problem, order=order)
        u_n = np.random.RandomState(0).rand(fp._basis.N)
        advection = fp._build_advection(u_n)
        max_col_sum = float(np.abs(np.asarray(advection.sum(axis=0)).ravel()).max())
        assert max_col_sum < 1e-12, f"FP advection (P{order}) not mass-conserving: max|col sum|={max_col_sum:.2e}"

    def test_gmsh_free_mesh_injection_is_returned_as_is(self):
        """generate_mesh() returns a pre-populated mesh_data unchanged (gmsh-free injected-mesh
        path / idempotent memoization) — gmsh is not importable in the default install."""
        from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
        from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

        mesh_data = skfem_to_meshdata(skfem.MeshTri.init_sqsymmetric().refined(1))
        geom = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
        geom.mesh_data = mesh_data
        assert geom.generate_mesh() is mesh_data


@pytest.mark.integration
class TestFEMBoundaryConditionResolution:
    """Seam 1 of the coupled-FEM chain: the FEM solver must receive a real ``BoundaryConditions``
    (or ``None``), NOT the boundary-handler metadata dict that ``UnstructuredMesh.
    get_boundary_conditions()`` used to return — that dict shadowed the real BC and crashed the
    bc_adapter with ``'dict' object has no attribute 'segments'``."""

    def test_mesh_bc_accessor_returns_bc_or_none_not_dict(self):
        from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
        from mfgarchon.geometry.boundary import no_flux_bc
        from mfgarchon.geometry.boundary.conditions import BoundaryConditions
        from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

        geom = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
        geom.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_sqsymmetric().refined(1))
        # no BC attached -> None (previously a {"type": "unstructured_mesh", ...} metadata dict)
        assert geom.get_boundary_conditions() is None
        # attached BC -> the real object
        geom.boundary_conditions = no_flux_bc(dimension=2)
        assert isinstance(geom.get_boundary_conditions(), BoundaryConditions)

    def test_fem_solver_receives_real_boundary_conditions(self):
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
        from mfgarchon.geometry.boundary.conditions import BoundaryConditions

        problem, _ = _fem_problem()  # helper attaches no_flux_bc to the geometry
        fp = FPFEMSolver(problem)
        assert isinstance(fp._bc, BoundaryConditions)
        assert fp._is_pure_neumann()  # no-flux is natural/Neumann

    def test_coupled_fem_no_flux_runs_and_conserves_mass(self):
        """First coupled FEM MFG through the real solver classes (manual Picard — the standard
        FixedPointIterator is still grid-only, seam 3). No-flux ⇒ mass conserved, confirming the
        -C^T advection fix end-to-end."""
        from mfgarchon.alg.numerical.fem.assembly import assemble_mass
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
        from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

        problem, mesh = _fem_problem()
        hjb, fp = HJBFEMSolver(problem), FPFEMSolver(problem)
        n_dof, n_t = fp._basis.N, problem.Nt
        mass_matrix = assemble_mass(fp._basis)
        m0 = np.exp(-10 * ((mesh.p[0] - 0.5) ** 2 + (mesh.p[1] - 0.5) ** 2))
        m0 /= (mass_matrix @ m0).sum()
        u_sol = np.zeros((n_t + 1, n_dof))
        m_sol = np.tile(m0, (n_t + 1, 1))
        for _ in range(8):
            u_new = np.asarray(hjb.solve_hjb_system(m_sol, np.zeros(n_dof), u_sol))
            m_new = np.asarray(fp.solve_fp_system(m0, u_new))
            u_sol = 0.5 * u_sol + 0.5 * u_new
            m_sol = 0.5 * m_sol + 0.5 * m_new
        assert np.all(np.isfinite(u_sol)) and np.all(np.isfinite(m_sol))
        assert np.all(m_sol >= -1e-9)
        masses = [float((mass_matrix @ m_sol[t]).sum()) for t in range(n_t + 1)]
        drift = abs(masses[-1] - masses[0]) / masses[0]
        assert drift < 1e-3, f"no-flux FEM mass drift {drift:.2%} (advection not conservative?)"


@pytest.mark.integration
class TestFEMFacetBoundaryTags:
    """Seam 2 of the coupled-FEM chain (#607): meshdata_to_skfem tags axis-aligned wall facets as
    named boundaries (x_min/x_max/...), so a BCSegment(boundary="x_min") resolves to the correct
    facet set. Before this, mesh.boundaries was None and the Dirichlet path crashed with
    "argument of type 'NoneType' is not iterable"."""

    def test_named_axis_walls_are_tagged(self):
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

        problem, _ = _fem_problem()
        fp = FPFEMSolver(problem)
        assert set(fp._skfem_mesh.boundaries) == {"x_min", "x_max", "y_min", "y_max"}

    def test_dirichlet_segment_resolves_to_correct_wall_dofs(self):
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
        from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
        from mfgarchon.core.mfg_components import MFGComponents
        from mfgarchon.core.mfg_problem import MFGProblem
        from mfgarchon.geometry.boundary.conditions import BCSegment, BCType, BoundaryConditions
        from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

        geom = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
        geom.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_sqsymmetric().refined(2))
        geom.boundary_conditions = BoundaryConditions(
            segments=[BCSegment(name="inlet", bc_type=BCType.DIRICHLET, boundary="x_min", value=0.0)],
            dimension=2,
        )
        components = MFGComponents(
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
            ),
        )
        problem = MFGProblem(geometry=geom, T=0.2, Nt=5, sigma=0.3, components=components, coupling_coefficient=0.5)
        fp = FPFEMSolver(problem)
        dofs, _vals = fp._dirichlet_dofs_and_values()  # previously raised NoneType-not-iterable
        assert len(dofs) > 0
        # every constrained dof is on the x_min wall, not the whole boundary
        assert np.allclose(fp._skfem_mesh.p[0, dofs], 0.0)
        assert len(dofs) < len(fp._skfem_mesh.boundary_nodes())  # a wall, not all boundary

    def test_untagged_mesh_falls_back_without_crashing(self):
        """If mesh.boundaries is None (no tagging), _find_segment_dofs must fall back to all
        boundary nodes, not raise."""
        from mfgarchon.alg.numerical.fem.assembly import create_basis
        from mfgarchon.alg.numerical.fem.bc_adapter import _find_segment_dofs
        from mfgarchon.geometry.boundary.conditions import BCSegment, BCType

        mesh = skfem.MeshTri.init_sqsymmetric().refined(1)  # raw skfem mesh: boundaries is None
        assert mesh.boundaries is None
        basis = create_basis(mesh, order=1)
        seg = BCSegment(name="d", bc_type=BCType.DIRICHLET, boundary="x_min", value=0.0)
        dofs = _find_segment_dofs(mesh, basis, seg)  # must not raise
        assert len(dofs) == len(mesh.boundary_nodes())  # fell back to all boundary nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

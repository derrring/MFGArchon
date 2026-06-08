"""Integration tests for the FEM solver *classes* (not raw assembly).

Before these, no test instantiated ``HJBFEMSolver`` / ``FPFEMSolver`` — ``test_fem_coupled_mfg.py``
hand-rolls a driftless Picard loop on raw ``K``/``M`` and never touches the solver classes, so the
FEM solver path was unverified. These tests pin the three things the FEM-readiness survey found
broken and that this change fixes:

1. the factory can build a FEM pair (the ``_scheme_family``/duality wiring),
2. the FP advection operator is mass-conserving (the ``-C^T`` fix),
3. a pre-built mesh can be injected without gmsh (the ``generate_mesh`` memoization).

Coupled FEM *through ``FixedPointIterator``* is NOT exercised here: the iterator currently requires
a ``CartesianGrid`` geometry, while the FEM solvers require an unstructured mesh, and the
``fem.bc_adapter`` expects a ``BoundaryConditions`` object where the solver receives a ``dict`` —
those two seams must be closed before an end-to-end coupled-FEM test is possible.
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

    def test_fp_advection_is_mass_conserving(self):
        """The FP advection operator must have zero column sums (mass conservation). The raw
        convective form ``C[i,j] = ∫ φ_i (v·∇φ_j)`` does NOT (column sums ≈ ∫ v·∇φ_j); the
        operator is assembled as ``-C^T`` whose column sums vanish since ``Σ_i φ_i = 1``. A random
        value function gives a non-divergence-free drift, so this fails on the prior ``+C``."""
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

        problem, _ = _fem_problem()
        fp = FPFEMSolver(problem)
        u_n = np.random.RandomState(0).rand(fp._basis.N)
        advection = fp._build_advection(u_n)
        max_col_sum = float(np.abs(np.asarray(advection.sum(axis=0)).ravel()).max())
        assert max_col_sum < 1e-12, f"FP advection not mass-conserving: max|col sum|={max_col_sum:.2e}"

    def test_gmsh_free_mesh_injection_is_returned_as_is(self):
        """generate_mesh() returns a pre-populated mesh_data unchanged (gmsh-free injected-mesh
        path / idempotent memoization) — gmsh is not importable in the default install."""
        from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
        from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

        mesh_data = skfem_to_meshdata(skfem.MeshTri.init_sqsymmetric().refined(1))
        geom = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
        geom.mesh_data = mesh_data
        assert geom.generate_mesh() is mesh_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

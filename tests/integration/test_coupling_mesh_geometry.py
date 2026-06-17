"""Coupled-FEM chain, seam 3: ``FixedPointIterator`` accepts unstructured-mesh geometry.

The iterator was grid-only (it required a ``CartesianGrid`` and used ``get_grid_shape()`` /
``get_grid_spacing()``), so coupled FEM could never run through the standard coupling loop — only
via a hand-rolled Picard. With the mesh branch (flat per-DOF state, unit volume element), an FEM
MFG solves through ``FixedPointIterator`` end-to-end. The grid path is unchanged (gated on
``isinstance(geometry, CartesianGrid)``); a byte-identical check on an FDM grid solve guards that.
"""

from __future__ import annotations

import pytest

import numpy as np

skfem = pytest.importorskip("skfem", reason="scikit-fem required for FEM tests")


def _fem_problem(refine: int = 2):
    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    geom = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    geom.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_sqsymmetric().refined(refine))
    geom.boundary_conditions = no_flux_bc(dimension=2)
    components = MFGComponents(
        m_initial=lambda x: float(np.exp(-10 * ((np.atleast_1d(x)[0] - 0.5) ** 2))),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
        ),
    )
    return MFGProblem(geometry=geom, T=0.2, Nt=5, sigma=0.3, components=components, coupling_coefficient=0.5)


@pytest.mark.integration
class TestFixedPointIteratorMeshGeometry:
    def test_fem_couples_through_fixed_point_iterator(self):
        """A FEM pair solves a coupled MFG through the STANDARD FixedPointIterator (no hand-rolled
        Picard). Result is finite, per-DOF-shaped (Nt+1, N_dof), non-negative, and mass-conserved
        under no-flux (the -C^T advection + the mesh-aware iterator working together)."""
        from mfgarchon.alg.numerical.coupling import FixedPointIterator
        from mfgarchon.alg.numerical.fem.assembly import assemble_mass
        from mfgarchon.factory.scheme_factory import NumericalScheme, create_paired_solvers

        problem = _fem_problem()
        hjb, fp = create_paired_solvers(problem, NumericalScheme.FEM_P1)
        res = FixedPointIterator(problem, hjb_solver=hjb, fp_solver=fp, relaxation=0.5).solve(
            max_iterations=15, tolerance=1e-5, verbose=False
        )
        m_sol = np.asarray(res.M)
        u_sol = np.asarray(res.U)
        assert m_sol.shape == (problem.Nt + 1, problem.num_spatial_points)
        assert np.all(np.isfinite(u_sol))
        assert np.all(np.isfinite(m_sol))
        assert np.all(m_sol >= -1e-9)
        mass_matrix = assemble_mass(fp._basis)
        masses = [float((mass_matrix @ m_sol[t]).sum()) for t in range(m_sol.shape[0])]
        drift = abs(masses[-1] - masses[0]) / masses[0]
        assert drift < 1e-3, f"no-flux FEM mass drift {drift:.2%} through the iterator"

    def test_iterator_rejects_unknown_geometry(self):
        """A geometry that is neither CartesianGrid nor unstructured mesh fails loud (not silently)."""
        from types import SimpleNamespace

        from mfgarchon.alg.numerical.coupling import FixedPointIterator

        problem = _fem_problem()
        # swap in a geometry-like object with an unrecognized type
        problem.geometry = SimpleNamespace(geometry_type="something_else")
        it = FixedPointIterator(problem, hjb_solver=object(), fp_solver=object())
        with pytest.raises(ValueError, match="CartesianGrid or unstructured mesh"):
            it.solve(max_iterations=1, verbose=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Issue #1489 (S2): the FEM HJB Newton correction must condense with a HOMOGENEOUS boundary lift.

The Newton correction ``delta`` has ``delta[dofs] = 0`` (``U_current`` already carries ``u = g``), so
condensing it with the actual Dirichlet values ``g`` adds a spurious ``-A[int,dofs]@g`` term that
corrupts every interior value. Vanishes for ``g = 0`` — which is why every prior FEM Dirichlet test
(all ``value=0.0``) missed it.
"""

from __future__ import annotations

import pytest

import numpy as np

skfem = pytest.importorskip("skfem", reason="scikit-fem required")


def _fem_hjb_solver_with_nonzero_dirichlet():
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.boundary.conditions import BCSegment, BCType, BoundaryConditions
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    geom = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    geom.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_sqsymmetric().refined(2))
    geom.boundary_conditions = BoundaryConditions(
        segments=[BCSegment(name="inlet", bc_type=BCType.DIRICHLET, boundary="x_min", value=1.5)],  # g != 0
        dimension=2,
    )
    components = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
        ),
    )
    problem = MFGProblem(geometry=geom, T=0.2, Nt=3, sigma=0.3, components=components, coupling_coefficient=1.0)
    return HJBFEMSolver(problem)


def test_homogeneous_condensation_drops_the_dirichlet_lift():
    solver = _fem_hjb_solver_with_nonzero_dirichlet()
    # Use the real stiffness so interior<->boundary coupling is nonzero (identity would hide the bug).
    jac = solver._K.tocsr()
    rhs = np.ones(jac.shape[0])

    _, rhs_hom = solver._apply_bc_to_system(jac, rhs, homogeneous=True)
    _, rhs_lift = solver._apply_bc_to_system(jac, rhs, homogeneous=False)

    dofs, vals = solver._dirichlet_dofs_and_values()
    assert np.any(vals != 0.0), "test requires a nonzero Dirichlet value to exercise the lift"
    interior = np.setdiff1d(np.arange(jac.shape[0]), np.asarray(dofs, dtype=int))

    # homogeneous=True => no boundary lift => rhs_int == rhs[interior]
    assert np.allclose(rhs_hom, rhs[interior]), "homogeneous condensation must drop the -A[int,dofs]@g lift"
    # non-homogeneous keeps the g-lift, so with g != 0 and coupling K they differ (the S2 corruption path)
    assert not np.allclose(rhs_hom, rhs_lift), "the two modes must differ for g != 0 (proves the lift is real)"

"""Issue #1252 (2026-06-10 audit): P2 mass-lumped nodal-gradient recovery is invalid.

Row-sum mass lumping (M_lumped = M.sum(axis=1)) assumes strictly positive lumped masses.
For P2+ Lagrange the vertex shape function integrates to ~0 over a triangle and to a negative
value over a tetrahedron, so the consistent-mass row sum at every vertex DOF is ~0 or < 0. The
old code clamped that to 1e-15 and 1/1e-15 = 1e15 turned the recovered vertex gradient into
garbage, silently feeding nonsense into H(grad u). _build_gradient_operators must now fail loud
for P2+ in the base WeakFormHJBSolver (protecting the meshless-Galerkin path), and
HJBFEMSolver overrides it to use a consistent-mass L2 projection for P2+ (#1252).

The fix exercises:
1. WeakFormHJBSolver._build_gradient_operators still fails loud for P2+ (base-class guard,
   called directly via _GradStub to confirm the meshless path is protected).
2. HJBFEMSolver._nodal_gradient with order=2 correctly recovers grad(x) = 1 at ALL DOFs
   (vertex + edge-midpoint) via the consistent-mass M^{-1} R_d solve.
3. P1 recovery is byte-identical after the refactor (no regression).
"""

import pytest

import numpy as np
from scipy import sparse

skfem = pytest.importorskip("skfem", reason="scikit-fem required for FEM gradient-recovery test")

from mfgarchon.alg.numerical.fem.assembly import assemble_mass, create_basis  # noqa: E402
from mfgarchon.alg.numerical.weak_form_hjb_solver import WeakFormHJBSolver  # noqa: E402


class _GradStub:
    """Minimal carrier exposing the attributes _build_gradient_operators reads/writes."""

    def __init__(self, M, n):
        self._M = M
        # _R_grad is only consumed AFTER the positivity guard; identity placeholders suffice.
        self._R_grad = [sparse.eye(n, format="csr"), sparse.eye(n, format="csr")]
        self._G_grad = None
        self._M_lumped_inv = None


def _mass_matrix(order):
    mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
    basis = create_basis(mesh, order=order)
    return assemble_mass(basis), basis.N


# ---------------------------------------------------------------------------
# Base-class guards (protect the meshless-Galerkin path)
# ---------------------------------------------------------------------------


def test_p2_gradient_lumping_fails_loud():
    """P2: lumped row sums are pathological (~0/negative) -> base class raises NotImplementedError.

    This verifies the WeakFormHJBSolver guard remains in place after the #1252 fix, so that
    any non-FEM subclass (e.g. MeshlessGalerkinHJBSolver) that inadvertently receives P2+
    operators still fails loud rather than silently returning garbage gradients.
    """
    M, n = _mass_matrix(order=2)
    m_lumped = np.asarray(M.sum(axis=1)).ravel()
    assert m_lumped.min() < 1e-12 * m_lumped.max(), "premise: P2 vertex DOFs should have ~0/negative lumped row sums"
    stub = _GradStub(M, n)
    with pytest.raises(NotImplementedError, match="lumping"):
        WeakFormHJBSolver._build_gradient_operators(stub)


def test_p1_gradient_lumping_ok():
    """P1: all lumped row sums strictly positive -> operators build without raising."""
    M, n = _mass_matrix(order=1)
    m_lumped = np.asarray(M.sum(axis=1)).ravel()
    assert m_lumped.min() > 1e-12 * m_lumped.max(), "P1 lumped row sums must be strictly positive"
    stub = _GradStub(M, n)
    WeakFormHJBSolver._build_gradient_operators(stub)  # must not raise
    assert stub._G_grad is not None
    assert len(stub._G_grad) == 2


# ---------------------------------------------------------------------------
# HJBFEMSolver pinning tests for #1252
# These tests FAIL pre-fix (HJBFEMSolver called base-class _build_gradient_operators
# which raised NotImplementedError for P2) and PASS after the consistent-mass fix.
# ---------------------------------------------------------------------------


def _make_fem_solver(order: int):
    """Build a minimal HJBFEMSolver on a unit-square MeshTri for gradient-recovery tests."""
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
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
        Nt=5,
        sigma=0.3,
        components=components,
        coupling_coefficient=0.5,
        boundary_conditions=no_flux_bc(dimension=2),
    )
    return HJBFEMSolver(problem, order=order)


def test_p2_hjb_fem_gradient_recovery_linear_x():
    """PINNING TEST for #1252 (FAILS pre-fix, PASSES after).

    HJBFEMSolver(order=2)._nodal_gradient(u) with u = x must recover grad_x ~ 1 and
    grad_y ~ 0 at EVERY DOF (vertex + edge-midpoint nodes) to within 1e-8. Pre-fix this
    raised NotImplementedError (fail-loud guard in base class). Post-fix HJBFEMSolver
    overrides _build_gradient_operators to use a consistent-mass L2 projection.
    """
    solver = _make_fem_solver(order=2)
    coords = solver._disc.dof_coordinates  # (N, 2)
    u_linear = coords[:, 0].copy()  # u = x, so exact grad = (1, 0)

    p_nodal = solver._nodal_gradient(u_linear)  # (N, 2)

    max_err_x = float(np.abs(p_nodal[:, 0] - 1.0).max())
    max_err_y = float(np.abs(p_nodal[:, 1]).max())
    assert max_err_x < 1e-8, (
        f"P2 consistent-mass gradient recovery: grad_x max error {max_err_x:.3e} (expected < 1e-8). "
        "Pre-fix this path raised NotImplementedError; now it must return 1 at all DOFs."
    )
    assert max_err_y < 1e-8, (
        f"P2 consistent-mass gradient recovery: grad_y max error {max_err_y:.3e} (expected < 1e-8)."
    )


def test_p2_hjb_fem_gradient_recovery_vertex_dofs():
    """PINNING TEST: vertex DOFs specifically must not collapse to ~0 (the #1252 corruption).

    The issue reported Tri P2 vertex rows with lumped mass ~1.7e-18 yielding grad ~1e-3
    instead of 1.0. This test extracts vertex-only DOFs from the P2 basis and checks them.
    """
    solver = _make_fem_solver(order=2)
    coords = solver._disc.dof_coordinates  # (N, 2)
    u_linear = coords[:, 0].copy()  # u = x

    p_nodal = solver._nodal_gradient(u_linear)  # (N, 2)

    # Vertex DOFs are identified by integer coordinates in the mesh (mesh.p columns
    # are mesh vertices). For a refined MeshTri, vertex DOFs index into basis.doflocs.
    mesh = solver._skfem_mesh
    n_vertices = mesh.p.shape[1]
    # P2 basis: first n_vertices DOFs are vertex nodes; remaining are edge-midpoints.
    vertex_grad_x = p_nodal[:n_vertices, 0]
    max_vertex_err = float(np.abs(vertex_grad_x - 1.0).max())
    assert max_vertex_err < 1e-8, (
        f"P2 vertex DOF gradient: max error {max_vertex_err:.3e} at vertex nodes "
        f"(issue #1252 reported ~1e-3 / -6e12 due to lumped-mass corruption). "
        f"n_vertices={n_vertices}, n_dof={solver._n_dof}"
    )


def test_p1_hjb_fem_gradient_recovery_unchanged():
    """P1 HJBFEMSolver gradient recovery is byte-identical after the #1252 refactor.

    The new _apply_gradient_operator hook must call the base-class lumped path for P1,
    producing the same result as the direct G_d @ u computation it replaced.
    """
    solver = _make_fem_solver(order=1)
    coords = solver._disc.dof_coordinates  # (N, 2)
    u_linear = coords[:, 0].copy()  # u = x

    p_nodal = solver._nodal_gradient(u_linear)  # (N, 2)

    max_err_x = float(np.abs(p_nodal[:, 0] - 1.0).max())
    max_err_y = float(np.abs(p_nodal[:, 1]).max())
    assert max_err_x < 1e-8, f"P1 gradient recovery: grad_x max error {max_err_x:.3e}"
    assert max_err_y < 1e-8, f"P1 gradient recovery: grad_y max error {max_err_y:.3e}"
    # Also confirm P1 uses lumped path (not consistent-mass)
    assert not solver._use_consistent_mass, "P1 must use lumped path, not consistent-mass"
    assert solver._M_lu is None, "P1 must not build LU factorisation"

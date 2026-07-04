"""
Finite Element Method (FEM) solver for the HJB equation on unstructured meshes.

Thin scikit-fem backend over the backend-agnostic ``WeakFormHJBSolver``: this
class supplies the mesh + Lagrange discretization (``FEMDiscretization``) and the
scikit-fem boundary-condition strategy; the implicit-Euler time stepping, Picard
linearization, and (semismooth) Newton iteration live in the base class.

Issue #773 (FEM); Issue #1131 Phase 2 (factored onto WeakFormHJBSolver).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from mfgarchon.alg.base_solver import SchemeFamily
from mfgarchon.alg.numerical.weak_form_hjb_solver import WeakFormHJBSolver
from mfgarchon.utils.mfg_logging import get_logger

from .discretization import FEMDiscretization
from .mesh_adapter import meshdata_to_skfem

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


class HJBFEMSolver(WeakFormHJBSolver):
    """FEM (mesh + Lagrange) HJB solver.

    Uses P1/P2 Lagrange elements on triangular/tetrahedral meshes. Nonlinearity
    handling (Picard default, ``use_newton=True`` for semismooth Newton) and time
    stepping are inherited from ``WeakFormHJBSolver``.

    Example:
        >>> from mfgarchon.alg.numerical.fem import HJBFEMSolver
        >>> solver = HJBFEMSolver(problem)
        >>> U = solver.solve_hjb_system(M_density, U_terminal, U_coupling_prev)
        >>> U = solver.solve_hjb_system(M_density, U_terminal, use_newton=True)
    """

    _scheme_family = SchemeFamily.FEM

    def __init__(self, problem: MFGProblem, order: int = 1) -> None:
        # Issue #1489: a non-mesh geometry (e.g. TensorProductGrid) has no `mesh_data` attribute at all,
        # so a direct `.mesh_data` access raised AttributeError BEFORE this guard — the message naming
        # TensorProductGrid was unreachable for its own case. getattr catches both the missing-attribute
        # and the None (ungenerated Mesh2D) cases.
        mesh_data = getattr(problem.geometry, "mesh_data", None)
        if mesh_data is None:
            raise ValueError(
                "HJBFEMSolver requires an unstructured mesh geometry with mesh_data (Mesh2D / Mesh3D); "
                f"got {type(problem.geometry).__name__}. A structured grid has no mesh — build a "
                "Mesh2D / Mesh3D, or use an FDM / FVM / GFDM solver for a TensorProductGrid."
            )
        self._skfem_mesh = meshdata_to_skfem(mesh_data)
        from .assembly import create_basis

        self._basis = create_basis(self._skfem_mesh, order=order)
        # P2+ gradient-recovery attributes: set before super().__init__ so that
        # _build_gradient_operators (called lazily on first solve) sees them defined.
        self._use_consistent_mass: bool = False
        self._M_lu: Any | None = None  # scipy factorized(M) callable, P2+ only (#1252)
        super().__init__(problem, FEMDiscretization(self._basis))
        self.hjb_method_name = "FEM"
        self.order = order
        logger.info(
            f"HJBFEMSolver initialized: {self._n_dof} DOFs, {self._skfem_mesh.t.shape[1]} elements, order={order}"
        )

    @property
    def basis(self):
        """scikit-fem Basis object."""
        return self._basis

    # --- P2+ consistent-mass gradient recovery (#1252) -----------------------
    def _build_gradient_operators(self) -> None:
        """Override: use consistent-mass L2 projection for P2+ Lagrange elements.

        Row-sum mass lumping (``WeakFormHJBSolver._build_gradient_operators``) is exact-
        order for P1 (all lumped masses strictly positive), but is invalid for P2+: the
        vertex shape function ``lambda(2 lambda - 1)`` integrates to 0 over a triangle
        and to a negative value over a tetrahedron, so the consistent-mass row sum at
        every vertex DOF is ~0 or < 0. The old clamp ``<1e-15 -> 1e-15`` turned invalid
        masses into 1e-15, yielding 1/1e-15 = 1e15 scale factors that corrupted the
        recovered vertex gradient (Tri: ~1e-3; Tet: ~-6e12 instead of 1.0).

        Fix (P2+): build a sparse LU factorisation of the consistent mass matrix M and
        solve ``M g_d = R_d u`` at each gradient evaluation via the precomputed factor.
        For P1 (all lumped masses positive): fall through to the cheap diagonal path in
        the base class. The meshless-Galerkin path (P1/MLS bases) is unaffected because
        it does not call this override.
        """
        if self._G_grad is not None:
            return
        M_lumped = np.asarray(self._M.sum(axis=1)).ravel()
        m_min, m_max = float(M_lumped.min()), float(M_lumped.max())
        if m_min < 1e-12 * m_max:
            # P2+: build LU once; gradient evaluations will call factorized(M)(R_d @ u).
            self._M_lu = factorized(self._M.tocsc())
            # _G_grad must be non-None to signal "operators built" (base-class guard).
            # Entries are None (sentinels); _apply_gradient_operator uses _M_lu + _R_grad.
            self._G_grad = [None] * len(self._R_grad)
            self._use_consistent_mass = True
            self._M_lumped_inv = None
            logger.info(
                "P2+ FEM: consistent-mass L2 projection for nodal gradient recovery "
                f"(min lumped mass {m_min:.3e}, max {m_max:.3e}; Issue #1252)"
            )
        else:
            super()._build_gradient_operators()

    def _apply_gradient_operator(self, d: int, u: NDArray) -> NDArray:
        """P2+ override: solve M g_d = R_d u via precomputed LU (#1252).

        For P1 the base-class lumped-diagonal path is used (self._use_consistent_mass
        is False). The meshless-Galerkin path never calls this override.
        """
        if self._use_consistent_mass:
            return self._M_lu(self._R_grad[d] @ u)
        return super()._apply_gradient_operator(d, u)

    def _hamiltonian_jacobian_term(self, dH_dp_d: NDArray, d: int) -> sparse.csr_matrix:
        """P2+ override: inexact Newton Jacobian for the consistent-mass path (#1252).

        The exact Jacobian term is ``M @ diag(dH/dp_d) @ M^{-1} @ R_d``. Forming
        ``M^{-1} @ R_d`` as a dense matrix is O(N^2) — prohibitive for production meshes.

        Approximation used here: ``diag(dH/dp_d) @ R_d`` (drops the ``M @ ... @ M^{-1}``
        wrapping). This is an inexact Jacobian; Newton converges to the correct residual-
        zero solution because the residual ``F(U) = ... + M H(M^{-1} R_d U)`` uses the
        full consistent-mass gradient. The Jacobian approximation only reduces the
        convergence rate from quadratic to superlinear (#1252 design note).

        For P1: the base-class exact term ``M @ diag(dH/dp_d) @ G_d`` is used.
        """
        if self._use_consistent_mass:
            return sparse.diags(dH_dp_d) @ self._R_grad[d]
        return super()._hamiltonian_jacobian_term(dH_dp_d, d)

    # --- scikit-fem boundary-condition strategy -------------------------------
    def _is_pure_neumann(self) -> bool:
        from .bc_adapter import is_pure_neumann

        return is_pure_neumann(self._bc)

    def _dirichlet_dofs_and_values(self) -> tuple[NDArray, NDArray]:
        from .bc_adapter import get_dirichlet_dofs_and_values

        return get_dirichlet_dofs_and_values(self._basis, self._bc)

    def _apply_bc_to_system(self, matrix, rhs):
        from .bc_adapter import apply_bc_to_fem_system

        return apply_bc_to_fem_system(matrix, rhs, self._basis, self._bc)

    def _robin_operator_terms(self, D: float):
        """Robin boundary operator augmentation (Issue #1237): the D-scaled boundary mass
        ``D*(alpha/beta)*int_dOmega phi_i phi_j`` (folded into ``M/dt + D*K``) and load
        ``D*(1/beta)*int_dOmega g phi_i`` (added to each RHS), assembled over the Robin facets
        via ``skfem.FacetBasis``. ``(None, None)`` when no Robin segment is present."""
        from .bc_adapter import assemble_robin_terms

        return assemble_robin_terms(self._basis, self._bc, D)


if __name__ == "__main__":
    """Smoke test: assemble on a unit-square mesh and check matrix shapes."""
    import skfem

    from .assembly import assemble_mass, assemble_stiffness, create_basis

    print("Testing HJBFEMSolver assembly path...")
    mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
    basis = create_basis(mesh, order=1)
    K = assemble_stiffness(basis)
    M = assemble_mass(basis)
    print(f"DOFs={basis.N}, K nnz={K.nnz}, M nnz={M.nnz}")
    print("Smoke test complete.")

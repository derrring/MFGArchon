"""
Finite Element Method (FEM) solver for the Fokker-Planck equation on unstructured
meshes.

Thin scikit-fem backend over the backend-agnostic ``WeakFormFPSolver``: this class
supplies the mesh + Lagrange discretization (``FEMDiscretization``), the scikit-fem
boundary-condition strategy, and the advection matrix built from the exact
quadrature-point gradient of the value function. Forward implicit-Euler stepping
and mass-conserving structure live in the base class.

Issue #773 (FEM); Issue #1131 Phase 2 (factored onto WeakFormFPSolver).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mfgarchon.alg.base_solver import SchemeFamily
from mfgarchon.alg.numerical.weak_form_fp_solver import WeakFormFPSolver
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.pde_coefficients import fp_drift_coefficient

from .discretization import FEMDiscretization
from .mesh_adapter import meshdata_to_skfem

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy import sparse

    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


class FPFEMSolver(WeakFormFPSolver):
    """FEM (mesh + Lagrange) Fokker-Planck solver.

    Mass conservation holds because the assembled operator has zero column sums: the
    diffusion stiffness already does (``sum_i K[i,j] = integral(grad(sum_i phi_i) . grad phi_j)
    = 0`` since ``sum_i phi_i = 1``), and the advection block is assembled as ``-C^T`` (see
    :meth:`_build_advection`) so its column sums vanish too. Forward time stepping is inherited
    from ``WeakFormFPSolver``.

    Example:
        >>> from mfgarchon.alg.numerical.fem import FPFEMSolver
        >>> solver = FPFEMSolver(problem)
        >>> M = solver.solve_fp_system(m_initial, U_solution)
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
                "FPFEMSolver requires an unstructured mesh geometry with mesh_data (Mesh2D / Mesh3D); "
                f"got {type(problem.geometry).__name__}. A structured grid has no mesh — build a "
                "Mesh2D / Mesh3D, or use an FDM / FVM / GFDM solver for a TensorProductGrid."
            )
        self._skfem_mesh = meshdata_to_skfem(mesh_data)
        from .assembly import create_basis

        self._basis = create_basis(self._skfem_mesh, order=order)
        super().__init__(problem, FEMDiscretization(self._basis))
        self.order = order
        logger.info(f"FPFEMSolver initialized: {self._n_dof} DOFs, {self._skfem_mesh.t.shape[1]} elements")

    @property
    def basis(self):
        return self._basis

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
        """Robin boundary operator augmentation, adjoint of the HJB term (Issue #1237).

        The FP Robin term is the symmetric boundary mass ``D*(alpha/beta)*int_dOmega phi_i phi_j``
        from integrating the FP diffusion operator ``-D*Delta m`` by parts, plus the boundary load
        ``D*(1/beta)*int_dOmega g phi_i``. Because the boundary mass is symmetric, this is identical
        to the HJB Robin term, so ``A_FP = A_HJB^T`` is preserved for the diffusion+Robin block.
        Assembled via ``skfem.FacetBasis``; ``(None, None)`` when no Robin segment is present."""
        from .bc_adapter import assemble_robin_terms

        return assemble_robin_terms(self._basis, self._bc, D)

    # --- advection from drift via exact quadrature-point gradient of U --------
    def _build_advection(self, U_n: NDArray, D: float = 0.0) -> sparse.csr_matrix:
        r"""Assemble the mass-conserving FP advection block for drift $v = -\text{coupling}\cdot\nabla U_n$.

        The raw convective form $C_{ij} = \int \phi_i\,(v\cdot\nabla\phi_j)\,dx$ (gradient on the
        TRIAL function) does NOT conserve mass: its column sums are
        $\sum_i C_{ij} = \int (v\cdot\nabla\phi_j)\,dx \ne 0$. Integrating $\text{div}(v\,m)$ by
        parts moves the gradient onto the test function, giving the operator $-C^{\top}$, whose
        column sums vanish ($\sum_i (-C^\top)_{ij} = -\int (v\cdot\nabla\phi_j)\sum_i\phi_i = 0$
        since $\sum_i \phi_i = 1$). $-C^\top$ is also the adjoint-consistency identity
        $A_{FP} = A_{HJB}^\top$, matching the meshless-Galerkin sibling (Issue #1131). Before this
        fix (Issue #1114-adjacent / FEM survey) FEM returned the un-transposed $+C$, giving ~20%+
        mass drift on a non-divergence-free drift.

        ``D`` (the current diffusion coefficient) is part of the base contract but unused by
        FEM, which adds no diffusion-scaled stabilization term."""
        import skfem
        from skfem import BilinearForm

        dim = self._skfem_mesh.p.shape[0]
        # Issue #1487/#1420 (G-017): the FP drift scale is the HJB optimal-control scale 1/control_cost,
        # single-sourced via fp_drift_coefficient — NOT the raw coupling_coefficient (which defaults to
        # 0.5 and silently diverges from 1/control_cost -> wrong equilibrium). Matches the 7 FDM/FVM/
        # particle/SL solvers; the weak-form family was the lone holdout.
        coupling = fp_drift_coefficient(self.problem)
        du = self._basis.interpolate(U_n)

        @BilinearForm
        def advection_form(u, v, w):
            result = 0.0
            for d in range(dim):
                v_d = -coupling * du.grad[d]
                result += v_d * u.grad[d]
            return result * v.value

        c_matrix = skfem.asm(advection_form, self._basis)
        return (-c_matrix.T).tocsr()


if __name__ == "__main__":
    """Smoke test: forward diffusion on a unit square mesh."""
    import skfem

    from .assembly import assemble_mass, assemble_stiffness, create_basis

    print("Testing FPFEMSolver assembly path...")
    mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
    basis = create_basis(mesh, order=1)
    K = assemble_stiffness(basis)
    M = assemble_mass(basis)
    print(f"DOFs={basis.N}, K nnz={K.nnz}, M nnz={M.nnz}")
    print("Smoke test complete.")

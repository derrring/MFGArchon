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

from mfgarchon.alg.numerical.weak_form_fp_solver import WeakFormFPSolver
from mfgarchon.utils.mfg_logging import get_logger

from .discretization import FEMDiscretization
from .mesh_adapter import meshdata_to_skfem

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy import sparse

    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


class FPFEMSolver(WeakFormFPSolver):
    """FEM (mesh + Lagrange) Fokker-Planck solver.

    Mass conservation is guaranteed by the Galerkin weak form (the test space
    contains constants). Forward time stepping is inherited from ``WeakFormFPSolver``.

    Example:
        >>> from mfgarchon.alg.numerical.fem import FPFEMSolver
        >>> solver = FPFEMSolver(problem)
        >>> M = solver.solve_fp_system(m_initial, U_solution)
    """

    def __init__(self, problem: MFGProblem, order: int = 1) -> None:
        mesh_data = problem.geometry.mesh_data
        if mesh_data is None:
            raise ValueError(
                "FPFEMSolver requires unstructured mesh geometry. Use Mesh2D or Mesh3D, not TensorProductGrid."
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

    # --- advection from drift via exact quadrature-point gradient of U --------
    def _build_advection(self, U_n: NDArray, D: float = 0.0) -> sparse.csr_matrix:
        r"""Assemble $\int (v \cdot \nabla\phi_j)\,\phi_i\,dx$ with $v = -\text{coupling}\cdot\nabla U_n$.

        ``D`` (the current diffusion coefficient) is part of the base contract but unused by
        FEM, which adds no diffusion-scaled stabilization term."""
        import skfem
        from skfem import BilinearForm

        dim = self._skfem_mesh.p.shape[0]
        coupling = getattr(self.problem, "coupling_coefficient", 0.5)
        du = self._basis.interpolate(U_n)

        @BilinearForm
        def advection_form(u, v, w):
            result = 0.0
            for d in range(dim):
                v_d = -coupling * du.grad[d]
                result += v_d * u.grad[d]
            return result * v.value

        return skfem.asm(advection_form, self._basis)


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

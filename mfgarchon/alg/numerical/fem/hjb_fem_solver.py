"""
Finite Element Method (FEM) solver for the HJB equation on unstructured meshes.

Thin scikit-fem backend over the backend-agnostic ``WeakFormHJBSolver``: this
class supplies the mesh + Lagrange discretization (``FEMDiscretization``) and the
scikit-fem boundary-condition strategy; the implicit-Euler time stepping, Picard
linearization, and (semismooth) Newton iteration live in the base class.

Issue #773 (FEM); Issue #1131 Phase 2 (factored onto WeakFormHJBSolver).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    def __init__(self, problem: MFGProblem, order: int = 1) -> None:
        mesh_data = problem.geometry.mesh_data
        if mesh_data is None:
            raise ValueError(
                "HJBFEMSolver requires unstructured mesh geometry. Use Mesh2D or Mesh3D, not TensorProductGrid."
            )
        self._skfem_mesh = meshdata_to_skfem(mesh_data)
        from .assembly import create_basis

        self._basis = create_basis(self._skfem_mesh, order=order)
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

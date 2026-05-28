"""
Finite Element Method (FEM) solver for Fokker-Planck equation on unstructured meshes.

Solves the FP equation forward in time:
    dm/dt + div(v * m) - D * Laplacian(m) = 0
    m(0, x) = m_initial(x)

where v = -coupling * grad(u) is the drift from the HJB solution.

Uses scikit-fem for assembly and implicit Euler time stepping.
Mass conservation is guaranteed by the weak formulation.

Issue #773 Phase 1: FPFEMSolver implementation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import spsolve

from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
from mfgarchon.utils.mfg_logging import get_logger

from .assembly import assemble_mass, assemble_stiffness, create_basis
from .discretization import FEMDiscretization
from .mesh_adapter import meshdata_to_skfem

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy import sparse

    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


class FPFEMSolver(BaseFPSolver):
    """
    FEM solver for the Fokker-Planck equation.

    Uses P1 finite elements with implicit Euler time stepping.
    Mass conservation is guaranteed by the Galerkin weak form:
    the test function space includes constants, so total mass
    integral(m * 1) is preserved.

    Example:
        >>> from mfgarchon.alg.numerical.fem import FPFEMSolver
        >>> solver = FPFEMSolver(problem)
        >>> M = solver.solve_fp_system(m_initial, U_solution)
    """

    def __init__(self, problem: MFGProblem, order: int = 1) -> None:
        super().__init__(problem)
        self.order = order

        mesh_data = problem.geometry.mesh_data
        if mesh_data is None:
            raise ValueError(
                "FPFEMSolver requires unstructured mesh geometry. Use Mesh2D or Mesh3D, not TensorProductGrid."
            )

        self._skfem_mesh = meshdata_to_skfem(mesh_data)
        self._basis = create_basis(self._skfem_mesh, order=order)
        self._disc = FEMDiscretization(self._basis)  # Issue #1131: assembly backend
        self._n_dof = self._basis.N

        self._K = self._disc.stiffness()
        self._M = self._disc.mass()

        # BC from problem geometry (same framework as FDM/GFDM)
        self._bc = getattr(problem.geometry, "boundary_conditions", None)

        logger.info(f"FPFEMSolver initialized: {self._n_dof} DOFs, {self._skfem_mesh.t.shape[1]} elements")

    @property
    def basis(self):
        return self._basis

    @property
    def n_dof(self) -> int:
        return self._n_dof

    def solve_fp_system(
        self,
        m_initial: NDArray,
        drift_field: NDArray | None = None,
        volatility_field: float | NDArray | None = None,
        **kwargs,
    ) -> NDArray:
        """
        Solve FP equation forward in time.

        Args:
            m_initial: Initial density m(0,x), shape (N_dof,)
            drift_field: Value function U for drift v = -coupling*grad(U),
                shape (Nt+1, N_dof). If None, pure diffusion.
            volatility_field: Diffusion D = sigma^2/2

        Returns:
            Density M(t,x), shape (Nt+1, N_dof)
        """
        Nt = self.problem.Nt
        dt = self.problem.dt
        N = self._n_dof

        # Diffusion coefficient
        if volatility_field is None:
            D = 0.5 * self.problem.sigma**2
        elif isinstance(volatility_field, (int, float)):
            D = float(volatility_field)
        else:
            D = float(np.mean(volatility_field))

        # Allocate solution
        M = np.zeros((Nt + 1, N))
        M[0] = m_initial[:N] if len(m_initial) >= N else np.pad(m_initial, (0, N - len(m_initial)))

        # Base system matrix: M/dt + D*K (no advection yet)
        A_base = self._M / dt + D * self._K

        # BC handling via framework adapter
        from .bc_adapter import apply_bc_to_fem_system, is_pure_neumann

        pure_neumann = is_pure_neumann(self._bc)

        for n in range(Nt):
            rhs = (self._M / dt) @ M[n]

            # Add advection if drift field provided
            if drift_field is not None:
                U_n = drift_field[n] if drift_field.ndim > 1 else drift_field
                A_advection = self._build_advection_from_drift(U_n)
                A_system = A_base + A_advection
            else:
                A_system = A_base

            if pure_neumann:
                # Natural BC (no-flux) — solve full system
                M[n + 1] = spsolve(A_system, rhs)
            else:
                # Has Dirichlet segments (e.g., absorbing boundaries)
                A_bc, rhs_bc = apply_bc_to_fem_system(A_system, rhs, self._basis, self._bc)
                from .bc_adapter import get_dirichlet_dofs_and_values

                d_dofs, d_vals = get_dirichlet_dofs_and_values(self._basis, self._bc)
                interior = np.setdiff1d(np.arange(N), d_dofs)
                M[n + 1, interior] = spsolve(A_bc, rhs_bc)
                M[n + 1, d_dofs] = d_vals

            # Enforce non-negativity
            M[n + 1] = np.maximum(M[n + 1], 0.0)

        return M

    def _build_advection_from_drift(self, U_n: NDArray) -> sparse.csr_matrix:
        """
        Build advection matrix from value function at timestep n.

        Computes div(v * m) where v = -coupling * grad(U).
        In weak form: integral(v . grad(phi_j) * phi_i * m) dx.

        For Picard linearization, we use the previous m and just build
        the matrix for the velocity field.
        """
        import skfem
        from skfem import BilinearForm

        # Compute gradient of U at nodes
        dim = self._skfem_mesh.p.shape[0]
        coupling = getattr(self.problem, "coupling_coefficient", 0.5)

        # Gradient via interpolation
        du = self._basis.interpolate(U_n)

        @BilinearForm
        def advection_form(u, v, w):
            result = 0.0
            for d in range(dim):
                # v_d = -coupling * dU/dx_d
                v_d = -coupling * du.grad[d]
                # Divergence form: integral(v_d * dm/dx_d * phi_i)
                result += v_d * u.grad[d]
            return result * v.value

        return skfem.asm(advection_form, self._basis)

    def solve_fp_step_adjoint_mode(
        self,
        M_current: NDArray,
        A_advection_T: sparse.csr_matrix,
        sigma: float | NDArray | None = None,
        time: float = 0.0,
    ) -> NDArray:
        """
        Solve single FP timestep with externally provided advection matrix.

        Compatible with BlockIterator's adjoint modes.
        """
        dt = self.problem.dt

        if sigma is None:
            D = 0.5 * self.problem.sigma**2
        elif isinstance(sigma, (int, float)):
            D = 0.5 * float(sigma) ** 2
        else:
            D = 0.5 * float(np.mean(sigma)) ** 2

        A_system = self._M / dt + A_advection_T + D * self._K
        rhs = (self._M / dt) @ M_current.ravel()

        M_next = spsolve(A_system, rhs)
        return np.maximum(M_next, 0.0).reshape(M_current.shape)


if __name__ == "__main__":
    """Smoke test: forward diffusion on unit square."""
    import skfem

    print("Testing FPFEMSolver...")

    mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
    basis = create_basis(mesh, order=1)

    K = assemble_stiffness(basis)
    M_mat = assemble_mass(basis)
    N = basis.N
    dt = 0.01
    D = 0.1

    # Forward heat equation: (M/dt + D*K) m^{n+1} = M/dt * m^n
    A = M_mat / dt + D * K

    # Initial condition: bump at center
    x, y = mesh.p
    m = np.exp(-20 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
    m /= m.sum() * (1.0 / N)  # Normalize

    initial_mass = M_mat @ m
    print(f"Initial total mass: {initial_mass.sum():.6f}")

    # Step forward
    for _step in range(10):
        rhs = (M_mat / dt) @ m
        m = spsolve(A, rhs)
        m = np.maximum(m, 0.0)

    final_mass = M_mat @ m
    print(f"Final total mass: {final_mass.sum():.6f}")
    print(f"Mass conservation: {abs(final_mass.sum() - initial_mass.sum()) / initial_mass.sum():.2e}")
    assert np.all(np.isfinite(m)), "Solution contains NaN/Inf"
    assert np.all(m >= 0), "Density went negative"
    print("Smoke test passed.")

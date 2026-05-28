"""
Finite Element Method (FEM) solver for HJB equation on unstructured meshes.

Solves the HJB equation backward in time:
    -du/dt + H(grad(u), x, t, m) - D * Laplacian(u) = 0
    u(T, x) = u_terminal(x)

Uses scikit-fem for stiffness/mass matrix assembly and implicit time stepping.
The Hamiltonian nonlinearity is handled via Picard linearization (fixed-point
iteration with the coupling layer).

Issue #773 Phase 1: HJBFEMSolver implementation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
from mfgarchon.utils.mfg_logging import get_logger

from .assembly import assemble_mass, assemble_stiffness, create_basis
from .discretization import FEMDiscretization
from .mesh_adapter import meshdata_to_skfem

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


class HJBFEMSolver(BaseHJBSolver):
    """
    FEM solver for the Hamilton-Jacobi-Bellman equation.

    Uses P1 (linear) finite elements on triangular/tetrahedral meshes.
    Time discretization: implicit Euler (backward in time).

    Two nonlinearity handling modes:
        - **Picard** (default): H is evaluated at the previous iterate's gradient.
          Fast per timestep but only linearly convergent.
        - **Newton** (use_newton=True): Full Newton iteration per timestep with
          analytical Jacobian from H.dp(). Supports nondifferentiable Hamiltonians
          via semismooth Newton -- H.dp() returns a Clarke subdifferential element
          through the finite-difference fallback (Issue #893).

    Example:
        >>> from mfgarchon.alg.numerical.fem import HJBFEMSolver
        >>> solver = HJBFEMSolver(problem)
        >>> U = solver.solve_hjb_system(M_density, U_terminal, U_coupling_prev)
        >>> # With Newton for nonsmooth H:
        >>> U = solver.solve_hjb_system(M_density, U_terminal, use_newton=True)
    """

    def __init__(self, problem: MFGProblem, order: int = 1) -> None:
        super().__init__(problem)
        self.hjb_method_name = "FEM"
        self.order = order

        # Build skfem mesh and basis from problem geometry
        mesh_data = problem.geometry.mesh_data
        if mesh_data is None:
            raise ValueError(
                "HJBFEMSolver requires unstructured mesh geometry. Use Mesh2D or Mesh3D, not TensorProductGrid."
            )

        self._skfem_mesh = meshdata_to_skfem(mesh_data)
        self._basis = create_basis(self._skfem_mesh, order=order)
        self._disc = FEMDiscretization(self._basis)  # Issue #1131: assembly backend
        self._n_dof = self._basis.N

        # Pre-assemble constant matrices
        self._K = self._disc.stiffness()  # Stiffness (Laplacian)
        self._M = self._disc.mass()  # Mass

        # Gradient projection matrices for Newton iteration (lazy-built)
        self._R_grad = self._disc.gradient_projection()
        self._G_grad: list[sparse.csr_matrix] | None = None
        self._M_lumped_inv: NDArray | None = None

        # BC from problem geometry (same framework as FDM/GFDM)
        self._bc = self.get_boundary_conditions()

        logger.info(
            f"HJBFEMSolver initialized: {self._n_dof} DOFs, {self._skfem_mesh.t.shape[1]} elements, order={order}"
        )

    @property
    def basis(self):
        """scikit-fem Basis object."""
        return self._basis

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        return self._n_dof

    def _build_gradient_operators(self) -> None:
        """Lazily build gradient operator matrices G_d = diag(1/M_lumped) @ R_d."""
        if self._G_grad is not None:
            return

        M_lumped = np.array(self._M.sum(axis=1)).ravel()
        M_lumped[M_lumped < 1e-15] = 1e-15
        self._M_lumped_inv = 1.0 / M_lumped

        self._G_grad = []
        for R_d in self._R_grad:
            G_d = sparse.diags(self._M_lumped_inv) @ R_d
            self._G_grad.append(G_d.tocsr())

    def _solve_timestep_newton(
        self,
        U_next: NDArray,
        m_n: NDArray,
        D: float,
        dt: float,
        t: float,
        rhs_coupling: NDArray,
        max_iterations: int = 30,
        tolerance: float = 1e-6,
    ) -> NDArray:
        """
        Solve one HJB timestep via Newton iteration (supports semismooth H).

        Residual: F(U) = (M/dt)(U - U_next) + D*K*U + M*H(grad(U)) - rhs_coupling
        Jacobian: J = M/dt + D*K + sum_d M @ diag(dH/dp_d) @ G_d

        For nondifferentiable Hamiltonians, H.dp() returns a Clarke subdifferential
        element via the finite-difference fallback, making this a semismooth Newton
        method without special-case logic.

        Args:
            U_next: Solution at next time level U^{n+1}, shape (N_dof,)
            m_n: Density at this time level, shape (N_dof,)
            D: Diffusion coefficient (sigma^2 / 2)
            dt: Time step size
            t: Current time
            rhs_coupling: Additional RHS from coupling f(m), shape (N_dof,)
            max_iterations: Maximum Newton iterations per timestep
            tolerance: Convergence tolerance (M-weighted norm of delta)

        Returns:
            Solution U^n at this time level, shape (N_dof,)
        """
        self._build_gradient_operators()

        H_class = self.problem.hamiltonian_class
        N = self._n_dof
        dim = len(self._G_grad)
        x_grid = self._skfem_mesh.p.T  # (N, dim)

        # Fixed part of Jacobian
        J_fixed = self._M / dt + D * self._K

        # BC setup
        from .bc_adapter import apply_bc_to_fem_system, get_dirichlet_dofs_and_values, is_pure_neumann

        pure_neumann = is_pure_neumann(self._bc)
        if not pure_neumann:
            d_dofs, d_vals = get_dirichlet_dofs_and_values(self._basis, self._bc)
            interior = np.setdiff1d(np.arange(N), d_dofs)
        else:
            d_dofs = np.array([], dtype=int)
            d_vals = np.array([])
            interior = np.arange(N)

        # Initial guess: U_next (from the next time level)
        U_current = U_next.copy()
        if not pure_neumann:
            U_current[d_dofs] = d_vals

        for k in range(max_iterations):
            # Compute gradient at nodes: p = grad(U_current)
            p_nodal = np.column_stack([G_d @ U_current for G_d in self._G_grad])  # (N, dim)

            # Evaluate Hamiltonian and its derivative
            H_vals = np.asarray(H_class(x_grid, m_n, p_nodal, t=t), dtype=float).ravel()
            dH_dp = np.asarray(H_class.dp(x_grid, m_n, p_nodal, t=t), dtype=float)
            if dH_dp.ndim == 1:
                dH_dp = dH_dp.reshape(-1, 1)  # (N, 1) for 1D

            # Residual: F = (M/dt)(U - U_next) + D*K*U + M*H - rhs_coupling
            residual = (
                (self._M / dt) @ (U_current - U_next) + D * (self._K @ U_current) + self._M @ H_vals - rhs_coupling
            )

            # Jacobian: J = M/dt + D*K + sum_d M @ diag(dH/dp_d) @ G_d
            J = J_fixed.copy()
            for d in range(dim):
                J = J + self._M @ sparse.diags(dH_dp[:, d]) @ self._G_grad[d]

            # Apply BC: zero residual and identity Jacobian at Dirichlet DOFs
            if pure_neumann:
                delta = spsolve(J.tocsc(), -residual)
            else:
                residual[d_dofs] = 0.0
                J_bc, res_bc = apply_bc_to_fem_system(J, -residual, self._basis, self._bc)
                delta = np.zeros(N)
                delta[interior] = spsolve(J_bc, res_bc)

            # Update
            U_current += delta

            # Convergence check (M-weighted norm)
            delta_norm = np.sqrt(np.abs(delta @ (self._M @ delta)))
            if delta_norm < tolerance:
                logger.debug(f"Newton converged in {k + 1} iterations (norm={delta_norm:.2e})")
                break
        else:
            logger.warning(f"Newton did not converge in {max_iterations} iterations (norm={delta_norm:.2e})")

        return U_current

    def solve_hjb_system(
        self,
        M_density: NDArray | None = None,
        U_terminal: NDArray | None = None,
        U_coupling_prev: NDArray | None = None,
        volatility_field: float | NDArray | None = None,
        use_newton: bool = False,
        max_newton_iterations: int = 30,
        newton_tolerance: float = 1e-6,
        # Deprecated names
        M_density_evolution_from_FP: NDArray | None = None,
        U_final_condition_at_T: NDArray | None = None,
        U_from_prev_picard: NDArray | None = None,
        **kwargs,
    ) -> NDArray:
        """
        Solve HJB system backward in time using FEM.

        Args:
            M_density: Density field from FP, shape (Nt+1, N_dof) or (N_dof,)
            U_terminal: Terminal condition u(T,x), shape (N_dof,)
            U_coupling_prev: Previous Picard iterate, shape (Nt+1, N_dof)
            volatility_field: Diffusion D = sigma^2/2 (None uses problem default)
            use_newton: If True, use Newton iteration per timestep instead of Picard
                linearization. Supports nondifferentiable Hamiltonians via semismooth
                Newton (Issue #893).
            max_newton_iterations: Max Newton iterations per timestep (default 30)
            newton_tolerance: Newton convergence tolerance (default 1e-6)

        Returns:
            Value function U(t,x), shape (Nt+1, N_dof)
        """
        # Handle deprecated parameter names
        if M_density is None and M_density_evolution_from_FP is not None:
            M_density = M_density_evolution_from_FP
        if U_terminal is None and U_final_condition_at_T is not None:
            U_terminal = U_final_condition_at_T
        if U_coupling_prev is None and U_from_prev_picard is not None:
            U_coupling_prev = U_from_prev_picard

        # Defaults
        Nt = self.problem.Nt
        dt = self.problem.dt
        N = self._n_dof

        if U_terminal is None:
            U_terminal = np.zeros(N)
        if M_density is None:
            M_density = np.ones((Nt + 1, N)) / N
        if U_coupling_prev is None:
            U_coupling_prev = np.zeros((Nt + 1, N))

        # Ensure M_density is (Nt+1, N)
        if M_density.ndim == 1:
            M_density = np.tile(M_density, (Nt + 1, 1))

        # Diffusion coefficient
        if volatility_field is None:
            D = 0.5 * self.problem.sigma**2
        elif isinstance(volatility_field, (int, float)):
            D = float(volatility_field)
        else:
            D = float(np.mean(volatility_field))

        # Allocate solution
        U = np.zeros((Nt + 1, N))
        U[Nt] = U_terminal

        # System matrix: (M/dt + D*K)
        # For implicit Euler backward: M/dt * u^n = M/dt * u^{n+1} - D*K*u^n - H_rhs
        # Rearranged: (M/dt + D*K) u^n = M/dt * u^{n+1} + source
        A_system = self._M / dt + D * self._K

        # Solve backward in time
        H_class = self.problem.hamiltonian_class

        for n in range(Nt - 1, -1, -1):
            if use_newton and H_class is not None:
                # Newton iteration per timestep (supports nonsmooth H)
                # The coupling RHS (e.g., f(m) source) is zero here because
                # f(m) is absorbed into H(x, m, p) in the Hamiltonian evaluation.
                rhs_coupling = np.zeros(N)
                U[n] = self._solve_timestep_newton(
                    U_next=U[n + 1],
                    m_n=M_density[n],
                    D=D,
                    dt=dt,
                    t=n * dt,
                    rhs_coupling=rhs_coupling,
                    max_iterations=max_newton_iterations,
                    tolerance=newton_tolerance,
                )
            else:
                # Picard linearization (original path)
                rhs = (self._M / dt) @ U[n + 1]

                if H_class is not None:
                    m_n = M_density[n]
                    x_grid = self._skfem_mesh.p.T
                    u_prev = U_coupling_prev[n]
                    p_prev = self._compute_nodal_gradient(u_prev)
                    H_values = np.asarray(H_class(x_grid, m_n, p_prev, t=n * dt), dtype=float).ravel()
                    rhs += self._M @ H_values

                from .bc_adapter import apply_bc_to_fem_system, is_pure_neumann

                if is_pure_neumann(self._bc):
                    U[n] = spsolve(A_system, rhs)
                else:
                    A_bc, rhs_bc = apply_bc_to_fem_system(A_system, rhs, self._basis, self._bc)
                    from .bc_adapter import get_dirichlet_dofs_and_values

                    d_dofs, d_vals = get_dirichlet_dofs_and_values(self._basis, self._bc)
                    interior = np.setdiff1d(np.arange(N), d_dofs)
                    U[n, interior] = spsolve(A_bc, rhs_bc)
                    U[n, d_dofs] = d_vals

        return U

    def _compute_nodal_gradient(self, u: NDArray) -> NDArray:
        """
        Compute gradient of FEM solution at nodes via L2 projection.

        Args:
            u: Nodal values, shape (N_dof,)

        Returns:
            Gradient at nodes, shape (N_dof, dim)
        """
        dim = self._skfem_mesh.p.shape[0]
        grad = np.zeros((self._n_dof, dim))

        # Use basis to compute element-wise gradients and project to nodes
        # This is a simple volume-weighted average
        basis = self._basis

        for d in range(dim):
            # Compute d(u)/dx_d at quadrature points, then project to nodes
            # Using the mass-lumped L2 projection: M_lumped * grad_d = integral(du/dx_d * phi_i)
            du_dx = basis.interpolate(u).grad[d]  # At quadrature points

            from skfem import LinearForm

            @LinearForm
            def grad_form(v, w, *, _du_dx=du_dx):
                return _du_dx * v.value

            rhs_d = grad_form.assemble(basis)

            # Mass-lumped solve (diagonal approximation)
            M_lumped = np.array(self._M.sum(axis=1)).ravel()
            M_lumped[M_lumped < 1e-15] = 1e-15
            grad[:, d] = rhs_d / M_lumped

        return grad


if __name__ == "__main__":
    """Smoke test: solve heat equation backward on unit square mesh."""
    import skfem

    print("Testing HJBFEMSolver...")

    # We can't use MFGProblem with unstructured mesh easily in smoke test,
    # so test the assembly components directly
    mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
    basis = create_basis(mesh, order=1)

    K = assemble_stiffness(basis)
    M = assemble_mass(basis)
    N = basis.N
    dt = 0.01
    D = 0.1

    # Backward heat equation: (M/dt + D*K) u^n = M/dt * u^{n+1}
    A = M / dt + D * K
    boundary = mesh.boundary_nodes()
    interior = np.setdiff1d(np.arange(N), boundary)

    # Terminal condition: bump at center
    x, y = mesh.p
    u_T = np.exp(-20 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

    # Step backward
    u = u_T.copy()
    for _step in range(10):
        rhs = (M / dt) @ u
        u_new = np.zeros(N)
        u_new[interior] = spsolve(A[np.ix_(interior, interior)], rhs[interior])
        u = u_new

    print(f"Mesh: {mesh.t.shape[1]} elements, {N} nodes")
    print(f"After 10 backward steps: u range [{u.min():.4f}, {u.max():.4f}]")
    assert np.all(np.isfinite(u)), "Solution contains NaN/Inf"
    print("Smoke test passed.")

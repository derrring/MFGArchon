"""
Backend-agnostic weak-form HJB solver.

Solves the HJB equation backward in time on the assembled weak-form operators of
a ``WeakFormDiscretization`` (stiffness ``K``, mass ``M``, gradient-projection
``R_d``), so one solver serves finite elements (mesh + Lagrange) and meshless
Galerkin (point cloud + MLS). A concrete subclass supplies only the
discretization and the boundary-condition strategy (the BC hooks).

Time discretization: implicit Euler (backward). Two nonlinearity modes:
- Picard (default): ``H`` evaluated at the previous iterate's gradient.
- Newton (``use_newton=True``): full/semismooth Newton per timestep with the
  Hamiltonian Jacobian ``H.dp()`` (Clarke element for nondifferentiable ``H``).

Issue #1131 Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.weak_form_discretization import WeakFormDiscretization
    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


class WeakFormHJBSolver(BaseHJBSolver):
    """HJB solver on assembled weak-form operators; BC strategy is subclass-supplied.

    Subclasses must implement the three boundary-condition hooks
    (``_is_pure_neumann``, ``_dirichlet_dofs_and_values``, ``_apply_bc_to_system``)
    for their discretization (e.g. scikit-fem condensation for meshes, collocation
    boundary nodes for point clouds).
    """

    def __init__(self, problem: MFGProblem, discretization: WeakFormDiscretization) -> None:
        super().__init__(problem)
        self._disc = discretization
        self._n_dof = discretization.n_dof
        self._K = discretization.stiffness()
        self._M = discretization.mass()
        self._R_grad = discretization.gradient_projection()
        self._G_grad: list[sparse.csr_matrix] | None = None
        self._M_lumped_inv: NDArray | None = None
        self._bc = self.get_boundary_conditions()

    @property
    def n_dof(self) -> int:
        return self._n_dof

    # --- Boundary-condition strategy (subclass-supplied) ----------------------
    def _is_pure_neumann(self) -> bool:
        raise NotImplementedError

    def _dirichlet_dofs_and_values(self) -> tuple[NDArray, NDArray]:
        raise NotImplementedError

    def _apply_bc_to_system(self, matrix, rhs):
        """Condense the linear system onto interior dofs for Dirichlet BC."""
        raise NotImplementedError

    def _weak_bc_terms(self, D: float):
        """Optional weak (Nitsche) boundary terms for non-interpolatory bases.

        Returns ``(A_extra, rhs_extra)`` to ADD to the full-size implicit operator
        and RHS, then solved on ALL dofs (no condensation), or ``(None, None)`` for
        no weak BC. Default no-op: FEM uses nodal condensation, so it never enters
        the weak path. Meshless Galerkin overrides this with the symmetric Nitsche
        terms (its MLS basis is non-interpolatory, so condensation is invalid)."""
        return None, None

    def _stabilization_terms(self, u: NDArray, D: float):
        """Optional symmetric stabilization operator added to BOTH the Newton residual
        (as ``S @ u``) and the Newton Jacobian (as ``S``), recomputed each Newton iterate.

        Returns a symmetric sparse matrix ``S`` or ``None``. Default no-op (FEM is
        unaffected and byte-identical). Meshless Galerkin overrides this to return the
        streamline-diffusion block; because the SAME symmetric ``S`` is added to the FP
        advection operator, ``A_FP = A_HJB^T`` is preserved (Issue #1145, Bug B)."""
        return None

    # --- Gradient recovery: mass-lumped L2 projection grad_d(u) = G_d @ u -----
    def _build_gradient_operators(self) -> None:
        if self._G_grad is not None:
            return
        M_lumped = np.array(self._M.sum(axis=1)).ravel()
        M_lumped[M_lumped < 1e-15] = 1e-15
        self._M_lumped_inv = 1.0 / M_lumped
        self._G_grad = [(sparse.diags(self._M_lumped_inv) @ R_d).tocsr() for R_d in self._R_grad]

    def _nodal_gradient(self, u: NDArray) -> NDArray:
        self._build_gradient_operators()
        return np.column_stack([G_d @ u for G_d in self._G_grad])

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
        """One HJB timestep via Newton iteration (supports semismooth H).

        Residual: F(U) = (M/dt)(U - U_next) + D*K*U + M*H(grad U) - rhs_coupling.
        Jacobian: J = M/dt + D*K + sum_d M @ diag(dH/dp_d) @ G_d.
        """
        self._build_gradient_operators()

        H_class = self.problem.hamiltonian_class
        N = self._n_dof
        dim = len(self._G_grad)
        x_grid = self._disc.dof_coordinates  # (N, dim)

        A_extra, rhs_extra = self._weak_bc_terms(D)
        weak_bc = A_extra is not None
        J_fixed = self._M / dt + D * self._K
        if weak_bc:
            J_fixed = J_fixed + A_extra

        pure_neumann = self._is_pure_neumann()
        condense = not pure_neumann and not weak_bc
        if condense:
            d_dofs, d_vals = self._dirichlet_dofs_and_values()
            interior = np.setdiff1d(np.arange(N), d_dofs)
        else:
            d_dofs = np.array([], dtype=int)
            d_vals = np.array([])
            interior = np.arange(N)

        U_current = U_next.copy()
        if condense:
            U_current[d_dofs] = d_vals

        delta_norm = np.inf
        for k in range(max_iterations):
            p_nodal = np.column_stack([G_d @ U_current for G_d in self._G_grad])  # (N, dim)

            H_vals = np.asarray(H_class(x_grid, m_n, p_nodal, t=t), dtype=float).ravel()
            dH_dp = np.asarray(H_class.dp(x_grid, m_n, p_nodal, t=t), dtype=float)
            if dH_dp.ndim == 1:
                dH_dp = dH_dp.reshape(-1, 1)

            residual = (
                (self._M / dt) @ (U_current - U_next) + D * (self._K @ U_current) + self._M @ H_vals - rhs_coupling
            )
            if weak_bc:
                residual = residual + A_extra @ U_current
                if rhs_extra is not None:
                    residual = residual - rhs_extra

            # Optional symmetric stabilization (e.g. streamline diffusion) for the
            # canonical HJB residual -u_t + H - (sigma^2/2) Delta u = 0; recomputed each
            # iterate (depends on the current gradient). The same S is added to the FP
            # advection block, preserving A_FP = A_HJB^T (default None -> FEM unaffected).
            S_stab = self._stabilization_terms(U_current, D)
            if S_stab is not None:
                residual = residual + S_stab @ U_current

            J = J_fixed.copy()
            if S_stab is not None:
                J = J + S_stab
            for d in range(dim):
                J = J + self._M @ sparse.diags(dH_dp[:, d]) @ self._G_grad[d]

            if condense:
                residual[d_dofs] = 0.0
                J_bc, res_bc = self._apply_bc_to_system(J, -residual)
                delta = np.zeros(N)
                delta[interior] = spsolve(J_bc, res_bc)
            else:
                delta = spsolve(J.tocsc(), -residual)

            U_current += delta

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
        """Solve the HJB system backward in time on the weak-form operators."""
        if M_density is None and M_density_evolution_from_FP is not None:
            M_density = M_density_evolution_from_FP
        if U_terminal is None and U_final_condition_at_T is not None:
            U_terminal = U_final_condition_at_T
        if U_coupling_prev is None and U_from_prev_picard is not None:
            U_coupling_prev = U_from_prev_picard

        Nt = self.problem.Nt
        dt = self.problem.dt
        N = self._n_dof

        if U_terminal is None:
            U_terminal = np.zeros(N)
        if M_density is None:
            M_density = np.ones((Nt + 1, N)) / N
        if U_coupling_prev is None:
            U_coupling_prev = np.zeros((Nt + 1, N))
        if M_density.ndim == 1:
            M_density = np.tile(M_density, (Nt + 1, 1))

        # volatility_field is the SDE volatility sigma; the PDE diffusion is D = sigma^2 / 2
        # (Conventions Index; Issue #811) -- matches the FP _diffusion_coefficient and adjoint mode.
        if volatility_field is None:
            D = 0.5 * self.problem.sigma**2
        elif isinstance(volatility_field, (int, float)):
            D = 0.5 * float(volatility_field) ** 2
        else:
            D = 0.5 * float(np.mean(volatility_field)) ** 2

        U = np.zeros((Nt + 1, N))
        U[Nt] = U_terminal

        A_system = self._M / dt + D * self._K
        A_extra, rhs_extra = self._weak_bc_terms(D)
        weak_bc = A_extra is not None
        if weak_bc:
            A_system = A_system + A_extra
        H_class = self.problem.hamiltonian_class

        for n in range(Nt - 1, -1, -1):
            if use_newton and H_class is not None:
                # f(m) is absorbed into H(x, m, p), so the coupling RHS is zero.
                U[n] = self._solve_timestep_newton(
                    U_next=U[n + 1],
                    m_n=M_density[n],
                    D=D,
                    dt=dt,
                    t=n * dt,
                    rhs_coupling=np.zeros(N),
                    max_iterations=max_newton_iterations,
                    tolerance=newton_tolerance,
                )
            else:
                rhs = (self._M / dt) @ U[n + 1]
                if H_class is not None:
                    p_prev = self._nodal_gradient(U_coupling_prev[n])
                    H_values = np.asarray(
                        H_class(self._disc.dof_coordinates, M_density[n], p_prev, t=n * dt), dtype=float
                    ).ravel()
                    rhs += self._M @ H_values

                if weak_bc:
                    if rhs_extra is not None:
                        rhs = rhs + rhs_extra
                    U[n] = spsolve(A_system, rhs)
                elif self._is_pure_neumann():
                    U[n] = spsolve(A_system, rhs)
                else:
                    A_bc, rhs_bc = self._apply_bc_to_system(A_system, rhs)
                    d_dofs, d_vals = self._dirichlet_dofs_and_values()
                    interior = np.setdiff1d(np.arange(N), d_dofs)
                    U[n, interior] = spsolve(A_bc, rhs_bc)
                    U[n, d_dofs] = d_vals

        return U

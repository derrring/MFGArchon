"""
Backend-agnostic weak-form Fokker-Planck solver.

Solves the FP equation forward in time on the assembled weak-form operators of a
``WeakFormDiscretization`` (stiffness ``K``, mass ``M``), so one solver serves
finite elements and meshless Galerkin. A concrete subclass supplies the
discretization, the boundary-condition strategy, and how the advection matrix is
built from the drift (the one piece that is genuinely backend-specific: FEM uses
the exact quadrature-point gradient of ``U``; meshless uses the protocol's
``advection`` with a recovered nodal gradient).

Time discretization: implicit Euler (forward). Mass conservation follows from the
Galerkin weak form (the test space contains constants).

Issue #1131 Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import spsolve

from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy import sparse

    from mfgarchon.alg.numerical.weak_form_discretization import WeakFormDiscretization
    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


class WeakFormFPSolver(BaseFPSolver):
    """FP solver on assembled weak-form operators; BC + advection are subclass hooks."""

    def __init__(self, problem: MFGProblem, discretization: WeakFormDiscretization) -> None:
        super().__init__(problem)
        self._disc = discretization
        self._n_dof = discretization.n_dof
        self._K = discretization.stiffness()
        self._M = discretization.mass()
        self._bc = getattr(problem.geometry, "boundary_conditions", None)

    @property
    def n_dof(self) -> int:
        return self._n_dof

    # --- Boundary-condition strategy (subclass-supplied) ----------------------
    def _is_pure_neumann(self) -> bool:
        raise NotImplementedError

    def _dirichlet_dofs_and_values(self) -> tuple[NDArray, NDArray]:
        raise NotImplementedError

    def _apply_bc_to_system(self, matrix, rhs):
        raise NotImplementedError

    # --- Advection from drift (subclass-supplied) -----------------------------
    def _build_advection(self, U_n: NDArray) -> sparse.csr_matrix:
        r"""Advection matrix for drift $v = -\text{coupling}\cdot\nabla U_n$ in divergence form."""
        raise NotImplementedError

    def _diffusion_coefficient(self, volatility_field) -> float:
        if volatility_field is None:
            return 0.5 * self.problem.sigma**2
        if isinstance(volatility_field, (int, float)):
            return float(volatility_field)
        return float(np.mean(volatility_field))

    def solve_fp_system(
        self,
        m_initial: NDArray,
        drift_field: NDArray | None = None,
        volatility_field: float | NDArray | None = None,
        **kwargs,
    ) -> NDArray:
        """Solve the FP equation forward in time on the weak-form operators."""
        Nt = self.problem.Nt
        dt = self.problem.dt
        N = self._n_dof

        D = self._diffusion_coefficient(volatility_field)

        M = np.zeros((Nt + 1, N))
        M[0] = m_initial[:N] if len(m_initial) >= N else np.pad(m_initial, (0, N - len(m_initial)))

        A_base = self._M / dt + D * self._K
        pure_neumann = self._is_pure_neumann()

        for n in range(Nt):
            rhs = (self._M / dt) @ M[n]

            if drift_field is not None:
                U_n = drift_field[n] if drift_field.ndim > 1 else drift_field
                A_system = A_base + self._build_advection(U_n)
            else:
                A_system = A_base

            if pure_neumann:
                M[n + 1] = spsolve(A_system, rhs)
            else:
                A_bc, rhs_bc = self._apply_bc_to_system(A_system, rhs)
                d_dofs, d_vals = self._dirichlet_dofs_and_values()
                interior = np.setdiff1d(np.arange(N), d_dofs)
                M[n + 1, interior] = spsolve(A_bc, rhs_bc)
                M[n + 1, d_dofs] = d_vals

            M[n + 1] = np.maximum(M[n + 1], 0.0)

        return M

    def solve_fp_step_adjoint_mode(
        self,
        M_current: NDArray,
        A_advection_T: sparse.csr_matrix,
        sigma: float | NDArray | None = None,
        time: float = 0.0,
    ) -> NDArray:
        """Single FP timestep with an externally provided (transposed) advection matrix.

        Used by BlockIterator's adjoint modes: the FP operator is supplied directly,
        e.g. as the transpose of the assembled HJB operator.
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

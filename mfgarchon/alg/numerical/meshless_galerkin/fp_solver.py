"""
Meshless Galerkin (MLS) Fokker-Planck solver.

Thin subclass over ``WeakFormFPSolver`` on a scattered collocation cloud. Forward
time stepping and mass-conserving structure are inherited; this class supplies the
discretization, Neumann BC, and the advection matrix.

Advection follows the current weak-form-family convention (option b, drift = U):
the velocity v = -coupling * grad(U) is recovered at the nodes via the
mass-lumped gradient projection and assembled with the protocol's advection.
(Convention alignment to drift = v = -grad_p H is tracked under #1043.)

Issue #1131 Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from mfgarchon.alg.base_solver import SchemeFamily
from mfgarchon.alg.numerical.meshless_galerkin.discretization import discretization_from_cloud
from mfgarchon.alg.numerical.weak_form_fp_solver import WeakFormFPSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.core.mfg_problem import MFGProblem


class MeshlessGalerkinFPSolver(WeakFormFPSolver):
    """Fokker-Planck on a scattered point cloud via Galerkin MLS (Type-A discrete duality)."""

    _scheme_family = SchemeFamily.MESHLESS_GALERKIN

    def __init__(
        self,
        problem: MFGProblem,
        collocation_points: NDArray,
        delta: float = 0.1,
        degree: int = 2,
        n_gauss: int = 4,
        backend: str = "numpy",
    ) -> None:
        disc = discretization_from_cloud(collocation_points, delta, degree, n_gauss, backend)
        super().__init__(problem, disc)
        self._G_grad: list[sparse.csr_matrix] | None = None

    def _gradient_operators(self) -> list[sparse.csr_matrix]:
        # G_d = diag(1/M_lumped) @ R_d : mass-lumped L2 projection of d/dx_d.
        if self._G_grad is None:
            M_lumped = np.array(self._M.sum(axis=1)).ravel()
            M_lumped[M_lumped < 1e-15] = 1e-15
            inv = 1.0 / M_lumped
            self._G_grad = [(sparse.diags(inv) @ R_d).tocsr() for R_d in self._disc.gradient_projection()]
        return self._G_grad

    def _is_pure_neumann(self) -> bool:
        from mfgarchon.alg.numerical.fem.bc_adapter import is_pure_neumann

        return is_pure_neumann(self._bc)

    def _dirichlet_dofs_and_values(self):
        raise NotImplementedError(
            "MeshlessGalerkinFPSolver supports Neumann/no-flux BC only; Dirichlet (Nitsche) deferred (#1131)."
        )

    def _apply_bc_to_system(self, matrix, rhs):
        raise NotImplementedError(
            "MeshlessGalerkinFPSolver supports Neumann/no-flux BC only; Dirichlet (Nitsche) deferred (#1131)."
        )

    def _build_advection(self, U_n: NDArray) -> sparse.csr_matrix:
        coupling = getattr(self.problem, "coupling_coefficient", 0.5)
        G = self._gradient_operators()
        grad_U = np.column_stack([G_d @ U_n for G_d in G])  # (N, dim)
        velocity = (-coupling * grad_U).T  # (dim, N): v = -coupling * grad(U)
        # FP weak form (Neumann, integrate by parts): the advection contributes
        # -C_b^T to the implicit operator (M/dt + D K - C_b^T), where
        # C_b[i,j] = integral(phi_i (b . grad phi_j)). The TRANSPOSE is the
        # adjoint-consistency identity A_FP = A_HJB^T and makes the test-function
        # gradient sum to zero (sum_i grad phi_i = 0), so column sums vanish and
        # integral(m) is conserved without renormalization. The MINUS comes from
        # the divergence integration by parts. Issue #1131.
        return (-self._disc.advection(velocity).T).tocsr()

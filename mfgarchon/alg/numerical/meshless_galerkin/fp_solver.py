"""
Meshless Galerkin (MLS) Fokker-Planck solver.

Thin subclass over ``WeakFormFPSolver`` on a scattered collocation cloud. Forward
time stepping and mass-conserving structure are inherited; this class supplies the
discretization, Neumann BC, and the advection matrix.

Advection follows the current weak-form-family convention (option b, drift = U):
the velocity v = -coupling * grad(U) is recovered at the nodes via the
mass-lumped gradient projection and assembled with the protocol's advection.
(Convention alignment to drift = v = -grad_p H is tracked under #1043.)

Boundary conditions: Neumann / no-flux (mass-conserving reflecting wall) and
absorbing ``m = 0`` on Dirichlet faces via symmetric Nitsche (#1138). The Nitsche
block is IDENTICAL to the HJB solver's (it is symmetric, hence its own transpose),
so the Type-A duality ``A_FP = A_HJB^T`` is preserved on the diffusion + Nitsche
block; mass leaves through ``Gamma_D`` (do not renormalize under absorbing). Only
the homogeneous case ``m = 0`` is supported. NOTE the advection operator carries no
boundary term, so ``Gamma_D`` is diffusively absorbing but advectively reflecting
(rigorous for ``b.n = 0`` on ``Gamma_D``); see ``nitsche.py``.

Issue #1131 Phase 2; Nitsche absorbing #1138.
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
        nitsche_penalty: float = 20.0,
    ) -> None:
        disc = discretization_from_cloud(collocation_points, delta, degree, n_gauss, backend)
        super().__init__(problem, disc)
        self._G_grad: list[sparse.csr_matrix] | None = None
        self._n_gauss = n_gauss
        self._nitsche_penalty = nitsche_penalty
        self._nitsche_cache: tuple | None = None
        self._nitsche_cache_D: float | None = None

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

    def _weak_bc_terms(self, D: float):
        """Symmetric Nitsche absorbing terms ``m = 0`` for the FP diffusion block.

        Returns ``(N_nitsche, None)`` -- the homogeneous case adds no RHS data. The
        block is assembled identically to the HJB solver (``include_data=False`` only
        skips the zero data vector), so it is symmetric and equals the HJB block,
        keeping ``A_FP = A_HJB^T``. ``(None, None)`` if no Dirichlet segments. Cached
        on ``D``."""
        if self._nitsche_cache is not None and self._nitsche_cache_D == D:
            return self._nitsche_cache
        from mfgarchon.alg.numerical.meshless_galerkin.nitsche import assemble_nitsche_terms

        terms = assemble_nitsche_terms(
            self._disc, self._bc, D, self._nitsche_penalty, self._n_gauss, include_data=False
        )
        self._nitsche_cache = terms
        self._nitsche_cache_D = D
        return terms

    def _dirichlet_dofs_and_values(self):
        raise NotImplementedError(
            "MeshlessGalerkinFPSolver imposes absorbing (m=0) BC weakly via Nitsche (_weak_bc_terms), "
            "not nodal condensation -- its MLS basis is non-interpolatory. Reaching this hook means an "
            "unsupported BC type (e.g. Robin, or inhomogeneous m=g!=0)."
        )

    def _apply_bc_to_system(self, matrix, rhs):
        raise NotImplementedError(
            "MeshlessGalerkinFPSolver imposes absorbing (m=0) BC weakly via Nitsche (_weak_bc_terms), "
            "not nodal condensation. Reaching this hook means an unsupported BC type."
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

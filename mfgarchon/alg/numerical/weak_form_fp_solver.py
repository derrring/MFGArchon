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

from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver, DriftConvention
from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy import sparse

    from mfgarchon.alg.numerical.weak_form_discretization import WeakFormDiscretization
    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


class WeakFormFPSolver(BaseFPSolver):
    """FP solver on assembled weak-form operators; BC + advection are subclass hooks."""

    # Issue #1043: this family takes the value function U and recovers α = -coupling·∇U on its
    # own quadrature/MLS basis (a genuine feature: avoids differentiating U on a coarse FP grid).
    _drift_convention = DriftConvention.VALUE_FUNCTION

    def __init__(self, problem: MFGProblem, discretization: WeakFormDiscretization) -> None:
        super().__init__(problem)
        self._disc = discretization
        self._n_dof = discretization.n_dof
        self._K = discretization.stiffness()
        self._M = discretization.mass()
        # Single source of truth for BCs (matches WeakFormHJBSolver); plain
        # getattr(geometry, "boundary_conditions") misses grids that expose BCs via
        # the accessor method (e.g. TensorProductGrid), silently dropping Dirichlet.
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
        raise NotImplementedError

    def _weak_bc_terms(self, D: float):
        """Optional weak (Nitsche) boundary terms for non-interpolatory bases.

        Returns ``(A_extra, rhs_extra)`` to ADD to the full-size operator and RHS,
        solved on ALL dofs, or ``(None, None)`` for no weak BC (default no-op; FEM
        uses condensation). Meshless Galerkin overrides this with the symmetric
        Nitsche block for absorbing (``m = 0``) boundaries -- the SAME block as the
        HJB solver, so the Type-A transpose identity ``A_FP = A_HJB^T`` is preserved.
        ``rhs_extra`` is ``None`` for the homogeneous absorbing case."""
        return None, None

    # --- Advection from drift (subclass-supplied) -----------------------------
    def _build_advection(self, U_n: NDArray, D: float) -> sparse.csr_matrix:
        r"""Advection matrix for drift $v = -\text{coupling}\cdot\nabla U_n$ in divergence form.

        ``D`` is the diffusion coefficient of the CURRENT solve (``_diffusion_coefficient``,
        volatility-aware), passed so a subclass that adds a diffusion-scaled stabilization
        term (e.g. meshless streamline diffusion) uses the same ``D`` as the stiffness block."""
        raise NotImplementedError

    def _diffusion_coefficient(self, volatility_field) -> float:
        # volatility_field is the SDE volatility sigma (codebase-wide convention); the PDE
        # diffusion coefficient is D = sigma^2 / 2 (Conventions Index; Issue #811). The
        # scalar/array branches previously returned the input as D, skipping the conversion
        # the None branch and solve_fp_step_adjoint_mode already apply.
        if volatility_field is None:
            return 0.5 * self.problem.sigma**2
        if isinstance(volatility_field, (int, float)):
            return 0.5 * float(volatility_field) ** 2
        return 0.5 * float(np.mean(volatility_field)) ** 2

    @deprecated_parameter(param_name="drift_field", since="v0.20.0", replacement="potential_field")
    def solve_fp_system(
        self,
        m_initial: NDArray,
        potential_field: NDArray | None = None,
        volatility_field: float | NDArray | None = None,
        drift_field: NDArray | None = None,  # DEPRECATED alias for potential_field (Issue #1043)
        **kwargs,
    ) -> NDArray:
        """Solve the FP equation forward in time on the weak-form operators.

        ``potential_field`` is the value function ``U(t,x)`` (Issue #1043); this solver recovers
        the advective velocity ``α = -coupling·∇U`` on its own quadrature/MLS basis. It was
        historically -- and misleadingly -- named ``drift_field``, but on this solver that input
        always meant ``U``, never the velocity; ``drift_field`` is kept as a deprecated alias.
        """
        # Issue #1043: drift_field is the deprecated name for the U (potential) input here.
        if drift_field is not None:
            if potential_field is not None:
                raise ValueError("Pass only potential_field; drift_field is its deprecated alias (Issue #1043).")
            potential_field = drift_field

        Nt = self.problem.Nt
        dt = self.problem.dt
        N = self._n_dof

        D = self._diffusion_coefficient(volatility_field)

        M = np.zeros((Nt + 1, N))
        M[0] = m_initial[:N] if len(m_initial) >= N else np.pad(m_initial, (0, N - len(m_initial)))

        A_base = self._M / dt + D * self._K
        A_extra, rhs_extra = self._weak_bc_terms(D)
        weak_bc = A_extra is not None
        if weak_bc:
            A_base = A_base + A_extra
        pure_neumann = self._is_pure_neumann()

        clip_warned = False
        for n in range(Nt):
            rhs = (self._M / dt) @ M[n]

            if potential_field is not None:
                U_n = potential_field[n] if potential_field.ndim > 1 else potential_field
                A_system = A_base + self._build_advection(U_n, D)
            else:
                A_system = A_base

            if weak_bc:
                if rhs_extra is not None:
                    rhs = rhs + rhs_extra
                M[n + 1] = spsolve(A_system, rhs)
            elif pure_neumann:
                M[n + 1] = spsolve(A_system, rhs)
            else:
                A_bc, rhs_bc = self._apply_bc_to_system(A_system, rhs)
                d_dofs, d_vals = self._dirichlet_dofs_and_values()
                interior = np.setdiff1d(np.arange(N), d_dofs)
                M[n + 1, interior] = spsolve(A_bc, rhs_bc)
                M[n + 1, d_dofs] = d_vals

            # Positivity clip. The Galerkin/MLS advection is not an M-matrix, so the
            # solve can produce density undershoots; the clip deletes them, which INJECTS
            # mass and violates conservation. Surface it once per solve rather than failing
            # silently (kernel fail-fast). Streamline diffusion suppresses the undershoots
            # at the source (meshless ``streamline_diffusion_scale > 0``, Issue #1145).
            if not clip_warned:
                injected = -float((self._M @ np.minimum(M[n + 1], 0.0)).sum())
                total = float((self._M @ np.maximum(M[n + 1], 0.0)).sum())
                if injected > 1e-6 * max(total, 1e-300):
                    logger.warning(
                        "FP positivity clip injected mass %.2e (%.1f%% of total) at step %d: the "
                        "Galerkin advection is not monotone; consider stabilization.",
                        injected,
                        100.0 * injected / max(total, 1e-300),
                        n,
                    )
                    clip_warned = True
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

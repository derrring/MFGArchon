"""
Regime-switching MFG iterator with Markov chain coupling.

Solves K coupled HJB-FP systems with inter-regime transition terms:

  HJB_k: -dv^k/dt + H^k(x, m^k, Dv^k) + sum_{j!=k} Q[k,j](v^k - v^j) = 0
  FP_k:  dm^k/dt - L^k[m^k] = sum_{j!=k} Q[j,k] m^j - Q[k,j] m^k

Unlike MultiPopulationIterator (independent populations coupled through
density in Hamiltonian), this handles OPERATOR-LEVEL coupling: each HJB
reads other regimes' value functions, each FP has mass transfer terms.

Issue #925: Part of Phase 2 (Generalized PDE & Institutional MFG Plan).

Design constraints (from Dev Plan Rev 4):
- Default Gauss-Seidel update (Constraint #3): uses updated v^j for j < k
- Cross-terms injected via source_term parameter (#921)
- FP source_term uses existing BaseFPSolver parameter (no FP changes needed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import BaseCouplingIterator

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.core.regime_switching import RegimeSwitchingConfig


@dataclass
class RegimeSwitchingResult:
    """Result container for regime-switching MFG."""

    values: list[NDArray]
    """Value functions v^k for each regime, shape (Nt+1, Nx) each."""

    densities: list[NDArray]
    """Density fields m^k for each regime, shape (Nt+1, Nx) each."""

    converged: bool
    """Whether the Picard iteration converged."""

    iterations: int
    """Number of Picard iterations performed."""

    error_history: list[float] = field(default_factory=list)
    """Max error across all regimes per iteration."""

    regime_config: RegimeSwitchingConfig | None = None
    """The regime switching configuration used."""


class RegimeSwitchingIterator(BaseCouplingIterator):
    """Picard iteration for Markov-switching MFG systems.

    Solves K coupled HJB equations (backward) and K coupled FP equations
    (forward) with inter-regime transition terms injected via source_term.

    Differs from MultiPopulationIterator in coupling structure:
    - Multi-population: K independent HJBs, coupled through m in Hamiltonian
    - Regime switching: K HJBs with explicit cross-terms Q[k,j](v^k - v^j)
                        K FPs with mass transfer Q[j,k]*m^j - Q[k,j]*m^k

    Parameters
    ----------
    problems : list[MFGProblem]
        One MFGProblem per regime. Each defines its own Hamiltonian, sigma, etc.
    regime_config : RegimeSwitchingConfig
        Transition rate matrix Q and regime metadata.
    hjb_solvers : list[BaseHJBSolver]
        One HJB solver per regime.
    fp_solvers : list[BaseFPSolver]
        One FP solver per regime.
    max_iterations : int
        Maximum Picard iterations (default 50).
    tolerance : float
        Convergence tolerance on max over regimes of both the value-function and the
        density change, max(|v^k_{n+1} - v^k_n|, |m^k_{n+1} - m^k_n|) (default 1e-5).
    damping : float
        Damping factor for Picard update (default 0.5).
    update_scheme : Literal["jacobi", "gauss_seidel"]
        Update order for regimes (default "gauss_seidel").
        Gauss-Seidel uses already-updated v^j for j < k (faster convergence).
        Jacobi uses all old values (parallelizable but slower).

    Example
    -------
    >>> from mfgarchon.core.regime_switching import RegimeSwitchingConfig
    >>> Q = np.array([[-0.1, 0.1], [0.2, -0.2]])
    >>> config = RegimeSwitchingConfig(transition_matrix=Q)
    >>> iterator = RegimeSwitchingIterator(
    ...     problems=[problem_high, problem_low],
    ...     regime_config=config,
    ...     hjb_solvers=[hjb_high, hjb_low],
    ...     fp_solvers=[fp_high, fp_low],
    ... )
    >>> result = iterator.solve()
    """

    def __init__(
        self,
        problems: list[MFGProblem],
        regime_config: RegimeSwitchingConfig,
        hjb_solvers: list[BaseHJBSolver],
        fp_solvers: list[BaseFPSolver],
        max_iterations: int = 50,
        tolerance: float = 1e-5,
        damping: float = 0.5,
        update_scheme: Literal["jacobi", "gauss_seidel"] = "gauss_seidel",
    ):
        # Use first problem as representative for base class
        super().__init__(problems[0])
        self._problems = problems
        self._regime = regime_config
        self._hjb = hjb_solvers
        self._fp = fp_solvers
        self._max_iter = max_iterations
        self._tol = tolerance
        self._damping = damping
        self._update_scheme = update_scheme

        # Validate dimensions
        K = regime_config.n_regimes
        if len(problems) != K:
            msg = f"Need {K} problems for {K} regimes, got {len(problems)}"
            raise ValueError(msg)
        if len(hjb_solvers) != K:
            msg = f"Need {K} HJB solvers for {K} regimes, got {len(hjb_solvers)}"
            raise ValueError(msg)
        if len(fp_solvers) != K:
            msg = f"Need {K} FP solvers for {K} regimes, got {len(fp_solvers)}"
            raise ValueError(msg)

        regime_config.validate()
        self._last_result: RegimeSwitchingResult | None = None

    def _make_hjb_source(
        self,
        k: int,
        K: int,
        Q: NDArray,
        Us_full: list[NDArray],
        Us_new: list[NDArray | None],
    ) -> Callable:
        """Build HJB source term for regime k with cross-coupling.

        Captures current state via explicit arguments (not closures over
        loop variables) to satisfy ruff B023.

        Note: Us_new is a mutable list reference. For Gauss-Seidel, this is
        intentional — when solving regime k, Us_new[j] for j < k contains
        the already-updated value function, providing the sequential update.
        """
        update_scheme = self._update_scheme
        dt_k = self._problems[k].dt
        Nt_k = self._problems[k].Nt

        def source(t: float, x: NDArray) -> NDArray:
            s = np.zeros(x.shape[0])
            n = min(round(t / dt_k), Nt_k) if dt_k > 0 else 0
            for j in range(K):
                if j != k:
                    # Gauss-Seidel: use updated v^j if already solved
                    if update_scheme == "gauss_seidel" and Us_new[j] is not None:
                        u_j = Us_new[j]
                    else:
                        u_j = Us_full[j]
                    u_k_n = Us_full[k][n] if n < Us_full[k].shape[0] else Us_full[k][-1]
                    u_j_n = u_j[n] if n < u_j.shape[0] else u_j[-1]
                    s += Q[k, j] * (u_k_n - u_j_n)
            return s

        return source

    def _make_fp_source(
        self,
        k: int,
        K: int,
        Q: NDArray,
        Ms: list[NDArray],
    ) -> Callable:
        """Build FP source term for regime k with mass transfer."""
        dt_k = self._problems[k].dt
        Nt_k = self._problems[k].Nt

        def source(t: float, x: NDArray) -> NDArray:
            n = min(round(t / dt_k), Nt_k) if dt_k > 0 else 0
            s = np.zeros(x.shape[0])
            for j in range(K):
                if j != k:
                    m_j = Ms[j] if Ms[j].ndim == 1 else (Ms[j][n] if n < Ms[j].shape[0] else Ms[j][-1])
                    m_k = Ms[k] if Ms[k].ndim == 1 else (Ms[k][n] if n < Ms[k].shape[0] else Ms[k][-1])
                    s += Q[j, k] * m_j  # inflow from j
                    s -= Q[k, j] * m_k  # outflow to j
            return s

        return source

    def solve(self) -> RegimeSwitchingResult:
        """Run Picard iteration over K coupled regime systems.

        Returns
        -------
        RegimeSwitchingResult
            Contains value functions, densities, convergence info.
        """
        K = self._regime.n_regimes
        Q = self._regime.transition_matrix

        # Initialize: terminal conditions and initial densities
        Us = [p.get_u_terminal() for p in self._problems]
        # Expand terminal to full time-space arrays
        Us_full = []
        for k in range(K):
            p = self._problems[k]
            Nt = p.Nt
            u_terminal = Us[k]
            U_k = np.zeros((Nt + 1, len(u_terminal)))
            U_k[-1] = u_terminal
            Us_full.append(U_k)

        Ms = [p.get_m_initial() for p in self._problems]

        error_history = []

        for iteration in range(self._max_iter):
            Us_new = [None] * K
            Ms_new = [None] * K

            # --- HJB step: solve K backward equations with cross-terms ---
            for k in range(K):
                hjb_source = self._make_hjb_source(k, K, Q, Us_full, Us_new)

                U_k = self._hjb[k].solve_hjb_system(
                    Ms[k]
                    if isinstance(Ms[k], np.ndarray) and Ms[k].ndim == 2
                    else np.tile(Ms[k], (self._problems[k].Nt + 1, 1)),
                    Us_full[k][-1],  # terminal condition
                    Us_full[k],  # previous iterate
                    source_term=hjb_source,
                )
                Us_new[k] = U_k

            # --- FP step: solve K forward equations with mass transfer ---
            for k in range(K):
                fp_source = self._make_fp_source(k, K, Q, Ms)

                m0_k = Ms[k][0] if Ms[k].ndim == 2 else Ms[k]
                M_k = self._fp[k].solve_fp_system(
                    m0_k,
                    drift_field=Us_new[k],  # Use updated value function
                    source_term=fp_source,
                )
                Ms_new[k] = M_k

            # --- Damping ---
            theta = self._damping
            for k in range(K):
                Us_new[k] = theta * Us_new[k] + (1 - theta) * Us_full[k]
                if Ms_new[k] is not None and Ms[k].ndim == Ms_new[k].ndim:
                    Ms_new[k] = theta * Ms_new[k] + (1 - theta) * Ms[k]

            # --- Convergence check ---
            # Gate on BOTH the value function AND the density change — the canonical (u, m)
            # criterion (see fixed_point_utils.check_convergence_criteria). Previously this
            # checked only U, so a regime whose density was still evolving while its value
            # function had stabilized (different timescales across regimes) reported
            # converged=True with a non-converged density (Issue #1043-class one-field defect).
            error_U = max(np.max(np.abs(Us_new[k] - Us_full[k])) for k in range(K))
            # On iteration 0 the initial Ms[k] is the 1D m_initial while Ms_new[k] is the 2D
            # trajectory (the ndim guard above skips iter-0 M-damping), so the M-change is
            # undefined → treat as not-converged.
            if all(Ms_new[k] is not None and Ms[k].ndim == Ms_new[k].ndim for k in range(K)):
                error_M = max(np.max(np.abs(Ms_new[k] - Ms[k])) for k in range(K))
            else:
                error_M = float("inf")
            error = max(error_U, error_M)
            error_history.append(error)

            Us_full = Us_new
            Ms = Ms_new

            if error < self._tol:
                self._last_result = RegimeSwitchingResult(
                    values=Us_full,
                    densities=Ms,
                    converged=True,
                    iterations=iteration + 1,
                    error_history=error_history,
                    regime_config=self._regime,
                )
                return self._last_result

        self._last_result = RegimeSwitchingResult(
            values=Us_full,
            densities=Ms,
            converged=False,
            iterations=self._max_iter,
            error_history=error_history,
            regime_config=self._regime,
        )
        return self._last_result

    def get_results(self) -> tuple:
        """Get computed solution arrays (required by BaseCouplingIterator)."""
        if self._last_result is not None:
            return self._last_result.values[0], self._last_result.densities[0]
        raise RuntimeError("No solution computed yet. Call solve() first.")

    def validate_solution(self) -> dict[str, Any]:
        """Placeholder for solution validation."""
        return {}

"""
Regime-switching MFG iterator with Markov chain coupling.

Solves K coupled HJB-FP systems with inter-regime transition terms:

  HJB_k: -dv^k/dt + H^k(x, m^k, Dv^k) + sum_{j!=k} Q[k,j](v^k - v^j) = 0
  FP_k:  dm^k/dt - L^k[m^k] = sum_{j!=k} Q[j,k] m^j - Q[k,j] m^k

Unlike MultiPopulationIterator (independent populations coupled through
density in Hamiltonian), this handles OPERATOR-LEVEL coupling: each HJB
reads other regimes' value functions, each FP has mass transfer terms.

Issue #925: Part of Phase 2 (Generalized PDE & Institutional MFG Plan).

The FP right-hand side is split before it reaches the solver: the off-diagonal inflow
goes through `source_term`, while the diagonal outflow `-q_k m^k` is carried by an
integrating factor, because a lagged sink defeats the scheme's positivity (Issue #1681).
See `_make_fp_source`.

Design constraints (from Dev Plan Rev 4):
- Default Gauss-Seidel update (Constraint #3): uses updated v^j for j < k
- Cross-terms injected via source_term parameter (#921)
- FP source_term uses existing BaseFPSolver parameter (no FP changes needed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import (
    BaseCouplingIterator,
    assert_bc_providers_resolvable,
    assert_paired_solver_sigma,
)
from mfgarchon.alg.numerical.coupling.fixed_point_utils import (
    fp_solver_sig_params,
    resolve_fp_drift_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.core.regime_switching import RegimeSwitchingConfig

# Largest q_k*T for which the diagonal integrating factor of Issue #1681 is accurate in
# float64. exp(50) is ~5e21: the source spans 21 orders across the horizon, which still
# leaves the early timesteps most of the mantissa. float64 does not overflow until ~709,
# so this is a precision bound, not a range one, and it is deliberately well inside it.
_MAX_OUTFLOW_HORIZON = 50.0


def _fp_boundary_conditions(fp_solver: Any, problem: Any) -> Any:
    """The BC object the FP solve will actually impose.

    NOT ``problem.geometry.boundary_conditions``. ``FPFDMSolver`` resolves its BC from a
    documented hierarchy (``fp_fdm.py``) in which an explicit ``boundary_conditions=``
    kwarg and ``problem.components.boundary_conditions`` both **outrank** geometry. The
    solvers are constructor arguments here, so that cascade has already run by the time
    this iterator is built, and the solver's own attribute is the single authoritative
    answer.

    #1802's first version of the guard read geometry instead, and both higher-priority
    routes walked past it: the predicate flagged the object the solver used and returned
    clean for the object the guard looked at. Falling back to geometry only when the
    solver exposes nothing keeps non-FDM solvers covered.
    """
    resolved = getattr(fp_solver, "boundary_conditions", None)
    if resolved is not None:
        return resolved
    return getattr(getattr(problem, "geometry", None), "boundary_conditions", None)


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
        # Guard EVERY regime's problem, not just problems[0] (self.problem): each regime solves its
        # own HJB and none of them resolves BC providers (Issue #1563).
        for _k, _p in enumerate(problems):
            assert_bc_providers_resolvable(_p, f"RegimeSwitchingIterator[regime {_k}]")
        self._regime = regime_config
        self._hjb = hjb_solvers
        self._fp = fp_solvers
        # Issue #1315: cache each per-regime FP solver's solve_fp_system signature so the FP
        # step can route the value function through the drift-convention dispatcher
        # (resolve_fp_drift_kwargs), exactly as the single-solver iterators do via
        # _init_solver_signatures. One set per regime k because regimes may use different FP
        # solver types.
        self._fp_sig_params_k = [fp_solver_sig_params(fp) for fp in fp_solvers]
        # Issue #1489 (S1): per-regime FP drift-input convention, parallel to _fp_sig_params_k, so
        # resolve_fp_drift_kwargs routes each regime by convention rather than param presence.
        self._fp_drift_convention_k = [getattr(fp, "_drift_convention", None) for fp in fp_solvers]
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
        # RFC #1574 C14 / Issue #1603: each regime HJB-FP pair is an adjoint pair; guard every pair
        # AFTER the count validation above, so a wrong-length list raises the clean count error first.
        for _k in range(K):
            assert_paired_solver_sigma(hjb_solvers[_k], fp_solvers[_k], f"RegimeSwitchingIterator[regime {_k}]")

        regime_config.validate()
        # Only meaningful once validate() has established the generator structure the
        # integrating factor relies on (non-negative off-diagonals, zero row sums).
        self._assert_outflow_horizon_representable(K, regime_config.transition_matrix)
        self._assert_fp_boundary_data_is_homogeneous(K, regime_config.transition_matrix)
        self._last_result: RegimeSwitchingResult | None = None

    def _assert_fp_boundary_data_is_homogeneous(self, K: int, Q: NDArray) -> None:
        """Refuse FP boundary data the integrating factor would silently rescale.

        ``m^k = exp(-q_k t) n^k`` reproduces the intended equation only for operations
        positively homogeneous of degree 1. The operator, the non-negativity clip and the
        mass-fabrication ratio all are. **Imposing inhomogeneous boundary data is not**:
        it is affine, so the solver pins ``n^k`` to ``g`` and the recovered density carries
        ``g * exp(-q_k t)`` instead of ``g``. Measured on a two-regime fixture with
        ``dirichlet_bc(value=0.2)``, ``m(T, x_min)`` came back 0.180967 and 0.163746 against
        the intended 0.2 -- ratios matching ``exp(-q_k T)`` to six digits at two different
        rates.

        Refusing is the honest disposition rather than the capability: carrying the factor
        into the boundary data means making it time-dependent inside the FP solve, which is
        a larger change than the positivity fix this guard belongs to (Issue #1805).

        Zero data of ANY type is fine -- ``g = 0`` makes the condition homogeneous, so
        no-flux, homogeneous Neumann and homogeneous Dirichlet all pass. ``bc_types=None``
        because the transform breaks on inhomogeneous data of *every* type, unlike #1686's
        gate, which honours all types but Neumann.

        Two things this guard got wrong on its first cut, both reachable by ordinary
        constructor arguments and both returning the rescaled density:

        - it read ``problem.geometry.boundary_conditions`` while the solve uses what the FP
          solver resolved, and geometry is only third in that hierarchy. See
          ``_fp_boundary_conditions``.
        - it iterated ``segments`` and never read ``default_bc``/``default_value``. A
          fall-through wall carries data too; #1686 had already closed that exact hole in
          ``base_solver.py`` 300 lines away, which is why the predicate is now shared
          rather than restated here.
        """
        from mfgarchon.geometry.boundary.bc_utils import describe_inhomogeneous_bc_data

        for k in range(K):
            if self._outflow_rate(k, K, Q) == 0.0:
                continue  # no factor is applied to this regime, so nothing is rescaled
            bc = _fp_boundary_conditions(self._fp[k], self._problems[k])
            offences = describe_inhomogeneous_bc_data(bc, bc_types=None)
            if offences:
                msg = (
                    f"RegimeSwitchingIterator[regime {k}]: the boundary conditions this FP solver "
                    f"will impose carry data that is not verifiably zero: {offences}. Since Issue "
                    "#1681 the diagonal outflow is carried by the factor exp(-q_k t), which is "
                    "exact only for boundary conditions homogeneous in the density; with data "
                    "g != 0 the solve would return g*exp(-q_k t) at the boundary instead of g, "
                    "silently. Use homogeneous data -- no-flux, or zero Dirichlet/Neumann, on "
                    "both the segments and the fall-through default. Inhomogeneous data on a "
                    "regime with q_k > 0 is Issue #1805."
                )
                raise ValueError(msg)

    def _assert_outflow_horizon_representable(self, K: int, Q: NDArray) -> None:
        """Refuse a horizon on which the diagonal integrating factor loses the density.

        ``_make_fp_source`` scales the inflow by ``exp(q_k t)`` and ``solve`` divides it
        back out. In real arithmetic that is exact for any ``q_k T``; in float64 the source
        spans ``exp(q_k T)`` across the horizon, so past a few tens of e-folds the early
        timesteps are computed against a source many orders below the late ones and the
        recovered density loses its leading digits. Stopping is the honest response --
        silently returning it is the defect class this fix exists to remove.
        """
        for k in range(K):
            q_k = self._outflow_rate(k, K, Q)
            span = q_k * self._problems[k].T
            if span > _MAX_OUTFLOW_HORIZON:
                msg = (
                    f"RegimeSwitchingIterator[regime {k}]: total outflow rate q_k={q_k:.4g} over "
                    f"horizon T={self._problems[k].T:.4g} gives q_k*T={span:.4g}, above the limit "
                    f"{_MAX_OUTFLOW_HORIZON:.0f} at which the diagonal integrating factor "
                    "(Issue #1681) stays accurate in float64. Shorten T, lower the transition "
                    "rates, or solve the horizon in segments and restart the iterator from the "
                    "intermediate densities."
                )
                raise ValueError(msg)

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
                    # The DPP chain term sum_j Q[k,j](v^k - v^j) sits on the HJB LHS
                    # (-v^k_t + H^k - (sigma^2/2)Lap + cross = 0). The HJB solver subtracts
                    # the source (Phi_U -= source_term, base_hjb), so source = -cross. Passing
                    # +cross flipped the inter-regime coupling sign (2026-06-10 audit, #1251);
                    # the FP source below already carries the correct +inflow/-outflow sign.
                    s -= Q[k, j] * (u_k_n - u_j_n)
            return s

        return source

    def _make_fp_source(
        self,
        k: int,
        K: int,
        Q: NDArray,
        Ms: list[NDArray],
    ) -> Callable:
        """Build the FP inflow source for regime k, pre-scaled by the integrating factor.

        The mass-transfer right-hand side splits into an off-diagonal inflow and a
        diagonal outflow::

            d_t m^k - L^k[m^k] = sum_{j!=k} Q[j,k] m^j  -  q_k m^k,   q_k = sum_{j!=k} Q[k,j]

        Only the first part is genuinely external. The second is linear in the unknown,
        and passing it as a lagged source is what drove ``divergence_upwind`` negative
        (Issue #1681): a positivity-preserving ``L^k`` guarantees non-negativity against a
        **non-negative** source, and ``-q_k m^k`` evaluated at the previous Picard iterate
        is neither non-negative nor proportional to the density actually present, so it
        subtracts mass the current iterate no longer has.

        Substituting ``m^k = exp(-q_k t) n^k`` removes the diagonal exactly::

            d_t n^k - L^k[n^k] = exp(+q_k t) * sum_{j!=k} Q[j,k] m^j

        ``RegimeSwitchingConfig.validate`` enforces ``Q[j,k] >= 0`` off-diagonal, so this
        source is non-negative and ``n^k >= 0`` follows from the scheme; ``solve`` then
        recovers ``m^k = exp(-q_k t) n^k >= 0``. The remaining lag is on the inflow only,
        where it costs accuracy rather than positivity.
        """
        dt_k = self._problems[k].dt
        Nt_k = self._problems[k].Nt
        q_k = self._outflow_rate(k, K, Q)

        def source(t: float, x: NDArray) -> NDArray:
            n = min(round(t / dt_k), Nt_k) if dt_k > 0 else 0
            s = np.zeros(x.shape[0])
            for j in range(K):
                if j != k:
                    m_j = Ms[j] if Ms[j].ndim == 1 else (Ms[j][n] if n < Ms[j].shape[0] else Ms[j][-1])
                    s += Q[j, k] * m_j  # inflow from j
            return np.exp(q_k * t) * s

        return source

    @staticmethod
    def _outflow_rate(k: int, K: int, Q: NDArray) -> float:
        """Total transition rate out of regime k, ``q_k = sum_{j!=k} Q[k,j]``."""
        return float(sum(Q[k, j] for j in range(K) if j != k))

    def _undo_integrating_factor(self, k: int, q_k: float, N_k: NDArray) -> NDArray:
        """Recover ``m^k = exp(-q_k t) n^k`` on regime k's time grid."""
        t_grid = np.arange(self._problems[k].Nt + 1) * self._problems[k].dt
        factor = np.exp(-q_k * t_grid)
        return N_k * factor.reshape((-1,) + (1,) * (N_k.ndim - 1))

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
                # Issue #1315 (Refs #1043): route the value function through the
                # drift-convention dispatcher instead of passing U as drift_field. After the
                # v0.18.6 rename drift_field is the velocity alpha*, not the U-potential; for a
                # smooth separable H the FP solver must receive U via potential_field and derive
                # the velocity internally. Passing drift_field=U is silent-wrong-equilibrium.
                # Mirrors FixedPointIterator.solve() / fictitious_play (#1299), per-regime: each
                # regime has its own problem (Hamiltonian/geometry) and FP solver signature.
                fp_kwargs: dict[str, Any] = {"source_term": fp_source}
                fp_sig_params_k = self._fp_sig_params_k[k]
                if fp_sig_params_k is not None:
                    drift_kwargs, use_positional_U = resolve_fp_drift_kwargs(
                        self._problems[k],
                        fp_sig_params_k,
                        None,
                        Us_new[k],
                        Ms[k],
                        drift_convention=self._fp_drift_convention_k[k],
                    )
                    fp_kwargs.update(drift_kwargs)
                    if use_positional_U:
                        N_k = self._fp[k].solve_fp_system(m0_k, Us_new[k], **fp_kwargs)
                    else:
                        N_k = self._fp[k].solve_fp_system(m0_k, **fp_kwargs)
                else:
                    N_k = self._fp[k].solve_fp_system(m0_k, Us_new[k], source_term=fp_source)
                # The solve returned n^k; the diagonal outflow lives in the integrating
                # factor, not in the source (see _make_fp_source, Issue #1681).
                Ms_new[k] = self._undo_integrating_factor(k, self._outflow_rate(k, K, Q), N_k)

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

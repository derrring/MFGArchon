"""
Multi-population Picard iterator for K-population MFG.

Issue #910 Phase 2: Coordinates K single-population solves in the
Picard fixed-point loop. Each iteration:
  1. Solve K HJB equations (each sees all current densities)
  2. Extract K drift fields via H_k.optimal_control()
  3. Solve K FP equations
  4. Damp all K density fields

Reuses standard HJB/FP solvers — this class only coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.multi_population import MultiPopulationProblem

logger = get_logger(__name__)


class MultiPopulationIterator:
    """Picard iteration for K-population MFG.

    Parameters
    ----------
    multi_problem : MultiPopulationProblem
        Container holding K single-population problems.
    hjb_solvers : list[BaseHJBSolver]
        One HJB solver per population.
    fp_solvers : list[BaseFPSolver]
        One FP solver per population.
    damping_factor : float
        Picard under-relaxation factor in (0, 1]. Default 0.5.

    Examples
    --------
    >>> iterator = MultiPopulationIterator(
    ...     multi_problem=multi,
    ...     hjb_solvers=[hjb_A, hjb_B],
    ...     fp_solvers=[fp_A, fp_B],
    ... )
    >>> result = iterator.solve(max_iterations=50, tolerance=1e-6)
    >>> result.U  # list of K value functions
    >>> result.M  # list of K density fields
    """

    @deprecated_parameter(param_name="damping_factor", since="v0.19.2", replacement="relaxation")
    def __init__(
        self,
        multi_problem: MultiPopulationProblem,
        hjb_solvers: list[BaseHJBSolver],
        fp_solvers: list[BaseFPSolver],
        relaxation: float = 0.5,
        # Legacy kwarg (deprecated since v0.19.2, removal v0.25.0)
        damping_factor: float | None = None,
    ):
        if damping_factor is not None:
            relaxation = damping_factor
        self.multi_problem = multi_problem
        self.hjb_solvers = hjb_solvers
        self.fp_solvers = fp_solvers
        self.relaxation = relaxation
        K = multi_problem.K

        if len(hjb_solvers) != K:
            raise ValueError(f"Need {K} HJB solvers, got {len(hjb_solvers)}")
        if len(fp_solvers) != K:
            raise ValueError(f"Need {K} FP solvers, got {len(fp_solvers)}")

        # Issue #1043: cache each FP solver's solve_fp_system signature so the drift/potential
        # convention can be resolved via the shared resolve_fp_drift_kwargs (same as single-pop).
        import inspect

        self._fp_sig_params: list[set[str] | None] = []
        for fp in fp_solvers:
            try:
                self._fp_sig_params.append(set(inspect.signature(fp.solve_fp_system).parameters))
            except (AttributeError, ValueError, TypeError):
                self._fp_sig_params.append(None)

    @property
    def damping_factor(self) -> float:
        """Deprecated alias for `relaxation` (v0.19.2+). Removal in v0.25.0."""
        return self.relaxation

    def solve(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> MultiPopulationResult:
        """Run Picard iteration over K populations.

        Returns
        -------
        MultiPopulationResult
            Contains U (list of value functions), M (list of densities),
            iterations, and convergence info.
        """
        K = self.multi_problem.K

        # Initialize from each population's problem
        M = []
        U = []
        for k in range(K):
            prob_k = self.multi_problem.get_population(k)
            Nt = prob_k.Nt
            grid_shape = prob_k.geometry.get_grid_shape()
            Nx = grid_shape[0]

            m0_k = prob_k.get_initial_m()
            M_k = np.zeros((Nt + 1, Nx))
            M_k[0] = m0_k
            for n in range(1, Nt + 1):
                M_k[n] = m0_k
            M.append(M_k)

            U_terminal_k = prob_k.get_final_u()
            U_k = np.zeros((Nt + 1, Nx))
            U_k[-1] = U_terminal_k
            U.append(U_k)

        # Picard iteration
        converged = False
        for iteration in range(max_iterations):
            M_old = [m.copy() for m in M]

            # Validate all populations have hamiltonian_class
            for k in range(K):
                if self.multi_problem.get_population(k).hamiltonian_class is None:
                    raise ValueError(
                        f"Population {k} ({self.multi_problem.population_names[k]}) "
                        "has no hamiltonian_class. Cannot compute drift velocity."
                    )

            # Build per-timestep stacked density for cross-coupling.
            # m_all_per_t[n] = concat(M[0][n], M[1][n], ..., M[K-1][n])
            m_all = np.concatenate(M, axis=-1)  # (Nt+1, K*Nx)

            # Step 1: Solve K HJB equations
            for k in range(K):
                solver_k = self.hjb_solvers[k]
                U_terminal_k = U[k][-1]

                # Issue #1071 (lock-faithful): pass the stacked cross-density trajectory ``m_all``
                # directly. The HJB solver indexes it at each integer timestep and feeds it to the
                # population's own Hamiltonian, which slices the other populations via
                # ``population_index`` — no BoundHamiltonian wrapper and no round(t/dt). Only
                # HJBFDMSolver threads this today (Issue #1157); other backends would silently
                # decouple (their solve reads the uncoupled ``problem.hamiltonian_class``), so fail
                # loud for K>1. K==1 has no cross-coupling: a standalone solve is byte-identical,
                # so no cross-density is sent regardless of backend.
                honors_override = getattr(solver_k, "_honors_multipop_hamiltonian_override", False)
                if K > 1 and not honors_override:
                    raise NotImplementedError(
                        f"Multi-population cross-coupling requires HJBFDMSolver for the HJB step "
                        f"(Issue #1157); population {k} uses {type(solver_k).__name__}, which does "
                        "not thread the cross-density trajectory into solve_hjb_system and "
                        "would silently decouple. Use HJBFDMSolver for multi-population MFG."
                    )
                if honors_override:
                    U[k] = solver_k.solve_hjb_system(
                        M[k], U_terminal_k, U[k], cross_density=(None if K == 1 else m_all)
                    )
                else:
                    U[k] = solver_k.solve_hjb_system(M[k], U_terminal_k, U[k])

            # Step 2: Solve K FP equations. The FP drift/potential convention is single-sourced
            # through resolve_fp_drift_kwargs (Issue #1043), identical to the single-population
            # FixedPointIterator — so a K=1 multi-population solve matches single-pop. The
            # cross-density-bound Hamiltonian is passed as h_class so the velocity (non-smooth H)
            # sees the other populations' density (its optimal_control maps t→trajectory row n;
            # Issue #1157). Network problems keep the U-as-drift path (FPNetworkSolver extracts
            # rates internally).
            from .fixed_point_utils import resolve_fp_drift_kwargs

            for k in range(K):
                prob_k = self.multi_problem.get_population(k)
                m0_k = M[k][0]
                H_k = prob_k.hamiltonian_class
                H_bound = H_k.bind_cross_density(m_all, dt=prob_k.dt) if hasattr(H_k, "bind_cross_density") else H_k
                fp_k = self.fp_solvers[k]

                if getattr(prob_k, "spatial_dimension", None) == 0:
                    M_new_k = fp_k.solve_fp_system(m0_k, drift_field=U[k], show_progress=False)
                else:
                    drift_kwargs, use_positional_U = resolve_fp_drift_kwargs(
                        prob_k, self._fp_sig_params[k], None, U[k], M[k], h_class=H_bound
                    )
                    if use_positional_U:
                        M_new_k = fp_k.solve_fp_system(m0_k, U[k], show_progress=False)
                    else:
                        M_new_k = fp_k.solve_fp_system(m0_k, show_progress=False, **drift_kwargs)

                M[k] = (1 - self.relaxation) * M_old[k] + self.relaxation * M_new_k

            # Check convergence
            errors = []
            for k in range(K):
                err_k = np.max(np.abs(M[k] - M_old[k]))
                errors.append(err_k)
            max_error = max(errors)

            logger.info(
                f"Multi-pop iter {iteration + 1}/{max_iterations}: "
                f"max_err={max_error:.4e}, per_pop={[f'{e:.2e}' for e in errors]}"
            )

            if max_error < tolerance:
                converged = True
                break

        return MultiPopulationResult(
            U=U,
            M=M,
            iterations=iteration + 1,
            converged=converged,
            errors=errors,
            population_names=self.multi_problem.population_names,
        )

    @staticmethod
    def _compute_drift_field(U, M, H_class, problem):
        """Deprecated: use _compute_velocity_field instead.

        Kept for backward compatibility. Returns synthetic U potential.
        """
        # Issue #915: dispatch on problem type
        spatial_dim = getattr(problem, "spatial_dimension", None)
        if spatial_dim == 0:
            # Network problem: pass U directly.
            # FPNetworkSolver computes flows from U differences.
            # TODO (#913): FPNetworkSolver should use H.optimal_control()
            return U

        # For smooth separable H, the FP solver's internal drift extraction
        # (-coupling_coefficient * ∇U) is already correct. Skip synthetic-U.
        from mfgarchon.core.hamiltonian import SeparableHamiltonian

        if isinstance(H_class, SeparableHamiltonian) and H_class.control_cost.is_smooth():
            return U

        # Non-smooth H, 1D only: synthetic U approach
        if U.ndim > 2:
            return U  # nD non-smooth: deferred

        # Continuous problem: synthetic U approach (same as FixedPointIterator)
        geometry = problem.geometry
        grid_spacing = geometry.get_grid_spacing()
        dx = grid_spacing[0]
        dt = problem.dt
        Nt = U.shape[0]
        Nx = U.shape[-1]
        coupling_coefficient = getattr(problem, "coupling_coefficient", 1.0)

        grad_U = np.gradient(U, dx, axis=-1)
        bounds = geometry.get_bounds()
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx).reshape(-1, 1)

        alpha_field = np.zeros_like(grad_U)
        for n in range(Nt):
            p = grad_U[n]
            m_n = M[n] if n < M.shape[0] else M[-1]
            alpha_field[n] = H_class.optimal_control(x_grid, m_n, p.reshape(-1, 1), t=n * dt).ravel()

        alpha_mid = 0.5 * (alpha_field[:, :-1] + alpha_field[:, 1:])
        increments = -alpha_mid * dx / coupling_coefficient
        U_synthetic = np.zeros_like(U)
        U_synthetic[:, 1:] = np.cumsum(increments, axis=-1)
        return U_synthetic


class MultiPopulationResult:
    """Result of multi-population MFG solve.

    Attributes
    ----------
    U : list[np.ndarray]
        Value functions, one per population. Each shape (Nt+1, Nx).
    M : list[np.ndarray]
        Density fields, one per population. Each shape (Nt+1, Nx).
    iterations : int
        Number of Picard iterations performed.
    converged : bool
        Whether tolerance was reached.
    errors : list[float]
        Final per-population errors.
    population_names : list[str]
        Names of populations.
    """

    def __init__(self, U, M, iterations, converged, errors, population_names):
        self.U = U
        self.M = M
        self.iterations = iterations
        self.converged = converged
        self.errors = errors
        self.population_names = population_names

    def __repr__(self):
        status = "converged" if self.converged else "not converged"
        return f"MultiPopulationResult({self.K} populations, {self.iterations} iterations, {status})"

    @property
    def K(self) -> int:
        return len(self.U)

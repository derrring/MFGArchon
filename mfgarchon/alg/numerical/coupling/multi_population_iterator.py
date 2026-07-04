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
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver, DriftConvention
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

        # Issue #1489 (S1): per-population FP drift-input convention, parallel to _fp_sig_params, so
        # resolve_fp_drift_kwargs routes each population by convention rather than param presence.
        self._fp_drift_convention: list[DriftConvention | None] = [
            getattr(fp, "_drift_convention", None) for fp in fp_solvers
        ]

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
                honors_cross_density = getattr(solver_k, "_honors_multipop_cross_density", False)
                if K > 1 and not honors_cross_density:
                    raise NotImplementedError(
                        f"Multi-population cross-coupling requires HJBFDMSolver for the HJB step "
                        f"(Issue #1071); population {k} uses {type(solver_k).__name__}, which does "
                        "not thread the cross-density trajectory into solve_hjb_system and "
                        "would silently decouple. Use HJBFDMSolver for multi-population MFG."
                    )
                if honors_cross_density:
                    U[k] = solver_k.solve_hjb_system(
                        M[k], U_terminal_k, U[k], cross_density=(None if K == 1 else m_all)
                    )
                else:
                    U[k] = solver_k.solve_hjb_system(M[k], U_terminal_k, U[k])

            # Step 2: Solve K FP equations. The FP drift/potential convention is single-sourced
            # through resolve_fp_drift_kwargs (Issue #1043), identical to the single-population
            # FixedPointIterator — so a K=1 multi-population solve matches single-pop. Issue #1071
            # (lock-faithful): the population's OWN (unbound) Hamiltonian is passed as h_class plus
            # the stacked cross-density trajectory m_all; the velocity (non-smooth H) sees the other
            # populations' density via cross_density[n] (sliced by population_index) — no
            # BoundHamiltonian wrapper, no round(t/dt). Smooth-separable H takes the potential_field=U
            # path (cross-coupling enters via the HJB only). Network problems keep the U-as-drift
            # path (FPNetworkSolver extracts rates internally).
            from .fixed_point_utils import resolve_fp_drift_kwargs

            for k in range(K):
                prob_k = self.multi_problem.get_population(k)
                m0_k = M[k][0]
                H_k = prob_k.hamiltonian_class
                fp_k = self.fp_solvers[k]

                if getattr(prob_k, "spatial_dimension", None) == 0:
                    M_new_k = fp_k.solve_fp_system(m0_k, drift_field=U[k], show_progress=False)
                else:
                    drift_kwargs, use_positional_U = resolve_fp_drift_kwargs(
                        prob_k,
                        self._fp_sig_params[k],
                        None,
                        U[k],
                        M[k],
                        h_class=H_k,
                        cross_density=m_all,
                        drift_convention=self._fp_drift_convention[k],
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

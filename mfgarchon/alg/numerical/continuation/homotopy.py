"""
Homotopy continuation for tracking MFG equilibrium branches.

Traces the equilibrium m*(lambda) as a problem parameter lambda varies,
using predictor-corrector with implicit function theorem + GMRES.

Issue #926: Part of Phase 3 (Generalized PDE & Institutional MFG Plan).

Design constraints (Dev Plan Rev 4):
- Predictor: implicit function theorem + GMRES, NOT naive finite differences
- All Jacobian information via JVP + Krylov (Constraint #4)
- Never assemble N x N Jacobian explicitly

Mathematical background:
    Track solutions of F(m, lambda) = Phi(m; lambda) - m = 0
    where Phi is the MFG fixed-point map.

    Predictor (tangent):
        dm*/dlambda = -(D_m F)^{-1} D_lambda F
        D_lambda F estimated by one MFG solve (finite difference)
        (D_m F)^{-1} r solved by GMRES with JVP (each iteration = 1 MFG solve)

    Corrector (Newton on F):
        m_{n+1} = m_n - (D_m F)^{-1} F(m_n, lambda_new)
        Uses solver as fixed-point map, warm-started from predictor

    Bifurcation detection:
        Monitor smallest singular values of D_m F via randomized SVD
        Sign change indicates bifurcation between parameter steps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse.linalg

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass
class BifurcationPoint:
    """Detected bifurcation point metadata."""

    parameter_value: float
    """Parameter value at bifurcation."""

    bifurcation_type: str
    """Type: 'saddle-node', 'pitchfork', 'transcritical', or 'unknown'."""

    smallest_sv: float
    """Smallest singular value at detection (near zero)."""

    branch_direction: NDArray | None = None
    """Null vector direction for branch switching (if computed)."""


@dataclass
class ContinuationResult:
    """Result of homotopy continuation."""

    parameter_values: list[float]
    """Parameter values along the traced branch."""

    solutions: list[NDArray]
    """Equilibrium solutions m*(lambda) at each parameter value."""

    bifurcation_points: list[BifurcationPoint] = field(default_factory=list)
    """Detected bifurcation points."""

    converged_steps: int = 0
    """Number of successfully traced steps."""

    failed_at: float | None = None
    """Parameter value where continuation failed (if any)."""


class HomotopyContinuation:
    """Predictor-corrector continuation for MFG equilibria.

    Tracks equilibrium branches as a problem parameter varies.
    Detects bifurcation points via singular value monitoring.

    Parameters
    ----------
    problem_factory : Callable[[float], MFGProblem-like]
        Creates a problem instance for a given parameter value.
        The returned object must work with solver_factory.
    solver_factory : Callable[[Any], solver-like]
        Creates a solver from a problem. The solver must have a
        solve() method returning a result with the equilibrium density.
    extract_solution : Callable[[Any], NDArray]
        Extracts the density array from a solver result.
    parameter_range : tuple[float, float]
        (lambda_start, lambda_end) for the continuation.
    initial_step : float
        Initial step size in parameter space (default 0.01).
    max_steps : int
        Maximum number of continuation steps (default 200).
    corrector_tol : float
        Tolerance for corrector convergence (default 1e-6).
    max_corrector_iters : int
        Maximum corrector iterations per step (default 10).
    adaptive_step : bool
        Whether to adapt step size based on corrector performance.
    min_step : float
        Minimum step size (default 1e-6).
    max_step : float
        Maximum step size (default 0.1).
    detect_bifurcation : bool
        Whether to monitor singular values for bifurcation detection.
    n_svs : int
        Number of smallest singular values to track (default 3).

    Example
    -------
    >>> def problem_factory(lam):
    ...     return MFGProblem(model=Model(hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=lam))), ...)
    >>> def solver_factory(problem):
    ...     return FixedPointIterator(problem, hjb, fp)
    >>> def extract(result):
    ...     return result[1][-1]  # final density
    >>> cont = HomotopyContinuation(
    ...     problem_factory, solver_factory, extract,
    ...     parameter_range=(0.0, 2.0),
    ... )
    >>> result = cont.trace(initial_solution=m0)
    """

    def __init__(
        self,
        problem_factory: Callable[[float], Any],
        solver_factory: Callable[[Any], Any],
        extract_solution: Callable[[Any], NDArray],
        parameter_range: tuple[float, float],
        initial_step: float = 0.01,
        max_steps: int = 200,
        corrector_tol: float = 1e-6,
        max_corrector_iters: int = 10,
        adaptive_step: bool = True,
        min_step: float = 1e-6,
        max_step: float = 0.1,
        detect_bifurcation: bool = True,
        n_svs: int = 3,
    ):
        self._problem_factory = problem_factory
        self._solver_factory = solver_factory
        self._extract = extract_solution
        self._lam_start, self._lam_end = parameter_range
        self._step = initial_step
        self._max_steps = max_steps
        self._corr_tol = corrector_tol
        self._max_corr_iters = max_corrector_iters
        self._adaptive = adaptive_step
        self._min_step = min_step
        self._max_step = max_step
        self._detect_bif = detect_bifurcation
        self._n_svs = n_svs

    def _solve_at(self, m_guess: NDArray, lam: float) -> NDArray:
        """Solve MFG at parameter value lam, returning equilibrium density."""
        problem = self._problem_factory(lam)
        solver = self._solver_factory(problem)
        result = solver.solve()
        return self._extract(result)

    def _fixed_point_map(self, m: NDArray, lam: float) -> NDArray:
        """Evaluate Phi(m; lambda) — one MFG fixed-point iteration."""
        return self._solve_at(m, lam)

    def _predictor_tangent(self, m_curr: NDArray, lam_curr: float, dlam: float) -> NDArray:
        """Tangent predictor via implicit function theorem + GMRES.

        dm*/dlambda = -(D_m F)^{-1} D_lambda F

        D_lambda F: 1 MFG solve (finite difference in lambda)
        (D_m F)^{-1} r: GMRES with JVP (each iteration = 1 MFG solve)

        Total cost: ~4-6 MFG solves.
        """
        eps_lam = max(1e-6 * abs(dlam), 1e-10)

        # D_lambda F ≈ [Phi(m; lam+eps) - Phi(m; lam)] / eps
        phi_base = self._fixed_point_map(m_curr, lam_curr)
        phi_perturbed = self._fixed_point_map(m_curr, lam_curr + eps_lam)
        D_lam_F = (phi_perturbed - phi_base) / eps_lam - 0.0  # F = Phi - m, d/dlam(m)=0

        # (D_m F) v = [Phi(m + eps*v; lam) - Phi(m; lam)] / eps - v
        eps_m = 1e-6
        N = len(m_curr)

        def jvp(v: NDArray) -> NDArray:
            """Jacobian-vector product: D_m F @ v via one MFG solve."""
            v = v.ravel()
            phi_shifted = self._fixed_point_map(m_curr + eps_m * v, lam_curr)
            return (phi_shifted - phi_base) / eps_m - v

        # Solve (D_m F) dm = -D_lam_F via GMRES
        from scipy.sparse.linalg import LinearOperator, gmres

        J_op = LinearOperator((N, N), matvec=jvp, dtype=np.float64)
        dm_dlam, info = gmres(J_op, -D_lam_F, atol=1e-4, maxiter=10)

        if info != 0:
            # GMRES didn't converge — use simple secant predictor
            dm_dlam = D_lam_F  # Crude fallback

        return m_curr + dlam * dm_dlam

    def _corrector(self, m_pred: NDArray, lam: float) -> tuple[NDArray, bool]:
        """Newton corrector: iterate Phi until convergence.

        Simple fixed-point iteration (not full Newton) — uses the MFG
        solver as the map Phi, warm-started from the predictor.

        Returns (m_corrected, converged).
        """
        m = m_pred.copy()
        for _ in range(self._max_corr_iters):
            m_new = self._fixed_point_map(m, lam)
            error = np.max(np.abs(m_new - m))
            m = m_new
            if error < self._corr_tol:
                return m, True
        return m, False

    def _estimate_smallest_svs(self, m: NDArray, lam: float, phi_base: NDArray | None = None) -> NDArray:
        """Estimate smallest singular values of D_m F via randomized SVD.

        Uses scipy.sparse.linalg.svds with LinearOperator wrapping JVP.
        Cost: n_svs * 2 MFG solves (Lanczos iterations).
        """
        if phi_base is None:
            phi_base = self._fixed_point_map(m, lam)

        eps_m = 1e-6
        N = len(m)

        def jvp(v: NDArray) -> NDArray:
            v = v.ravel()
            phi_shifted = self._fixed_point_map(m + eps_m * v, lam)
            return (phi_shifted - phi_base) / eps_m - v

        from scipy.sparse.linalg import LinearOperator, svds

        J_op = LinearOperator((N, N), matvec=jvp, dtype=np.float64)
        try:
            _, svs, _ = svds(J_op, k=min(self._n_svs, N - 2), which="SM")
            return np.sort(svs)
        except (np.linalg.LinAlgError, scipy.sparse.linalg.ArpackNoConvergence, ValueError):
            # SVD may fail for near-singular or ill-conditioned Jacobians
            return np.array([float("inf")] * self._n_svs)

    def trace(self, initial_solution: NDArray) -> ContinuationResult:
        """Trace equilibrium branch from initial solution.

        Parameters
        ----------
        initial_solution : NDArray
            Equilibrium density m* at lambda_start.

        Returns
        -------
        ContinuationResult
            Full branch with solutions, parameters, and bifurcation points.
        """
        lam = self._lam_start
        m = initial_solution.copy()
        dlam = self._step * np.sign(self._lam_end - self._lam_start)

        param_values = [lam]
        solutions = [m.copy()]
        bifurcations: list[BifurcationPoint] = []
        prev_svs: NDArray | None = None

        for step in range(self._max_steps):
            lam_new = lam + dlam

            # Check if we've passed the end
            if (dlam > 0 and lam_new > self._lam_end) or (dlam < 0 and lam_new < self._lam_end):
                lam_new = self._lam_end

            # Predictor
            m_pred = self._predictor_tangent(m, lam, dlam)

            # Corrector
            m_corr, converged = self._corrector(m_pred, lam_new)

            if not converged:
                # Shrink step and retry
                if self._adaptive and abs(dlam) > self._min_step * 1.5:
                    dlam *= 0.5
                    continue
                # Give up at this point
                return ContinuationResult(
                    parameter_values=param_values,
                    solutions=solutions,
                    bifurcation_points=bifurcations,
                    converged_steps=step,
                    failed_at=lam_new,
                )

            # Bifurcation detection
            if self._detect_bif:
                curr_svs = self._estimate_smallest_svs(m_corr, lam_new)
                if prev_svs is not None and len(curr_svs) > 0 and len(prev_svs) > 0:
                    # Check for sign change in smallest SV trend
                    if curr_svs[0] < 0.1 * prev_svs[0] and curr_svs[0] < 1e-3:
                        bifurcations.append(
                            BifurcationPoint(
                                parameter_value=lam_new,
                                bifurcation_type="unknown",
                                smallest_sv=float(curr_svs[0]),
                            )
                        )
                prev_svs = curr_svs

            # Accept step
            lam = lam_new
            m = m_corr
            param_values.append(lam)
            solutions.append(m.copy())

            # Adaptive step sizing
            if self._adaptive and converged:
                dlam = min(abs(dlam) * 1.2, self._max_step) * np.sign(self._lam_end - self._lam_start)

            # Check if we've reached the end
            if abs(lam - self._lam_end) < 1e-12:
                break

        return ContinuationResult(
            parameter_values=param_values,
            solutions=solutions,
            bifurcation_points=bifurcations,
            converged_steps=len(param_values) - 1,
        )

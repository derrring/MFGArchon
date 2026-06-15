"""
MFG System Residual Computation.

Issue #492 Phase 1: Foundation for Newton family solvers.

Computes the residual F(U, M) of the coupled MFG system:
    F_HJB(U, M) = HJB_solve(M) - U
    F_FP(U, M) = FP_solve(U) - M

At a fixed point (equilibrium), F(U, M) = 0.

Usage:
    residual_computer = MFGResidual(problem, hjb_solver, fp_solver)
    F = residual_computer.compute_residual(U, M)  # Returns flattened residual
    ||F|| < tol indicates convergence

References:
    - Achdou & Capuzzo-Dolcetta (2010): Mean field games: Numerical methods
    - Nocedal & Wright (2006): Numerical Optimization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.mfg_logging import get_logger

from .fixed_point_utils import resolve_fp_drift_kwargs
from .source_composition import compose_fp_source, compose_hjb_source

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


class MFGResidual:
    """
    Computes residuals of the coupled MFG system.

    The MFG system at equilibrium satisfies:
        U = HJB_solve(M)  (value function solves HJB given density)
        M = FP_solve(U)   (density solves FP given value function)

    The residual measures deviation from this fixed point:
        F(U, M) = [HJB_solve(M) - U, FP_solve(U) - M]

    Source / nonlocal / obstacle terms (Issue #1361):
        ``source_term_hjb``, ``source_term_fp``, ``nonlocal_operator``, and
        ``obstacle`` are composed via the single-source helpers
        :func:`source_composition.compose_hjb_source` /
        :func:`source_composition.compose_fp_source` — the *same* copy the Picard
        ``FixedPointIterator`` consumes (no private second copy; that was the bug
        class behind #1259 and #1285).

        Design choice (option A): the source is composed from the ``(U, M)``
        **arguments** of :meth:`compute_hjb_output` / :meth:`compute_fp_output`,
        not frozen at an outer iterate. The finite-difference Jacobian
        (``NewtonMFGSolver`` drives ``NewtonSolver`` with an FD Jacobian) then
        differentiates through the source dependence automatically — no separate
        analytic Jacobian term is needed. At convergence ``U = U_new`` and
        ``M = M_new``, so the residual root coincides with the Picard fixed point
        including the source/nonlocal/obstacle terms.

        Obstacle handling matches the Picard path exactly: the approximate
        ``v = 0`` penalty ``(1/eps) * max(0, psi)`` (see ``compose_hjb_source``).
        Proper handling is the ``PenaltyHJBSolver`` wrapper (#924); both coupling
        paths use the same approximation so they do not silently diverge.

    Attributes:
        problem: MFG problem definition
        hjb_solver: HJB solver instance
        fp_solver: FP solver instance
        M_initial: Initial density condition
        U_terminal: Terminal value function

    Example:
        >>> residual_computer = MFGResidual(problem, hjb_solver, fp_solver)
        >>> U, M = initial_guess(problem)
        >>> F = residual_computer.compute_residual(U, M)
        >>> print(f"Residual norm: {np.linalg.norm(F):.2e}")
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        *,
        volatility_field: float | NDArray | Any | None = None,
        drift_field: NDArray | Any | None = None,
    ):
        """
        Initialize MFG residual computer.

        Args:
            problem: MFG problem definition
            hjb_solver: HJB solver instance
            fp_solver: FP solver instance
            volatility_field: Optional diffusion override (Phase 2.3)
            drift_field: Optional drift override for non-MFG problems
        """
        self.problem = problem
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.volatility_field = volatility_field
        self.drift_field = drift_field

        # Cache problem dimensions
        self.num_time_steps = problem.Nt + 1
        self.spatial_shape = tuple(problem.geometry.get_grid_shape())
        self.solution_shape = (self.num_time_steps, *self.spatial_shape)

        # Cache initial/terminal conditions
        self.M_initial: NDArray | None = None
        self.U_terminal: NDArray | None = None
        self._initialize_conditions()

        # Cache solver method signatures for parameter passing
        self._hjb_sig_params: set[str] | None = None
        self._fp_sig_params: set[str] | None = None
        self._cache_solver_signatures()

        # Evaluation counter for diagnostics
        self.residual_evaluations = 0

    def _initialize_conditions(self) -> None:
        """Initialize initial density and terminal value from problem."""
        shape = self.spatial_shape

        # Try to get initial density
        try:
            self.M_initial = self.problem.get_m_initial()
            if self.M_initial.shape != shape:
                self.M_initial = self.M_initial.reshape(shape)
        except AttributeError:
            try:
                self.M_initial = self.problem.m_initial  # Issue #670: unified naming
                if self.M_initial is not None and self.M_initial.shape != shape:
                    self.M_initial = self.M_initial.reshape(shape)
            except AttributeError:
                # Default: uniform density
                self.M_initial = np.ones(shape) / np.prod(shape)
                logger.warning("No initial density found, using uniform")

        # Try to get terminal value
        try:
            self.U_terminal = self.problem.get_u_terminal()
            if self.U_terminal.shape != shape:
                self.U_terminal = self.U_terminal.reshape(shape)
        except AttributeError:
            try:
                self.U_terminal = self.problem.u_terminal  # Issue #670: unified naming
                if self.U_terminal is not None and self.U_terminal.shape != shape:
                    self.U_terminal = self.U_terminal.reshape(shape)
            except AttributeError:
                # Default: zero terminal cost
                self.U_terminal = np.zeros(shape)

    def _cache_solver_signatures(self) -> None:
        """Cache solver method signatures for parameter passing."""
        import inspect

        try:
            sig = inspect.signature(self.hjb_solver.solve_hjb_system)
            self._hjb_sig_params = set(sig.parameters.keys())
        except AttributeError:
            self._hjb_sig_params = None

        try:
            sig = inspect.signature(self.fp_solver.solve_fp_system)
            self._fp_sig_params = set(sig.parameters.keys())
        except AttributeError:
            self._fp_sig_params = None

    def compute_hjb_output(self, M: NDArray, U_prev: NDArray) -> NDArray:
        """
        Compute HJB solver output for given density.

        Issue #1361: composes ``source_term_hjb`` / ``nonlocal_operator`` /
        ``obstacle`` from the ``(M, U_prev)`` arguments via the single-source
        :func:`source_composition.compose_hjb_source` (shared with Picard) and
        passes it as ``source_term=`` when the solver accepts it. Composing from
        the arguments (option A) lets the finite-difference Jacobian differentiate
        through the source dependence; at convergence this matches the Picard
        fixed point.

        Args:
            M: Current density field (Nt+1, *spatial_shape)
            U_prev: Previous value function (for Newton linearization)

        Returns:
            U_new: Value function from HJB solve
        """
        kwargs: dict[str, Any] = {}

        if self._hjb_sig_params is not None:
            if "show_progress" in self._hjb_sig_params:
                kwargs["show_progress"] = False
            if "volatility_field" in self._hjb_sig_params and self.volatility_field is not None:
                kwargs["volatility_field"] = self.volatility_field
            if "source_term" in self._hjb_sig_params:
                hjb_source = compose_hjb_source(self.problem, M, U_prev)
                if hjb_source is not None:
                    kwargs["source_term"] = hjb_source

        return self.hjb_solver.solve_hjb_system(M, self.U_terminal, U_prev, **kwargs)

    def compute_fp_output(self, U: NDArray, M: NDArray | None = None) -> NDArray:
        """
        Compute FP solver output for given value function.

        Issue #1233: the drift/potential convention is single-sourced through
        :func:`resolve_fp_drift_kwargs`, identical to the Picard ``FixedPointIterator``.
        Previously this method always passed the value function as ``drift_field``,
        which the v0.18.6 rename redefined as the *velocity* — so the Newton residual
        was inconsistent with the Picard fixed point and the solvers diverged.

        Issue #1361: composes ``source_term_fp`` from the ``(M, U)`` arguments via
        the single-source :func:`source_composition.compose_fp_source` (shared with
        Picard) and passes it as ``source_term=`` when the solver accepts it.

        Args:
            U: Current value function (Nt+1, *spatial_shape)
            M: Current density (Nt+1, *spatial_shape), used both for non-smooth $H$
                to evaluate the density-dependent velocity and for the time-``t``
                density slice of ``source_term_fp``. Defaults to the time-broadcast
                initial density when not supplied.

        Returns:
            M_new: Density field from FP solve
        """
        if self._fp_sig_params is None:
            return self.fp_solver.solve_fp_system(self.M_initial, U)

        params = self._fp_sig_params
        kwargs: dict[str, Any] = {}
        if "show_progress" in params:
            kwargs["show_progress"] = False
        if "volatility_field" in params and self.volatility_field is not None:
            kwargs["volatility_field"] = self.volatility_field

        if M is None:
            M = np.broadcast_to(self.M_initial, self.solution_shape) if self.M_initial is not None else U

        if "source_term" in params:
            fp_source = compose_fp_source(self.problem, M, U)
            if fp_source is not None:
                kwargs["source_term"] = fp_source

        drift_kwargs, use_positional_U = resolve_fp_drift_kwargs(self.problem, params, self.drift_field, U, M)
        kwargs.update(drift_kwargs)

        if use_positional_U:
            return self.fp_solver.solve_fp_system(self.M_initial, U, **kwargs)
        return self.fp_solver.solve_fp_system(self.M_initial, **kwargs)

    def compute_residual(
        self,
        U: NDArray,
        M: NDArray,
        *,
        return_components: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray, NDArray]:
        """
        Compute MFG system residual F(U, M).

        The residual is defined as:
            F_HJB = HJB_solve(M) - U
            F_FP = FP_solve(U) - M
            F = [F_HJB, F_FP]  (flattened)

        Args:
            U: Value function (Nt+1, *spatial_shape)
            M: Density field (Nt+1, *spatial_shape)
            return_components: If True, return (F, F_HJB, F_FP) tuple

        Returns:
            F: Flattened residual vector (2 * total_size,)
            Or tuple (F, F_HJB, F_FP) if return_components=True
        """
        self.residual_evaluations += 1

        # Compute solver outputs
        U_new = self.compute_hjb_output(M, U)
        M_new = self.compute_fp_output(U, M)

        # Compute residuals (difference from fixed point)
        F_HJB = U_new - U
        F_FP = M_new - M

        # Flatten for Newton solver
        F = np.concatenate([F_HJB.flatten(), F_FP.flatten()])

        if return_components:
            return F, F_HJB, F_FP
        return F

    def compute_residual_norm(
        self,
        U: NDArray,
        M: NDArray,
        *,
        norm_type: str = "l2",
    ) -> float:
        """
        Compute norm of MFG residual.

        Args:
            U: Value function
            M: Density field
            norm_type: 'l2', 'linf', or 'relative'

        Returns:
            Residual norm
        """
        F = self.compute_residual(U, M)

        if norm_type == "l2":
            return float(np.linalg.norm(F))
        elif norm_type == "linf":
            return float(np.max(np.abs(F)))
        elif norm_type == "relative":
            state_norm = np.linalg.norm(np.concatenate([U.flatten(), M.flatten()]))
            return float(np.linalg.norm(F) / (state_norm + 1e-10))
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def pack_state(self, U: NDArray, M: NDArray) -> NDArray:
        """
        Pack (U, M) into a single flattened state vector.

        Args:
            U: Value function (Nt+1, *spatial_shape)
            M: Density field (Nt+1, *spatial_shape)

        Returns:
            x: Flattened state vector [U_flat, M_flat]
        """
        return np.concatenate([U.flatten(), M.flatten()])

    def unpack_state(self, x: NDArray) -> tuple[NDArray, NDArray]:
        """
        Unpack flattened state vector into (U, M).

        Args:
            x: Flattened state vector [U_flat, M_flat]

        Returns:
            (U, M): Reshaped value function and density
        """
        total_size = np.prod(self.solution_shape)
        U_flat = x[:total_size]
        M_flat = x[total_size:]

        U = U_flat.reshape(self.solution_shape)
        M = M_flat.reshape(self.solution_shape)

        return U, M

    def residual_function(self, x: NDArray) -> NDArray:
        """
        Residual function in form suitable for Newton solver.

        Args:
            x: Flattened state [U_flat, M_flat]

        Returns:
            F(x): Flattened residual
        """
        U, M = self.unpack_state(x)
        return self.compute_residual(U, M)

    def get_initial_guess(self) -> NDArray:
        """
        Get initial guess for Newton iteration.

        Returns cold start initialization: U = U_terminal, M = M_initial
        propagated through time.

        Returns:
            x0: Flattened initial state
        """
        U = np.zeros(self.solution_shape)
        M = np.zeros(self.solution_shape)

        # Set boundary conditions
        if self.U_terminal is not None:
            U[-1] = self.U_terminal
        if self.M_initial is not None:
            M[0] = self.M_initial

        return self.pack_state(U, M)

    def reset_evaluation_count(self) -> None:
        """Reset residual evaluation counter."""
        self.residual_evaluations = 0


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing MFGResidual...")

    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import TensorProductGrid

    # Create simple 1D problem
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(geometry=geometry, T=0.5, Nt=10, sigma=0.1)

    # Create solvers
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    # Create residual computer
    residual = MFGResidual(problem, hjb_solver, fp_solver)

    print(f"  Solution shape: {residual.solution_shape}")
    print(f"  State size: {2 * np.prod(residual.solution_shape)}")

    # Get initial guess
    x0 = residual.get_initial_guess()
    print(f"  Initial guess shape: {x0.shape}")

    # Compute residual
    F = residual.residual_function(x0)
    print(f"  Initial residual norm: {np.linalg.norm(F):.2e}")
    print(f"  Residual evaluations: {residual.residual_evaluations}")

    # Test pack/unpack
    U, M = residual.unpack_state(x0)
    x_repacked = residual.pack_state(U, M)
    assert np.allclose(x0, x_repacked), "Pack/unpack should be identity"
    print("  Pack/unpack: OK")

    print("MFGResidual smoke tests passed!")

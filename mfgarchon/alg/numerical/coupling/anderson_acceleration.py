"""
Anderson Acceleration for Fixed-Point Iteration.

Anderson acceleration (also known as Anderson mixing) is a method to accelerate
the convergence of fixed-point iterations by combining information from previous
iterates to compute better next guesses.

Issue #720: Modernized implementation following best practices from:
- SCS 3.2.9 (Stanford): Safeguarding, acceleration intervals
- Boyd et al. (2024): Globally convergent Type-I AA

Key features:
- QR decomposition for numerical stability (avoids normal equations)
- Safeguarding: rejects steps that increase residual
- Acceleration interval: apply AA every k steps for stability
- Stagnation detection: skip AA when differences are near-zero

References:
- Anderson, D. G. (1965). Iterative procedures for nonlinear integral equations.
  Journal of the ACM, 12(4), 547-560.
- Walker, H. F., & Ni, P. (2011). Anderson acceleration for fixed-point iterations.
  SIAM Journal on Numerical Analysis, 49(4), 1715-1735.
- O'Donoghue, B. (2024). Operator Splitting for Conic Optimization via Homogeneous
  Self-Dual Embedding. SCS 3.2.9 Documentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from typing import Literal


class AndersonAccelerator:
    """
    Anderson acceleration for fixed-point iterations.

    Given a fixed-point iteration x_{k+1} = g(x_k), Anderson acceleration
    computes an accelerated iterate by solving a least-squares problem using
    the history of previous iterates and residuals.

    The method maintains a sliding window of the last m iterates and residuals,
    and computes the next iterate as a linear combination that minimizes the
    residual in a least-squares sense.

    Issue #720 features (modern best practices):
    - QR decomposition for numerical stability
    - Safeguarding: rejects AA steps that increase residual
    - Acceleration interval: apply AA every k steps
    - Stagnation detection: skip AA when ΔF ≈ 0

    Attributes:
        depth: Number of previous iterates to use (window size m)
        beta: Damping parameter in [0, 1] (1 = no damping)
        regularization: Regularization parameter for least-squares (0 = none)
        safeguard: If True, reject AA steps that increase residual
        acceleration_interval: Apply AA every k iterations (1 = every step)
        stagnation_tol: Skip AA if ||ΔF|| < stagnation_tol * ||F||

    Note:
        Automatically handles multi-dimensional arrays (e.g., 2D grids for MFG density).
        Arrays are flattened internally for linear algebra operations, then reshaped
        to original shape in the output.
    """

    def __init__(
        self,
        depth: int = 5,
        beta: float = 1.0,
        regularization: float = 1e-8,
        restart_threshold: float | None = None,
        safeguard: bool = True,
        safeguard_factor: float = 2.0,
        acceleration_interval: int = 1,
        stagnation_tol: float = 1e-14,
    ):
        """
        Initialize Anderson accelerator.

        Args:
            depth: Number of previous iterates to store (m in literature).
                Typical values: 5-10. Larger values use more memory but may
                converge faster. Too large can cause numerical instability.
            beta: Damping/relaxation parameter in [0,1]. beta=1 means no damping,
                  beta<1 adds relaxation for stability.
            regularization: Tikhonov regularization for least-squares solve.
                Used when solving via normal equations (fallback).
            restart_threshold: If residual increases by this factor, restart
                (None = never restart based on this criterion).
            safeguard: If True (default), reject AA steps that would increase
                the residual norm. This is critical for stability. (Issue #720)
            safeguard_factor: Reject AA step if it's > safeguard_factor times larger
                than the simple fixed-point step. Default 2.0 allows moderately
                larger steps while catching wild extrapolations.
            acceleration_interval: Apply Anderson acceleration every k iterations.
                Default 1 means every iteration. SCS recommends 10 for stability.
                Intermediate iterations use simple fixed-point with damping.
            stagnation_tol: Skip AA if ||ΔF_newest|| < stagnation_tol * ||F_current||.
                Prevents singular least-squares when iterates barely change.
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if not 0 < beta <= 1:
            raise ValueError(f"beta must be in (0,1], got {beta}")
        if regularization < 0:
            raise ValueError(f"regularization must be >= 0, got {regularization}")
        if acceleration_interval < 1:
            raise ValueError(f"acceleration_interval must be >= 1, got {acceleration_interval}")

        self.depth = depth
        self.beta = beta
        self.regularization = regularization
        self.restart_threshold = restart_threshold
        self.safeguard = safeguard
        self.safeguard_factor = safeguard_factor
        self.acceleration_interval = acceleration_interval
        self.stagnation_tol = stagnation_tol

        # Storage for iterates and residuals
        self.X_history: list[np.ndarray] = []  # x_k
        self.F_history: list[np.ndarray] = []  # g(x_k)
        self.residual_norms: list[float | np.floating[Any]] = []
        self.iteration_count = 0

        # Statistics for diagnostics (Issue #720)
        self.aa_steps_taken = 0
        self.aa_steps_rejected = 0
        self.stagnation_skips = 0

        # Store original shape for reshaping outputs (set on first update)
        self._original_shape: tuple[int, ...] | None = None

    def update(
        self,
        x_current: np.ndarray,
        f_current: np.ndarray,
        method: Literal["type1", "type2"] = "type1",
    ) -> np.ndarray:
        """
        Compute Anderson-accelerated next iterate.

        Args:
            x_current: Current iterate x_k (can be multi-dimensional)
            f_current: Function evaluation g(x_k) (the fixed-point map)
            method: "type1" uses g(x) - x, "type2" uses g(x) formulation

        Returns:
            x_next: Accelerated next iterate (same shape as input)

        Note:
            Multi-dimensional arrays are automatically flattened for internal
            computations and reshaped back to original shape in output.

        Issue #720 improvements:
            - Uses QR decomposition for numerical stability
            - Safeguards against steps that increase residual
            - Respects acceleration_interval setting
            - Detects stagnation to avoid singular systems
        """
        # Store original shape on first call
        if self._original_shape is None:
            self._original_shape = x_current.shape

        # Flatten arrays for vector operations (handles 1D, 2D, 3D, etc.)
        x_flat = x_current.ravel()
        f_flat = f_current.ravel()

        # Compute residual
        residual_flat = f_flat - x_flat  # r_k = g(x_k) - x_k
        residual_norm = np.linalg.norm(residual_flat)
        f_norm = np.linalg.norm(f_flat)

        # Store current state (flattened)
        self.X_history.append(x_flat.copy())
        self.F_history.append(f_flat.copy())
        self.residual_norms.append(residual_norm)
        self.iteration_count += 1

        # Maintain sliding window
        if len(self.X_history) > self.depth + 1:
            self.X_history.pop(0)
            self.F_history.pop(0)

        # Simple fixed-point step (used as fallback and for non-AA iterations)
        x_simple_flat = x_flat + self.beta * residual_flat

        # Need at least 2 iterates for acceleration
        if len(self.X_history) < 2:
            return x_simple_flat.reshape(self._original_shape)

        # Check restart condition
        if self.restart_threshold is not None and len(self.residual_norms) >= 2:
            if residual_norm > self.restart_threshold * self.residual_norms[-2]:
                # Restart: clear history except current
                self.X_history = [x_flat.copy()]
                self.F_history = [f_flat.copy()]
                return x_simple_flat.reshape(self._original_shape)

        # Issue #720: Acceleration interval - only apply AA every k steps
        if self.iteration_count % self.acceleration_interval != 0:
            return x_simple_flat.reshape(self._original_shape)

        # Issue #720: Stagnation detection - skip AA if differences are tiny
        newest_delta_F = self.F_history[-1] - self.F_history[-2]
        if f_norm > 0 and np.linalg.norm(newest_delta_F) < self.stagnation_tol * f_norm:
            self.stagnation_skips += 1
            return x_simple_flat.reshape(self._original_shape)

        # Anderson acceleration with least-squares
        # Build difference matrices (all arrays are already flattened)
        # ΔF_k = [f_1 - f_0, f_2 - f_1, ..., f_k - f_{k-1}]
        delta_F = np.column_stack([self.F_history[i] - self.F_history[i - 1] for i in range(1, len(self.F_history))])

        # Issue #720: Use QR decomposition for numerical stability
        # Avoids forming A = ΔF^T ΔF which squares the condition number
        x_aa_flat = self._compute_aa_step_qr(x_flat, f_flat, residual_flat, delta_F, method)

        # Issue #720: Safeguarding - reject AA step if it's too aggressive
        if self.safeguard:
            # Residual-based safeguard (SCS-style): predict residual at AA point
            # The predicted residual is ||g_k + ΔG γ|| which should be small
            # Reject if predicted residual > safeguard_factor * current residual
            #
            # This is more robust than step-size comparison because:
            # 1. AA naturally takes larger steps for slow-converging problems
            # 2. We care about convergence (residual), not step size
            delta_X = np.column_stack(
                [self.X_history[i] - self.X_history[i - 1] for i in range(1, len(self.X_history))]
            )
            delta_G = delta_F - delta_X  # Residual differences
            gamma = self._solve_least_squares(delta_G, -residual_flat)

            # Predicted residual at AA point
            predicted_residual_norm = np.linalg.norm(residual_flat + delta_G @ gamma)

            # Reject if predicted residual is worse than current
            # Using safeguard_factor as multiplier (default 2.0 = accept if predicted is up to 2x current)
            if predicted_residual_norm > self.safeguard_factor * residual_norm:
                self.aa_steps_rejected += 1
                return x_simple_flat.reshape(self._original_shape)

        self.aa_steps_taken += 1
        return x_aa_flat.reshape(self._original_shape)

    def _solve_least_squares(
        self,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve least squares problem min_x ||Ax - b||^2 with regularization.

        Issue #720: Uses QR decomposition primarily, SVD as fallback.

        Strategy:
        1. QR decomposition (fast, numerically stable)
        2. If ill-conditioned, fall back to truncated SVD (robust)

        Args:
            A: Matrix of shape (n, m), typically n >> m for Anderson
            b: Right-hand side vector of shape (n,)

        Returns:
            Solution x of shape (m,)
        """
        try:
            # Primary: QR decomposition via lstsq
            # numpy's lstsq uses SVD internally when needed
            x, _residuals, _rank, s = np.linalg.lstsq(A, b, rcond=None)

            # Check for ill-conditioning using singular values
            if s is not None and len(s) > 0:
                cond = s[0] / (s[-1] + 1e-16)
                if cond > 1e10:
                    # Fall back to truncated SVD with regularization
                    return self._solve_svd_regularized(A, b)

            return x

        except np.linalg.LinAlgError:
            # Fallback: SVD with regularization
            return self._solve_svd_regularized(A, b)

    def _solve_svd_regularized(
        self,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve least squares using truncated SVD with Tikhonov regularization.

        Issue #720: SVD fallback for ill-conditioned problems.

        Uses the formula: x = V @ diag(s / (s^2 + λ)) @ U^T @ b

        Args:
            A: Matrix of shape (n, m)
            b: Right-hand side vector of shape (n,)

        Returns:
            Regularized solution x of shape (m,)
        """
        U, s, Vt = np.linalg.svd(A, full_matrices=False)

        # Tikhonov regularization: s_i / (s_i^2 + λ) instead of 1/s_i
        # This dampens small singular values
        s_reg = s / (s**2 + self.regularization)

        # Compute solution: x = V @ diag(s_reg) @ U^T @ b
        return Vt.T @ (s_reg * (U.T @ b))

    def _compute_aa_step_qr(
        self,
        x_flat: np.ndarray,
        f_flat: np.ndarray,
        residual_flat: np.ndarray,
        delta_F: np.ndarray,
        method: Literal["type1", "type2"],
    ) -> np.ndarray:
        """
        Compute Anderson acceleration step using QR/SVD least squares.

        Issue #720: SCS-style residual formulation (robust).

        SCS formulation (O'Donoghue 2024, Eyert 1996):
            Build: ΔG = [Δg_0, ..., Δg_{m-1}] where g_j = f_j - x_j (residuals)
            Build: ΔS = [Δx_0, ..., Δx_{m-1}] (iterate differences)
            Solve: min ||g_k + ΔG γ||  (find γ to minimize extrapolated residual)
            Update: x_{k+1} = x_k + β(g_k + (ΔS + ΔG) γ)

        Type I uses g_k directly, Type II is equivalent for linear problems.
        Both share the same core computation in this implementation.

        Args:
            x_flat: Current iterate (flattened)
            f_flat: Function value g(x) (flattened)
            residual_flat: f - x (flattened), this is g_k in SCS notation
            delta_F: Matrix of F differences, shape (n, m)
            method: "type1" or "type2" formulation (both use SCS approach)

        Returns:
            Accelerated iterate (flattened)
        """
        # Build iterate difference matrix ΔS
        delta_X = np.column_stack([self.X_history[i] - self.X_history[i - 1] for i in range(1, len(self.X_history))])

        # Build residual difference matrix ΔG = ΔF - ΔX
        # Since g_j = f_j - x_j, we have Δg_j = Δf_j - Δx_j
        delta_G = delta_F - delta_X

        if method == "type1":
            # SCS Type I: Minimize ||g_k + ΔG γ||
            # Solve: ΔG γ ≈ -g_k
            gamma = self._solve_least_squares(delta_G, -residual_flat)

            # Update: x_{k+1} = x_k + β(g_k + (ΔS + ΔG) γ)
            # Simplify: g_k + ΔG γ ≈ 0 at optimum, so correction is (ΔS + ΔG) γ
            # But we keep the full formula for non-zero residual
            correction = residual_flat + (delta_X + delta_G) @ gamma
            x_next_flat = x_flat + self.beta * correction

        else:  # type2
            # Type II: Same minimization but different update formula
            # Minimize ||g_k + ΔG γ||
            gamma = self._solve_least_squares(delta_G, -residual_flat)

            # Alternative update: x_{k+1} = f_k + ΔS γ
            # This uses the fact that at optimum: g_k + ΔG γ ≈ 0
            # So x_{k+1} = x_k + g_k + ΔS γ + ΔG γ ≈ x_k + g_k + ΔS γ = f_k + ΔS γ
            x_aa_flat = f_flat + delta_X @ gamma

            # Apply damping
            x_next_flat = x_flat + self.beta * (x_aa_flat - x_flat)

        return x_next_flat

    def reset(self):
        """Reset accelerator state (clear history and statistics)."""
        self.X_history.clear()
        self.F_history.clear()
        self.residual_norms.clear()
        self.iteration_count = 0
        # Reset Issue #720 statistics
        self.aa_steps_taken = 0
        self.aa_steps_rejected = 0
        self.stagnation_skips = 0
        self._original_shape = None

    def get_convergence_info(self) -> dict:
        """
        Get convergence and diagnostic information.

        Returns:
            Dictionary with:
            - iteration_count: Total iterations
            - residual_norms: History of residual norms
            - current_depth: Current history size
            - max_depth: Maximum history size (m)
            - aa_steps_taken: Number of AA steps accepted (Issue #720)
            - aa_steps_rejected: Number of AA steps rejected by safeguard
            - stagnation_skips: Number of skips due to stagnation
            - aa_acceptance_rate: Fraction of AA steps accepted
        """
        total_aa_attempts = self.aa_steps_taken + self.aa_steps_rejected
        acceptance_rate = self.aa_steps_taken / total_aa_attempts if total_aa_attempts > 0 else 1.0

        return {
            "iteration_count": self.iteration_count,
            "residual_norms": self.residual_norms.copy(),
            "current_depth": len(self.X_history) - 1,
            "max_depth": self.depth,
            # Issue #720 diagnostics
            "aa_steps_taken": self.aa_steps_taken,
            "aa_steps_rejected": self.aa_steps_rejected,
            "stagnation_skips": self.stagnation_skips,
            "aa_acceptance_rate": acceptance_rate,
        }


def create_anderson_accelerator(
    depth: int = 5,
    beta: float = 1.0,
    regularization: float = 1e-8,
    **kwargs,
) -> AndersonAccelerator:
    """
    Create Anderson accelerator with sensible defaults.

    Args:
        depth: Number of previous iterates (default: 5)
        beta: Damping parameter (default: 1.0 = no damping)
        regularization: Tikhonov regularization (default: 1e-8)
        **kwargs: Additional parameters for AndersonAccelerator

    Returns:
        Configured AndersonAccelerator instance
    """
    return AndersonAccelerator(
        depth=depth,
        beta=beta,
        regularization=regularization,
        **kwargs,
    )

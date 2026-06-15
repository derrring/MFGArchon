from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse

from mfgarchon.alg.base_solver import BaseNumericalSolver, SchemeFamily
from mfgarchon.alg.numerical.hjb_solvers.h_eval import eval_dH_dp_batch, eval_H_batch
from mfgarchon.backends.compat import backend_aware_assign, backend_aware_copy, has_nan_or_inf
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.geometry.boundary import BoundaryConditions

# BC-aware gradient computation (Issue #542 fix)
# Validated in: mfg-research/experiments/crowd_evacuation_2d/runners/exp14b_fdm_bc_fix_validation.py
# Issue #638: Import Robin BC ghost cell computation
# Issue #625: Migrated from tensor_calculus to operators/stencils (tensor_calculus deprecated v0.18.0)
from mfgarchon.geometry.boundary import no_flux_bc, pad_array_with_ghosts
from mfgarchon.operators.stencils.finite_difference import (
    gradient_central,
    gradient_upwind,
    laplacian_with_bc,
)
from mfgarchon.utils.pde_coefficients import CoefficientField, get_spatial_grid

logger = get_logger(__name__)
if TYPE_CHECKING:
    from mfgarchon.backends.base_backend import BaseBackend
    from mfgarchon.config import BaseConfig
    from mfgarchon.core.mfg_problem import MFGProblem

    # from mfgarchon.utils.aux_func import npart, ppart # Not needed here if problem provides jacobian parts

# Clipping limit for p_values ONLY when using numerical FD for Jacobian H-part (fallback)
P_VALUE_CLIP_LIMIT_FD_JAC = 1e6

# Default Newton solver parameters shared across HJB solvers (Issue #966)
DEFAULT_NEWTON_MAX_ITERATIONS: int = 30
DEFAULT_NEWTON_TOLERANCE: float = 1e-6


def _compute_gradient_array_1d(
    U_array: np.ndarray,
    Dx: float,
    bc: BoundaryConditions | None = None,
    upwind: bool = False,
    time: float = 0.0,
) -> np.ndarray:
    """
    Compute gradient for entire 1D array using BC-aware computation.

    Uses operators/stencils with ghost cell padding from geometry/boundary.
    Falls back to periodic BC if bc is None (backward compatibility).

    Args:
        U_array: Solution array of shape (Nx,)
        Dx: Spatial grid spacing
        bc: Boundary conditions. If None, uses periodic BC.
        upwind: If True, use upwind scheme for HJB stability
        time: Current time for time-dependent BCs

    Returns:
        Gradient array of shape (Nx,) with du/dx at each point

    Note:
        Issue #542 fix. Validated in:
        mfg-research/experiments/crowd_evacuation_2d/runners/exp14b_fdm_bc_fix_validation.py
        Achieves 23x error reduction (47.98% -> 2.06%) for Tower-on-Beach problem.

        Issue #625: Migrated from tensor_calculus to operators/stencils.
    """
    Nx = len(U_array)
    if Nx <= 1 or abs(Dx) < 1e-14:
        return np.zeros(Nx)

    # Apply ghost cells if BC provided
    if bc is not None:
        u_work = pad_array_with_ghosts(U_array, bc, ghost_depth=1, time=time)
    else:
        u_work = U_array

    # Compute gradient with selected scheme
    if upwind:
        grad = gradient_upwind(u_work, axis=0, h=Dx)
    else:
        grad = gradient_central(u_work, axis=0, h=Dx)

    # Extract interior if ghost cells were added
    if bc is not None:
        grad = grad[1:-1]

    return grad


def _compute_laplacian_1d(
    U_array: np.ndarray,
    Dx: float,
    bc: BoundaryConditions | None = None,
    domain_bounds: np.ndarray | None = None,
    time: float = 0.0,
) -> np.ndarray:
    """
    Compute Laplacian for entire 1D array using BC-aware computation.

    Delegates to operators/stencils/finite_difference.laplacian_with_bc
    for unified ghost cell handling, matching the nD path (Issue #639).

    Falls back to periodic BC if bc is None (backward compatibility).

    Args:
        U_array: Solution array of shape (Nx,)
        Dx: Spatial grid spacing
        bc: Boundary conditions. If None, uses periodic BC.
        domain_bounds: Unused (kept for backward compatibility)
        time: Current time for time-dependent BCs

    Returns:
        Laplacian array of shape (Nx,) with d^2u/dx^2 at each point
    """
    Nx = len(U_array)
    if Nx <= 1 or abs(Dx) < 1e-14:
        return np.zeros(Nx)

    # Delegate to unified stencil infrastructure (Issue #639)
    return laplacian_with_bc(U_array, [Dx], bc=bc, time=time)


def _get_bc_info_1d(
    bc: BoundaryConditions,
    side: str,
    time: float = 0.0,
) -> tuple:
    """
    Extract full BC info for a given side from BoundaryConditions (Issue #638).

    Returns (bc_type, value, alpha, beta) to support Robin BC:
        alpha*u + beta*du/dn = value

    For non-Robin BC, alpha=1.0, beta=0.0 (Dirichlet-like encoding).

    Args:
        bc: Boundary conditions object
        side: "left" or "right"
        time: Current time for time-dependent BCs

    Returns:
        Tuple (BCType, value, alpha, beta)
    """
    from mfgarchon.geometry.boundary.types import BCType, BoundaryFace

    target_face = BoundaryFace(0, "min" if side == "left" else "max")
    default_alpha, default_beta = 1.0, 0.0

    # Priority 1: Try segment-based access (supports Robin with alpha/beta)
    # Issue #946: Match via seg.face instead of string comparison
    try:
        for seg in bc.segments:
            seg_face = seg.face
            if seg_face is not None and seg_face == target_face:
                value = seg.value
                if callable(value):
                    value = value(time)
                value = value if value is not None else 0.0
                alpha = getattr(seg, "alpha", default_alpha)
                beta = getattr(seg, "beta", default_beta)
                return seg.bc_type, value, alpha, beta
        # No matching segment - use default
        default_value = getattr(bc, "default_value", 0.0)
        return bc.default_type, default_value, default_alpha, default_beta
    except AttributeError:
        pass  # No segments attribute, try unified interface

    # Priority 2: Try unified interface (get_boundary_type method)
    # These methods still take string keys (public API)
    boundary_key = target_face.to_string()
    try:
        bc_type = bc.get_boundary_type(boundary_key)
        bc_value = bc.get_boundary_value(boundary_key, time=time)
        if bc_value is None:
            bc_value = 0.0
        return bc_type, bc_value, default_alpha, default_beta
    except AttributeError:
        pass

    # Default to Neumann zero
    return BCType.NEUMANN, 0.0, default_alpha, default_beta


class BaseHJBSolver(BaseNumericalSolver):
    """Base class for Hamilton-Jacobi-Bellman equation solvers."""

    # Scheme family trait for duality validation (Issue #580)
    _scheme_family = SchemeFamily.GENERIC

    def __init__(self, problem: MFGProblem, config: BaseConfig | None = None) -> None:
        # Maintain backward compatibility - if no config provided, create a minimal one
        if config is None:
            config = type("MinimalConfig", (), {})()

        super().__init__(problem, config)
        self.hjb_method_name = "BaseHJB"

        # Validate solver compatibility if problem supports it (Phase 3.1.5)
        self._validate_problem_compatibility()

    def _validate_problem_compatibility(self) -> None:
        """
        Validate that this solver is compatible with the problem.

        This method checks if the problem has solver compatibility detection
        (Phase 3.1 unified interface) and validates compatibility if available.
        For older problems without this feature, validation is skipped.
        """
        # Issue #543 Phase 2: Replace hasattr with try/except
        try:
            # Get solver type identifier from subclass
            solver_type = self._get_solver_type_id()
            if solver_type is not None:
                self.problem.validate_solver_type(solver_type)
        except AttributeError:
            # Backward compatibility: problem doesn't have validate_solver_type
            return
        except ValueError as e:
            # Re-raise with solver class information
            raise ValueError(f"Cannot use {self.__class__.__name__} with this problem.\n\n{e!s}") from e

    def _get_solver_type_id(self) -> str | None:
        """
        Get solver type identifier for compatibility checking.

        Subclasses should override this to return their type identifier.
        Returns None if solver type cannot be determined (skips validation).
        """
        # Map class names to solver type IDs
        class_name = self.__class__.__name__
        type_mapping = {
            "HJBFDMSolver": "fdm",
            "HJBSemiLagrangianSolver": "semi_lagrangian",
            "HJBWENOSolver": "semi_lagrangian",  # WENO is a semi-Lagrangian variant
            "HJBGFDMSolver": "gfdm",
            "HJBNetworkSolver": "network_solver",
            "HJBDGMSolver": "dgm",
            "HJBPINNSolver": "pinn",
        }
        return type_mapping.get(class_name)

    # === Hamiltonian single source of truth (Issue #1071) ===

    def _hamiltonian_control_cost(self):
        """Control-cost component of the problem's Hamiltonian -- the #1071 single source.

        The Hamiltonian class (``problem.hamiltonian_class``) is the canonical source of
        the control-cost weight ``lambda`` and the optimal control ``alpha* = -dH/dp``.
        This returns its ``ControlCostBase`` component, which exposes ``.lambda_``,
        ``.optimal_control(p)``, ``.dp(p)`` and ``.evaluate(p)``.

        Fail loud when the problem carries no Hamiltonian class (or a Hamiltonian without a
        control-cost component): there is then no canonical control cost to read, and a
        solver must not silently fall back to a scalar placeholder (Issue #1071, the root
        of the #1247 Howard lambda desync).
        """
        H_class = getattr(self.problem, "hamiltonian_class", None)
        if H_class is None:
            raise ValueError(
                f"{type(self).__name__}: problem.hamiltonian_class is None; cannot derive the "
                "control-cost lambda / optimal control from the Hamiltonian single source "
                "(Issue #1071)."
            )
        control_cost = getattr(H_class, "control_cost", None)
        if control_cost is None:
            raise ValueError(
                f"{type(self).__name__}: the Hamiltonian {type(H_class).__name__} exposes no "
                "control_cost component; cannot derive lambda / optimal control (Issue #1071)."
            )
        return control_cost

    def _control_cost_lambda(self) -> float:
        """Control-cost weight ``lambda`` from the Hamiltonian single source (Issue #1071).

        Prefers the Hamiltonian's control cost (``hamiltonian_class.control_cost.lambda_``)
        -- the canonical source of ``alpha* = -p/lambda``. Falls back to ``problem.lambda_``
        ONLY when the problem carries no Hamiltonian class (the legacy LQ fast path, where
        ``problem.lambda_`` is then the sole available source).

        This redirect removes the silent desync where a solver read ``problem.lambda_``
        while the components Hamiltonian's control cost said otherwise -- the root of the
        #1247 Howard defects. For the matched ``lambda=1`` paper case the two agree, so the
        golden byte-identity gate is preserved.
        """
        H_class = getattr(self.problem, "hamiltonian_class", None)
        if H_class is not None:
            control_cost = getattr(H_class, "control_cost", None)
            if control_cost is not None:
                lambda_val = float(control_cost.lambda_)
                if lambda_val <= 0:
                    raise ValueError(
                        f"Control cost lambda must be positive, got {lambda_val} from "
                        f"{type(H_class).__name__}.control_cost (Issue #1071)."
                    )
                return lambda_val
        # Legacy fallback: no Hamiltonian class -- problem.lambda_ is the only source.
        lambda_val = getattr(self.problem, "lambda_", 1.0)
        if lambda_val is None:
            lambda_val = 1.0
        if lambda_val <= 0:
            raise ValueError(
                f"Control cost parameter lambda_ must be positive, got {lambda_val}. "
                "Set problem.lambda_ to a positive value."
            )
        return float(lambda_val)

    # Implementation of BaseMFGSolver abstract methods
    def solve(self) -> np.ndarray:
        """
        Solve standalone HJB equation (single-agent optimal control).

        This method solves the HJB equation in standalone mode without Mean Field Game
        coupling. It uses uniform density (no population effects) and is suitable for:
        - Single-agent optimal control problems
        - HJB solver comparison and benchmarking
        - Algorithm testing and validation

        For Mean Field Games with population coupling, use through MFG fixed-point
        solver which calls solve_hjb_system() with density from Fokker-Planck equation.

        Returns:
            np.ndarray: Value function U(t,x) of shape (Nt+1, Nx+1) for 1D problems
                       or (Nt+1, *spatial_shape) for nD problems.

        Example:
            >>> from mfgarchon.core.mfg_problem import MFGProblem
            >>> from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
            >>> from mfgarchon.geometry import TensorProductGrid
            >>> grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101])
            >>> problem = MFGProblem(geometry=grid, Nt=50, T=1.0)
            >>> solver = HJBFDMSolver(problem)
            >>> U = solver.solve()  # Standalone HJB solution
            >>> print(U.shape)  # (51, 101)
        """
        # Create uniform density for standalone mode (no MFG coupling)
        # Using geometry-first API (works for any dimension)
        Nt_points = self.problem.Nt + 1  # Time points (Nt intervals + 1)

        # Get spatial info from geometry
        spatial_shape = self.problem.geometry.get_grid_shape()  # tuple of Nx_points per dim
        bounds = self.problem.geometry.get_bounds()

        # Compute domain volume for normalization
        volume = 1.0
        for dim in range(len(spatial_shape)):
            volume *= bounds[1][dim] - bounds[0][dim]

        # Create uniform density: shape (Nt_points, *spatial_shape)
        full_shape = (Nt_points, *spatial_shape)
        m_uniform = np.ones(full_shape) / volume

        # Get terminal condition from problem
        U_terminal = self.problem.get_final_u()

        # Initial guess for nonlinear solver: repeat terminal condition across time
        U_prev_picard = np.tile(U_terminal, (Nt_points,) + (1,) * len(spatial_shape))

        # Solve using the specific solver's implementation
        return self.solve_hjb_system(m_uniform, U_terminal, U_prev_picard)

    def validate_solution(self) -> dict[str, float]:
        """Validate the HJB solution."""
        # Basic validation - can be extended by specific solvers
        return {
            "hjb_residual": 0.0,  # Placeholder
            "boundary_error": 0.0,  # Placeholder
            "terminal_condition_error": 0.0,  # Placeholder
        }

    def discretize(self) -> None:
        """Set up spatial and temporal discretization for HJB equation."""
        # Most HJB solvers will discretize based on the problem geometry
        # This is a placeholder that can be overridden

    @abstractmethod
    def solve_hjb_system(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
        volatility_field: float | np.ndarray | None = None,
        source_term: Callable | None = None,
    ) -> np.ndarray:
        """
        Solve the HJB system given density evolution and boundary conditions.

        This is the main method that specific HJB solvers must implement.

        The HJB equation is:
            -∂u/∂t + H(∇u, x, t, m) - S(t, x) = 0

        where S is an optional source term (default: 0), and H includes the
        diffusion term -(σ²/2)Δu internally.

        The source_term enables:
        - Non-local operators: J[v] for jump-diffusion MFG
        - Regime-switching cross-terms: λ_{kj}(v^k - v^j)
        - Penalty for variational inequalities: (1/ε)max(0, Ψ-v)
        - Common Noise Lions derivative correction (Layer 2)

        Args:
            M_density_evolution_from_FP: Density field m(t,x) from Fokker-Planck equation
                                        For MFG: from previous Picard iteration
                                        For standalone: uniform density
            U_final_condition_at_T: Terminal condition u(T,x) at final time
            U_from_prev_picard: Value function from previous Picard iteration
                               For MFG: actual previous iterate U^{k-1}
                               For standalone: initial guess (zeros, terminal condition, etc.)
            volatility_field: Diffusion coefficient specification (optional):
                              None: use problem.sigma (backward compatible)
                              float: constant isotropic σ
                              ndarray: spatially/temporally varying
                              Callable: state-dependent σ(t, x, m)
            source_term: Optional source/sink term S(t, x) in the HJB equation.
                         Callable signature: source_term(t: float, x: ndarray) -> ndarray
                         where x has shape (N, d) and return has shape (N,) or matching grid.
                         Subtracted from H in the residual: F(u) = (u-u_next)/dt + H - S = 0.
                         Default: None (classical MFG, no source term).

        Returns:
            np.ndarray: Value function U(t,x) solution

        Note:
            For MFG consistency, the same volatility_field should be used in both
            HJB and FP solvers. The coupling solver handles this synchronization.
        """


def _calculate_derivatives(
    U_array: np.ndarray,
    i: int,
    Dx: float,
    Nx: int,
    clip: bool = False,
    clip_limit: float = P_VALUE_CLIP_LIMIT_FD_JAC,
    upwind: bool = False,
    precomputed_gradient: np.ndarray | None = None,
) -> dict[tuple[int], float]:
    """
    Calculate derivatives using standard tuple multi-index notation.

    This function computes 1D derivatives and returns them in tuple notation:
    - derivs[(0,)] = u (function value)
    - derivs[(1,)] = ∂u/∂x (first derivative)

    Supports both central difference (default) and upwind discretization:
    - Central: p = (p_forward + p_backward) / 2 (second-order accurate)
    - Upwind: Godunov scheme based on characteristic direction (monotone)

    Args:
        U_array: Solution array
        i: Spatial index
        Dx: Spatial grid spacing
        Nx: Number of spatial points
        clip: Whether to clip derivative values
        clip_limit: Maximum absolute value for clipping
        upwind: If True, use Godunov upwind discretization for HJB stability
        precomputed_gradient: Optional precomputed gradient array from _compute_gradient_array_1d.
                              If provided, uses this instead of local computation.
                              This enables BC-aware gradients (Issue #542 fix).

    Returns:
        Dictionary with tuple keys: {(0,): u, (1,): p}

    See:
        - docs/gradient_notation_standard.md
        - mfgarchon.core.DerivativeTensors for modern derivative representation
    """
    # Backend compatibility - tensor to scalar conversion (Issue #543 acceptable)
    # PyTorch tensors have .item() method, NumPy arrays don't
    # hasattr checks throughout this function are for external library compatibility
    if hasattr(U_array[i], "item"):
        u_i = U_array[i].item()
    else:
        u_i = float(U_array[i])

    # Handle edge cases
    if Nx == 1:
        return {(0,): u_i, (1,): 0.0}

    if abs(Dx) < 1e-14:
        return {(0,): u_i, (1,): np.nan}

    # Use precomputed gradient if available (Issue #542 fix for BC-aware computation)
    if precomputed_gradient is not None:
        p_value = float(precomputed_gradient[i])
        if np.isnan(p_value) or np.isinf(p_value):
            return {(0,): u_i, (1,): np.nan}
        if clip:
            p_value = np.clip(p_value, -clip_limit, clip_limit)
        return {(0,): u_i, (1,): p_value}

    # Legacy path: compute derivatives locally with periodic BC (% Nx indexing)
    # Extract neighbor values
    # Backend compatibility - tensor to scalar conversion (Issue #543 acceptable)
    if hasattr(U_array[(i + 1) % Nx], "item"):
        u_ip1 = U_array[(i + 1) % Nx].item()
    else:
        u_ip1 = float(U_array[(i + 1) % Nx])

    if hasattr(U_array[(i - 1 + Nx) % Nx], "item"):  # Issue #543 acceptable
        u_im1 = U_array[(i - 1 + Nx) % Nx].item()
    else:
        u_im1 = float(U_array[(i - 1 + Nx) % Nx])

    # Check for NaN/Inf in values
    if np.isinf(u_i) or np.isinf(u_ip1) or np.isinf(u_im1) or np.isnan(u_i) or np.isnan(u_ip1) or np.isnan(u_im1):
        return {(0,): u_i, (1,): np.nan}

    # Compute forward and backward differences
    p_forward = (u_ip1 - u_i) / Dx
    p_backward = (u_i - u_im1) / Dx

    # Check for NaN/Inf in derivatives
    if np.isinf(p_forward) or np.isnan(p_forward):
        p_forward = np.nan
    if np.isinf(p_backward) or np.isnan(p_backward):
        p_backward = np.nan

    # Choose discretization based on upwind flag
    if np.isnan(p_forward) or np.isnan(p_backward):
        p_value = np.nan
    elif upwind:
        # Godunov upwind: choose based on characteristic direction
        # For typical MFG Hamiltonian H = |p|²/(2σ²) + V(x,m), characteristic velocity = p/σ²
        # Use backward difference if p ≥ 0 (info from left), forward if p < 0 (info from right)
        p_central_for_sign = (p_forward + p_backward) / 2.0
        if p_central_for_sign >= 0:
            p_value = p_backward  # Characteristic from left
        else:
            p_value = p_forward  # Characteristic from right
    else:
        # Central difference (default, second-order accurate)
        p_value = (p_forward + p_backward) / 2.0

    # Clip if requested
    if clip and not np.isnan(p_value):
        p_value = np.clip(p_value, -clip_limit, clip_limit)

    return {(0,): u_i, (1,): p_value}


def _calculate_p_values(
    U_array: np.ndarray,
    i: int,
    Dx: float,
    Nx: int,
    clip: bool = False,
    clip_limit: float = P_VALUE_CLIP_LIMIT_FD_JAC,
) -> dict[str, float]:
    """
    Legacy wrapper for _calculate_derivatives() using string keys.

    DEPRECATED: Use _calculate_derivatives() with tuple notation instead.

    This function maintains backward compatibility for code expecting
    string keys {"forward": ..., "backward": ...}.

    Returns:
        Dictionary with string keys {" forward": p, "backward": p}

    See:
        - _calculate_derivatives() for tuple notation
        - mfgarchon.core.DerivativeTensors for modern derivative representation
    """
    # Call new tuple-based function
    derivs = _calculate_derivatives(U_array, i, Dx, Nx, clip=clip, clip_limit=clip_limit)

    # Convert to legacy string-keyed format: {(1,): p} -> {"forward": p, "backward": p}
    p = derivs.get((1,), 0.0)
    return {"forward": p, "backward": p}


def _clip_p_values(p_values: dict[str, float], clip_limit: float) -> dict[str, float]:  # Helper for FD Jac
    clipped_p_values = {}
    for key, p_val in p_values.items():
        if np.isnan(p_val) or np.isinf(p_val):
            clipped_p_values[key] = np.nan
        else:
            clipped_p_values[key] = np.clip(p_val, -clip_limit, clip_limit)
    return clipped_p_values


def compute_hjb_residual(
    U_n_current_newton_iterate: np.ndarray,  # U_kp1_n in notebook's getFnU_withM
    U_n_plus_1_from_hjb_step: np.ndarray,  # U_kp1_np1 in notebook
    M_density_at_n_plus_1: np.ndarray,  # M_k_np1 in notebook
    problem: MFGProblem,
    t_idx_n: int,  # Time index for U_n
    backend=None,  # Backend for MPS/CUDA support
    sigma_at_n: float | np.ndarray | None = None,  # Diffusion at time t_n
    use_upwind: bool = True,  # Use Godunov upwind (True) or central (False)
    bc: BoundaryConditions | None = None,  # Boundary conditions (Issue #542 fix)
    domain_bounds: np.ndarray | None = None,  # Domain bounds for BC
    current_time: float = 0.0,  # Current time for time-dependent BCs
    bc_values: dict[str, float] | None = None,  # Issue #574: Per-boundary BC values
    source_term: np.ndarray | None = None,  # MMS verification: pre-evaluated S(t, x)
    active_hamiltonian=None,  # Issue #1157: multi-pop cross-density bound H (None => problem's own)
) -> np.ndarray:
    Nx = problem.geometry.get_grid_shape()[0]
    dx = problem.geometry.get_grid_spacing()[0]
    dt = problem.dt

    # Handle diffusion field - NumPy will broadcast scalar automatically
    if sigma_at_n is None:
        sigma = problem.sigma  # Backward compatible (scalar)
    elif isinstance(sigma_at_n, (int, float)):
        sigma = sigma_at_n  # Keep as scalar (not float()) for broadcasting
    else:
        # Spatially varying diffusion array
        sigma = sigma_at_n

    if backend is not None:
        Phi_U = backend.zeros((Nx,))
    else:
        Phi_U = np.zeros(Nx)

    if has_nan_or_inf(U_n_current_newton_iterate, backend):
        if backend is not None:
            return backend.full((Nx,), float("nan"))
        return np.full(Nx, np.nan)

    # Time derivative: (U_n_current - U_{n+1})/dt
    # Notebook FnU[i] += -(Ukp1_np1[i] - Ukp1_n[i])/dt;  (U_n - U_{n+1})/dt
    if abs(dt) < 1e-14:
        if not np.allclose(U_n_current_newton_iterate, U_n_plus_1_from_hjb_step, rtol=1e-9, atol=1e-9):
            pass
    else:
        time_deriv_term = (U_n_current_newton_iterate - U_n_plus_1_from_hjb_step) / dt
        if has_nan_or_inf(time_deriv_term, backend):
            if backend is not None:
                return backend.full((Nx,), float("nan"))
            Phi_U[:] = np.nan
            return Phi_U
        Phi_U += time_deriv_term

    # Diffusion term: -(sigma^2/2) * (U_n_current)_xx
    # Notebook FnU[i] += - ((sigma**2)/2.) * (Ukp1_n[i+1]-2*Ukp1_n[i]+Ukp1_n[i-1])/(dx**2)
    if abs(dx) > 1e-14 and Nx > 1:
        # Backend compatibility - PyTorch .roll() vs NumPy BC-aware Laplacian (Issue #543 acceptable)
        if backend is not None and hasattr(U_n_current_newton_iterate, "roll"):
            # PyTorch tensors have .roll() method (no BC support for GPU yet)
            U_xx = (
                U_n_current_newton_iterate.roll(-1)
                - 2 * U_n_current_newton_iterate
                + U_n_current_newton_iterate.roll(1)
            ) / dx**2
        else:
            # NumPy: use BC-aware Laplacian (Issue #542 fix)
            U_xx = _compute_laplacian_1d(
                U_n_current_newton_iterate,
                dx,
                bc=bc,
                domain_bounds=domain_bounds,
                time=current_time,
            )
        if has_nan_or_inf(U_xx, backend):
            if backend is not None:
                return backend.full((Nx,), float("nan"))
            Phi_U[:] = np.nan
            return Phi_U

        # Apply diffusion term (NumPy broadcasts scalar automatically)
        # Works for both constant σ (scalar) and σ(x,t) (array)
        Phi_U += -diffusion_from_volatility(sigma, kind="field") * U_xx

    # Precompute BC-aware gradient for entire array (Issue #542 fix)
    # This avoids repeated per-point computation with wrong BC
    precomputed_grad = None
    if bc is not None and backend is None:
        precomputed_grad = _compute_gradient_array_1d(
            U_n_current_newton_iterate, dx, bc=bc, upwind=use_upwind, time=current_time
        )

    # For m-coupling term, original notebook passed gradUkn, gradUknim1 (from prev Picard iter)
    # but mdmH_withM itself didn't use them. We'll pass an empty dict for now.
    U_n_derivatives_for_m_coupling: dict[str, Any] = {}  # Not used by MFGProblem's term

    # Issue #789: Batch Hamiltonian path — single H_class() call replaces per-point loop
    # Conditions: precomputed_grad available (BC-aware), no backend (NumPy), H_class exists
    # Issue #1157: a multi-population bound Hamiltonian (active_hamiltonian) is honored
    # only on this batch path; the per-point fallback below calls problem.H() (the
    # uncoupled Hamiltonian), which would silently drop the cross-density coupling.
    H_class = active_hamiltonian if active_hamiltonian is not None else problem.hamiltonian_class
    if active_hamiltonian is not None and not (precomputed_grad is not None and backend is None):
        raise NotImplementedError(
            "Multi-population Hamiltonian override requires the batch Hamiltonian path "
            "(precomputed_grad available, backend=None); the per-point problem.H() fallback "
            "cannot honor the cross-density coupling and would silently decouple (Issue #1157)."
        )
    if precomputed_grad is not None and backend is None and H_class is not None:
        # Build batch arrays
        x_grid = problem.geometry.get_spatial_grid()  # (Nx, 1)
        m_grid = np.asarray(M_density_at_n_plus_1, dtype=float)  # (Nx,)
        p_grid = precomputed_grad.reshape(-1, 1)  # (Nx, 1)

        # Single batch call — eliminates per-point problem.H() overhead
        H_values = eval_H_batch(H_class, x_grid, m_grid, p_grid, current_time).ravel()

        # Mask: propagate NaN from existing Phi_U or from NaN gradients
        nan_mask = np.isnan(Phi_U) | np.isnan(precomputed_grad) | ~np.isfinite(H_values)
        Phi_U[~nan_mask] += H_values[~nan_mask]
        Phi_U[nan_mask & ~np.isnan(Phi_U)] = np.nan  # New NaN from grad/H

        # Subtract source term (MMS verification): F(u) = ... - S = 0
        if source_term is not None:
            Phi_U -= source_term

        return Phi_U

    # Fallback: per-point loop for backend != None or missing precomputed_grad
    for i in range(Nx):
        # Backend compatibility - tensor to scalar conversion (Issue #543 acceptable)
        phi_val = Phi_U[i].item() if hasattr(Phi_U[i], "item") else float(Phi_U[i])
        if np.isnan(phi_val):
            continue

        derivs = _calculate_derivatives(
            U_n_current_newton_iterate, i, dx, Nx, clip=False, upwind=use_upwind, precomputed_gradient=precomputed_grad
        )

        if np.any(np.isnan(list(derivs.values()))):
            Phi_U[i] = float("nan")
            continue

        # Backend compatibility - tensor to scalar conversion (Issue #543 acceptable)
        m_val = (
            M_density_at_n_plus_1[i].item()
            if hasattr(M_density_at_n_plus_1[i], "item")
            else float(M_density_at_n_plus_1[i])
        )
        hamiltonian_val = problem.H(x_idx=i, m_at_x=m_val, derivs=derivs, t_idx=t_idx_n)
        if np.isnan(hamiltonian_val) or np.isinf(hamiltonian_val):
            Phi_U[i] = float("nan")
            continue
        else:
            Phi_U[i] += hamiltonian_val

        # m-coupling term (Issue #673: always None, kept for API compatibility)
        m_coupling_term = problem.get_hjb_residual_m_coupling_term(
            M_density_at_n_plus_1, U_n_derivatives_for_m_coupling, i, t_idx_n
        )
        if m_coupling_term is not None:
            if np.isnan(m_coupling_term) or np.isinf(m_coupling_term):
                Phi_U[i] = float("nan")
                continue
            Phi_U[i] += m_coupling_term

    # Subtract source term (MMS verification): F(u) = ... - S = 0
    if source_term is not None:
        Phi_U -= source_term

    return Phi_U


def compute_hjb_jacobian(
    U_n_current_newton_iterate: np.ndarray,  # U_new_n_tmp in notebook Newton step
    U_k_n_from_prev_picard: np.ndarray,  # U_k_n (or Uoldn) in notebook Jacobian
    M_density_at_n_plus_1: np.ndarray,
    problem: MFGProblem,
    t_idx_n: int,
    backend=None,  # Backend for MPS/CUDA support
    sigma_at_n: float | np.ndarray | None = None,  # Diffusion at time t_n
    use_upwind: bool = True,  # Use Godunov upwind (True) or central (False)
    bc: BoundaryConditions | None = None,  # Boundary conditions (Issue #542 fix)
    domain_bounds: np.ndarray | None = None,  # Domain bounds for BC
    current_time: float = 0.0,  # Current time for time-dependent BCs
    active_hamiltonian=None,  # Issue #1157: multi-pop cross-density bound H (None => problem's own)
) -> sparse.csr_matrix:
    Nx = problem.geometry.get_grid_shape()[0]
    dx = problem.geometry.get_grid_spacing()[0]
    dt = problem.dt
    eps = 1e-7

    # Handle diffusion field - NumPy will broadcast scalar automatically
    if sigma_at_n is None:
        sigma = problem.sigma  # Backward compatible (scalar)
    elif isinstance(sigma_at_n, (int, float)):
        sigma = sigma_at_n  # Keep as scalar (not float()) for broadcasting
    else:
        # Spatially varying diffusion array
        sigma = sigma_at_n

    # For Jacobian, we always need NumPy arrays for scipy.sparse
    # Convert backend arrays to NumPy if needed
    if backend is not None:
        from mfgarchon.backends.compat import to_numpy

        U_n_np = to_numpy(U_n_current_newton_iterate, backend)
    else:
        U_n_np = U_n_current_newton_iterate

    J_D = np.zeros(Nx)
    J_L = np.zeros(Nx)
    J_U = np.zeros(Nx)

    if has_nan_or_inf(U_n_current_newton_iterate, backend):
        return sparse.diags([np.full(Nx, np.nan)], [0], shape=(Nx, Nx)).tocsr()

    # Time derivative part: d/dU_n_current[j] of (U_n_current[i] - U_{n+1}[i])/dt
    if abs(dt) > 1e-14:
        J_D += 1.0 / dt

    # Diffusion part: d/dU_n_current[j] of -(sigma^2/2) * (U_n_current)_xx[i]
    # NumPy broadcasts scalar σ automatically to match array shapes
    if abs(dx) > 1e-14 and Nx > 1:
        # Diagonal: ∂/∂U[i] of -(σ²/2)(U[i-1] - 2U[i] + U[i+1])/dx² = σ²/dx²
        J_D += sigma**2 / dx**2
        # Off-diagonal: coefficient for U[i±1] terms
        val_off_diag_diff = -(sigma**2) / (2 * dx**2)
        J_L += val_off_diag_diff
        J_U += val_off_diag_diff
        # Note: For spatially varying σ(x), this assumes σ is smooth.
        # More accurate treatment would include ∂σ/∂x terms (Phase 3 extension)

    # Hamiltonian part: analytical Jacobian via chain rule (Issue #789)
    # dΦ/dU_j = (dH/dp)_i * (dp_i/dU_j), where dp/dU comes from the gradient stencil.
    # Issue #1157: the multi-pop bound Hamiltonian is honored only on the batch dp() path;
    # the per-point problem.H() fallback below would silently drop the cross-density coupling.
    H_class = active_hamiltonian if active_hamiltonian is not None else problem.hamiltonian_class
    if active_hamiltonian is not None and not (backend is None and H_class is not None and abs(dx) > 1e-14 and Nx > 1):
        raise NotImplementedError(
            "Multi-population Hamiltonian override requires the batch Jacobian path "
            "(backend=None, Nx>1); the per-point problem.H() fallback cannot honor the "
            "cross-density coupling and would silently decouple (Issue #1157)."
        )
    if backend is None and H_class is not None and abs(dx) > 1e-14 and Nx > 1:
        # Compute BC-aware gradient for stencil direction
        precomputed_grad = _compute_gradient_array_1d(U_n_np, dx, bc=bc, upwind=use_upwind, time=current_time)

        # Build batch arrays for H_class.dp()
        x_grid = problem.geometry.get_spatial_grid()  # (Nx, 1)
        m_grid = np.asarray(M_density_at_n_plus_1, dtype=float)  # (Nx,)
        p_grid = precomputed_grad.reshape(-1, 1)  # (Nx, 1)

        # Single batch call: dH/dp at all grid points
        dH_dp = eval_dH_dp_batch(
            H_class, x_grid, m_grid, p_grid, current_time
        ).ravel()  # (Nx,) — squeeze the 1D momentum dimension

        # Stencil coefficients: dp_i/dU_j depends on upwind direction
        # Godunov upwind: p >= 0 uses backward (U_i - U_{i-1})/dx,
        #                 p < 0  uses forward  (U_{i+1} - U_i)/dx
        inv_dx = 1.0 / dx
        if use_upwind:
            backward_mask = precomputed_grad >= 0  # (Nx,)
            # Diagonal: dp_i/dU_i
            J_D += dH_dp * np.where(backward_mask, inv_dx, -inv_dx)
            # Lower: dp_i/dU_{i-1} (backward stencil only)
            J_L += dH_dp * np.where(backward_mask, -inv_dx, 0.0)
            # Upper: dp_i/dU_{i+1} (forward stencil only)
            J_U += dH_dp * np.where(backward_mask, 0.0, inv_dx)
        else:
            # Central difference: p_i = (U_{i+1} - U_{i-1}) / (2*dx)
            half_inv_dx = inv_dx / 2.0
            # dp_i/dU_i = 0 (central difference has no diagonal stencil)
            J_L += dH_dp * (-half_inv_dx)
            J_U += dH_dp * half_inv_dx
    else:
        # Fallback: per-point numerical FD Jacobian for backend or no H_class
        for i in range(Nx):
            U_perturbed_p_i = U_n_np.copy()
            U_perturbed_p_i[i] += eps
            U_perturbed_m_i = U_n_np.copy()
            U_perturbed_m_i[i] -= eps

            derivs_p_i = _calculate_derivatives(
                U_perturbed_p_i,
                i,
                dx,
                Nx,
                clip=True,
                clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                upwind=use_upwind,
            )
            derivs_m_i = _calculate_derivatives(
                U_perturbed_m_i,
                i,
                dx,
                Nx,
                clip=True,
                clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                upwind=use_upwind,
            )

            H_p_i = np.nan
            H_m_i = np.nan
            if not (np.any(np.isnan(list(derivs_p_i.values())))):
                H_p_i = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_p_i, t_idx=t_idx_n)
            if not (np.any(np.isnan(list(derivs_m_i.values())))):
                H_m_i = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_m_i, t_idx=t_idx_n)

            if not (np.isnan(H_p_i) or np.isnan(H_m_i) or np.isinf(H_p_i) or np.isinf(H_m_i)):
                J_D[i] += (H_p_i - H_m_i) / (2 * eps)

            if Nx > 1:
                im1 = (i - 1 + Nx) % Nx
                U_perturbed_p_im1 = U_n_np.copy()
                U_perturbed_p_im1[im1] += eps
                U_perturbed_m_im1 = U_n_np.copy()
                U_perturbed_m_im1[im1] -= eps
                derivs_p_im1 = _calculate_derivatives(
                    U_perturbed_p_im1,
                    i,
                    dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                    upwind=use_upwind,
                )
                derivs_m_im1 = _calculate_derivatives(
                    U_perturbed_m_im1,
                    i,
                    dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                    upwind=use_upwind,
                )
                H_p_im1 = np.nan
                H_m_im1 = np.nan
                if not (np.any(np.isnan(list(derivs_p_im1.values())))):
                    H_p_im1 = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_p_im1, t_idx=t_idx_n)
                if not (np.any(np.isnan(list(derivs_m_im1.values())))):
                    H_m_im1 = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_m_im1, t_idx=t_idx_n)
                if not (np.isnan(H_p_im1) or np.isnan(H_m_im1) or np.isinf(H_p_im1) or np.isinf(H_m_im1)):
                    J_L[i] += (H_p_im1 - H_m_im1) / (2 * eps)

                ip1 = (i + 1) % Nx
                U_perturbed_p_ip1 = U_n_np.copy()
                U_perturbed_p_ip1[ip1] += eps
                U_perturbed_m_ip1 = U_n_np.copy()
                U_perturbed_m_ip1[ip1] -= eps
                derivs_p_ip1 = _calculate_derivatives(
                    U_perturbed_p_ip1,
                    i,
                    dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                    upwind=use_upwind,
                )
                derivs_m_ip1 = _calculate_derivatives(
                    U_perturbed_m_ip1,
                    i,
                    dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                    upwind=use_upwind,
                )
                H_p_ip1 = np.nan
                H_m_ip1 = np.nan
                if not (np.any(np.isnan(list(derivs_p_ip1.values())))):
                    H_p_ip1 = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_p_ip1, t_idx=t_idx_n)
                if not (np.any(np.isnan(list(derivs_m_ip1.values())))):
                    H_m_ip1 = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_m_ip1, t_idx=t_idx_n)
                if not (np.isnan(H_p_ip1) or np.isnan(H_m_ip1) or np.isinf(H_p_ip1) or np.isinf(H_m_ip1)):
                    J_U[i] += (H_p_ip1 - H_m_ip1) / (2 * eps)

    # Assemble sparse Jacobian
    # The original notebook used rolled J_L and J_U for its spdiags call:
    # spdiags([np.roll(ML,-1),MD,np.roll(MU,1)],[-1,0,1],Nx,Nx)
    # where ML[j] was dRes[j]/dU[j-1] and MU[j] was dRes[j]/dU[j+1]
    # So, np.roll(J_L, -1)[k] = J_L[k+1] = dRes[k+1]/dU[k] (for offset -1)
    # And np.roll(J_U, 1)[k] = J_U[k-1] = dRes[k-1]/dU[k] (for offset 1)
    # This matches the spdiags convention if J_L and J_U are the direct band contributions.

    J_L_for_spdiags = np.roll(J_L, -1) if Nx > 1 else J_L
    J_U_for_spdiags = np.roll(J_U, 1) if Nx > 1 else J_U

    diagonals_data = [J_L_for_spdiags, J_D, J_U_for_spdiags] if Nx > 1 else [J_D]
    offsets = [-1, 0, 1] if Nx > 1 else [0]

    try:
        Jac = sparse.spdiags(diagonals_data, offsets, Nx, Nx, format="csr")
    except ValueError:
        fallback_diag = np.ones(Nx) * (1.0 / dt if abs(dt) > 1e-14 else 1.0)
        Jac = sparse.diags([fallback_diag], [0], shape=(Nx, Nx)).tocsr()

    return Jac.tocsr()


def newton_hjb_step(
    U_n_current_newton_iterate: np.ndarray,  # U_new_n_tmp in notebook
    U_n_plus_1_from_hjb_step: np.ndarray,  # U_new_np1 in notebook
    U_k_n_from_prev_picard: np.ndarray,  # U_k_n in notebook
    M_density_at_n_plus_1: np.ndarray,
    problem: MFGProblem,
    t_idx_n: int,
    backend=None,  # Add backend parameter for MPS/CUDA support
    sigma_at_n: float | np.ndarray | None = None,  # Diffusion at time t_n
    use_upwind: bool = True,  # Use Godunov upwind (True) or central (False)
    bc: BoundaryConditions | None = None,  # Boundary conditions (Issue #542 fix)
    domain_bounds: np.ndarray | None = None,  # Domain bounds for BC
    current_time: float = 0.0,  # Current time for time-dependent BCs
    bc_values: dict[str, float] | None = None,  # Issue #574
    source_term: np.ndarray | None = None,  # MMS verification
    active_hamiltonian=None,  # Issue #1157: multi-pop cross-density bound H
) -> tuple[np.ndarray, float]:
    dx = problem.geometry.get_grid_spacing()[0]
    dx_norm = dx if abs(dx) > 1e-12 else 1.0

    if has_nan_or_inf(U_n_current_newton_iterate, backend):
        return U_n_current_newton_iterate, np.inf

    residual_F_U = compute_hjb_residual(
        U_n_current_newton_iterate,
        U_n_plus_1_from_hjb_step,
        M_density_at_n_plus_1,
        problem,
        t_idx_n,
        backend,
        sigma_at_n,
        use_upwind,
        bc=bc,
        domain_bounds=domain_bounds,
        current_time=current_time,
        bc_values=bc_values,  # Issue #574
        source_term=source_term,
        active_hamiltonian=active_hamiltonian,  # Issue #1157
    )
    if has_nan_or_inf(residual_F_U, backend):
        return U_n_current_newton_iterate, np.inf

    # Jacobian uses U_k_n_from_prev_picard for its H-part if problem provides specific terms
    jacobian_J_U = compute_hjb_jacobian(
        U_n_current_newton_iterate,  # For time/diffusion deriv
        U_k_n_from_prev_picard,  # For specific H-deriv
        M_density_at_n_plus_1,
        problem,
        t_idx_n,
        backend,
        sigma_at_n,
        use_upwind,
        bc=bc,
        domain_bounds=domain_bounds,
        current_time=current_time,
        active_hamiltonian=active_hamiltonian,  # Issue #1157
    )
    if np.any(np.isnan(jacobian_J_U.data)) or np.any(np.isinf(jacobian_J_U.data)):
        return U_n_current_newton_iterate, np.inf

    delta_U = np.zeros_like(U_n_current_newton_iterate)
    l2_error_of_step = np.inf
    try:
        Nx = problem.geometry.get_grid_shape()[0]
        if not jacobian_J_U.nnz > 0 and Nx > 0:
            pass
        else:
            # Original notebook's RHS for Newton solve was effectively:
            # b = Jac * U_current_newton_iterate - Residual(U_current_newton_iterate)
            # And it solved Jac * U_next_newton_iterate = b
            # This is equivalent to Jac * delta_U = -Residual
            # where delta_U = U_next_newton_iterate - U_current_newton_iterate
            delta_U = sparse.linalg.spsolve(jacobian_J_U, -residual_F_U)

        if np.any(np.isnan(delta_U)) or np.any(np.isinf(delta_U)):
            delta_U = np.zeros_like(U_n_current_newton_iterate)
        else:
            l2_error_of_step = np.linalg.norm(delta_U) * np.sqrt(dx_norm)

    except (ValueError, RuntimeError) as e:
        # Issue #547: Replace silent fallback with logged warning
        # Sparse solver can fail due to singular matrix, shape mismatch, etc.
        logger.warning(
            "Newton iteration linear solve failed: %s. "
            "Using zero update (delta_U = 0) for this step. "
            "This may indicate numerical instability.",
            e,
        )
        # delta_U already initialized to zeros above, l2_error_of_step = inf

    max_delta_u_norm = 1e2
    current_delta_u_norm = np.linalg.norm(delta_U) * np.sqrt(dx_norm)
    if current_delta_u_norm > max_delta_u_norm and current_delta_u_norm > 1e-9:
        delta_U = delta_U * (max_delta_u_norm / current_delta_u_norm)
        l2_error_of_step = np.linalg.norm(delta_U) * np.sqrt(dx_norm)

    U_n_next_newton_iterate = U_n_current_newton_iterate + delta_U

    return U_n_next_newton_iterate, l2_error_of_step


def solve_hjb_timestep_newton(
    U_n_plus_1_from_hjb_step: np.ndarray,  # U_new[n+1] in notebook
    U_k_n_from_prev_picard: np.ndarray,  # U_k[n] in notebook
    M_density_at_n_plus_1: np.ndarray,  # M_k[n+1] in notebook
    problem: MFGProblem,
    max_newton_iterations: int | None = None,
    newton_tolerance: float | None = None,
    t_idx_n: int | None = None,  # time index for U_n being solved
    backend: BaseBackend | None = None,
    sigma_at_n: float | np.ndarray | None = None,  # Diffusion at time t_n
    use_upwind: bool = True,  # Use Godunov upwind (True) or central (False)
    bc: BoundaryConditions | None = None,  # Boundary conditions (Issue #542 fix)
    domain_bounds: np.ndarray | None = None,  # Domain bounds for BC
    current_time: float = 0.0,  # Current time for time-dependent BCs
    bc_values: dict[str, float] | None = None,  # Issue #574: Per-boundary BC values
    source_term: np.ndarray | None = None,  # MMS verification
    active_hamiltonian=None,  # Issue #1157: multi-pop cross-density bound H
) -> np.ndarray:
    """
    Solve HJB timestep using Newton's method.

    Args:
        U_n_plus_1_from_hjb_step: Solution at next time step
        U_k_n_from_prev_picard: Solution from previous Picard iteration
        M_density_at_n_plus_1: Density at next time step
        problem: MFG problem instance
        max_newton_iterations: Maximum Newton iterations (new parameter name)
        newton_tolerance: Newton convergence tolerance (new parameter name)
        t_idx_n: Time index for current solution
        bc_values: Per-boundary BC values (Issue #574)
    """
    # Set defaults if still None
    if max_newton_iterations is None:
        max_newton_iterations = DEFAULT_NEWTON_MAX_ITERATIONS
    if newton_tolerance is None:
        newton_tolerance = DEFAULT_NEWTON_TOLERANCE

    # Initial guess for Newton for U_n is U_{n+1} (from current HJB backward step)
    # Use backend-aware copy from compatibility layer
    U_n_current_newton_iterate = backend_aware_copy(U_n_plus_1_from_hjb_step, backend)

    if has_nan_or_inf(U_n_current_newton_iterate, backend):
        return U_n_current_newton_iterate

    final_l2_error = np.inf
    converged = False

    for iiter in range(max_newton_iterations):
        # Ensure t_idx_n is not None
        if t_idx_n is None:
            t_idx_n = 0  # Default time index

        U_n_next_newton_iterate, l2_error = newton_hjb_step(
            U_n_current_newton_iterate,
            U_n_plus_1_from_hjb_step,
            U_k_n_from_prev_picard,  # Pass U from prev Picard for Jacobian
            M_density_at_n_plus_1,
            problem,
            t_idx_n,
            backend,  # Pass backend for MPS/CUDA support
            sigma_at_n,  # Pass diffusion field
            use_upwind,  # Pass advection scheme flag
            bc=bc,
            domain_bounds=domain_bounds,
            current_time=current_time,
            bc_values=bc_values,  # Issue #574
            source_term=source_term,
            active_hamiltonian=active_hamiltonian,  # Issue #1157
        )

        if has_nan_or_inf(U_n_next_newton_iterate, backend):
            break

        if iiter > 0 and l2_error > final_l2_error * 0.9999 and l2_error > newton_tolerance:
            break

        U_n_current_newton_iterate = U_n_next_newton_iterate
        final_l2_error = l2_error

        if l2_error < newton_tolerance:
            converged = True
            break

    if not converged and max_newton_iterations > 0 and not (np.isnan(final_l2_error) or np.isinf(final_l2_error)):
        pass

    # Enforce BC on solution (Issue #542)
    # BC-aware Laplacian uses ghost cells for derivatives, but boundary values must be explicitly set
    if bc is not None:
        from mfgarchon.geometry.boundary.types import BCType

        # Compute grid spacing for Neumann BC enforcement
        if domain_bounds is not None:
            Nx = len(U_n_current_newton_iterate)
            # domain_bounds is shape (1, 2) for 1D: [[x_min, x_max]]
            if domain_bounds.ndim == 2:
                dx = (domain_bounds[0, 1] - domain_bounds[0, 0]) / Nx
            else:
                dx = (domain_bounds[1] - domain_bounds[0]) / Nx
        else:
            dx = 1.0  # Fallback (should not happen in practice)

        # Left boundary - Issue #638: Use _get_bc_info_1d for Robin BC support
        left_type, left_value, left_alpha, left_beta = _get_bc_info_1d(bc, "left", current_time)

        if left_type == BCType.DIRICHLET:
            # Dirichlet: Set boundary value directly
            if backend is not None:
                U_n_current_newton_iterate[0] = backend.array([left_value])[0]
            else:
                U_n_current_newton_iterate[0] = left_value
        elif left_type == BCType.NEUMANN:
            # Neumann: Set boundary value to satisfy gradient constraint
            # Forward difference: (u[1] - u[0]) / dx = g  =>  u[0] = u[1] - g*dx
            if backend is not None:
                U_n_current_newton_iterate[0] = U_n_current_newton_iterate[1] - backend.array([left_value * dx])[0]
            else:
                U_n_current_newton_iterate[0] = U_n_current_newton_iterate[1] - left_value * dx
        elif left_type == BCType.ROBIN:
            # Issue #638: Robin BC enforcement: alpha*u + beta*du/dn = g
            # At left boundary, outward normal is -x, so du/dn = (u[0] - u[1])/dx
            # alpha*u[0] + beta*(u[0] - u[1])/dx = g
            # u[0]*(alpha + beta/dx) = g + beta*u[1]/dx
            # u[0] = (g + beta*u[1]/dx) / (alpha + beta/dx)
            denom = left_alpha + left_beta / dx
            if abs(denom) > 1e-14:
                u1 = U_n_current_newton_iterate[1]
                if backend is not None:
                    u1_val = u1.item() if hasattr(u1, "item") else float(u1)
                    new_val = (left_value + left_beta * u1_val / dx) / denom
                    U_n_current_newton_iterate[0] = backend.array([new_val])[0]
                else:
                    U_n_current_newton_iterate[0] = (left_value + left_beta * u1 / dx) / denom
        elif left_type == BCType.PERIODIC:
            # Periodic: No enforcement needed - values wrap around via ghost cells
            pass
        else:
            raise ValueError(
                f"Unsupported BC type '{left_type}' at left boundary for enforcement. "
                f"Supported types: DIRICHLET, NEUMANN, ROBIN, PERIODIC."
            )

        # Right boundary - Issue #638: Use _get_bc_info_1d for Robin BC support
        right_type, right_value, right_alpha, right_beta = _get_bc_info_1d(bc, "right", current_time)

        if right_type == BCType.DIRICHLET:
            # Dirichlet: Set boundary value directly
            if backend is not None:
                U_n_current_newton_iterate[-1] = backend.array([right_value])[0]
            else:
                U_n_current_newton_iterate[-1] = right_value
        elif right_type == BCType.NEUMANN:
            # Neumann: Set boundary value to satisfy gradient constraint
            # Backward difference: (u[-1] - u[-2]) / dx = g  =>  u[-1] = u[-2] + g*dx
            if backend is not None:
                U_n_current_newton_iterate[-1] = U_n_current_newton_iterate[-2] + backend.array([right_value * dx])[0]
            else:
                U_n_current_newton_iterate[-1] = U_n_current_newton_iterate[-2] + right_value * dx
        elif right_type == BCType.ROBIN:
            # Issue #638: Robin BC enforcement: alpha*u + beta*du/dn = g
            # At right boundary, outward normal is +x, so du/dn = (u[-1] - u[-2])/dx
            # alpha*u[-1] + beta*(u[-1] - u[-2])/dx = g
            # u[-1]*(alpha + beta/dx) = g + beta*u[-2]/dx
            # u[-1] = (g + beta*u[-2]/dx) / (alpha + beta/dx)
            denom = right_alpha + right_beta / dx
            if abs(denom) > 1e-14:
                u_m2 = U_n_current_newton_iterate[-2]
                if backend is not None:
                    u_m2_val = u_m2.item() if hasattr(u_m2, "item") else float(u_m2)
                    new_val = (right_value + right_beta * u_m2_val / dx) / denom
                    U_n_current_newton_iterate[-1] = backend.array([new_val])[0]
                else:
                    U_n_current_newton_iterate[-1] = (right_value + right_beta * u_m2 / dx) / denom
        elif right_type == BCType.PERIODIC:
            # Periodic: No enforcement needed - values wrap around via ghost cells
            pass
        else:
            raise ValueError(
                f"Unsupported BC type '{right_type}' at right boundary for enforcement. "
                f"Supported types: DIRICHLET, NEUMANN, ROBIN, PERIODIC."
            )

    return U_n_current_newton_iterate


def solve_hjb_system_backward(
    M_density_from_prev_picard: np.ndarray,  # M_k in notebook
    U_final_condition_at_T: np.ndarray,
    U_from_prev_picard: np.ndarray,  # U_k in notebook
    problem: MFGProblem,
    max_newton_iterations: int | None = None,
    newton_tolerance: float | None = None,
    backend: BaseBackend | None = None,
    volatility_field: float | np.ndarray | None = None,  # Diffusion field
    use_upwind: bool = True,  # Use Godunov upwind (True) or central (False)
    bc: BoundaryConditions | None = None,  # Boundary conditions (Issue #542 fix)
    domain_bounds: np.ndarray | None = None,  # Domain bounds for BC
    bc_values: dict[str, float] | None = None,  # Issue #574: Per-boundary BC values
    source_term: Callable | None = None,  # MMS verification: S(t, x_grid) -> values
    active_hamiltonian=None,  # Issue #1157: multi-pop cross-density bound H
) -> np.ndarray:
    """
    Solve HJB system backward in time using Newton's method.

    Args:
        M_density_from_prev_picard: Density from previous Picard iteration
        U_final_condition_at_T: Terminal condition for value function
        U_from_prev_picard: Value function from previous Picard iteration
        problem: MFG problem instance
        max_newton_iterations: Maximum Newton iterations (new parameter name)
        newton_tolerance: Newton convergence tolerance (new parameter name)
        bc_values: Per-boundary Neumann BC values (Issue #574):
            {"x_min": gradient_left, "x_max": gradient_right}
            For adjoint-consistent BC. Default: None (standard BC with 0 gradient).
    """
    # Set defaults if still None
    if max_newton_iterations is None:
        max_newton_iterations = DEFAULT_NEWTON_MAX_ITERATIONS
    if newton_tolerance is None:
        newton_tolerance = DEFAULT_NEWTON_TOLERANCE

    Nt = problem.Nt + 1
    Nx = problem.geometry.get_grid_shape()[0]

    # Use backend.zeros() instead of xp.zeros() to ensure correct device
    if backend is not None:
        U_solution_this_picard_iter = backend.zeros((Nt, Nx))
    else:
        U_solution_this_picard_iter = np.zeros((Nt, Nx))  # U_new in notebook
    if Nt == 0:
        return U_solution_this_picard_iter

    # Use backend-aware nan/inf checking and assignment
    if has_nan_or_inf(U_final_condition_at_T, backend):
        backend_aware_assign(U_solution_this_picard_iter, (Nt - 1, slice(None)), float("nan"), backend)
    else:
        backend_aware_assign(U_solution_this_picard_iter, (Nt - 1, slice(None)), U_final_condition_at_T, backend)

    if Nt == 1:
        return U_solution_this_picard_iter

    for n_idx_hjb in range(Nt - 2, -1, -1):  # Solves for U_solution_this_picard_iter at t_idx_n = n_idx_hjb
        U_n_plus_1_current_picard = U_solution_this_picard_iter[n_idx_hjb + 1, :]

        # Backend-aware nan/inf checking
        if has_nan_or_inf(U_n_plus_1_current_picard, backend):
            backend_aware_assign(
                U_solution_this_picard_iter, (n_idx_hjb, slice(None)), U_n_plus_1_current_picard, backend
            )
            continue

        # BUG #7 FIX: Use M at the same time index as we're solving for U
        # Previously used M[n+1], now correctly uses M[n] to match HJB equation structure
        M_n_prev_picard = M_density_from_prev_picard[n_idx_hjb, :]  # M_k[n] - FIXED from n_idx_hjb+1
        if has_nan_or_inf(M_n_prev_picard, backend):
            backend_aware_assign(U_solution_this_picard_iter, (n_idx_hjb, slice(None)), float("nan"), backend)
            continue

        U_n_prev_picard = U_from_prev_picard[n_idx_hjb, :]  # U_k[n] from notebook
        if has_nan_or_inf(U_n_prev_picard, backend):
            backend_aware_assign(U_solution_this_picard_iter, (n_idx_hjb, slice(None)), float("nan"), backend)
            continue

        # Extract or evaluate diffusion using CoefficientField abstraction
        diffusion = CoefficientField(volatility_field, problem.sigma, "volatility_field", dimension=1)
        grid = get_spatial_grid(problem)
        sigma_at_n = diffusion.evaluate_at(timestep_idx=n_idx_hjb, grid=grid, density=M_n_prev_picard, dt=problem.dt)

        # Handle backend compatibility for NaN/Inf checking in callable results
        if diffusion.is_callable() and isinstance(sigma_at_n, np.ndarray):
            if has_nan_or_inf(sigma_at_n, backend):
                raise ValueError(f"Callable volatility_field returned NaN/Inf at timestep {n_idx_hjb}")

        # Compute current time for time-dependent BCs
        current_time = n_idx_hjb * problem.dt

        # Evaluate source term at current timestep (if provided)
        if source_term is not None:
            x_grid = problem.geometry.get_spatial_grid()  # (Nx, d) ndarray
            source_at_n = source_term(current_time, x_grid).ravel()
        else:
            source_at_n = None

        U_new_n = solve_hjb_timestep_newton(
            U_n_plus_1_current_picard,  # U_new[n+1]
            U_n_prev_picard,  # U_k[n] (for Jacobian)
            M_n_prev_picard,  # M_k[n] - FIXED: now uses correct time index
            problem,
            max_newton_iterations=max_newton_iterations,
            newton_tolerance=newton_tolerance,
            t_idx_n=n_idx_hjb,
            backend=backend,  # Pass backend for acceleration
            sigma_at_n=sigma_at_n,  # Pass diffusion at time n
            use_upwind=use_upwind,  # Pass advection scheme flag
            bc=bc,  # Pass BC for Issue #542 fix
            domain_bounds=domain_bounds,
            current_time=current_time,
            bc_values=bc_values,  # Issue #574: Per-boundary BC values
            source_term=source_at_n,
            active_hamiltonian=active_hamiltonian,  # Issue #1157
        )
        backend_aware_assign(U_solution_this_picard_iter, (n_idx_hjb, slice(None)), U_new_n, backend)

        # Check for NaN introduction during Newton step
        if has_nan_or_inf(U_solution_this_picard_iter[n_idx_hjb, :], backend) and not has_nan_or_inf(
            U_n_plus_1_current_picard, backend
        ):
            logger.warning("U_solution became NaN after Newton step for t_idx_n=%d", n_idx_hjb)

    return U_solution_this_picard_iter


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing BaseHJBSolver...")

    # Test base class and helper functions availability
    assert BaseHJBSolver is not None
    assert compute_hjb_residual is not None
    assert compute_hjb_jacobian is not None
    assert newton_hjb_step is not None
    assert solve_hjb_timestep_newton is not None
    assert solve_hjb_system_backward is not None
    print("  Base HJB solver class and helpers available")

    # Test that BaseHJBSolver is abstract
    from mfgarchon import MFGProblem
    from mfgarchon.geometry import TensorProductGrid

    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=5, sigma=0.1)

    try:
        base_solver = BaseHJBSolver(problem)
        # Should fail because solve_hjb_system is abstract
        base_solver.solve_hjb_system(None, None, None)
        raise AssertionError("Should have raised NotImplementedError")
    except (TypeError, NotImplementedError):
        print("  BaseHJBSolver correctly abstract")

    print("Smoke tests passed!")

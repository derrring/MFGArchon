#!/usr/bin/env python3
"""
Semi-Lagrangian HJB Solver for Mean Field Games

This module implements a semi-Lagrangian method for solving the Hamilton-Jacobi-Bellman
equation in MFG problems. The method follows characteristics backward in time and uses
interpolation to compute values at departure points.

The HJB equation solved is:
    -∂u/∂t + H(x, ∇u, m) - σ²/2 Δu = 0    in [0,T) × Ω
    u(T, x) = g(x)                         at t = T

equivalently ∂u/∂t = H(x, ∇u, m) - σ²/2 Δu (cost-to-go convention; u flows backward
in time from terminal data g). The semi-Lagrangian scheme discretizes this as:
    (u^n - û^{n+1}) / (-Δt) + H(x, ∇û^{n+1}, m^{n+1}) - σ²/2 Δû^{n+1} = 0
    u^n = û^{n+1} - Δt · H(x, ∇û^{n+1}, m^{n+1}) + ... (diffusion handled by chosen method)

where û^{n+1} is the value of u^{n+1} at the characteristic departure point.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from mfgarchon.alg.numerical.hjb_solvers.h_eval import eval_dH_dp_batch, eval_H_batch
from mfgarchon.geometry.boundary.applicator_fdm import FDMApplicator
from mfgarchon.geometry.boundary.applicator_interpolation import InterpolationApplicator
from mfgarchon.geometry.boundary.bc_utils import (
    bc_type_to_geometric_operation,
    checked_bc_type_string,
)
from mfgarchon.geometry.boundary.types import BCType
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.pde_coefficients import check_adi_compatibility, diffusion_from_volatility

from .base_hjb import BaseHJBSolver
from .hjb_sl_adi import (
    adi_diffusion_step,
    solve_crank_nicolson_diffusion_1d,
)
from .hjb_sl_characteristics import (
    apply_boundary_conditions_1d,
    apply_boundary_conditions_nd,
    reflect_into_domain,
    trace_characteristic_backward_1d,
    trace_characteristic_backward_nd,
)
from .hjb_sl_interpolation import (
    interpolate_nearest_neighbor,
    interpolate_value_1d,
    interpolate_value_nd,
    interpolate_value_rbf_fallback,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.boundary.conditions import BoundaryConditions

from mfgarchon.core.derivatives import DerivativeTensors

logger = get_logger(__name__)
try:
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def _checked_bc_type_string(bc) -> str:
    """Collapse ``bc`` to the single BC type the SL fold applies to every axis, or refuse.

    Thin wrapper over :func:`checked_bc_type_string`, which is the one owner of this collapse for
    every solver whose fold is per-axis blind (Issues #1560, #1697). It lives here only to bind the
    consumer name and the suggested alternative; the logic, including the ``default_bc`` union that
    a segments-only guard would miss, belongs to ``bc_utils``.
    """
    return checked_bc_type_string(
        bc,
        consumer="HJBSemiLagrangianSolver",
        alternative=(
            "Use one BC type across axes, or HJB-FDM/GFDM which resolve BC per wall (Issue #1560 / RFC #1574 Phase 0)."
        ),
    )


class HJBSemiLagrangianSolver(BaseHJBSolver):
    """
    Semi-Lagrangian method for solving Hamilton-Jacobi-Bellman equations.

    The semi-Lagrangian method discretizes the HJB equation by following
    characteristics backward in time and interpolating values at departure points.
    This approach is particularly stable for convection-dominated problems.

    Key features:
    - Stable for large time steps
    - Handles discontinuous solutions well
    - Natural upwind discretization
    - Monotone and conservative

    Dimension support:
    - 1D: Full support (production-ready)
    - nD (2D/3D/4D+): Full support (2025-11-02)
      - Interpolation: RegularGridInterpolator (complete)
      - Diffusion: nD Laplacian (complete)
      - Characteristic tracing: Vector form (complete)
      - Optimal control: Vector optimization (complete)

    Required Geometry Traits (Issue #596 Phase 2.1):
        - SupportsGradient: Provides ∇U operator for optimal control computation

    Compatible Geometries:
        - TensorProductGrid (structured grids)
        - ImplicitDomain (SDF-based domains)
        - Any geometry implementing SupportsGradient trait
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.SL

    # BoundaryCapable protocol (Issue #1456): the SL diffusion sub-step (CN/ADI) is zero-flux
    # (no-flux / Neumann g=0) and the characteristic foot wraps for periodic; Dirichlet / Robin /
    # absorbing are silently collapsed to Neumann on the default path, so they fail loud here.
    _SUPPORTED_BC_TYPES: frozenset = frozenset({BCType.NO_FLUX, BCType.NEUMANN, BCType.PERIODIC})

    @property
    def supported_bc_types(self) -> frozenset:
        """BC types this solver supports (BoundaryCapable protocol)."""
        return self._SUPPORTED_BC_TYPES

    def __init__(
        self,
        problem: MFGProblem,
        interpolation_method: str = "linear",
        optimization_method: str = "brent",
        characteristic_solver: str = "explicit_euler",
        diffusion_method: str = "adi",
        use_rbf_fallback: bool = True,
        rbf_kernel: str = "thin_plate_spline",
        use_jax: bool | None = None,
        tolerance: float = 1e-8,
        max_char_iterations: int = 100,
        check_cfl: bool = True,
        enable_adaptive_substepping: bool = True,
        max_substeps: int = 100,
        cfl_target: float = 0.9,
        gradient_clip_threshold: float | None = None,
        enable_gradient_monitoring: bool = True,
        ode_rtol: float = 1e-6,
        ode_atol: float = 1e-8,
    ):
        """
        Initialize semi-Lagrangian HJB solver.

        Args:
            problem: MFG problem instance
            interpolation_method: Method for interpolating values
                - 'linear': Linear interpolation (fastest, C⁰ continuous)
                - 'cubic': Cubic spline interpolation (slower, C² continuous)
                - 'quintic': Quintic interpolation (slowest, highest accuracy, nD only)
                - 'nearest': Nearest neighbor (for debugging)
            optimization_method: Method for Hamiltonian optimization ('brent', 'golden')
            characteristic_solver: Method for solving characteristics
                - 'explicit_euler': First-order explicit Euler (fastest, least accurate)
                - 'rk2': Second-order Runge-Kutta midpoint method
                - 'rk4': Fourth-order Runge-Kutta via scipy.solve_ivp (most accurate)
            diffusion_method: Method for handling diffusion term (default: 'adi')
                - 'adi': ADI (Alternating Direction Implicit) splitting (default)
                - 'explicit': Explicit Laplacian (simple, requires small dt)
                - 'stochastic': Stochastic characteristic with Brownian motion (high-dim friendly).
                  Explicit-alpha* (alpha* = -grad u^{n+1} at the grid node).
                - 'canonical_cs': Canonical Carlini-Silva 2014 SL with the IMPLICIT-alpha* DPP
                  fixed point (Issue #1058). Per grid point, alpha* is the minimizer of the DPP
                  objective (NOT the at-grid gradient), so it is consistent with the departure
                  point it induces -- the hypothesis under which CS 2014 prove unconditional
                  stability for monotone (Q1/linear) interpolation. Requires
                  interpolation_method='linear'. Per-point optimization (slower) in exchange for
                  unconditional stability; targets reflecting/no-flux boundaries.
                - 'none': No diffusion (for testing or zero-diffusion problems)
            use_rbf_fallback: Use RBF interpolation as fallback for boundary cases
            rbf_kernel: RBF kernel function
                - 'thin_plate_spline': Smooth, no free parameters (recommended)
                - 'multiquadric': Good for scattered data
                - 'gaussian': Localized influence
            use_jax: Whether to use JAX acceleration (auto-detect if None)
            tolerance: Convergence tolerance for optimization
            max_char_iterations: Maximum iterations for characteristic solving
            check_cfl: Whether to check CFL condition and issue warnings (default: True).
                CFL = max|grad(u)| * dt / dx. Warns if CFL > 1.0.
            enable_adaptive_substepping: Whether to automatically subdivide time steps
                when CFL > 1.0 to maintain stability (default: True). When enabled,
                the solver will use smaller internal time steps while preserving the
                overall time discretization.
            max_substeps: Maximum number of substeps per time step when adaptive
                substepping is enabled (default: 100). If more substeps are needed,
                a warning is issued and the solver proceeds with max_substeps.
            cfl_target: Target CFL number for adaptive substepping (default: 0.9).
                When CFL > 1.0, the time step is subdivided to achieve CFL ≤ cfl_target.
            gradient_clip_threshold: Safety threshold for gradient clipping (default: None).
                If provided, gradients exceeding this threshold will be clipped to prevent
                overflow in p² terms. Recommended: 1e6 for strong coupling problems.
                When None, no clipping is performed.
            enable_gradient_monitoring: Enable detailed gradient statistics tracking (default: True).
                Records when and where gradient clipping occurs for debugging. Disable for
                performance if clipping monitoring is not needed.
            ode_rtol: Relative tolerance for scipy.solve_ivp when characteristic_solver='rk4'.
                Default 1e-6.
            ode_atol: Absolute tolerance for scipy.solve_ivp when characteristic_solver='rk4'.
                Default 1e-8.
        """
        super().__init__(problem)
        self.hjb_method_name = "Semi-Lagrangian"

        # Solver configuration
        self.interpolation_method = interpolation_method
        self.optimization_method = optimization_method
        self.characteristic_solver = characteristic_solver
        self.diffusion_method = diffusion_method
        self.use_rbf_fallback = use_rbf_fallback
        self.rbf_kernel = rbf_kernel
        self.tolerance = tolerance
        self.max_char_iterations = max_char_iterations
        self.check_cfl = check_cfl
        self.enable_adaptive_substepping = enable_adaptive_substepping
        self.max_substeps = max_substeps
        self.cfl_target = cfl_target
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

        # Gradient clipping configuration (Issue #583)
        self.gradient_clip_threshold = gradient_clip_threshold
        self.enable_gradient_monitoring = enable_gradient_monitoring

        # Issue #1049: Carlini-Silva 2014 prove unconditional stability of the
        # deterministic 2-direction averaging SL scheme (here: diffusion_method=
        # "stochastic") **specifically for Q1 (linear, monotone) interpolation**.
        # Cubic interpolation is not covered by the CS 2014 proof and is non-monotone
        # (Issue #1033 documents the exponential blow-up on Towel-on-Beach).
        # The previous validation actively rejected the proof-applicable combination.
        # Now: warn (don't reject) when cubic+stochastic is selected, since that
        # combination violates the monotone-scheme requirement of CS 2014.
        if self.diffusion_method == "stochastic" and self.interpolation_method in ("cubic", "quintic"):
            import warnings

            warnings.warn(
                f"diffusion_method='stochastic' with interpolation_method='{self.interpolation_method}' "
                "is NOT covered by the Carlini-Silva 2014 stability proof, which "
                "requires monotone (Q1/linear) interpolation. Cubic/quintic can "
                "violate the monotone-scheme requirement of Barles-Souganidis and "
                "produce exponential blow-up on stiff problems (see Issue #1033). "
                "Recommended: interpolation_method='linear'. mfgarchon's cubic "
                "path now uses `PchipInterpolator` (monotonic Hermite) which is "
                "more stable than `CubicSpline` but still outside the formal proof.",
                UserWarning,
                stacklevel=2,
            )

        # Issue #1058: canonical Carlini-Silva SL with implicit-alpha* DPP fixed point.
        # CS 2014's stability proof requires monotone (Q1/linear) interpolation; cubic/quintic
        # are non-monotone and outside the proof (Issue #1033/#1049). Fail fast rather than
        # silently running an unproven combination.
        if self.diffusion_method == "canonical_cs" and self.interpolation_method != "linear":
            raise ValueError(
                f"diffusion_method='canonical_cs' requires interpolation_method='linear' "
                f"(monotone Q1), got '{self.interpolation_method}'. The Carlini-Silva 2014 "
                f"stability proof only covers monotone interpolation; cubic/quintic break it."
            )

        # Issue #1547 / RFC #1574 Phase 0: the characteristic-foot velocity dH/dp = p/lambda traces
        # departures x - (grad_u/lambda)*dt, i.e. alpha* = -grad_u/lambda (MINIMIZE). A MAXIMIZE
        # control cost has alpha* = +grad_u/lambda, so the feet would be traced in the wrong
        # direction; the MAXIMIZE-quadratic H is smooth so the non-smooth DPP reroute never fires and
        # the wrong-signed foot path is taken silently. Fail loud (mirrors the HJBGFDMSolver Howard
        # gate) rather than run the wrong scheme. MAXIMIZE support on the SL path is deferred.
        _sl_h_class = getattr(self.problem, "hamiltonian_class", None)
        _sl_control_cost = getattr(_sl_h_class, "control_cost", None)
        if _sl_control_cost is not None and getattr(_sl_control_cost, "sign", 1) != 1:
            raise NotImplementedError(
                "HJBSemiLagrangianSolver traces characteristic feet with the MINIMIZE-signed velocity "
                "alpha* = -grad(u)/lambda, but the Hamiltonian's control cost is MAXIMIZE "
                "(alpha* = +grad(u)/lambda). The feet would move in the wrong direction and the solve "
                "would converge to a different equilibrium. MAXIMIZE is not yet supported on the "
                "semi-Lagrangian path (Issue #1547 / RFC #1574); use HJB-FDM/GFDM, or MINIMIZE."
            )

        # Gradient clipping statistics tracking
        self._reset_gradient_stats()

        # JAX acceleration
        self.use_jax = use_jax if use_jax is not None else JAX_AVAILABLE
        if self.use_jax and not JAX_AVAILABLE:
            logger.warning("JAX not available, falling back to NumPy")
            self.use_jax = False

        # Detect problem dimension (inherited from BaseNumericalSolver, Issue #633)
        self.dimension = self._detect_dimension()

        # Create boundary condition applicators
        # FDMApplicator: for ghost cell operations (gradient computation)
        self.bc_applicator = FDMApplicator(dimension=self.dimension)
        # InterpolationApplicator: for post-interpolation BC enforcement (Issue #636)
        self.interp_bc_applicator = InterpolationApplicator(dimension=self.dimension)

        # Validate geometry capabilities (Issue #596 Phase 2.1)
        # Semi-Lagrangian solver requires gradient operator for optimal control computation
        from mfgarchon.geometry.protocols import SupportsGradient

        if not isinstance(problem.geometry, SupportsGradient):
            raise TypeError(
                f"HJB Semi-Lagrangian solver requires geometry with SupportsGradient trait for ∇U computation. "
                f"{type(problem.geometry).__name__} does not implement this trait. "
                f"Compatible geometries: TensorProductGrid, ImplicitDomain."
            )

        # Precompute grid and time parameters (dimension-agnostic)
        if self.dimension == 1:
            # 1D problem: Use geometry API
            bounds = problem.geometry.get_bounds()
            xmin, xmax = bounds[0][0], bounds[1][0]
            Nx = problem.geometry.get_grid_shape()[0]
            self.x_grid = np.linspace(xmin, xmax, Nx)
            self.dt = problem.dt
            self.dx = problem.geometry.get_grid_spacing()[0]
            self.grid = None  # 1D uses direct arrays, not grid object
        else:
            # nD problem: Use CartesianGrid interface
            from mfgarchon.geometry.base import CartesianGrid

            if not isinstance(problem.geometry, CartesianGrid):
                raise ValueError(
                    f"Multi-dimensional problem requires CartesianGrid geometry. "
                    f"Got {type(problem.geometry).__name__} (dimension={self.dimension})"
                )
            self.grid = problem.geometry  # Geometry IS the grid
            self.dt = problem.dt
            # Grid spacing: vector of spacings in each dimension
            self.dx = np.array(self.grid.get_grid_spacing())
            # Grid shape: use get_grid_shape() for CartesianGrid interface compatibility
            self._grid_shape = tuple(self.grid.get_grid_shape())
            self._num_points_total = int(np.prod(self._grid_shape))
            self.x_grid = None  # Not used for nD

            # Check ADI compatibility for nD diffusion
            adi_ok, adi_msg = check_adi_compatibility(problem.sigma)
            self._adi_compatible = adi_ok
            if not adi_ok:
                logger.warning(
                    f"Diffusion tensor not ADI-compatible: {adi_msg}. "
                    f"ADI scheme may be inaccurate. Consider using more timesteps "
                    f"or implementing Craig-Sneyd scheme for mixed derivatives."
                )
            else:
                logger.info(f"ADI diffusion enabled for nD solve: {adi_msg}")

        # Setup JAX functions if available
        if self.use_jax:
            self._setup_jax_functions()

        # Issue #1456: fail loud now if the (geometry/problem) BC requests a type SL cannot honor
        # (Dirichlet/Robin/absorbing — otherwise silently collapsed to the zero-flux Neumann
        # diffusion). None (BC resolved later) is a no-op; SL re-reads get_boundary_conditions().
        self._validate_bc_support(self.get_boundary_conditions())

        # Issue #1560 / RFC #1574 Phase 0: even when every segment type is individually supported, the
        # SL characteristic fold and ADI diffusion collapse a MIXED per-axis BC to segments[0]'s single
        # geometric operation (reflect vs wrap) applied to ALL axes (get_bc_type_string returns only the
        # first segment) — so e.g. no-flux walls on one axis + periodic on another is silently reduced
        # to one op, and reordering the segments flips the physics. Per-axis handling is a follow-up;
        # for now fail loud when the segments do not agree on a single geometric operation.
        _sl_bc = self.get_boundary_conditions()
        _sl_segments = getattr(_sl_bc, "segments", None)
        if _sl_segments:
            _sl_ops = set()
            for _seg in _sl_segments:
                _seg_type = str(getattr(_seg.bc_type, "value", _seg.bc_type))
                _sl_ops.add(bc_type_to_geometric_operation(_seg_type))
            _sl_default = getattr(_sl_bc, "default_bc", None)
            if _sl_default is not None:
                _d = str(getattr(_sl_default, "value", _sl_default))
                _sl_ops.add(bc_type_to_geometric_operation(_d))
            if len(_sl_ops) > 1:
                raise NotImplementedError(
                    f"HJBSemiLagrangianSolver does not support a mixed per-axis boundary condition whose "
                    f"segments map to different geometric operations ({sorted(_sl_ops)}). The characteristic "
                    f"fold and ADI diffusion apply the FIRST segment's single operation to every axis "
                    f"(order-sensitive silent collapse). Use a single BC type across axes, or HJB-FDM/GFDM "
                    f"which resolve BC per wall (Issue #1560 / RFC #1574 Phase 0)."
                )

    # _detect_dimension() inherited from BaseNumericalSolver (Issue #633)

    def _setup_jax_functions(self):
        """Setup JAX-accelerated functions for performance."""
        if not self.use_jax:
            return

        @jit
        def jax_interpolate_linear(x_points, y_values, x_query):
            """JAX-accelerated linear interpolation."""
            return jnp.interp(x_query, x_points, y_values)

        @jit
        def jax_solve_characteristic_euler(x_current, p_optimal, dt):
            """JAX-accelerated characteristic solving using Euler method."""
            return x_current - p_optimal * dt

        self._jax_interpolate = jax_interpolate_linear
        self._jax_solve_characteristic = jax_solve_characteristic_euler

    def _reset_gradient_stats(self):
        """Reset gradient clipping statistics for new solve (Issue #583)."""
        self.gradient_stats = {
            "count": 0,  # Total number of clipped spatial points
            "max_gradient": 0.0,  # Maximum gradient magnitude encountered
            "locations": [],  # List of {t_idx, spatial_idx, gradient_value, density_value}
            "by_timestep": {},  # {t_idx: count} - clipping events per timestep
        }

    def _log_gradient_clipping_summary(self):
        """Log detailed summary of gradient clipping events (Issue #583)."""
        from mfgarchon.utils.mfg_logging import get_logger

        logger_local = get_logger(__name__)

        if self.gradient_stats["count"] == 0:
            if self.gradient_clip_threshold is not None:
                logger_local.info(
                    f"No gradient clipping required - all gradients remained below threshold "
                    f"({self.gradient_clip_threshold:.2e}). Max gradient: {self.gradient_stats['max_gradient']:.2e}"
                )
            return

        # Gradient clipping occurred
        logger_local.warning("=" * 60)
        logger_local.warning("GRADIENT CLIPPING SUMMARY (Issue #583)")
        logger_local.warning("=" * 60)
        logger_local.warning(f"Total clipped points: {self.gradient_stats['count']}")
        logger_local.warning(f"Max gradient encountered: {self.gradient_stats['max_gradient']:.2e}")
        logger_local.warning(f"Clip threshold: {self.gradient_clip_threshold:.2e}")

        # Temporal distribution
        if self.gradient_stats["by_timestep"]:
            logger_local.warning("\nClipping by timestep (first 10):")
            sorted_timesteps = sorted(self.gradient_stats["by_timestep"].keys())[:10]
            for t_idx in sorted_timesteps:
                count = self.gradient_stats["by_timestep"][t_idx]
                logger_local.warning(f"  t={t_idx}: {count} points clipped")

            if len(self.gradient_stats["by_timestep"]) > 10:
                logger_local.warning(f"  ... and {len(self.gradient_stats['by_timestep']) - 10} more timesteps")

        # Spatial hotspots (if tracked)
        if self.gradient_stats["locations"] and self.enable_gradient_monitoring:
            logger_local.warning("\nFirst few clipping locations:")
            for loc in self.gradient_stats["locations"][:5]:
                density_str = f"{loc['density_value']:.2e}" if loc["density_value"] is not None else "N/A"
                logger_local.warning(
                    f"  t={loc['t_idx']}, x_idx={loc['spatial_idx']}, "
                    f"||∇U||={loc['gradient_value']:.2e}, "
                    f"m={density_str}"
                )

            if len(self.gradient_stats["locations"]) > 5:
                logger_local.warning(f"  ... and {len(self.gradient_stats['locations']) - 5} more locations")

        logger_local.warning("=" * 60)
        logger_local.warning(
            "RECOMMENDATION: Gradient clipping is a SAFETY NET, not a solution. "
            "Enable adaptive Picard damping (adaptive_damping=True in FixedPointIterator) "
            "or use weaker coupling to prevent gradient amplification at the source."
        )

    def _clip_gradient_with_monitoring(
        self,
        grad_u: np.ndarray | tuple[np.ndarray, ...],
        t_idx: int | None = None,
        m_density: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Clip gradients and track where clipping occurs (Issue #583).

        Args:
            grad_u: Gradient array(s) from _compute_gradient
            t_idx: Current timestep index for location tracking (optional)
            m_density: Density values for correlation analysis (optional)

        Returns:
            Clipped gradient with same structure as input
        """
        if self.gradient_clip_threshold is None:
            return grad_u  # No clipping

        # Import logging tools
        from mfgarchon.utils.mfg_logging import get_logger

        logger_local = get_logger(__name__)

        if self.dimension == 1:
            # 1D gradient clipping
            grad_norm = np.abs(grad_u)
            grad_max = np.max(grad_norm)

            # Update max gradient stat
            self.gradient_stats["max_gradient"] = max(self.gradient_stats["max_gradient"], float(grad_max))

            # Identify where clipping is needed
            clip_mask = grad_norm > self.gradient_clip_threshold

            if np.any(clip_mask):
                clip_indices = np.where(clip_mask)[0]
                n_clipped = len(clip_indices)
                self.gradient_stats["count"] += n_clipped

                # Track by timestep
                if t_idx is not None and self.enable_gradient_monitoring:
                    if t_idx not in self.gradient_stats["by_timestep"]:
                        self.gradient_stats["by_timestep"][t_idx] = 0
                    self.gradient_stats["by_timestep"][t_idx] += n_clipped

                    # Store first few locations (avoid memory explosion)
                    if len(self.gradient_stats["locations"]) < 100:
                        for idx in clip_indices[:10]:  # First 10 per timestep
                            self.gradient_stats["locations"].append(
                                {
                                    "t_idx": int(t_idx),
                                    "spatial_idx": int(idx),
                                    "gradient_value": float(grad_norm[idx]),
                                    "density_value": float(m_density[idx]) if m_density is not None else None,
                                }
                            )

                    # Log clipping event with location info
                    x_values = self.x_grid[clip_indices]
                    logger_local.warning(
                        f"Gradient clipped at t={t_idx}: {n_clipped} points, "
                        f"max ||∇U||={grad_max:.2e}, "
                        f"locations: x={x_values[:5].tolist()}"  # First 5 x-coordinates
                    )

                # Perform clipping
                grad_u_clipped = np.clip(grad_u, -self.gradient_clip_threshold, self.gradient_clip_threshold)
                return grad_u_clipped

            return grad_u

        else:
            # nD gradient clipping
            grad_components = list(grad_u)  # Convert tuple to list for modification
            grad = np.stack(grad_components, axis=0)
            grad_norm = np.sqrt(np.sum(grad**2, axis=0))
            grad_max = np.max(grad_norm)

            # Update max gradient stat
            self.gradient_stats["max_gradient"] = max(self.gradient_stats["max_gradient"], float(grad_max))

            # Identify where clipping is needed
            clip_mask = grad_norm > self.gradient_clip_threshold

            if np.any(clip_mask):
                clip_indices = np.argwhere(clip_mask)  # Returns (N, d) array
                n_clipped = len(clip_indices)
                self.gradient_stats["count"] += n_clipped

                # Track by timestep
                if t_idx is not None and self.enable_gradient_monitoring:
                    if t_idx not in self.gradient_stats["by_timestep"]:
                        self.gradient_stats["by_timestep"][t_idx] = 0
                    self.gradient_stats["by_timestep"][t_idx] += n_clipped

                    # Store first few locations
                    if len(self.gradient_stats["locations"]) < 100:
                        for idx_tuple in clip_indices[:10]:
                            idx_tuple_int = tuple(int(i) for i in idx_tuple)
                            self.gradient_stats["locations"].append(
                                {
                                    "t_idx": int(t_idx),
                                    "spatial_idx": idx_tuple_int,
                                    "gradient_value": float(grad_norm[idx_tuple_int]),
                                    "density_value": float(m_density[idx_tuple_int]) if m_density is not None else None,
                                }
                            )

                    # Log clipping event
                    logger_local.warning(
                        f"Gradient clipped at t={t_idx}: {n_clipped} points, max ||∇U||={grad_max:.2e}"
                    )

                # Perform clipping component-wise
                for d in range(self.dimension):
                    grad_components[d] = np.where(
                        clip_mask,
                        grad_components[d] * self.gradient_clip_threshold / (grad_norm + 1e-16),
                        grad_components[d],
                    )

                return tuple(grad_components)

            return grad_u

    def _compute_gradient(
        self,
        u_values: np.ndarray,
        check_cfl: bool = True,
        t_idx: int | None = None,
        m_density: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Compute gradient ∇u for optimal control using trait-based geometry operators (Issue #596 Phase 2.1).

        For standard MFG with quadratic control cost, the optimal control is:
            α*(x,t) = ∇u(x,t)

        Uses geometry.get_gradient_operator() which automatically handles:
        - Boundary conditions via ghost cells
        - Scheme selection (central differences for Semi-Lagrangian)
        - Multi-dimensional stencils

        Args:
            u_values: Value function array
                - 1D: shape (Nx+1,)
                - nD: shape (Nx1+1, Nx2+1, ..., Nxd+1)
            check_cfl: Whether to check CFL condition (default: True)
            t_idx: Current timestep index for gradient clipping monitoring (optional, Issue #583)
            m_density: Density values for gradient clipping correlation analysis (optional, Issue #583)

        Returns:
            gradient: Gradient array(s), optionally clipped if gradient_clip_threshold is set
                - 1D: shape (Nx+1,) - scalar gradient at each point
                - nD: tuple of d arrays, each shape (Nx1+1, ..., Nxd+1)

        Note:
            Uses central differences for characteristic tracing (Semi-Lagrangian scheme).
            Boundary conditions are automatically enforced by gradient operators.
            Issues CFL warning if max|∇u|·dt/dx > 1.
            Gradient clipping (Issue #583): Clips gradients > gradient_clip_threshold to prevent overflow.
        """
        # Get gradient operators from geometry (Issue #596 Phase 2.1)
        # Semi-Lagrangian uses central differences for gradient computation
        grad_ops = self.problem.geometry.get_gradient_operator(scheme="central")

        if self.dimension == 1:
            # 1D gradient computation via operator
            grad_u = grad_ops[0](u_values)

            # Apply gradient clipping (Issue #583)
            grad_u_clipped = self._clip_gradient_with_monitoring(grad_u, t_idx=t_idx, m_density=m_density)

            # CFL check (after clipping to get realistic CFL with clipped gradients)
            if check_cfl and self.check_cfl:
                max_grad = np.max(np.abs(grad_u_clipped))
                cfl = max_grad * self.dt / self.dx
                if cfl > 1.0:
                    logger.warning(
                        f"CFL condition violated: max|∇u|·dt/dx = {cfl:.3f} > 1.0. "
                        f"Consider reducing dt or increasing dx. "
                        f"max|∇u| = {max_grad:.3f}, dt = {self.dt:.6f}, dx = {self.dx:.6f}"
                    )

            return grad_u_clipped

        else:
            # nD gradient computation via operators
            grad_components = []
            for d in range(self.dimension):
                grad_axis = grad_ops[d](u_values)
                grad_components.append(grad_axis)

            # Apply gradient clipping (Issue #583)
            grad_components_clipped = self._clip_gradient_with_monitoring(
                tuple(grad_components), t_idx=t_idx, m_density=m_density
            )

            # CFL check (after clipping)
            if check_cfl and self.check_cfl:
                grad = np.stack(grad_components_clipped, axis=0)
                magnitude = np.sqrt(np.sum(grad**2, axis=0))
                max_grad = np.max(magnitude)
                min_spacing = np.min(self.dx)
                cfl = max_grad * self.dt / min_spacing
                if cfl > 1.0:
                    logger.warning(
                        f"CFL condition violated: max|∇u|·dt/dx_min = {cfl:.3f} > 1.0. "
                        f"Consider reducing dt or increasing grid spacing. "
                        f"max|∇u| = {max_grad:.3f}, dt = {self.dt:.6f}, dx_min = {min_spacing:.6f}"
                    )

            # Return as tuple of arrays (one per dimension)
            return grad_components_clipped

    def _compute_cfl_and_substeps(self, u_values: np.ndarray, dt_target: float) -> tuple[float, int, float]:
        """
        Compute CFL number and determine optimal number of substeps.

        When the CFL condition (CFL = max|grad(u)| * dt / dx) exceeds 1.0,
        this method computes how many substeps are needed to maintain
        CFL <= cfl_target (default 0.9).

        Uses trait-based gradient operators for consistent computation (Issue #596 Phase 2.1).

        Args:
            u_values: Current value function array
            dt_target: Target time step (full time step to subdivide)

        Returns:
            Tuple of (cfl_number, n_substeps, dt_substep):
                - cfl_number: The CFL number with the target dt
                - n_substeps: Number of substeps needed (1 if CFL <= 1.0)
                - dt_substep: Time step to use for each substep
        """
        # Compute gradient using trait-based operators (reuse _compute_gradient with CFL check disabled)
        grad_result = self._compute_gradient(u_values, check_cfl=False)

        if self.dimension == 1:
            # 1D CFL computation
            grad_u = grad_result
            max_grad = np.max(np.abs(grad_u))
            cfl = max_grad * dt_target / self.dx
            dx_eff = self.dx
        else:
            # nD CFL computation
            grad_components = grad_result  # Tuple of gradient arrays
            grad = np.stack(grad_components, axis=0)
            magnitude = np.sqrt(np.sum(grad**2, axis=0))
            max_grad = np.max(magnitude)
            dx_eff = np.min(self.dx)
            cfl = max_grad * dt_target / dx_eff

        # Determine substeps needed
        if cfl <= 1.0 or not self.enable_adaptive_substepping:
            return cfl, 1, dt_target

        # Compute substeps to achieve CFL <= cfl_target
        n_substeps = int(np.ceil(cfl / self.cfl_target))
        n_substeps = min(n_substeps, self.max_substeps)

        if n_substeps >= self.max_substeps:
            logger.warning(
                f"CFL = {cfl:.2f} requires {int(np.ceil(cfl / self.cfl_target))} substeps, "
                f"capped at max_substeps={self.max_substeps}. "
                f"Stability may be compromised. Consider reducing dt or increasing grid resolution."
            )

        dt_substep = dt_target / n_substeps
        actual_cfl = max_grad * dt_substep / dx_eff

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Adaptive substepping: CFL={cfl:.2f} -> {actual_cfl:.2f} ({n_substeps} substeps, dt={dt_substep:.6f})"
            )

        return cfl, n_substeps, dt_substep

    # _get_boundary_conditions() removed (Issue #634): was duplicating
    # BaseMFGSolver.get_boundary_conditions() from base_solver.py:175-234.
    # All callers now use the inherited get_boundary_conditions().

    def _get_bc_type_string(self, bc) -> str | None:
        """
        Extract BC type string from BoundaryConditions object.

        Args:
            bc: BoundaryConditions object or None

        Returns:
            BC type string ("periodic", "dirichlet", "neumann") or None

        Note:
            Issue #545: Replace hasattr pattern for BCType enum value extraction.
            Used in characteristic tracing and diffusion term computation.
        """
        if bc is None:
            return None

        # Try to get default_bc attribute
        try:
            bc_type_enum = bc.default_bc
            if bc_type_enum is None:
                return None

            # Try to get .value attribute (BCType enum)
            try:
                return bc_type_enum.value
            except AttributeError:
                # Fall back to string conversion
                return str(bc_type_enum)
        except AttributeError:
            return None

    def _get_per_boundary_bc_types(self, bc) -> tuple[str | None, str | None]:
        """
        Get BC type strings for each boundary (1D: xmin and xmax).

        For mixed BCs (e.g., Neumann at x=0, Dirichlet at x=L), this method
        queries the BC type at each boundary separately.

        Args:
            bc: BoundaryConditions object or None

        Returns:
            Tuple of (bc_type_at_xmin, bc_type_at_xmax)

        Note:
            For uniform BCs, both values will be the same.
            For mixed BCs, values may differ per boundary.
        """
        if bc is None:
            return (None, None)

        # Try to use get_bc_type_at_boundary method for per-boundary queries
        try:
            bc_type_min_enum = bc.get_bc_type_at_boundary("x_min")
            bc_type_max_enum = bc.get_bc_type_at_boundary("x_max")

            # Extract string values from BCType enums
            bc_type_min = bc_type_min_enum.value if bc_type_min_enum is not None else None
            bc_type_max = bc_type_max_enum.value if bc_type_max_enum is not None else None

            return (bc_type_min, bc_type_max)
        except AttributeError:
            pass

        # Fallback: use uniform BC type for both boundaries
        bc_type = self._get_bc_type_string(bc)
        return (bc_type, bc_type)

    def _enforce_boundary_conditions(self, U: np.ndarray, time: float = 0.0) -> np.ndarray:
        """
        Enforce boundary conditions on solution array (dimension-agnostic).

        For Semi-Lagrangian, BC enforcement after each timestep ensures:
        - **Neumann** (du/dn=0): 2nd-order extrapolation preserving zero gradient
        - **Dirichlet** (u=g): u[boundary] = g (prescribed value)

        This explicit enforcement is critical because Semi-Lagrangian's
        interpolation-based approach doesn't naturally preserve BCs.

        Uses InterpolationApplicator (Issue #636) for unified BC handling
        across all dimensions.

        Args:
            U: Solution array of shape (Nx,) for 1D, (Ny, Nx) for 2D, etc.
            time: Current time for time-dependent BC values

        Returns:
            Solution with BCs enforced (modified in-place)
        """
        bc = self.get_boundary_conditions()
        if bc is None:
            return U

        # Use InterpolationApplicator for dimension-agnostic BC enforcement
        return self.interp_bc_applicator.enforce_values(U, bc, time=time)

    def solve_hjb_system(
        self,
        M_density: np.ndarray | None = None,
        U_terminal: np.ndarray | None = None,
        U_coupling_prev: np.ndarray | None = None,
        volatility_field: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve the HJB system using semi-Lagrangian method.

        The semi-Lagrangian discretization of the HJB equation:
            ∂u/∂t + H(x, ∇u, m) - σ²/2 Δu = 0

        is solved by following characteristics backward in time:
            1. For each grid point x_i at time t^{n+1}
            2. Find optimal control p* that minimizes H(x_i, p, m^{n+1})
            3. Trace characteristic backward: X(t^n) = x_i - p* Δt
            4. Interpolate u^n at departure point X(t^n)
            5. Update: u^{n+1}_i = û^n(X(t^n)) - Δt[H(...) - σ²/2 Δu]

        Args:
            M_density: (Nt, *spatial_shape) density from FP solver
            U_terminal: (*spatial_shape,) terminal condition u(T, x)
            U_coupling_prev: (Nt, *spatial_shape) previous coupling iteration estimate
            volatility_field: Optional diffusion coefficient override

        Returns:
            (Nt, *grid_shape) solution array for value function
        """
        # Issue #1316: the semi-Lagrangian solver reads diffusion from problem.sigma at
        # multiple scattered sites (the advection-diffusion split, ADI, Crank-Nicolson),
        # with no single sigma chokepoint to redirect. Honoring a volatility_field that
        # differs from problem.sigma would require threading it through all of them; doing
        # nothing would silently solve HJB with problem.sigma while FP uses the field,
        # breaking the Picard correspondence. Fail loud instead of accept-and-ignore. A
        # scalar field equal to problem.sigma is the iterator's redundant forwarding of
        # problem.volatility_field (Issue #1248) and is accepted as a no-op.
        if volatility_field is not None and not (
            np.isscalar(volatility_field) and float(volatility_field) == float(self.problem.sigma)
        ):
            raise NotImplementedError(
                "HJBSemiLagrangianSolver cannot honor a volatility_field that differs from "
                "problem.sigma: it reads diffusion from problem.sigma at multiple sites with no "
                "single chokepoint (Issue #1316). A spatially-varying or mismatched field would "
                "make HJB solve a different diffusion than FP, breaking the Picard fixed point. "
                "Use HJBGFDMSolver (which consumes volatility_field) or set problem.sigma to match."
            )

        # Validate required parameters
        if M_density is None:
            raise ValueError("M_density is required")
        if U_terminal is None:
            raise ValueError("U_terminal is required")
        if U_coupling_prev is None:
            raise ValueError("U_coupling_prev is required")

        # Issue #1071 / fail-fast: a missing Hamiltonian must fail loud HERE, before the
        # timestep loop — otherwise the batch path silently zeros H (pure transport of the
        # terminal data). (The per-point loops used to swallow the resulting per-point raise
        # as well; since #1635 they propagate it, but the batch path still needs this gate.)
        # MFGProblem construction requires a Hamiltonian, so this only fires on a duck-typed
        # or externally-nulled problem; it must not silently solve the wrong physics.
        if self.problem.hamiltonian_class is None:
            raise ValueError(
                "HJBSemiLagrangianSolver: problem.hamiltonian_class is None. Specify a Hamiltonian "
                "explicitly, e.g. MFGComponents(hamiltonian=SeparableHamiltonian(...)). The solver "
                "will not silently substitute the LQ default H=0.5*|p|^2 (Issue #1071, fail-fast)."
            )

        # Reset gradient clipping statistics for this solve (Issue #583)
        self._reset_gradient_stats()

        # Handle multi-dimensional grids
        # M_density has shape (Nt_points, *spatial_shape) where Nt_points = Nt + 1
        shape = M_density.shape
        Nt_points = shape[0]  # Number of time points (includes t=0 and t=T)
        grid_shape = shape[1:]  # Remaining dimensions

        # Output shape: (Nt_points, *grid_shape) - same as input
        U_solution = np.zeros((Nt_points, *grid_shape))

        # Set final condition at t=T (last index)
        U_solution[-1] = U_terminal

        total_points = np.prod(grid_shape)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"Starting semi-Lagrangian HJB solve: {Nt_points} time points, {total_points} spatial points ({grid_shape})"
            )
            if self.gradient_clip_threshold is not None:
                logger.info(f"Gradient clipping enabled: threshold = {self.gradient_clip_threshold:.2e}")

        # Solve backward in time using semi-Lagrangian method
        # Loop from second-to-last index down to 0
        total_substeps_used = 0
        for n in range(Nt_points - 2, -1, -1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Solving time step {n}/{Nt_points - 2}")

            # Index for density and coupling arrays
            m_idx = min(n + 1, Nt_points - 1)
            u_prev_idx = min(n, Nt_points - 1)

            # Compute CFL and determine substeps needed for this time step.
            # DPP and canonical-CS paths don't trace explicit characteristics (the per-point
            # optimization / DPP fixed point is unconditionally stable, Issue #1058), so CFL
            # substepping is neither needed nor applied -- this is what lets canonical_cs use
            # a large dt directly.
            if self._use_dpp or self.diffusion_method == "canonical_cs":
                cfl, n_substeps, dt_substep = 0.0, 1, self.dt
            else:
                cfl, n_substeps, dt_substep = self._compute_cfl_and_substeps(U_solution[n + 1], self.dt)
            total_substeps_used += n_substeps

            if n_substeps == 1:
                # No substepping needed - use standard time step
                U_solution[n] = self._solve_timestep_semi_lagrangian(
                    U_solution[n + 1],  # u^{n+1} (from output array, always valid)
                    M_density[m_idx],  # m^{n+1} or last available density
                    U_coupling_prev[u_prev_idx],  # u_k^n for coupling terms
                    n,  # time index
                )
            else:
                # Adaptive substepping: subdivide the time step
                U_current = U_solution[n + 1].copy()
                for substep in range(n_substeps):
                    U_current = self._solve_timestep_semi_lagrangian_with_dt(
                        U_current,
                        M_density[m_idx],
                        U_coupling_prev[u_prev_idx],
                        n,
                        dt_substep,
                    )
                    # Check for numerical issues after each substep
                    if np.any(np.isnan(U_current) | np.isinf(U_current)):
                        error_msg = (
                            f"Semi-Lagrangian solver failed at time step {n}/{Nt_points - 2}, "
                            f"substep {substep + 1}/{n_substeps} with NaN/Inf values. "
                            f"CFL was {cfl:.2f}, using {n_substeps} substeps with dt={dt_substep:.6f}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                U_solution[n] = U_current
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Time step {n}: used {n_substeps} substeps (CFL={cfl:.2f})")

            # Check for numerical issues
            if np.any(np.isnan(U_solution[n]) | np.isinf(U_solution[n])):
                error_msg = (
                    f"Semi-Lagrangian solver failed at time step {n}/{Nt_points - 2} with NaN/Inf values. "
                    "Possible causes:\n"
                    "  1. CFL condition violated (try smaller dt or enable adaptive_substepping=True)\n"
                    "  2. Grid too coarse for solution features\n"
                    "  3. Hamiltonian evaluation issues\n"
                    "  4. Interpolation errors near boundaries"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        if logger.isEnabledFor(logging.INFO):
            final_residual = np.linalg.norm(U_solution[1] - U_solution[0])
            logger.info(f"Semi-Lagrangian HJB solve completed. Final residual: {final_residual:.2e}")
            if self.enable_adaptive_substepping and total_substeps_used > Nt_points:
                logger.info(
                    f"Adaptive substepping used {total_substeps_used} total substeps for {Nt_points} time points"
                )

        # Log gradient clipping summary (Issue #583)
        if self.gradient_clip_threshold is not None or self.gradient_stats["count"] > 0:
            self._log_gradient_clipping_summary()

        return U_solution

    def _advect_pointwise(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        grad_u: np.ndarray,
        t_val: float,
        dt: float,
    ) -> np.ndarray:
        """One backward semi-Lagrangian advection sweep, node by node.

        Issue #1413: lambda-scaled foot (x - dt*dH/dp = x - dt*p/lambda) followed by the
        Lax-Oleinik value update, matching the vectorized batch path.

        Single owner for the two byte-identical copies of this loop (Issue #1635): the
        rk4/other fallback of the fixed-dt path, and the CFL-substepping path.

        A per-node failure propagates. The previous handlers caught Exception and assigned
        ``U_star[i] = U_next[i]`` -- the value at t^{n+1}, i.e. no update at all for that
        node. That substitution is finite by construction, so the NaN/Inf guard in
        solve_hjb_system could not see it and the solver returned a plausible, silently
        wrong value function with no machine-readable trace. The nD siblings catch no
        exceptions at all; they do still substitute a stale value on NaN/Inf, counting the
        affected nodes and escalating to a raise past a 10% threshold -- though the with-dt
        sibling logs nothing below it, leaving that path as invisible as this one was.
        Tracked in #1641.
        """
        Nx = len(U_next)
        # Issue #1547: dH/dp from the Hamiltonian, batched once per sweep rather than a hardcoded
        # p/lambda per node (see _characteristic_foot_velocity).
        vel_all = self._characteristic_foot_velocity(
            self.x_grid.reshape(-1, 1), M_next, grad_u.reshape(-1, 1), t_val
        ).reshape(-1)
        U_star = np.zeros(Nx)
        for i in range(Nx):
            x_i = self.x_grid[i]
            try:
                vel_i = vel_all[i]
                x_departure = self._trace_characteristic_backward(x_i, vel_i, dt)
                u_departure = self._interpolate_value(U_next, x_departure)
                U_star[i] = self._sl_value_update(
                    u_departure, np.array([x_i]), M_next[i], np.array([grad_u[i]]), t_val, dt
                )
            except NotImplementedError:
                # An unsupported configuration is not a per-point numerical failure; let the
                # declared type reach the caller instead of being retyped as RuntimeError.
                raise
            except Exception as e:
                raise RuntimeError(
                    f"Semi-Lagrangian update failed at grid point {i} (x={x_i:.6g}, t={t_val:.6g}, dt={dt:.6g}): {e}"
                ) from e
        return U_star

    def _solve_timestep_semi_lagrangian(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        U_prev_picard: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """
        Solve one timestep using semi-Lagrangian method (supports 1D and nD).

        Args:
            U_next: Value function at next time step
                - 1D: shape (Nx,)
                - nD: shape matching grid.num_points
            M_next: Density at next time step (same shape as U_next)
            U_prev_picard: Value from previous Picard iteration (for coupling)
            time_idx: Current time index

        Returns:
            Value function at current time step (same shape as U_next)
        """
        # Issue #1058: canonical Carlini-Silva SL with implicit-alpha* DPP fixed point.
        # Dispatched first so an explicit diffusion_method='canonical_cs' request is honored
        # regardless of whether a Lagrangian-driven DPP path would also apply.
        if self.diffusion_method == "canonical_cs":
            return self._solve_timestep_canonical_cs(U_next, M_next, time_idx, dt=self.dt)

        # Issue #909: L-based DPP path for non-smooth Lagrangians
        if self._use_dpp:
            return self._solve_timestep_dpp(U_next, M_next, time_idx)

        # Issue #1026: Carlini-Silva stochastic-characteristic SL bypasses splitting
        if self.diffusion_method == "stochastic":
            return self._solve_timestep_stochastic_sl(U_next, M_next, time_idx, dt=self.dt)

        if self.dimension == 1:
            # 1D solve with operator splitting: characteristics + Crank-Nicolson diffusion

            # Compute gradient for optimal control: α* = ∇u
            # Pass timestep and density for gradient clipping monitoring (Issue #583)
            grad_u = self._compute_gradient(U_next, check_cfl=True, t_idx=time_idx, m_density=M_next)

            # Issue #930: Vectorized advection — batch characteristic tracing + interpolation
            # For explicit_euler/rk2, characteristic is x_departure = x - p*dt (vectorizable)
            if self.characteristic_solver in ("explicit_euler", "rk2"):
                # Step 1a: Batch departure points.
                # Issue #1071 fail-fast on a missing Hamiltonian, hoisted above the foot (Issue
                # #1547) because the foot itself now needs H: never silently drop the H term.
                H_class = self.problem.hamiltonian_class
                if H_class is None:
                    raise ValueError(
                        "HJBSemiLagrangianSolver: problem.hamiltonian_class is None in the batch "
                        "Hamiltonian path. Specify a Hamiltonian explicitly (Issue #1071, fail-fast)."
                    )
                x_batch = self.x_grid.reshape(-1, 1)  # (Nx, 1)
                p_batch = grad_u.reshape(-1, 1)  # (Nx, 1)
                # Issue #1547: dH/dp from the Hamiltonian, not a hardcoded p/lambda.
                vel = self._characteristic_foot_velocity(x_batch, M_next, p_batch, time_idx * self.dt).reshape(-1)
                x_departures = self.x_grid - vel * self.dt

                # Apply boundary conditions (vectorized)
                bc = self.get_boundary_conditions()
                bc_op = bc_type_to_geometric_operation(_checked_bc_type_string(bc))
                bounds = self.problem.geometry.get_bounds()
                xmin, xmax = bounds[0][0], bounds[1][0]
                if bc_op == "reflect":
                    # Issue #1161: mirror-reflect out-of-bounds feet (no-flux/Neumann),
                    # not np.clip — clamping collapsed them onto the wall node.
                    x_departures = reflect_into_domain(x_departures, xmin, xmax)
                elif bc_op == "wrap":
                    # Periodic: wrap around
                    L = xmax - xmin
                    x_departures = xmin + (x_departures - xmin) % L

                # Step 1b: Batch interpolation
                from scipy.interpolate import CubicSpline, interp1d

                if self.interpolation_method == "cubic":
                    interp_fn = CubicSpline(self.x_grid, U_next, bc_type="not-a-knot")
                    u_departures = interp_fn(x_departures)
                else:
                    interp_fn = interp1d(self.x_grid, U_next, kind="linear", fill_value="extrapolate")
                    u_departures = interp_fn(x_departures)

                # Step 1d: Lax-Oleinik value update (Issue #1413)
                U_star = self._sl_value_update(u_departures, x_batch, M_next, p_batch, time_idx * self.dt, self.dt)

            else:
                # Fallback: per-point loop for rk4 or other methods. Issue #1413: λ-scaled foot
                # (x - dt·∂H/∂p = x - dt·p/λ) + Lax-Oleinik value update, matching the batch path.
                U_star = self._advect_pointwise(U_next, M_next, grad_u, time_idx * self.dt, self.dt)

            # Step 2: Diffusion (using configured method)
            U_current = self._apply_diffusion(U_star, self.dt)

            # Step 3: Enforce boundary conditions on solution using the applicator
            bc = self.get_boundary_conditions()
            if bc:
                time = time_idx * self.dt
                U_current = self.bc_applicator.enforce_values(
                    U_current, boundary_conditions=bc, spacing=(self.dx,), time=time
                )

            return U_current

        else:
            # nD solve with operator splitting: advection + ADI diffusion
            # Reshape arrays to grid shape for easier indexing
            if U_next.ndim == 1:
                # Infer grid shape from array size (handles both full grid and interior points)
                total_points = U_next.size
                expected_full = int(np.prod(self._grid_shape))

                if total_points == expected_full:
                    grid_shape = tuple(self._grid_shape)
                else:
                    # Interior points only (num_points - 1 in each dimension)
                    grid_shape = tuple(n - 1 for n in self._grid_shape)

                U_next_shaped = U_next.reshape(grid_shape)
                M_next_shaped = M_next.reshape(grid_shape)
            else:
                U_next_shaped = U_next
                M_next_shaped = M_next
                grid_shape = U_next_shaped.shape

            # Step 1: Advection pass - compute u_star for all points
            # u_star = u(X(t-dt)) - dt * H(x, p*, m)
            U_star = np.zeros_like(U_next_shaped)

            # Compute gradient for optimal control: alpha* = grad(u)
            # Returns tuple of gradient components, each with shape grid_shape
            # Pass timestep and density for gradient clipping monitoring (Issue #583)
            grad_components = self._compute_gradient(
                U_next_shaped, check_cfl=True, t_idx=time_idx, m_density=M_next_shaped
            )

            # Track errors for diagnostics
            error_count = 0
            total_points = int(np.prod(grid_shape))

            # Issue #1413: hoist the time value for the Lax-Oleinik update in the loop.
            t_val = time_idx * self.dt
            # Issue #1547: dH/dp from the Hamiltonian, batched once per step (see
            # _characteristic_foot_velocity) rather than a hardcoded p/lambda per node.
            vel_grid = self._nd_foot_velocity_field(grid_shape, grad_components, M_next_shaped, t_val)
            # Iterate over all grid points for advection
            for multi_idx in np.ndindex(grid_shape):
                # Get spatial coordinates for this grid point
                x_current = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
                m_current = M_next_shaped[multi_idx]

                # Extract momentum p = ∇u (vector for nD)
                p_optimal = np.array([grad_components[d][multi_idx] for d in range(self.dimension)])

                # Issue #1413: trace along the characteristic velocity ∂H/∂p, then apply the
                # Lax-Oleinik value update (the foot carries advection; cost = dt·H_control -
                # dt·(V+f)). Replaces the inconsistent `u_departure - dt·H` with a non-λ-scaled
                # foot (Issue #575/#1413). Issue #1547: ∂H/∂p is the Hamiltonian's, not p/λ.
                vel = vel_grid[multi_idx]
                x_departure = self._trace_characteristic_backward(x_current, vel, self.dt)
                u_departure = self._interpolate_value(U_next_shaped, x_departure)
                u_star_val = self._sl_value_update(u_departure, x_current, m_current, p_optimal, t_val, self.dt)

                # Check for numerical issues
                if np.isnan(u_star_val) or np.isinf(u_star_val):
                    error_count += 1
                    if error_count <= 5:
                        logger.warning(f"NaN/Inf at grid point {multi_idx}: u_departure={u_departure:.3e}")
                    U_star[multi_idx] = U_next_shaped[multi_idx]  # Fallback
                else:
                    U_star[multi_idx] = u_star_val

            # Report error summary if any occurred in advection
            if error_count > 0:
                error_pct = 100 * error_count / total_points
                if error_pct > 10:
                    raise ValueError(
                        f"Semi-Lagrangian advection failed: {error_count}/{total_points} points ({error_pct:.1f}%) "
                        f"had NaN/Inf values at time step {time_idx}. Check grid resolution and time step."
                    )
                else:
                    logger.warning(
                        f"Semi-Lagrangian advection: {error_count}/{total_points} points ({error_pct:.1f}%) "
                        f"had NaN/Inf values at time step {time_idx}"
                    )

            # Step 2: Diffusion pass (using configured method)
            U_current_shaped = self._apply_diffusion(U_star, self.dt)

            # Return flattened if input was flattened
            if U_next.ndim == 1:
                return U_current_shaped.ravel()
            else:
                return U_current_shaped

    def _solve_timestep_semi_lagrangian_with_dt(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        U_prev_picard: np.ndarray,
        time_idx: int,
        dt: float,
    ) -> np.ndarray:
        """
        Solve one timestep using semi-Lagrangian method with custom time step.

        This is the same as _solve_timestep_semi_lagrangian but allows specifying
        a custom dt for adaptive substepping.

        Args:
            U_next: Value function at next time step
            M_next: Density at next time step
            U_prev_picard: Value from previous Picard iteration
            time_idx: Current time index
            dt: Time step to use (allows custom dt for substepping)

        Returns:
            Value function at current time step
        """
        # Issue #1058: canonical Carlini-Silva SL with implicit-alpha* DPP fixed point.
        if self.diffusion_method == "canonical_cs":
            return self._solve_timestep_canonical_cs(U_next, M_next, time_idx, dt=dt)

        # Issue #909: L-based DPP path for non-smooth Lagrangians
        if self._use_dpp:
            return self._solve_timestep_dpp(U_next, M_next, time_idx, dt=dt)

        # Issue #1026: Carlini-Silva stochastic-characteristic SL bypasses splitting
        if self.diffusion_method == "stochastic":
            return self._solve_timestep_stochastic_sl(U_next, M_next, time_idx, dt=dt)

        if self.dimension == 1:
            # 1D solve with operator splitting

            # Compute gradient for optimal control
            # Pass timestep and density for gradient clipping monitoring (Issue #583)
            grad_u = self._compute_gradient(U_next, check_cfl=False, t_idx=time_idx, m_density=M_next)

            # Step 1: Advection along characteristics (Issue #1413: λ-scaled foot + Lax-Oleinik)
            U_star = self._advect_pointwise(U_next, M_next, grad_u, time_idx * self.dt, dt)

            # Step 2: Diffusion with custom dt
            U_current = self._apply_diffusion(U_star, dt)

            # Step 3: Enforce boundary conditions on solution
            U_current = self._enforce_boundary_conditions(U_current)

            return U_current

        else:
            # nD solve with operator splitting
            if U_next.ndim == 1:
                total_points = U_next.size
                expected_full = int(np.prod(self._grid_shape))

                if total_points == expected_full:
                    grid_shape = tuple(self._grid_shape)
                else:
                    grid_shape = tuple(n - 1 for n in self._grid_shape)

                U_next_shaped = U_next.reshape(grid_shape)
                M_next_shaped = M_next.reshape(grid_shape)
            else:
                U_next_shaped = U_next
                M_next_shaped = M_next
                grid_shape = U_next_shaped.shape

            U_star = np.zeros_like(U_next_shaped)
            # Pass timestep and density for gradient clipping monitoring (Issue #583)
            grad_components = self._compute_gradient(
                U_next_shaped, check_cfl=False, t_idx=time_idx, m_density=M_next_shaped
            )

            error_count = 0
            total_points = int(np.prod(grid_shape))

            t_val = time_idx * self.dt
            # Issue #1547: dH/dp from the Hamiltonian, batched once (see _nd_foot_velocity_field).
            vel_grid = self._nd_foot_velocity_field(grid_shape, grad_components, M_next_shaped, t_val)
            for multi_idx in np.ndindex(grid_shape):
                x_current = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
                m_current = M_next_shaped[multi_idx]
                p_optimal = np.array([grad_components[d][multi_idx] for d in range(self.dimension)])

                # Issue #1413: characteristic foot (∂H/∂p) + Lax-Oleinik value update.
                vel = vel_grid[multi_idx]
                x_departure = self._trace_characteristic_backward(x_current, vel, dt)
                u_departure = self._interpolate_value(U_next_shaped, x_departure)
                u_star_val = self._sl_value_update(u_departure, x_current, m_current, p_optimal, t_val, dt)

                if np.isnan(u_star_val) or np.isinf(u_star_val):
                    error_count += 1
                    U_star[multi_idx] = U_next_shaped[multi_idx]
                else:
                    U_star[multi_idx] = u_star_val

            if error_count > 0:
                error_pct = 100 * error_count / total_points
                if error_pct > 10:
                    raise ValueError(
                        f"Semi-Lagrangian advection failed: {error_count}/{total_points} points ({error_pct:.1f}%) "
                        f"had NaN/Inf values at time step {time_idx}."
                    )

            # ADI diffusion with custom dt
            U_current_shaped = self._apply_diffusion(U_star, dt)

            # Enforce boundary conditions (Issue #636 - nD support)
            U_current_shaped = self._enforce_boundary_conditions(U_current_shaped)

            if U_next.ndim == 1:
                return U_current_shaped.ravel()
            else:
                return U_current_shaped

    # === Carlini-Silva stochastic-characteristic SL (Issue #1026) ===

    def _brownian_foot_offset(self, sqrt_dt: float) -> np.ndarray:
        """Per-axis Brownian foot offset ``c_ax`` for the 2d-departure SL diffusion quadrature.

        Both the ``'stochastic'`` and ``'canonical_cs'`` SL steps place ``2d`` feet
        ``x ± c_ax·e_ax`` (one ± pair per axis) and average them with uniform weight ``1/(2d)``.
        Taylor: ``(1/2d)·Σ_ax[u(x + c_ax e_ax) + u(x − c_ax e_ax)] = u + (1/2d)·Σ_ax c_ax²·∂²u/∂x_ax² + O(c⁴)``.
        Recovering the canonical anisotropic viscosity ``(1/2)·Σ_ax σ_ax²·∂²u/∂x_ax²·dt`` — the
        ``-(σ²/2)·Δu`` term of the HJB residual (``base_hjb.py``) — requires
        ``c_ax = √d·σ_ax·√dt`` (weak-Euler direction tree: ``E[ξ ξᵀ] = I·dt`` over the ``2d``
        one-axis feet). The ``√d`` is an exact identity at ``d = 1`` and restores the ``1/d``
        diffusion deficit that under-diffused every ``d ≥ 2`` SL solve (Issue #1543: 2× in 2D,
        3× in 3D). ``diffusion_method='adi'`` is a separate path and is unaffected.

        Single owner for the departure offset — the ``'stochastic'`` and ``'canonical_cs'`` paths
        must not re-derive ``σ·√dt`` independently (the divergence that was Issue #1543).
        """
        d = self.dimension
        sigma = self.problem.sigma
        if isinstance(sigma, np.ndarray):
            sigma_diag = np.asarray(sigma, dtype=float).ravel()
            if sigma_diag.size != d:
                raise ValueError(
                    f"Diagonal sigma must have {d} entries, got {sigma_diag.size}. "
                    "Full-tensor sigma not supported by semi-Lagrangian SL."
                )
        else:
            sigma_diag = np.full(d, float(sigma))
        return np.sqrt(d) * sigma_diag * sqrt_dt

    def _solve_timestep_stochastic_sl(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        time_idx: int,
        dt: float | None = None,
    ) -> np.ndarray:
        """Carlini-Silva (2014) semi-Lagrangian step with stochastic characteristics.

        The diffusion enters directly through 2*d stochastic departure points
        (one pair per spatial dimension), eliminating the operator-splitting
        diffusion solve. For a separable Hamiltonian H = H_control(p) + V(x) + f(m)
        the characteristic velocity is dH/dp (= p/lambda for the quadratic control
        cost) and the Lax-Oleinik update is (Issue #1413)

            U^n_i = (1/(2d)) * sum_{k=1..d} [I[U^{n+1}](y_k^+) + I[U^{n+1}](y_k^-)]
                    + dt * H_control(p_i) - dt * (V + f)
                  = u_avg + dt * (H(x_i, p_i, m) - 2 * H(x_i, 0, m))

        with y_k^pm = x_i - (dH/dp)_i * dt +/- sqrt(d) * sigma * sqrt(dt) * e_k (the
        sqrt(d) restores the full (sigma^2/2) Lap(u) under the 1/(2d) average, Issue #1543). (The prior
        `alpha* = -nabla u` foot with `- dt*H` was lambda=1-only on the foot and
        double-counted the kinetic term ~3x; see Issue #575/#1413.)

        Validation: see mfg-research/experiments/crowd_evacuation_2d/minors/archive/
        exp14_towel_1d_benchmark/subs/exp14e_solver_comparison/, where the
        decoupled-from-diffusion form (diffusion_method='adi' default) gave
        O(h) convergence on the 1D Boltzmann-Gibbs equilibrium against the
        Carlini-Silva theoretical O(h^2). Reproducing the CS rate requires
        this stochastic-characteristic path.

        References:
            Carlini, E., & Silva, F. J. (2014). A semi-Lagrangian scheme for a
            degenerate second order MFG system. Discrete and Continuous
            Dynamical Systems, 35(9), 4269-4292.

        Args:
            U_next: Value function at next time step (shape (Nx,) for 1D, or
                grid shape / flattened for nD).
            M_next: Density at next time step (matching shape).
            time_idx: Current time index (used in Hamiltonian evaluation).
            dt: Time step. Defaults to self.dt; pass explicitly for adaptive
                substepping.

        Returns:
            Value function at current time step, same shape as U_next.

        Notes:
            Issue #1049: previously rejected interpolation_method="linear" here,
            inverted from CS 2014's stability requirement. The "linear" path is
            now allowed; warning issued at __init__ when cubic/quintic is used
            with stochastic (the unproven combination).
        """
        if dt is None:
            dt = self.dt

        return self._stochastic_sl_step(U_next, M_next, time_idx, dt)

    def _stochastic_sl_step(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        time_idx: int,
        dt: float,
    ) -> np.ndarray:
        """Dimension-agnostic stochastic SL step (Carlini-Silva 2014).

        Issue #1050: unifies the former ``_stochastic_sl_step_1d`` and
        ``_stochastic_sl_step_nd`` into one method handling d ∈ {1, 2, 3, ...}.
        One Brownian-quadrature step: ``2*d`` departures per node (a
        ``±√d·σ_ax·√dt`` pair per axis from the drift foot ``x − p·dt``; the
        ``√d`` makes the ``1/(2d)`` average recover the full ``(σ²/2)Δu``,
        Issue #1543), interpolate ``u^{n+1}`` at each foot, average over the
        ``2*d`` directions, subtract ``dt·H``.

        The shared structure (drift + Brownian departures, boundary fold,
        averaging, batch Hamiltonian) is written once, so the 1D fixes
        #1033/#1048/#1049 and the nD fixes #1054 now live in a single path.
        The merge is byte-identical to both former methods (Issue #1050
        verification: 1D/nD, shaped/flat, linear/cubic, time-dependent H).

        Two backend choices stay dimension-dependent — not because the
        algorithm differs, but because collapsing them changes the numerics
        (verified non-byte-identical):

        - **Interpolation backend**: 1D uses ``numpy.interp`` (linear) /
          ``PchipInterpolator`` (cubic), which *clamp* out-of-bounds feet to
          the endpoint value; nD uses ``RegularGridInterpolator``, which
          *extrapolates* (``fill_value=None``). Under reflect/wrap BC every
          foot is in-bounds and the two agree to ~1 ULP; they diverge only for
          ``clamp`` BC (Dirichlet/none), where clamp vs. extrapolation are
          genuinely different policies. The per-dimension backend is kept so
          each stays byte-identical to its pre-#1050 behavior.
        - **Final BC enforcement**: 1D uses ``FDMApplicator`` (Neumann
          2nd-order extrapolation); nD uses ``InterpolationApplicator`` (via
          ``_enforce_boundary_conditions``). These give materially different
          boundary values (O(1e-1)); the split predates #1050 and pervades the
          SL solver (the ADI path has the same fork). Reconciling it is a
          separate concern, out of scope for this refactor.

        See ``_solve_timestep_stochastic_sl`` for the scheme references.
        """
        from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

        d = self.dimension
        sqrt_dt = float(np.sqrt(dt))

        # --- Grid coordinates, shaped fields, grid shape (dim-agnostic) ---
        if d == 1:
            # 1D stores direct arrays (self.grid is None); the coordinate tuple
            # is just (x_grid,) and fields are already flat.
            grid_coords = (self.x_grid,)
            U_shaped = U_next
            M_shaped = M_next
            grid_shape = U_next.shape
            flat_input = False
        else:
            # Reshape to grid form (matches the Strang-splitting nD path)
            if U_next.ndim == 1:
                total_points = U_next.size
                expected_full = int(np.prod(self._grid_shape))
                if total_points == expected_full:
                    grid_shape = tuple(self._grid_shape)
                else:
                    grid_shape = tuple(n - 1 for n in self._grid_shape)
                U_shaped = U_next.reshape(grid_shape)
                M_shaped = M_next.reshape(grid_shape)
                flat_input = True
            else:
                U_shaped = U_next
                M_shaped = M_next
                grid_shape = U_shaped.shape
                flat_input = False
            grid_coords = tuple(self.grid.coordinates)

        # --- Per-axis Brownian foot offset c_ax = √d·σ_ax·√dt (Issue #1543, single source) ---
        foot_offset = self._brownian_foot_offset(sqrt_dt)

        # Optimal control α* = -p where p = ∇u^{n+1}; drift foot x_drift = x − p·dt.
        grad = self._compute_gradient(U_shaped, check_cfl=True, t_idx=time_idx, m_density=M_shaped)
        grad_components = (grad,) if d == 1 else grad

        # --- Boundary fold for the Brownian feet ---
        # Issue #1048 (1D) / #1054 (nD): REFLECT feet for Neumann BC (not clamp),
        # wrap for periodic. reflect_into_domain is the correct per-axis fold
        # (identity in-bounds); the earlier center-flip mirrored about the midpoint.
        bc = self.get_boundary_conditions()
        bc_op = bc_type_to_geometric_operation(_checked_bc_type_string(bc))
        bounds = self.problem.geometry.get_bounds()
        x_min = np.asarray(bounds[0], dtype=float)
        x_max = np.asarray(bounds[1], dtype=float)
        L_axis = x_max - x_min

        # --- Build the 2*d departures per node (one ± pair per axis) ---
        n_total = int(np.prod(grid_shape))
        mesh = np.meshgrid(*grid_coords, indexing="ij")
        # x_drift_flat[i, ax] = x_current[ax] − dt · ∂H/∂p[ax] for node i
        x_positions_flat = np.stack([mesh[ax].ravel() for ax in range(d)], axis=1)
        p_flat = np.stack([grad_components[ax].ravel() for ax in range(d)], axis=1)
        # Issue #1413: drift along the characteristic velocity ∂H/∂p (not raw p).
        # Issue #1547: ∂H/∂p comes from the Hamiltonian, not a hardcoded p/λ.
        vel_flat = self._characteristic_foot_velocity(x_positions_flat, M_shaped.ravel(), p_flat, time_idx * dt)
        x_drift_flat = x_positions_flat - vel_flat * dt

        all_departures = np.empty((2 * d * n_total, d), dtype=float)
        for ax in range(d):
            offset = np.zeros(d)
            offset[ax] = foot_offset[ax]
            block_start = 2 * ax * n_total
            all_departures[block_start : block_start + n_total] = x_drift_flat + offset[None, :]
            all_departures[block_start + n_total : block_start + 2 * n_total] = x_drift_flat - offset[None, :]

        if bc_op == "reflect":
            all_departures = reflect_into_domain(all_departures, x_min, x_max)
        elif bc_op == "wrap":
            all_departures = x_min + (all_departures - x_min) % L_axis

        # --- Interpolate u^{n+1} at every foot (dim-dependent backend, see docstring) ---
        # Issue #1033/#1054: monotone dispatch — cubic/quintic → PCHIP (monotone
        # Hermite) to avoid the non-monotone CubicSpline blow-up; linear is the
        # canonical Carlini-Silva interpolant (Issue #1049).
        if self.interpolation_method == "linear":
            interp_method = "linear"
        elif self.interpolation_method in ("cubic", "quintic"):
            interp_method = "pchip"
        else:
            interp_method = "linear"

        if d == 1:
            # numpy.interp / PchipInterpolator preserve the 1D clamp-at-boundary
            # semantics. extrapolate=False propagates nan for an out-of-range query,
            # which (after the reflect/wrap fold) would signal a real upstream bug.
            if interp_method == "linear":
                all_u = np.interp(all_departures[:, 0], self.x_grid, U_next)
            else:
                all_u = PchipInterpolator(self.x_grid, U_next, extrapolate=False)(all_departures[:, 0])
        else:
            interp_fn = RegularGridInterpolator(
                grid_coords, U_shaped, method=interp_method, bounds_error=False, fill_value=None
            )
            all_u = interp_fn(all_departures)

        # Average over the 2*d Brownian directions
        u_avg = all_u.reshape(2 * d, n_total).mean(axis=0).reshape(grid_shape)

        # --- Lax-Oleinik value update (Issue #1413; single source, Issue #1071) ---
        # Reuses the node positions / momenta already stacked for the drift foot above.
        x_batch = x_positions_flat
        p_batch = p_flat
        H_class = self.problem.hamiltonian_class
        if H_class is None:
            raise ValueError(
                "HJBSemiLagrangianSolver (stochastic CS): problem.hamiltonian_class is None. "
                "Specify a Hamiltonian explicitly (Issue #1071, fail-fast)."
            )
        U_current = self._sl_value_update(u_avg.ravel(), x_batch, M_shaped.ravel(), p_batch, time_idx * dt, dt).reshape(
            grid_shape
        )

        # --- Enforce BC on the result (dim-dependent applicator, see docstring) ---
        if d == 1:
            if bc:
                U_current = self.bc_applicator.enforce_values(
                    U_current, boundary_conditions=bc, spacing=(self.dx,), time=time_idx * dt
                )
            return U_current

        U_current = self._enforce_boundary_conditions(U_current)
        return U_current.ravel() if flat_input else U_current

    # === Canonical Carlini-Silva SL with implicit-alpha* DPP (Issue #1058) ===

    def _solve_timestep_canonical_cs(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        time_idx: int,
        dt: float | None = None,
    ) -> np.ndarray:
        r"""Canonical Carlini-Silva (2014) SL step with the implicit-$\alpha^*$ DPP fixed point.

        Unlike ``diffusion_method="stochastic"`` (explicit $\alpha^* = -\nabla u^{n+1}(x_i)$,
        the gradient at the *grid* node), this path solves the **implicit** dynamic-programming
        principle: at each node $x_i$ the optimal control is the per-point minimizer of the DPP
        objective, so $\alpha^*$ is consistent with the departure point it induces. This is the
        hypothesis under which CS 2014 prove unconditional stability and convergence for monotone
        (Q1/linear) interpolation; the explicit at-grid gradient is an approximation that
        diverges from it on stiff problems (Issue #1058).

        For the standard separable Hamiltonian $H(x,p,m) = \tfrac12 |p|^2 + h(x,m)$ (unit
        control weight; $h(x,m) = H(x,m,p{=}0,t)$ is the potential + coupling part, obtained
        single-source via ``eval_H_batch`` at $p=0$), the per-node objective is

        .. math::
            \varphi(\alpha) = \tfrac{\mathrm{d}t}{2}\,|\alpha|^2 - \mathrm{d}t\, h(x_i, m_i)
              + \frac{1}{2d} \sum_{k=1}^{d}
                \bigl[ I_h u^{n+1}(y_k^+) + I_h u^{n+1}(y_k^-) \bigr],

        with stochastic departures
        $y_k^\pm = x_i + \alpha\,\mathrm{d}t \pm \sigma_k\sqrt{\mathrm{d}t}\,e_k$ (folded into
        the domain by the boundary operation), $I_h$ the linear (monotone Q1) interpolant of
        $u^{n+1}$, and $u^n(x_i) = \min_\alpha \varphi(\alpha) = \varphi(\alpha^*)$. $\alpha^*$
        is *implicit* because the departure points -- hence $I_h$ -- depend on $\alpha$. The
        $- \mathrm{d}t\,h$ sign (rather than $+$) follows from the cost-to-go backward
        convention $\partial_t u = H - \tfrac{\sigma^2}{2}\Delta u$ used throughout this solver:
        the running Lagrangian is $L = \tfrac12|\alpha|^2 - h$ (Legendre dual of $H$).

        The minimization is over the control at each node (1D: a vectorized fixed-iteration
        golden-section search solving all nodes' independent 1D problems at once;
        nD: ``scipy.optimize.minimize`` L-BFGS-B per node) -- the cost CS trade for
        unconditional stability. Diffusion enters through the $2d$ Brownian departures (added variance
        $\sigma^2\mathrm{d}t$ per step), so no operator-splitting diffusion solve is used
        (``_apply_diffusion`` is bypassed, as for ``"stochastic"``). The scheme targets
        reflecting / no-flux (Neumann) boundaries (the CS 2014 setting); the reflected
        departures realize zero-flux without a separate boundary enforcement pass.

        Issue #1058. Canonical CS implicit-alpha*; validated in
        mfg-research/.../exp08_towel_2d_validation/_preflight_1d/cs_sl_canonical_implicit_1d.py
        (KL=3.3e-2 on 1D Towel-on-Beach, 4x faster than FDM).

        References:
            Carlini, E., & Silva, F. J. (2014). A semi-Lagrangian scheme for a degenerate
            second order MFG system. ESAIM: M2AN 49(6), 1567-1604.

        Args:
            U_next: Value function at the next time step (shape ``(Nx,)`` for 1D, grid shape or
                flattened for nD).
            M_next: Density at the next time step (matching shape).
            time_idx: Current time index (used in the Hamiltonian evaluation).
            dt: Time step. Defaults to ``self.dt``; pass explicitly for substepping.

        Returns:
            Value function at the current time step, same shape as ``U_next``.
        """
        if dt is None:
            dt = self.dt
        return self._canonical_cs_step(U_next, M_next, time_idx, dt)

    def _canonical_cs_step(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        time_idx: int,
        dt: float,
    ) -> np.ndarray:
        """Per-node implicit-alpha* DPP minimization (see ``_solve_timestep_canonical_cs``).

        Reuses the stochastic-SL departure-point machinery (diagonal volatility, boundary fold)
        but replaces the explicit ``alpha* = -grad u`` drift with a per-node minimizer of the
        DPP objective ``phi(alpha)``. 1D uses a vectorized fixed-iteration golden-section search
        that minimizes all nodes' independent 1D problems simultaneously; nD uses L-BFGS-B
        (``minimize``) per node over the control vector.
        """
        from scipy.interpolate import RegularGridInterpolator, interp1d

        d = self.dimension
        sqrt_dt = float(np.sqrt(dt))
        t_n = time_idx * dt

        # Boundary fold for the stochastic departures (reflect = no-flux/Neumann, the CS
        # setting; wrap = periodic; clamp otherwise). Shared with the stochastic SL path.
        bc = self.get_boundary_conditions()
        bc_op = bc_type_to_geometric_operation(_checked_bc_type_string(bc))
        bounds = self.problem.geometry.get_bounds()
        x_min = np.asarray(bounds[0], dtype=float)
        x_max = np.asarray(bounds[1], dtype=float)
        span = x_max - x_min

        def _fold(points: np.ndarray) -> np.ndarray:
            """Fold departure coordinates (``(..., d)``) back into ``[x_min, x_max]``."""
            if bc_op == "reflect":
                return reflect_into_domain(points, x_min, x_max)
            if bc_op == "wrap":
                return x_min + (points - x_min) % span
            return np.clip(points, x_min, x_max)

        # Per-axis Brownian foot offset c_ax = √d·σ_ax·√dt (Issue #1543, single source; shared with stochastic SL).
        foot_offset = self._brownian_foot_offset(sqrt_dt)

        H_class = self.problem.hamiltonian_class
        # Issue #1420: control-cost lambda from the single source. The DPP running cost is
        # L(alpha) = (lambda/2)|alpha|^2, so the per-node minimizer gives alpha* = -grad(u)/lambda.
        # Hardcoding (1/2)|alpha|^2 (lambda=1) below undercut the solver's lambda != 1 support.
        lam = self._control_cost_lambda()
        # Allow up to 4x natural-traversal speed (scale-invariant heuristic, per the validated
        # prototype): alpha is bounded by 4 * domain_length / T per axis, scaled by 1/lambda since
        # the optimal control alpha* = -grad(u)/lambda grows as 1/lambda (no-op at lambda=1).
        alpha_bound = 4.0 * span / float(self.problem.T) / lam

        if d == 1:
            Nx = len(U_next)
            diff_off = float(foot_offset[0])
            bound = float(alpha_bound[0])

            # h_i = H(x_i, m_i, p=0, t_n): potential + coupling, single-source via H_class.
            if H_class is not None:
                x_batch = self.x_grid.reshape(-1, 1)
                p_zero = np.zeros((Nx, 1))
                h = eval_H_batch(H_class, x_batch, M_next, p_zero, t_n).ravel()
            else:
                h = np.zeros(Nx)

            # Build the monotone (Q1/linear) interpolant once per backward step.
            interp_fn = interp1d(self.x_grid, U_next, kind="linear", bounds_error=False, fill_value="extrapolate")

            # Vectorized per-node DPP minimization. All Nx nodes solve the SAME 1D objective
            # phi_i(alpha) up to the per-node offset (x_i, h_i); the prior per-point
            # ``minimize_scalar`` (Brent) loop is replaced by one fixed-iteration
            # golden-section search carried over the whole alpha array, so each iteration is a
            # single batched interpolation of all nodes' feet instead of Nx separate scipy
            # calls. phi is unimodal in the same sense Brent assumes (convex 0.5*dt*alpha^2
            # term plus the monotone Q1 interpolant of u^{n+1}); golden section localizes the
            # minimizer with the same bracket tolerance as Brent's ``xatol``.
            def phi_vec(alpha: np.ndarray) -> np.ndarray:
                """phi over all nodes for a per-node control array (shape (Nx,))."""
                y_drift = self.x_grid + alpha * dt
                feet_plus = _fold((y_drift + diff_off).reshape(-1, 1)).ravel()
                feet_minus = _fold((y_drift - diff_off).reshape(-1, 1)).ravel()
                u_pm = 0.5 * (interp_fn(feet_plus) + interp_fn(feet_minus))
                return 0.5 * lam * dt * alpha * alpha - dt * h + u_pm

            invphi = (np.sqrt(5.0) - 1.0) / 2.0  # 0.618...
            invphi2 = 1.0 - invphi  # 0.382...
            a = np.full(Nx, -bound)
            b = np.full(Nx, bound)
            # Iterations to shrink the bracket from width 2*bound to xatol=self.tolerance,
            # matching the convergence target of the Brent call this replaces.
            width0 = 2.0 * bound
            n_iter = max(1, int(np.ceil(np.log(self.tolerance / width0) / np.log(invphi))))
            x1 = a + invphi2 * (b - a)
            x2 = a + invphi * (b - a)
            f1 = phi_vec(x1)
            f2 = phi_vec(x2)
            for _ in range(n_iter):
                mask = f1 < f2  # min in [a, x2]; else min in [x1, b]
                new_a = np.where(mask, a, x1)
                new_b = np.where(mask, x2, b)
                new_x1 = np.where(mask, new_a + invphi2 * (new_b - new_a), x2)
                new_x2 = np.where(mask, x1, new_a + invphi * (new_b - new_a))
                # One new interior point per node (the other is carried with its value).
                eval_point = np.where(mask, new_x1, new_x2)
                f_eval = phi_vec(eval_point)
                new_f1 = np.where(mask, f_eval, f2)
                new_f2 = np.where(mask, f1, f_eval)
                a, b, x1, x2, f1, f2 = new_a, new_b, new_x1, new_x2, new_f1, new_f2

            # u^n(x_i) = phi(alpha*): the lowest evaluated DPP value at the implicit optimizer.
            return np.minimum(f1, f2)

        # --- nD: per-node vector minimization over the control alpha in R^d ---
        if U_next.ndim == 1:
            total_points = U_next.size
            expected_full = int(np.prod(self._grid_shape))
            grid_shape = (
                tuple(self._grid_shape) if total_points == expected_full else tuple(n - 1 for n in self._grid_shape)
            )
            U_shaped = U_next.reshape(grid_shape)
            M_shaped = M_next.reshape(grid_shape)
            flat_input = True
        else:
            U_shaped = U_next
            M_shaped = M_next
            grid_shape = U_shaped.shape
            flat_input = False

        grid_coords = tuple(self.grid.coordinates)
        interp_fn = RegularGridInterpolator(grid_coords, U_shaped, method="linear", bounds_error=False, fill_value=None)

        # The 2*d departure offsets: +/- c_ax along each axis -> (2d, d); c_ax = √d·σ_ax·√dt (Issue #1543).
        axis_offsets = np.diag(foot_offset)  # (d, d), row k = offset along axis k
        depart_offsets = np.concatenate([axis_offsets, -axis_offsets], axis=0)  # (2d, d)

        # h batch over all nodes: h(x_i, m_i) = H(x_i, m_i, p=0, t_n).
        mesh = np.meshgrid(*grid_coords, indexing="ij")
        x_flat = np.stack([mesh[ax].ravel() for ax in range(d)], axis=1)  # (n_total, d)
        if H_class is not None:
            p_zero = np.zeros((x_flat.shape[0], d))
            h_flat = eval_H_batch(H_class, x_flat, M_shaped.ravel(), p_zero, t_n).ravel()
        else:
            h_flat = np.zeros(x_flat.shape[0])

        box = [(-float(alpha_bound[ax]), float(alpha_bound[ax])) for ax in range(d)]
        U_current = np.empty(grid_shape)
        for flat_i, multi_idx in enumerate(np.ndindex(grid_shape)):
            x_i = x_flat[flat_i]
            h_i = float(h_flat[flat_i])

            def phi_nd(alpha_vec: np.ndarray, _xi: np.ndarray = x_i, _hi: float = h_i) -> float:
                drift = _xi + np.asarray(alpha_vec) * dt  # (d,)
                feet = _fold(drift[None, :] + depart_offsets)  # (2d, d)
                u_pm = interp_fn(feet)  # (2d,)
                return 0.5 * lam * dt * float(np.dot(alpha_vec, alpha_vec)) - dt * _hi + float(u_pm.mean())

            res = minimize(
                phi_nd,
                np.zeros(d),
                bounds=box,
                method="L-BFGS-B",
                options={"ftol": self.tolerance, "maxiter": 100},
            )
            U_current[multi_idx] = float(res.fun)

        return U_current.ravel() if flat_input else U_current

    # === L-based DPP formulation (Issue #909) ===

    def _solve_timestep_dpp(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        time_idx: int,
        dt: float | None = None,
    ) -> np.ndarray:
        """Solve one timestep via L-based Dynamic Programming Principle.

        u^n(x_i) = min_alpha { dt * L(x_i, alpha, m, t) + u^{n+1}(x_i + alpha*dt) }

        This avoids computing grad_u entirely — the optimization is over the
        control alpha, not the momentum p. Handles non-smooth L naturally:
        - Quadratic: closed-form alpha* = -p/lambda (falls back to H-based equivalent)
        - L1 / bang-bang: compare values at alpha in {-1, 0, 1}
        - Bounded: scalar optimization over [-a_max, a_max]
        - Finite action set: compare K candidate values

        Diffusion is handled identically via operator splitting after the
        advection/optimization step.

        Parameters
        ----------
        U_next : np.ndarray
            Value function at next time step. Shape (Nx,) for 1D, grid_shape for nD.
        M_next : np.ndarray
            Density at next time step (same shape).
        time_idx : int
            Current time index.
        dt : float or None
            Time step. If None, uses self.dt.

        Returns
        -------
        np.ndarray
            Value function at current time step.
        """
        if dt is None:
            dt = self.dt

        L_class = self.problem.lagrangian_class
        t_value = time_idx * self.problem.T / self.problem.Nt

        bounds = L_class.control_bounds() or (-10.0, 10.0)

        if self.dimension == 1:
            Nx = len(U_next)
            U_star = np.zeros(Nx)

            # Detect special structure for fast paths
            from mfgarchon.core.hamiltonian import L1ControlCost, SeparableLagrangian

            # The admissible set A is READ from its single owner,
            # ControlCostBase.effective_domain() (Issue #1642, B3) -- it is never
            # re-derived here. This previously carried its own ladder holding a
            # bang-bang interval literal for L1 and reading the max-control
            # attribute off Bounded directly, i.e. a second owner of A that could
            # drift from the first and that a regularized cost silently defeated.
            #
            # What remains local is only QUADRATURE -- how densely to sample A --
            # which is a genuine structural distinction, not a duplicated
            # quantity: a piecewise-linear L_ctrl is optimized at a vertex of A
            # or at its kink, a quadratic one needs interior samples. Narrowing
            # that last isinstance to a control-cost capability is Issue #1651.
            fast_candidates = None
            if isinstance(L_class, SeparableLagrangian):
                cc = L_class.control_cost
                domain = cc.effective_domain()
                if domain is None:
                    # A = R^d: no box to sample. QuadraticControlCost lands here
                    # and has the closed form alpha* = -grad_u/lambda anyway, so
                    # fall through to the scalar optimization path.
                    fast_candidates = None
                elif isinstance(cc, L1ControlCost):
                    # Bang-bang: L_ctrl is piecewise linear, so the optimum sits
                    # at an endpoint of A or at the kink alpha = 0.
                    fast_candidates = np.array([domain[0], 0.0, domain[1]])
                else:
                    # Quadratic-on-A costs (BoundedControlCost, and its
                    # Moreau-Yosida envelope): sample endpoints + interior.
                    fast_candidates = np.linspace(domain[0], domain[1], 11)

            for i in range(Nx):
                x_i = self.x_grid[i]
                x_arr = np.array([x_i])
                m_i = M_next[i]

                if fast_candidates is not None:
                    # Evaluate DPP cost at each candidate
                    best_val = np.inf
                    for alpha in fast_candidates:
                        x_next = x_i + alpha * dt
                        # Apply boundary handling
                        x_next = self._apply_boundary_to_point(x_next)
                        u_next = self._interpolate_value(U_next, x_next)
                        L_val = float(L_class(x_arr, np.array([alpha]), m_i, t_value))
                        cost = dt * L_val + u_next
                        if cost < best_val:
                            best_val = cost
                    U_star[i] = best_val
                else:
                    # Scalar optimization over alpha
                    def dpp_cost(alpha, _xi=x_i, _xa=x_arr, _mi=m_i):
                        x_next = _xi + alpha * dt
                        x_next = self._apply_boundary_to_point(x_next)
                        u_next = self._interpolate_value(U_next, x_next)
                        L_val = float(L_class(_xa, np.array([alpha]), _mi, t_value))
                        return dt * L_val + u_next

                    result = minimize_scalar(
                        dpp_cost, bounds=bounds, method="bounded", options={"xatol": self.tolerance}
                    )
                    U_star[i] = result.fun

            # Diffusion step (same as H-based)
            U_current = self._apply_diffusion(U_star, dt)

            # Enforce boundary conditions
            bc = self.get_boundary_conditions()
            if bc:
                time = time_idx * self.dt
                U_current = self.bc_applicator.enforce_values(
                    U_current, boundary_conditions=bc, spacing=(self.dx,), time=time
                )

            return U_current

        else:
            # nD DPP
            if U_next.ndim == 1:
                total_points = U_next.size
                expected_full = int(np.prod(self._grid_shape))
                grid_shape = (
                    tuple(self._grid_shape) if total_points == expected_full else tuple(n - 1 for n in self._grid_shape)
                )
                U_next_shaped = U_next.reshape(grid_shape)
                M_next_shaped = M_next.reshape(grid_shape)
            else:
                U_next_shaped = U_next
                M_next_shaped = M_next
                grid_shape = U_next_shaped.shape

            U_star = np.zeros_like(U_next_shaped)

            for multi_idx in np.ndindex(grid_shape):
                x_current = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
                m_current = M_next_shaped[multi_idx]

                _bnd_min, _bnd_max = self.grid.get_bounds()  # Issue #1056: uniform accessor

                def dpp_cost_nd(alpha_vec, _xc=x_current, _mc=m_current, _lo=_bnd_min, _hi=_bnd_max):
                    x_next = _xc + alpha_vec * dt
                    # Clip to domain bounds (per-axis; the prior self.grid.bounds[0][d] mis-indexed
                    # the ad-hoc .bounds shape for d>=1 -- latent nD bug, Issue #1056).
                    for d in range(self.dimension):
                        x_next[d] = np.clip(x_next[d], _lo[d], _hi[d])
                    u_next = self._interpolate_value(U_next_shaped, x_next)
                    L_val = float(L_class(_xc, alpha_vec, _mc, t_value))
                    return dt * L_val + u_next

                alpha0 = np.zeros(self.dimension)
                result = minimize(
                    dpp_cost_nd,
                    alpha0,
                    bounds=[bounds] * self.dimension,
                    method="L-BFGS-B",
                    options={"ftol": self.tolerance, "maxiter": 100},
                )

                U_star[multi_idx] = result.fun if result.success else self._interpolate_value(U_next_shaped, x_current)

            U_current_shaped = self._apply_diffusion(U_star, dt)
            U_current_shaped = self._enforce_boundary_conditions(U_current_shaped)

            return U_current_shaped.ravel() if U_next.ndim == 1 else U_current_shaped

    def _apply_boundary_to_point(self, x: float) -> float:
        """Apply 1D boundary handling to a single point (reflect or clip)."""
        bounds = self.problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]

        bc = self.get_boundary_conditions()
        bc_type = _checked_bc_type_string(bc)
        bc_op = bc_type_to_geometric_operation(bc_type)

        return apply_boundary_conditions_1d(x, xmin=xmin, xmax=xmax, bc_type=bc_op)

    @property
    def _use_dpp(self) -> bool:
        """Whether to use L-based DPP instead of H-based characteristics.

        Uses DPP when:
        1. problem.lagrangian_class is available, AND
        2. Either no hamiltonian_class, or the Lagrangian's control cost is non-smooth
        """
        L_class = self.problem.lagrangian_class
        if L_class is None:
            return False

        # If no H available, DPP is the only option
        H_class = self.problem.hamiltonian_class
        if H_class is None:
            return True

        # If H is non-smooth, prefer DPP (avoids grad_u at kinks)
        is_smooth = getattr(H_class, "is_smooth", lambda: True)
        return bool(callable(is_smooth) and not is_smooth())

    def _characteristic_foot_velocity(self, x: np.ndarray, m: np.ndarray, p: np.ndarray, t: float) -> np.ndarray:
        """Characteristic velocity ``dH/dp`` over a batch of nodes (Issue #1547 / RFC #1574 Phase 1).

        The departure foot of the semi-Lagrangian step is ``x - dt * dH/dp``. Every foot site in
        this solver used to hardcode ``dH/dp = p/lambda`` -- the QUADRATIC-control-cost form -- while
        the Lax-Oleinik value term routed through ``eval_H_batch`` and evaluated the true ``H``. For
        any Hamiltonian whose real ``dH/dp`` differs (multiplicative congestion
        ``dH/dp = p/(lambda*c(m))``, additive congestion, a non-quadratic control cost) that hybrid
        is not a discretization of the requested PDE: it solves without complaint and converges to
        the WRONG LIMIT. On a frozen-density pure-HJB refinement against ``HJBFDMSolver`` the stock
        error plateaued at ``max|SL-FDM| ~ 0.206`` from Nx=51 to Nx=201 instead of decreasing.

        ``dH/dp`` now comes from the same single source the FDM and GFDM solvers already use
        (``eval_dH_dp_batch`` -> ``HamiltonianBase.evaluate_dp`` -> ``H_class.dp``), so the
        characteristic velocity has one owner across the whole HJB family rather than a private
        quadratic copy here. That also makes the fix self-extending: a hand-rolled
        ``HamiltonianBase`` subclass that overrides ``dp()`` is honored automatically, with no
        marker attribute to spell correctly.

        Byte-identity at the quadratic case: for ``SeparableHamiltonian(QuadraticControlCost)``,
        ``dp`` is ``control_cost.dp(p) == p / lambda_`` and ``_control_cost_lambda()`` returns
        ``float(control_cost.lambda_)`` -- the same Python float -- so this returns the same array,
        elementwise, as the expression it replaces.

        Args:
            x: Node positions, shape ``(N, d)``.
            m: Density at those nodes, shape ``(N,)``.
            p: Momentum ``grad(u)`` at those nodes, shape ``(N, d)``.
            t: Time at which to evaluate.

        Returns:
            ``dH/dp``, shape ``(N, d)``.
        """
        H_class = self.problem.hamiltonian_class
        if H_class is None:
            # Legacy no-Hamiltonian LQ path: p/lambda IS the definition of the scheme there, and
            # problem.lambda_ is the only available source (Issue #1071). The value term fail-louds
            # separately if this path is reached without any usable H.
            return np.asarray(p, dtype=float) / self._control_cost_lambda()

        vel = np.asarray(eval_dH_dp_batch(H_class, x, m, p, t), dtype=float)
        p_arr = np.asarray(p, dtype=float)
        if vel.size != p_arr.size:
            raise ValueError(
                f"{type(H_class).__name__}.dp() returned shape {vel.shape} for momentum of shape "
                f"{p_arr.shape}; the semi-Lagrangian characteristic foot needs one dH/dp component "
                f"per momentum component. Fix the Hamiltonian's dp() to return the documented "
                f"(N, d) batch shape (Issue #1547 / RFC #1574)."
            )
        return vel.reshape(p_arr.shape)

    def _nd_foot_velocity_field(
        self,
        grid_shape: tuple[int, ...],
        grad_components: Sequence[np.ndarray],
        m_shaped: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """``dH/dp`` on every node of an nD grid, shaped ``grid_shape + (d,)`` (Issue #1547).

        Batches the per-node ``dH/dp`` the nD advection loops need into ONE Hamiltonian call. The
        coordinate array is built from ``self.grid.coordinates`` truncated to ``grid_shape`` per
        axis, which reproduces exactly the ``self.grid.coordinates[ax][multi_idx[ax]]`` lookup the
        loops perform -- including the reduced ``n-1`` shape the periodic nD path uses.
        """
        d = self.dimension
        coord_axes = [np.asarray(self.grid.coordinates[ax], dtype=float)[: grid_shape[ax]] for ax in range(d)]
        mesh = np.meshgrid(*coord_axes, indexing="ij")
        x_flat = np.stack([mesh[ax].ravel() for ax in range(d)], axis=1)
        p_flat = np.stack([np.asarray(grad_components[ax]).ravel() for ax in range(d)], axis=1)
        vel_flat = self._characteristic_foot_velocity(x_flat, np.asarray(m_shaped).ravel(), p_flat, t)
        return vel_flat.reshape((*tuple(grid_shape), d))

    def _trace_characteristic_backward(
        self, x_current: np.ndarray | float, p_optimal: np.ndarray | float, dt: float
    ) -> np.ndarray | float:
        """
        Trace characteristic backward in time to find departure point (supports 1D and nD).

        Delegates to hjb_sl_characteristics module functions.

        Args:
            x_current: Current spatial position
                - 1D: scalar float
                - nD: array of shape (dimension,), e.g., [x, y] for 2D
            p_optimal: Optimal control value
                - 1D: scalar float
                - nD: array of shape (dimension,), e.g., [px, py] for 2D
            dt: Time step size

        Returns:
            Departure point X(t-dt)
                - 1D: scalar float
                - nD: array of shape (dimension,)
        """
        if self.dimension == 1:
            # 1D characteristic tracing
            jax_fn = self._jax_solve_characteristic if self.use_jax else None
            x_departure = trace_characteristic_backward_1d(
                x_current,
                p_optimal,
                dt,
                method=self.characteristic_solver,
                use_jax=self.use_jax,
                jax_solve_fn=jax_fn,
                ode_rtol=self.ode_rtol,
                ode_atol=self.ode_atol,
            )

            # Apply boundary conditions
            # Issue #702: Use centralized bc_utils for consistent BC handling
            bc = self.get_boundary_conditions()
            bc_type = _checked_bc_type_string(bc)
            bc_op = bc_type_to_geometric_operation(bc_type)

            bounds = self.problem.geometry.get_bounds()
            xmin, xmax = bounds[0][0], bounds[1][0]
            return apply_boundary_conditions_1d(
                x_departure,
                xmin=xmin,
                xmax=xmax,
                bc_type=bc_op,
            )

        else:
            # nD characteristic tracing
            x_departure = trace_characteristic_backward_nd(
                x_current,
                p_optimal,
                dt,
                dimension=self.dimension,
                method=self.characteristic_solver,
                ode_rtol=self.ode_rtol,
                ode_atol=self.ode_atol,
            )

            # Issue #702: Use centralized bc_utils for consistent BC handling
            bc = self.get_boundary_conditions()
            bc_type = _checked_bc_type_string(bc)
            bc_op = bc_type_to_geometric_operation(bc_type)

            return apply_boundary_conditions_nd(
                x_departure,
                bounds=np.column_stack(self.grid.get_bounds()),  # Issue #1056: uniform accessor
                bc_type=bc_op,
            )

    def _interpolate_value(self, U_values: np.ndarray, x_query: np.ndarray | float) -> float:
        """
        Interpolate value function at query point (supports 1D and nD).

        Delegates to hjb_sl_interpolation module functions.

        Args:
            U_values: Value function on grid
                - 1D: shape (Nx,)
                - nD: shape matching grid.num_points, e.g., (Nx, Ny) for 2D
            x_query: Query point for interpolation
                - 1D: scalar float
                - nD: array of shape (dimension,), e.g., [x, y] for 2D

        Returns:
            Interpolated value at query point
        """
        if self.dimension == 1:
            # 1D interpolation
            jax_fn = self._jax_interpolate if self.use_jax else None
            bounds = self.problem.geometry.get_bounds()
            xmin, xmax = bounds[0][0], bounds[1][0]
            return interpolate_value_1d(
                U_values,
                x_query,
                self.x_grid,
                method=self.interpolation_method,
                xmin=xmin,
                xmax=xmax,
                use_jax=self.use_jax,
                jax_interpolate_fn=jax_fn,
            )

        else:
            # nD interpolation
            grid_coords = tuple(self.grid.coordinates)
            grid_shape = tuple(self._grid_shape)

            try:
                return interpolate_value_nd(
                    U_values,
                    x_query,
                    grid_coords,
                    grid_shape,
                    method=self.interpolation_method,
                )
            except Exception as e:
                logger.debug(f"nD interpolation failed at x={x_query}: {e}")

                # Try RBF fallback if enabled
                if self.use_rbf_fallback:
                    try:
                        return interpolate_value_rbf_fallback(
                            U_values,
                            x_query,
                            grid_coords,
                            grid_shape,
                            rbf_kernel=self.rbf_kernel,
                        )
                    except Exception as rbf_error:
                        logger.debug(f"RBF fallback failed: {rbf_error}")

                # Final fallback: nearest neighbor
                return interpolate_nearest_neighbor(
                    U_values,
                    x_query,
                    grid_coords,
                    grid_shape,
                )

    def _compute_diffusion_term(self, U_values: np.ndarray, idx: int | tuple) -> float:
        """
        Compute discrete Laplacian (diffusion term) at grid point (supports 1D and nD).

        1D: Uses standard finite difference (U[i+1] - 2*U[i] + U[i-1]) / dx²
        nD: Computes Laplacian as sum over dimensions: Δu = Σ_d ∂²u/∂x_d²

        Args:
            U_values: Value function array
                - 1D: shape (Nx,)
                - nD: shape matching grid.num_points
            idx: Grid point index
                - 1D: scalar integer i
                - nD: tuple of indices, e.g., (i, j) for 2D

        Returns:
            Discrete Laplacian value
        """
        if self.dimension == 1:
            # 1D Laplacian: Use existing logic
            i = int(idx)
            Nx = len(U_values)

            if Nx <= 2:
                return 0.0

            # Handle boundary points - get BC type once
            # Issue #545: Use centralized BC retrieval (NO hasattr)
            bc = self.get_boundary_conditions()
            bc_type = self._get_bc_type_string(bc)

            if i == 0:
                if bc_type == "periodic":
                    laplacian = (U_values[1] - 2 * U_values[0] + U_values[-1]) / self.dx**2
                else:
                    laplacian = (U_values[1] - U_values[0]) / self.dx**2

            elif i == Nx - 1:
                if bc_type == "periodic":
                    laplacian = (U_values[0] - 2 * U_values[-1] + U_values[-2]) / self.dx**2
                else:
                    laplacian = (U_values[-1] - U_values[-2]) / self.dx**2

            else:
                # Central difference for interior points
                laplacian = (U_values[i + 1] - 2 * U_values[i] + U_values[i - 1]) / self.dx**2

            return laplacian

        else:
            # nD Laplacian: Sum of second derivatives in each dimension
            # Δu = ∂²u/∂x₁² + ∂²u/∂x₂² + ...

            # Ensure U_values is reshaped to grid shape
            if U_values.ndim == 1:
                U_shaped = U_values.reshape(self._grid_shape)
            else:
                U_shaped = U_values

            # Get multi-index
            if isinstance(idx, (tuple, list)):
                multi_idx = tuple(idx)
            else:
                # Convert flat index to multi-index
                multi_idx = self.grid.get_multi_index(int(idx))

            laplacian = 0.0

            # Compute second derivative in each dimension
            for d in range(self.dimension):
                # Check if we're at a boundary in this dimension
                at_lower_bound = multi_idx[d] == 0
                at_upper_bound = multi_idx[d] == self._grid_shape[d] - 1

                # Create index tuples for neighbors
                idx_center = list(multi_idx)
                idx_plus = list(multi_idx)
                idx_minus = list(multi_idx)

                if at_lower_bound or at_upper_bound:
                    # Boundary: use one-sided difference (assume Neumann BC)
                    if at_lower_bound:
                        idx_plus[d] = multi_idx[d] + 1
                        u_center = U_shaped[tuple(idx_center)]
                        u_plus = U_shaped[tuple(idx_plus)]
                        # One-sided: (u_plus - u_center) / dx²
                        second_deriv = (u_plus - u_center) / self.dx[d] ** 2
                    else:  # at_upper_bound
                        idx_minus[d] = multi_idx[d] - 1
                        u_center = U_shaped[tuple(idx_center)]
                        u_minus = U_shaped[tuple(idx_minus)]
                        # One-sided: (u_center - u_minus) / dx²
                        second_deriv = (u_center - u_minus) / self.dx[d] ** 2

                else:
                    # Interior: central difference
                    idx_plus[d] = multi_idx[d] + 1
                    idx_minus[d] = multi_idx[d] - 1

                    u_center = U_shaped[tuple(idx_center)]
                    u_plus = U_shaped[tuple(idx_plus)]
                    u_minus = U_shaped[tuple(idx_minus)]

                    # Central: (u_plus - 2*u_center + u_minus) / dx²
                    second_deriv = (u_plus - 2 * u_center + u_minus) / self.dx[d] ** 2

                laplacian += second_deriv

            return float(laplacian)

    def _sl_value_update(
        self,
        u_at_foot: np.ndarray | float,
        x: np.ndarray,
        m: np.ndarray | float,
        p: np.ndarray,
        t: float,
        dt: float,
    ) -> np.ndarray | float:
        """Consistent semi-Lagrangian value update (Issue #1413).

        For the backward HJB ``-∂_t u + H(x, ∇u, m) = 0`` with a separable
        ``H = H_control(p) + V(x) + f(m)``, the Lax-Oleinik / DPP step is

            u^n(x) = u^{n+1}(x - dt·∂_pH) + dt·H_control(p) - dt·(V + f),

        where the departure foot ``x - dt·∂_pH`` carries the advection (``∂_pH`` is the
        characteristic velocity, ``= p/λ`` for the quadratic control cost). With the
        departure value ``u_at_foot = u^{n+1}(foot)`` already interpolated and
        ``V + f = H(x, m, p=0)``:

            u^n = u_at_foot + dt·(H(p) - H(0)) - dt·H(0) = u_at_foot + dt·(H(p) - 2·H(0)).

        This replaces the prior ``u_at_foot - dt·H(p)`` (with a non-λ-scaled foot), which
        double-counted the kinetic term (≈3× too large) and was λ=1-only on the foot —
        ~24% off the analytic Hopf-Lax solution even at λ=1 (FDM matches it to 0.6%). See
        Issue #575 (which corrected the state term but left the kinetic error) and Issue
        #1413. ``H`` comes from the single source (``eval_H_batch`` → ``hamiltonian_class``).
        """
        H_class = self.problem.hamiltonian_class
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        H_full = np.asarray(eval_H_batch(H_class, x, m, p_arr, t), dtype=float)
        H_state = np.asarray(eval_H_batch(H_class, x, m, np.zeros_like(p_arr), t), dtype=float)
        return u_at_foot + dt * (H_full - 2.0 * H_state)

    def _evaluate_hamiltonian(self, x: np.ndarray | float, p: np.ndarray | float, m: float, time_idx: int) -> float:
        """
        Evaluate Hamiltonian H(x, p, m) at given point (supports 1D and nD).

        Uses DerivativeTensors for consistency with all solvers.
        See archon-notes/development/guides/NAMING_CONVENTIONS.md (mfg-research, private) "Derivative Tensor Standard" section.

        Args:
            x: Spatial position
                - 1D: scalar float
                - nD: array of shape (dimension,)
            p: Control/momentum value (gradient ∇u)
                - 1D: scalar float
                - nD: array of shape (dimension,)
            m: Density value
            time_idx: Time index

        Returns:
            Hamiltonian value
        """
        # Compute time value
        t_value = time_idx * self.problem.T / self.problem.Nt if time_idx is not None else 0.0

        # Issue #902/#1071: route the H value through the single-source batch shim
        # (eval_H_batch -> HamiltonianBase.evaluate_H) rather than calling H_class
        # directly, so the batch-call glue (dtype/shape) has one home — matching the
        # batch SL paths above. Byte-identical: evaluate_H is np.asarray(H_class(...),
        # dtype=float), so float() of it equals the prior float(H_class(...)).
        H_class = self.problem.hamiltonian_class
        if H_class is not None:
            x_vec = np.atleast_1d(x)
            p_vec = np.atleast_1d(p)
            return float(eval_H_batch(H_class, x_vec, m, p_vec, t_value))

        # Legacy fallbacks for problems without hamiltonian_class
        derivs = self._build_derivative_tensors(p)
        x_idx = self._position_to_index(x)

        try:
            return self.problem.H(x_idx, m, derivs=derivs, t_idx=time_idx)
        except AttributeError:
            pass

        try:
            return self.problem.hamiltonian(x_idx, m, derivs=derivs, t=t_value)
        except (AttributeError, TypeError):
            pass

        try:
            return self.problem.hamiltonian(np.atleast_1d(x), m, np.atleast_1d(p), t_value)
        except (AttributeError, TypeError) as e:
            logger.debug(f"Legacy Hamiltonian signature failed: {e}")

        # Issue #1071 / fail-fast: do NOT silently fall back to the LQ default
        # H = ½|p|² + C·m. That substitutes the WRONG physics for any non-LQ problem and
        # returns a plausible-but-incorrect solution with no error (the exact silent-fallback
        # class this codebase forbids). Fail loud and tell the caller to supply a Hamiltonian.
        raise ValueError(
            "HJB semi-Lagrangian: no Hamiltonian available (problem.hamiltonian_class is None "
            "and no legacy problem.H / problem.hamiltonian succeeded). Specify one explicitly, "
            "e.g. MFGComponents(hamiltonian=SeparableHamiltonian(...)). The solver will not "
            "silently substitute the LQ default H=0.5*|p|^2 + C*m (Issue #1071, fail-fast)."
        )

    def _build_derivative_tensors(self, p: np.ndarray | float) -> DerivativeTensors:
        """
        Build DerivativeTensors from gradient array/scalar.

        Args:
            p: Gradient value(s)
                - 1D: scalar float
                - nD: array of shape (dimension,)

        Returns:
            DerivativeTensors with gradient tensor
        """
        if self.dimension == 1:
            p_scalar = float(p) if np.ndim(p) > 0 else p
            grad = np.array([p_scalar])
        else:
            grad = np.atleast_1d(p).astype(float)

        return DerivativeTensors.from_gradient(grad)

    def _position_to_index(self, x: np.ndarray | float) -> int | tuple[int, ...]:
        """
        Convert spatial position to grid index.

        Args:
            x: Spatial position

        Returns:
            Grid index (int for 1D, tuple for nD)
        """
        # Issue #1056: uniform get_bounds() accessor (was a polymorphic .bounds-primary path with a
        # get_bounds() fallback; get_bounds() gives the same mins, so this is byte-identical).
        mins, _maxs = self.problem.geometry.get_bounds()
        grid_shape = self.problem.geometry.get_grid_shape()

        if self.dimension == 1:
            x_scalar = float(x) if np.ndim(x) > 0 else x
            xmin = mins[0]
            Nx = grid_shape[0] - 1
            # dx is scalar for 1D
            dx = self.dx if np.isscalar(self.dx) else self.dx[0]
            x_idx = int((x_scalar - xmin) / dx)
            return int(np.clip(x_idx, 0, Nx))
        else:
            x_vec = np.atleast_1d(x)
            indices = []
            for i in range(self.dimension):
                xmin_i = mins[i]
                Nx_i = grid_shape[i] - 1
                # dx is array for nD
                dx_i = self.dx[i]
                idx = int((x_vec[i] - xmin_i) / dx_i)
                indices.append(int(np.clip(idx, 0, Nx_i)))
            return tuple(indices)

    def _solve_crank_nicolson_diffusion(self, U_star: np.ndarray, dt: float, sigma: float) -> np.ndarray:
        """
        Solve diffusion step using Crank-Nicolson (unconditionally stable).

        Delegates to hjb_sl_adi.solve_crank_nicolson_diffusion_1d.

        Args:
            U_star: Intermediate solution after advection step
            dt: Time step size
            sigma: Diffusion coefficient

        Returns:
            Solution after implicit diffusion step
        """
        bc_op = self._get_diffusion_bc_type()
        return solve_crank_nicolson_diffusion_1d(U_star, dt, sigma, self.x_grid, bc_type=bc_op)

    def _get_diffusion_bc_type(self) -> str:
        """Get BC type string for diffusion step ('neumann' or 'periodic')."""
        bc = self.get_boundary_conditions()
        bc_type = _checked_bc_type_string(bc)
        if bc_type == "periodic":
            return "periodic"
        return "neumann"

    def _adi_diffusion_step(self, U_star: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply ADI (Alternating Direction Implicit) diffusion for nD grids.

        Delegates to hjb_sl_adi.adi_diffusion_step.

        Args:
            U_star: Intermediate solution after advection step, shape (N1, N2, ..., Nd)
            dt: Time step size

        Returns:
            Solution after ADI diffusion step, same shape as U_star
        """
        if self.dimension == 1:
            # For 1D, use standard Crank-Nicolson
            return self._solve_crank_nicolson_diffusion(U_star, dt, self.problem.sigma)

        bc_op = self._get_diffusion_bc_type()
        return adi_diffusion_step(
            U_star,
            dt,
            self.problem.sigma,
            self.dx,
            tuple(self._grid_shape),
            bc_type=bc_op,
        )

    def _apply_diffusion(self, U_star: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply diffusion step using the configured method.

        Args:
            U_star: Solution after advection step
            dt: Time step size

        Returns:
            Solution after diffusion step
        """
        if self.diffusion_method == "none":
            # No diffusion - just return advected solution
            return U_star

        elif self.diffusion_method == "explicit":
            # Explicit Laplacian: u^n = u* + dt * σ²/2 * Δu*
            # Simple but requires small dt for stability (dt < dx²/(2*d*σ²))
            return self._explicit_diffusion_step(U_star, dt)

        elif self.diffusion_method == "stochastic":
            # Issue #1026: stochastic SL bypasses _apply_diffusion entirely; the
            # Carlini-Silva update bakes diffusion into the SL averaging step.
            # Reaching this branch indicates broken dispatch in
            # _solve_timestep_semi_lagrangian.
            raise NotImplementedError(
                "_apply_diffusion should not be called when diffusion_method='stochastic'. "
                "The Carlini-Silva 2014 SL update incorporates diffusion via 2d "
                "stochastic departure points, replacing the operator-splitting "
                "diffusion step. Check dispatch in _solve_timestep_semi_lagrangian."
            )

        elif self.diffusion_method == "canonical_cs":
            # Issue #1058: canonical CS (implicit-alpha* DPP) bakes diffusion into the
            # per-point DPP minimization via 2d stochastic departure points, just like
            # 'stochastic'. Reaching this branch indicates broken dispatch.
            raise NotImplementedError(
                "_apply_diffusion should not be called when diffusion_method='canonical_cs'. "
                "The canonical Carlini-Silva 2014 SL update (implicit-alpha* DPP) incorporates "
                "diffusion via 2d stochastic departure points, replacing the operator-splitting "
                "diffusion step. Check dispatch in _solve_timestep_semi_lagrangian."
            )

        else:  # "adi" (default)
            if self.dimension == 1:
                return self._solve_crank_nicolson_diffusion(U_star, dt, self.problem.sigma)
            else:
                return self._adi_diffusion_step(U_star, dt)

    def _explicit_diffusion_step(self, U_star: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply explicit diffusion step using discrete Laplacian.

        Uses central differences for Laplacian:
            Δu ≈ Σ_d (u_{i+1,d} - 2*u_i + u_{i-1,d}) / dx_d²

        Note: This is conditionally stable. Requires:
            dt < dx²/(2*d*σ²) where d is dimension

        Args:
            U_star: Solution after advection step
            dt: Time step size

        Returns:
            Solution after explicit diffusion step
        """
        sigma = self.problem.sigma
        sigma_sq_half = diffusion_from_volatility(sigma)

        if self.dimension == 1:
            dx = self.dx
            # 1D Laplacian with Neumann BC
            laplacian = np.zeros_like(U_star)
            laplacian[1:-1] = (U_star[2:] - 2 * U_star[1:-1] + U_star[:-2]) / dx**2
            # Neumann BC: du/dx = 0 at boundaries
            laplacian[0] = (U_star[1] - U_star[0]) / dx**2
            laplacian[-1] = (U_star[-2] - U_star[-1]) / dx**2
        else:
            # nD Laplacian
            laplacian = np.zeros_like(U_star)
            for d in range(self.dimension):
                dx_d = self.dx[d]
                # Second derivative along axis d
                laplacian += np.gradient(np.gradient(U_star, dx_d, axis=d), dx_d, axis=d)

        # Explicit update: u^n = u* + dt * σ²/2 * Δu*
        U_new = U_star + dt * sigma_sq_half * laplacian
        return U_new

    # ═══════════════════════════════════════════════════════════════════
    # BoundaryHandler Protocol (Issue #545)
    # ═══════════════════════════════════════════════════════════════════

    def get_boundary_indices(self) -> np.ndarray:
        """
        Identify boundary points in solver's discretization.

        Returns:
            Array of integer indices identifying boundary grid points.

        Note:
            For Semi-Lagrangian method, boundary points are those on the
            domain boundary where characteristic tracing requires clamping.

        Implementation:
            - 1D: First and last grid points [0, N-1]
            - nD: All points on boundary faces (any coordinate at min/max)
        """
        if self.dimension == 1:
            # 1D: First and last grid points
            Nx = len(self.x_grid)
            return np.array([0, Nx - 1], dtype=np.int64)
        else:
            # nD: All grid points on boundary faces
            # For tensor grid: point is on boundary if any coordinate is at min/max
            boundary_mask = np.zeros(self._grid_shape, dtype=bool)

            # Mark boundary faces in each dimension
            for d in range(self.dimension):
                # Lower boundary face (index 0 along dimension d)
                slices_lower = [slice(None)] * self.dimension
                slices_lower[d] = 0
                boundary_mask[tuple(slices_lower)] = True

                # Upper boundary face (index -1 along dimension d)
                slices_upper = [slice(None)] * self.dimension
                slices_upper[d] = -1
                boundary_mask[tuple(slices_upper)] = True

            # Return flat indices of boundary points
            return np.flatnonzero(boundary_mask.ravel())

    def apply_boundary_conditions(
        self,
        values: np.ndarray,
        bc: BoundaryConditions,
        time: float = 0.0,
    ) -> np.ndarray:
        """
        Apply boundary conditions to solution values.

        For Semi-Lagrangian method, BC enforcement is handled during
        characteristic tracing (clamping departure points to domain).
        This method is a no-op adapter for protocol compliance.

        Args:
            values: Solution values at all grid points
            bc: Boundary conditions object (from mfgarchon.geometry.boundary)
            time: Current time (unused for Semi-Lagrangian)

        Returns:
            Solution values (unchanged, BC enforced during characteristic tracing)

        Note:
            Semi-Lagrangian enforces BCs during characteristic tracing via
            hjb_sl_characteristics.apply_boundary_conditions_1d/nd(), not as
            a post-processing step. This method exists for protocol compliance.
        """
        # Semi-Lagrangian enforces BCs during characteristic tracing,
        # not as a post-processing step. Return values unchanged.
        return values

    def get_bc_type_for_point(self, point_idx: int) -> str:
        """
        Determine BC type for a specific grid point.

        Args:
            point_idx: Index of grid point

        Returns:
            BC type string: "periodic", "dirichlet", "neumann", or "none"

        Note:
            For Semi-Lagrangian solver with uniform BCs, returns the same
            BC type for all boundary points. Mixed BC support would require
            querying BC segments based on point spatial coordinates.
        """
        bc = self.get_boundary_conditions()
        if bc is None:
            return "none"

        # For uniform BC (most common case)
        bc_type_str = self._get_bc_type_string(bc)
        if bc_type_str is not None:
            return bc_type_str

        # For mixed BC, would need to query BC segments
        # (Semi-Lagrangian typically uses uniform BC)
        return "none"

    def get_solver_info(self) -> dict[str, Any]:
        """Return solver configuration information."""
        return {
            "method": "Semi-Lagrangian",
            "interpolation": self.interpolation_method,
            "optimization": self.optimization_method,
            "characteristic_solver": self.characteristic_solver,
            "use_jax": self.use_jax,
            "tolerance": self.tolerance,
            "max_iterations": self.max_char_iterations,
            "adaptive_substepping": self.enable_adaptive_substepping,
            "max_substeps": self.max_substeps,
            "cfl_target": self.cfl_target,
        }


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBSemiLagrangianSolver...")

    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    def _smoke_components_1d():
        """MFGComponents for the 1D smoke tests."""
        return MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
        )

    def _smoke_components_2d():
        """MFGComponents for the 2D smoke tests."""
        return MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
        )

    # Test 1: Solver initialization
    print("\n1. Testing solver initialization...")
    geometry_1d = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[51],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    problem = MFGProblem(
        geometry=geometry_1d,
        T=1.0,
        Nt=100,
        diffusion=0.5 * 0.1**2,
        components=_smoke_components_1d(),
    )
    solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear", optimization_method="brent")

    assert solver.dimension == 1
    assert solver.hjb_method_name == "Semi-Lagrangian"
    assert solver.interpolation_method == "linear"
    print("   1D solver initialization: OK")

    # Test 2: 1D Crank-Nicolson diffusion (used by 1D solver)
    print("\n2. Testing 1D Crank-Nicolson diffusion...")
    # Create a smooth test function (Gaussian)
    x = np.linspace(0, 1, 51)
    U_test = np.exp(-50 * (x - 0.5) ** 2)

    # Apply diffusion for one timestep
    dt = 0.01
    sigma = 0.1
    U_diffused = solver._solve_crank_nicolson_diffusion(U_test, dt, sigma)

    assert U_diffused.shape == U_test.shape
    assert not np.any(np.isnan(U_diffused))
    assert not np.any(np.isinf(U_diffused))
    # Diffusion should smooth the peak
    assert U_diffused.max() < U_test.max()
    print(f"   Peak before diffusion: {U_test.max():.4f}")
    print(f"   Peak after diffusion: {U_diffused.max():.4f}")
    print("   1D Crank-Nicolson: OK")

    # Test 3: 2D solver initialization with ADI compatibility check
    print("\n3. Testing 2D solver with ADI...")

    geometry_2d = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[20, 20],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    problem_2d = MFGProblem(
        geometry=geometry_2d,
        T=0.5,
        Nt=50,
        diffusion=0.5 * 0.1**2,
        components=_smoke_components_2d(),
    )

    solver_2d = HJBSemiLagrangianSolver(problem_2d, interpolation_method="linear")

    assert solver_2d.dimension == 2
    # Issue #545: Direct attribute access in test code (will raise AttributeError if missing)
    assert solver_2d._adi_compatible  # Scalar sigma should be ADI compatible
    print(f"   ADI compatible: {solver_2d._adi_compatible}")
    print("   2D solver initialization: OK")

    # Test 4: ADI diffusion step directly
    print("\n4. Testing ADI diffusion step...")
    # Create 2D Gaussian test function
    grid_shape = tuple(solver_2d._grid_shape)
    x = np.linspace(0, 1, grid_shape[0])
    y = np.linspace(0, 1, grid_shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    U_2d_test = np.exp(-50 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

    # Apply ADI diffusion
    U_2d_diffused = solver_2d._adi_diffusion_step(U_2d_test, dt=0.01)

    assert U_2d_diffused.shape == U_2d_test.shape
    assert not np.any(np.isnan(U_2d_diffused))
    assert not np.any(np.isinf(U_2d_diffused))
    # Diffusion should smooth the peak
    assert U_2d_diffused.max() < U_2d_test.max()
    print(f"   Peak before ADI diffusion: {U_2d_test.max():.4f}")
    print(f"   Peak after ADI diffusion: {U_2d_diffused.max():.4f}")
    print("   ADI diffusion step: OK")

    # Test 5: ADI preserves mass (integral)
    print("\n5. Testing ADI mass conservation...")
    dx_2d = solver_2d.dx  # Grid spacing array
    mass_before = np.sum(U_2d_test) * dx_2d[0] * dx_2d[1]
    mass_after = np.sum(U_2d_diffused) * dx_2d[0] * dx_2d[1]
    mass_error = abs(mass_after - mass_before) / mass_before
    print(f"   Mass before: {mass_before:.6f}")
    print(f"   Mass after: {mass_after:.6f}")
    print(f"   Relative error: {mass_error:.2e}")
    # With Neumann BC, mass should be approximately conserved
    assert mass_error < 0.05, f"Mass error too large: {mass_error}"
    print("   Mass conservation: OK")

    # Test 6: ADI with anisotropic sigma (diagonal tensor) - SKIPPED
    # The diagonal-sigma API was changed to require a spatial field (Nx, Ny) or
    # (Nt, Nx, Ny) rather than a per-axis vector (d,). Restoring this smoke test
    # requires migrating the volatility-field API; out of scope here.
    print("\n6. Anisotropic ADI smoke test SKIPPED (volatility-field API change)")

    # Test 7: BoundaryHandler protocol compliance (Issue #545)
    print("\n7. Testing BoundaryHandler protocol...")
    from mfgarchon.geometry.boundary import validate_boundary_handler

    # Validate protocol compliance
    assert validate_boundary_handler(solver), "1D solver should implement BoundaryHandler"
    assert validate_boundary_handler(solver_2d), "2D solver should implement BoundaryHandler"
    print("   Protocol validation: OK")

    # Test get_boundary_indices()
    boundary_indices_1d = solver.get_boundary_indices()
    assert len(boundary_indices_1d) == 2, "1D should have 2 boundary points"
    assert boundary_indices_1d[0] == 0, "First boundary point should be 0"
    assert boundary_indices_1d[-1] == 50, "Last boundary point should be Nx-1"
    print(f"   1D boundary indices: {boundary_indices_1d}")

    boundary_indices_2d = solver_2d.get_boundary_indices()
    assert len(boundary_indices_2d) > 0, "2D should have boundary points"
    print(f"   2D boundary count: {len(boundary_indices_2d)}")

    # Test get_bc_type_for_point()
    bc_type = solver.get_bc_type_for_point(0)
    valid_bc_types = ["periodic", "dirichlet", "neumann", "no_flux", "robin", "none"]
    assert bc_type in valid_bc_types, f"Invalid BC type: {bc_type}"
    print(f"   BC type for point 0: {bc_type}")

    # Test apply_boundary_conditions() (no-op adapter)
    U_test_1d = np.ones(51)
    U_result = solver.apply_boundary_conditions(U_test_1d, None)
    assert np.array_equal(U_result, U_test_1d), "apply_boundary_conditions should be no-op for Semi-Lagrangian"
    print("   apply_boundary_conditions (no-op): OK")

    print("   BoundaryHandler protocol: OK")

    # Test 8: Gradient clipping (Issue #583)
    print("\n8. Testing gradient clipping (Issue #583)...")
    solver_clipped = HJBSemiLagrangianSolver(
        problem,
        gradient_clip_threshold=1e6,
        enable_gradient_monitoring=True,
    )
    assert solver_clipped.gradient_clip_threshold == 1e6, "Clip threshold not set"
    assert solver_clipped.enable_gradient_monitoring, "Monitoring not enabled"
    print(f"   Clip threshold: {solver_clipped.gradient_clip_threshold:.0e}")

    # Test 1D clipping
    test_grad = np.array([1e5, 2e6, 5e5, 3e6, 1e4])  # 2e6, 3e6 exceed threshold
    solver_clipped._reset_gradient_stats()
    clipped_grad = solver_clipped._clip_gradient_with_monitoring(test_grad, t_idx=0, m_density=np.ones(5))
    assert np.max(np.abs(clipped_grad)) <= 1e6, "1D clipping failed"
    assert solver_clipped.gradient_stats["count"] == 2, "Expected 2 clips"
    print(f"   1D clipping: {solver_clipped.gradient_stats['count']} points clipped, OK")

    # Test 2D clipping with direction preservation (use 2D solver)
    solver_clipped_2d = HJBSemiLagrangianSolver(
        problem_2d,  # Use 2D problem
        gradient_clip_threshold=1e6,
        enable_gradient_monitoring=True,
    )
    grid_shape = (5, 5)
    grad_x = np.ones(grid_shape) * 1e5
    grad_y = np.ones(grid_shape) * 1e5
    grad_x[2, 2] = 3e6
    grad_y[2, 2] = 4e6  # Norm = 5e6 at (2,2)
    solver_clipped_2d._reset_gradient_stats()
    clipped_x, clipped_y = solver_clipped_2d._clip_gradient_with_monitoring(
        (grad_x, grad_y), t_idx=0, m_density=np.ones(grid_shape)
    )
    norm_after = np.sqrt(clipped_x[2, 2] ** 2 + clipped_y[2, 2] ** 2)
    assert norm_after <= 1e6 + 1e-6, "2D clipping failed"
    # Check direction preserved (3:4 ratio)
    assert abs(clipped_x[2, 2] / clipped_y[2, 2] - 0.75) < 1e-10, "Direction not preserved"
    print("   2D clipping with direction preservation: OK")

    # Test no-clip path
    solver_no_clip = HJBSemiLagrangianSolver(problem, gradient_clip_threshold=None)
    result = solver_no_clip._clip_gradient_with_monitoring(test_grad)
    assert np.array_equal(result, test_grad), "No-clip path failed"
    print("   No-clip path (threshold=None): OK")

    print("   Gradient clipping (Issue #583): OK")

    # Test 9: Carlini-Silva stochastic SL (Issue #1026)
    print("\n9. Testing Carlini-Silva stochastic SL (Issue #1026)...")

    # 9a: linear + stochastic is the canonical Carlini-Silva 2014 combo — accepted (Issue #1049)
    solver_linear_cs = HJBSemiLagrangianSolver(
        problem, interpolation_method="linear", diffusion_method="stochastic", check_cfl=False
    )
    assert solver_linear_cs.interpolation_method == "linear"
    assert solver_linear_cs.diffusion_method == "stochastic"
    print("   9a: linear + stochastic accepted (canonical CS 2014, Issue #1049): OK")

    # 9a': cubic + stochastic is outside the CS 2014 monotone proof — warns, not rejected (Issue #1049/#1033)
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as _w:
        _warnings.simplefilter("always")
        HJBSemiLagrangianSolver(problem, interpolation_method="cubic", diffusion_method="stochastic", check_cfl=False)
    assert any(issubclass(rec.category, UserWarning) for rec in _w), (
        "Expected UserWarning for cubic + stochastic (outside CS 2014 proof)"
    )
    print("   9a': cubic + stochastic warns (outside CS 2014 proof): OK")

    # 9b: cubic + stochastic instantiates and dispatches correctly
    solver_cs = HJBSemiLagrangianSolver(
        problem,
        interpolation_method="cubic",
        diffusion_method="stochastic",
        check_cfl=False,
    )
    assert solver_cs.diffusion_method == "stochastic"
    print("   9b: cubic + stochastic instantiated: OK")

    # 9c: _apply_diffusion raises NotImplementedError under stochastic
    try:
        solver_cs._apply_diffusion(np.zeros(11), 0.01)
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError:
        print("   9c: _apply_diffusion raises under stochastic: OK")

    # 9d: constant terminal -> constant solution (no-drift, no-curvature sanity)
    from mfgarchon.core.hamiltonian import HamiltonianBase, OptimizationSense

    class _ZeroH(HamiltonianBase):
        def __init__(self):
            super().__init__(sense=OptimizationSense.MINIMIZE)

        def __call__(self, x, m, p, t=0.0):
            p_arr = np.atleast_1d(np.asarray(p, dtype=float))
            return np.zeros(p_arr.shape[:-1]) if p_arr.ndim > 0 else 0.0

        def gradient_p(self, x, m, p, t=0.0):
            return np.zeros_like(np.asarray(p, dtype=float))

        def density_derivative(self, x, m, p, t=0.0):
            return 0.0

    grid_const = TensorProductGrid(
        dimension=1,
        bounds=[(-1.0, 1.0)],
        Nx_points=[51],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    components_const = MFGComponents(
        hamiltonian=_ZeroH(),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 1.0,
    )
    problem_const = MFGProblem(
        geometry=grid_const,
        T=0.1,
        Nt=20,
        diffusion=0.045,
        components=components_const,
    )
    solver_const = HJBSemiLagrangianSolver(
        problem_const,
        interpolation_method="cubic",
        diffusion_method="stochastic",
        check_cfl=False,
    )
    U_T_const = np.ones(51)
    M_dummy = np.ones((problem_const.Nt + 1, 51))
    U_const = solver_const.solve_hjb_system(
        M_density=M_dummy,
        U_terminal=U_T_const,
        U_coupling_prev=np.zeros((problem_const.Nt + 1, 51)),
    )
    assert np.allclose(U_const[0], 1.0, atol=1e-10), (
        f"Constant terminal not preserved: U[0] range [{U_const[0].min()}, {U_const[0].max()}]"
    )
    print("   9d: constant terminal preserved (H=0, sigma>0): OK")

    # 9e: stochastic and ADI produce numerically equivalent results on H=0
    #     Gaussian backward heat. Both schemes solve the same equation;
    #     only their discretization paths differ.
    sigma_test = 0.3
    T_test = 0.5
    beta_T = 1.0
    N_test, Nt_test = 100, 200
    grid_g = TensorProductGrid(
        dimension=1,
        bounds=[(-5.0, 5.0)],
        Nx_points=[N_test + 1],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    x_g = grid_g.get_spatial_grid().flatten()
    components_g = MFGComponents(
        hamiltonian=_ZeroH(),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: float(np.exp(-(x[0] ** 2) / (2 * beta_T)) / np.sqrt(2 * np.pi * beta_T)),
    )
    problem_g = MFGProblem(
        geometry=grid_g,
        T=T_test,
        Nt=Nt_test,
        diffusion=sigma_test**2 / 2,
        components=components_g,
    )
    U_T_g = np.exp(-(x_g**2) / (2 * beta_T)) / np.sqrt(2 * np.pi * beta_T)
    M_dummy_g = np.ones((Nt_test + 1, N_test + 1))
    solver_st = HJBSemiLagrangianSolver(
        problem_g,
        interpolation_method="cubic",
        diffusion_method="stochastic",
        check_cfl=False,
    )
    solver_adi = HJBSemiLagrangianSolver(
        problem_g,
        interpolation_method="cubic",
        diffusion_method="adi",
        check_cfl=False,
    )
    U_st = solver_st.solve_hjb_system(
        M_density=M_dummy_g,
        U_terminal=U_T_g,
        U_coupling_prev=np.zeros((Nt_test + 1, N_test + 1)),
    )
    U_adi = solver_adi.solve_hjb_system(
        M_density=M_dummy_g,
        U_terminal=U_T_g,
        U_coupling_prev=np.zeros((Nt_test + 1, N_test + 1)),
    )
    discrepancy = np.max(np.abs(U_st[0] - U_adi[0]))
    assert discrepancy < 5e-3, f"Stochastic and ADI diverge on H=0 Gaussian: max diff = {discrepancy:.3e}"
    print(f"   9e: stochastic vs ADI on H=0 Gaussian, max diff = {discrepancy:.3e}: OK")

    print("   Carlini-Silva stochastic SL (Issue #1026): OK")

    print("\nAll smoke tests passed!")

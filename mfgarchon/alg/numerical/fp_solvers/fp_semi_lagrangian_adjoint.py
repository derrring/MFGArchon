"""
Adjoint Semi-Lagrangian Fokker-Planck Solver for Mean Field Games.

This module implements the Forward (Adjoint) Semi-Lagrangian method for solving
the Fokker-Planck equation. This is the mathematically correct dual to the
Backward SL used for HJB, ensuring discrete duality for MFG convergence.

The FP equation solved is (divergence form):
    dm/dt + div(alpha * m) = sigma^2/2 * Laplacian(m)    in [0,T] x Omega
    m(0, x) = m0(x)                                       at t = 0

Key differences from Backward SL:
- **Forward tracing**: x_dest = x + α*dt (where does mass go?)
- **Splatting**: Mass is scattered to destination cells (transpose of interpolation)
- **Conservative**: Mass conservation is exact by construction
- **Duality**: Preserves ∫ m (S φ) dx = ∫ (S* m) φ dx with HJB operator S

References:
    - Carlini & Silva (2014): Semi-Lagrangian schemes for MFG
    - The discrete FP operator is M^{n+1} = I_{interp}^T @ M^n where I_{interp}
      is the interpolation matrix used in HJB.

Issue #578: Adjoint SL implementation for proper SL-SL MFG coupling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from scipy.linalg import solve_banded

from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import (
    adi_diffusion_step,
    solve_crank_nicolson_diffusion_1d,
)
from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import (
    apply_boundary_conditions_1d,
)
from mfgarchon.geometry.boundary.bc_utils import (
    bc_type_to_geometric_operation,
    get_bc_type_string,
)
from mfgarchon.geometry.boundary.types import BCType
from mfgarchon.utils.deprecation import deprecated, deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility, fp_drift_coefficient

from .base_fp import BaseFPSolver, DriftConvention
from .fp_sl_splatting import splat_1d, splat_nd

if TYPE_CHECKING:
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import BoundaryConditions

logger = get_logger(__name__)


class FPSLSolver(BaseFPSolver):
    """
    Forward Semi-Lagrangian solver for Fokker-Planck equations.

    This is the recommended FP solver for use with HJB Semi-Lagrangian solvers,
    as it provides discrete adjoint consistency (Issue #710).

    The Forward SL method asks "Where does mass go?" and scatters (splats) mass
    to destination cells. This is the adjoint of the Backward SL interpolation
    used in HJB solvers, ensuring discrete duality for MFG.

    Algorithm (operator splitting):
        1. Advection: Forward trace x_dest = x + α*dt, scatter mass via splatting
        2. Diffusion: Crank-Nicolson implicit solve

    Key Properties:
        - Mass conservation is exact (scatter weights sum to 1)
        - Density peaks form naturally from converging flow
        - Discrete duality with HJB Backward SL is preserved
        - No Jacobian correction needed (conservation is intrinsic)

    Splatting Methods (Issue #708):
        - 'linear': 2-point stencil (1D) / 2^d corners (nD), preserves positivity
        - 'cubic': 4-point Catmull-Rom stencil, O(dx³) accuracy (1D only)
        - 'quintic': 6-point Lagrange stencil, O(dx⁵) accuracy (1D only)

    Important: The interpolation_method must match the HJB solver's method
    to maintain exact discrete adjoint consistency.

    Dimension support:
        - 1D: Full support with linear/cubic/quintic splatting
        - nD: Full support with linear splatting + ADI diffusion

    .. versionchanged:: 0.17.6
        Renamed from ``FPSLAdjointSolver`` to ``FPSLSolver`` (Issue #710).
        The old name is still available as a deprecated alias.
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.SL  # Forward SL (adjoint of HJB Backward SL)
    _drift_convention = DriftConvention.VALUE_FUNCTION  # Issue #1043: takes U via potential_field

    # BoundaryCapable protocol (Issue #1456): the CN/ADI diffusion sub-step is zero-flux
    # (no-flux / Neumann g=0) and the advection wraps for periodic; Dirichlet / Robin / absorbing
    # are silently collapsed to Neumann downstream, so they fail loud here instead.
    _SUPPORTED_BC_TYPES: frozenset = frozenset({BCType.NO_FLUX, BCType.NEUMANN, BCType.PERIODIC})

    @property
    def supported_bc_types(self) -> frozenset:
        """BC types this solver supports (BoundaryCapable protocol)."""
        return self._SUPPORTED_BC_TYPES

    def __init__(
        self,
        problem: MFGProblem,
        boundary_conditions: BoundaryConditions | None = None,
        interpolation_method: str = "linear",
    ):
        """
        Initialize Adjoint Semi-Lagrangian FP solver.

        Args:
            problem: MFG problem instance
            boundary_conditions: Optional boundary conditions override.
                If None, uses boundary conditions from problem.geometry.
                The advection step uses reflecting BC for mass conservation.
            interpolation_method: Splatting method (adjoint of interpolation)
                - 'linear': Linear splatting (fastest, preserves positivity)
                - 'cubic': Cubic splatting (O(dx³), may produce negatives)
                - 'quintic': Quintic splatting (O(dx⁵), may produce negatives)
                Must match the HJB solver's interpolation_method for adjoint consistency.
        """
        super().__init__(problem)
        self.fp_method_name = "Adjoint Semi-Lagrangian"

        # Detect problem dimension
        self.dimension = self._detect_dimension()  # Issue #633: Use inherited method

        # Validate interpolation method
        valid_methods_1d = ("linear", "cubic", "quintic")
        valid_methods_nd = ("linear",)  # Only linear for nD currently

        if self.dimension == 1:
            if interpolation_method not in valid_methods_1d:
                raise ValueError(
                    f"Unknown interpolation_method: {interpolation_method}. For 1D, use one of {valid_methods_1d}."
                )
        else:
            if interpolation_method not in valid_methods_nd:
                raise ValueError(f"For nD problems, only 'linear' splatting is supported. Got: {interpolation_method}.")
        self.interpolation_method = interpolation_method

        # Positivity-clip diagnostic state (Issue #1147 class). Reset per solve_fp_system call;
        # tracks whether the once-per-solve mass-injection warning has already fired.
        self._clip_warned = False

        # Precompute grid parameters (dimension-agnostic)
        self.dt = problem.dt

        if self.dimension == 1:
            # 1D problem: Use geometry API
            bounds = problem.geometry.get_bounds()
            self.xmin, self.xmax = bounds[0][0], bounds[1][0]
            Nx = problem.geometry.get_grid_shape()[0]
            self.x_grid = np.linspace(self.xmin, self.xmax, Nx)
            self.dx = problem.geometry.get_grid_spacing()[0]
            self.Nx = Nx
            # nD attributes set to None for 1D
            self.grid = None
            self.grid_shape = (Nx,)
            self.bounds = [(self.xmin, self.xmax)]
            self.spacing = np.array([self.dx])
            self.grid_coordinates = (self.x_grid,)
        else:
            # nD problem: Requires TensorProductGrid for per-axis coordinates
            from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

            if not isinstance(problem.geometry, TensorProductGrid):
                raise TypeError(
                    f"Multi-dimensional FP semi-Lagrangian adjoint requires TensorProductGrid. "
                    f"Got {type(problem.geometry).__name__} (dimension={self.dimension})"
                )

            # Grid shape and spacing
            self.grid_shape = problem.geometry.get_grid_shape()
            self.spacing = np.array(problem.geometry.get_grid_spacing())

            # Grid coordinates for each dimension
            self.grid_coordinates = tuple(problem.geometry.coordinates)

            # Domain bounds
            self.bounds = [(self.grid_coordinates[d][0], self.grid_coordinates[d][-1]) for d in range(self.dimension)]

            # Store grid reference
            self.grid = problem.geometry

            # 1D attributes set to None for nD
            self.x_grid = None
            self.xmin = None
            self.xmax = None
            self.dx = None
            self.Nx = None

            logger.info(
                f"FPSLSolver initialized for {self.dimension}D: shape={self.grid_shape}, spacing={self.spacing}"
            )

        # Boundary conditions
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            self.boundary_conditions = self._get_boundary_conditions_from_problem()
        # Issue #1456: fail loud now if the resolved BC requests a type this solver cannot honor
        # (Dirichlet/Robin would otherwise be silently collapsed to the zero-flux Neumann stencil).
        self._validate_bc_support(self.boundary_conditions)

    def _get_boundary_conditions_from_problem(self) -> BoundaryConditions | None:
        """Get boundary conditions from problem or geometry."""
        try:
            return self.problem.geometry.boundary_conditions
        except AttributeError:
            pass
        try:
            return self.problem.geometry.get_boundary_conditions()
        except AttributeError:
            pass
        return None

    def _get_bc_operation_type(self) -> str:
        """
        Get boundary operation type from boundary conditions.

        Issue #702: Uses centralized bc_utils for consistent BC handling.

        Returns:
            Geometric operation: 'reflect', 'clamp', or 'periodic'
        """
        bc_type = get_bc_type_string(self.boundary_conditions)
        return bc_type_to_geometric_operation(bc_type)

    def _get_diffusion_bc_type(self) -> str:
        """Return diffusion BC type for CN/ADI: 'periodic' or 'neumann'.

        Issue #1257, 2026-06-10 audit: FP-SL diffusion sub-step must use the
        same BC type as the advection sub-step so a periodic domain does not
        acquire a spurious Neumann seam-flux.  Mirrors HJB-SL
        _get_diffusion_bc_type() (hjb_semi_lagrangian.py:2238).
        """
        if self._get_bc_operation_type() == "periodic":
            return "periodic"
        return "neumann"

    @deprecated_parameter(param_name="drift_field", since="v0.18.6", replacement="potential_field")
    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        potential_field: np.ndarray | None = None,
        volatility_field: float | np.ndarray | None = None,
        show_progress: bool | None = None,
        # Deprecated parameters
        drift_field: np.ndarray | None = None,  # Deprecated: renamed to potential_field
    ) -> np.ndarray:
        """
        Solve FP system forward in time using Adjoint Semi-Lagrangian method.

        The Forward SL discretization uses mass splatting instead of interpolation:
            1. Forward trace: x_dest = x + α*dt
            2. Scatter mass to destination cells with linear weights
            3. Apply diffusion via Crank-Nicolson

        Args:
            M_initial: Initial density m0(x). Shape: (Nx,)
            potential_field: Value function U from HJB (potential for drift).
                - np.ndarray: Shape (Nt+1, Nx) - U values at each time step
                  The drift velocity is computed as alpha = -grad(U)
            drift_field: DEPRECATED. Renamed to potential_field.
            volatility_field: Optional volatility coefficient σ (SDE noise) override.
                Note: Internally converted to diffusion D = σ²/2 for FP equation.
            show_progress: Show progress bar during solve

        Returns:
            Density evolution M(t,x). Shape: (Nt+1, Nx)
        """
        if M_initial is None:
            raise ValueError("M_initial is required")

        # Handle deprecated drift_field -> potential_field (v0.18.6)
        if drift_field is not None:
            if potential_field is not None:
                raise ValueError("Cannot specify both potential_field and drift_field")
            potential_field = drift_field

        if potential_field is None:
            raise ValueError(
                "potential_field (value function U) is required for Adjoint SL FP. Pass the U solution from HJB solver."
            )

        # Handle volatility (Issue #717: unified API)
        if volatility_field is None:
            sigma = self.problem.sigma
        elif isinstance(volatility_field, (int, float)):
            sigma = float(volatility_field)
        else:
            raise NotImplementedError("Array/callable volatility_field not yet supported")

        # Determine number of time steps from potential_field
        Nt_points = potential_field.shape[0]

        # Allocate solution array (dimension-agnostic)
        if self.dimension == 1:
            M_shape = (Nt_points, self.Nx)
        else:
            M_shape = (Nt_points, *self.grid_shape)

        # Reset the once-per-solve positivity-clip warning (Issue #1147 class).
        self._clip_warned = False

        M = np.zeros(M_shape)
        M[0] = M_initial.copy().reshape(self.grid_shape if self.dimension > 1 else -1)

        # Progress bar
        from mfgarchon.utils.progress import create_progress_bar, should_show_progress

        timestep_range = create_progress_bar(
            range(Nt_points - 1),
            verbose=should_show_progress(show_progress),
            desc="FP-SL Adjoint",
        )

        # Forward time stepping (dimension-agnostic dispatch)
        for n in timestep_range:
            if self.dimension == 1:
                # 1D solve
                U_n = potential_field[n, :]
                alpha = self._compute_velocity_1d(U_n)
                M[n + 1, :] = self._adjoint_sl_step_1d(M[n, :], alpha, self.dt, sigma)
            else:
                # nD solve
                U_n = potential_field[n].reshape(self.grid_shape)
                alpha = self._compute_velocity_nd(U_n)
                M[n + 1] = self._adjoint_sl_step_nd(M[n].reshape(self.grid_shape), alpha, self.dt, sigma)

        return M

    def _compute_velocity_1d(self, U: np.ndarray) -> np.ndarray:
        """Compute the optimal-control drift alpha* = -grad(U) / control_cost for 1D.

        Issue #1420 / G-017 / S0-03: coefficient single-sourced via ``fp_drift_coefficient``
        (= 1/control_cost), not hardcoded to 1. Byte-identical when control_cost == 1.
        """
        return -fp_drift_coefficient(self.problem) * np.gradient(U, self.dx)

    def _compute_velocity_nd(self, U: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Compute the optimal-control drift alpha* = -grad(U) / control_cost for nD.

        Issue #1420 / G-017 / S0-03: coefficient single-sourced via ``fp_drift_coefficient``
        (= 1/control_cost), not hardcoded to 1. Returns a tuple, one array per dimension.
        """
        # Use np.gradient with spacing for each dimension
        c = fp_drift_coefficient(self.problem)
        gradients = np.gradient(U, *[self.spacing[d] for d in range(self.dimension)])
        if self.dimension == 1:
            return (-c * gradients,)
        return tuple(-c * g for g in gradients)

    def _adjoint_sl_step_1d(
        self,
        m: np.ndarray,
        alpha: np.ndarray,
        dt: float,
        sigma: float,
    ) -> np.ndarray:
        """
        One Adjoint Semi-Lagrangian step for 1D Fokker-Planck equation.

        Operator splitting:
            1. Forward advection with splatting (mass-conservative)
            2. Diffusion via Crank-Nicolson with zero-flux BC

        Args:
            m: Current density, shape (Nx,)
            alpha: Velocity field, shape (Nx,)
            dt: Time step
            sigma: Diffusion coefficient

        Returns:
            Density at next time step, shape (Nx,)
        """
        # Step 1: Forward Advection (Splatting)
        # =====================================
        # Forward trace: where does mass at x go?
        x_dest = self.x_grid + alpha * dt

        # Issue #702: Apply boundary conditions based on problem BC type
        # This ensures adjoint consistency with HJB-SL solver
        # Uses shared BC operations from hjb_sl_characteristics module
        bc_op = self._get_bc_operation_type()
        apply_bc = np.vectorize(lambda x: apply_boundary_conditions_1d(x, self.xmin, self.xmax, bc_op))
        x_dest_bounded = apply_bc(x_dest)

        # Issue #708: Splatting is the transpose of interpolation
        # Use the same method as HJB solver for exact adjoint consistency
        # - linear: 2-point, weights (1-w, w)
        # - cubic: 4-point, Catmull-Rom weights
        # - quintic: 6-point, Lagrange weights
        m_star = splat_1d(
            m=m,
            x_dest=x_dest_bounded,
            x_grid=self.x_grid,
            dx=self.dx,
            xmin=self.xmin,
            xmax=self.xmax,
            method=self.interpolation_method,
        )

        # Ensure non-negativity (only matters for cubic/quintic which may oscillate)
        if self.interpolation_method != "linear":
            m_star = self._clip_nonneg(m_star)

        # Step 2: Diffusion via Crank-Nicolson
        # =====================================
        # Issue #1257, 2026-06-10 audit: the diffusion BC must match the advection BC.
        # On a periodic domain the advection step already wraps mass across the seam;
        # pairing it with a Neumann/zero-flux diffusion sub-step produces an O(1) seam
        # flux error every step and breaks adjoint consistency with HJB-SL (which
        # threads _get_diffusion_bc_type into both its CN and ADI).  Use
        # solve_crank_nicolson_diffusion_1d from hjb_sl_adi (which has both
        # 'periodic' and 'neumann' branches) so the periodic case gets the
        # Sherman-Morrison circulant solve instead of the zero-flux stencil.
        diff_bc = self._get_diffusion_bc_type()
        if diff_bc == "periodic":
            return solve_crank_nicolson_diffusion_1d(m_star, dt, sigma, self.x_grid, bc_type="periodic")

        # Neumann / zero-flux path (preserved from Issue #708: FV stencil for mass
        # conservation; the standard ghost-point strong-form method breaks ∫m).
        D = diffusion_from_volatility(sigma)
        r = D * dt / (self.dx**2)

        # Build RHS: (I + r/2 * L_fv) * m_star
        # L_fv uses zero-flux (one-sided) boundary stencil
        rhs = np.zeros(self.Nx)
        # Interior points: standard 3-point stencil
        rhs[1:-1] = m_star[1:-1] + (r / 2) * (m_star[:-2] - 2 * m_star[1:-1] + m_star[2:])
        # Boundary points: zero-flux (finite volume) stencil
        # L[0] = (m[1] - m[0])/dx^2, so (I + r/2*L)[0] = m[0] + r/2*(m[1] - m[0])
        rhs[0] = m_star[0] + (r / 2) * (m_star[1] - m_star[0])
        rhs[-1] = m_star[-1] + (r / 2) * (m_star[-2] - m_star[-1])

        # Build tridiagonal matrix (I - r/2 * L_fv) for Crank-Nicolson
        # Interior: (I - r/2*L) has [r/2, 1+r, r/2] pattern
        # Boundary: zero-flux gives [-1, 1]/dx^2, so (I - r/2*L) has [1+r/2, -r/2]
        ab = np.zeros((3, self.Nx))
        # Main diagonal
        ab[1, :] = 1 + r  # Interior: 1 + r
        ab[1, 0] = 1 + r / 2  # Left boundary: 1 + r/2
        ab[1, -1] = 1 + r / 2  # Right boundary: 1 + r/2
        # Upper diagonal (superdiagonal)
        ab[0, 1:] = -r / 2  # Interior
        ab[0, 1] = -r / 2  # Left boundary (same coefficient)
        # Lower diagonal (subdiagonal)
        ab[2, :-1] = -r / 2  # Interior
        ab[2, -2] = -r / 2  # Right boundary (same coefficient)

        # Solve tridiagonal system
        m_new = solve_banded((1, 1), ab, rhs)

        # Ensure non-negativity
        m_new = self._clip_nonneg(m_new)

        return m_new

    def _adjoint_sl_step_nd(
        self,
        m: np.ndarray,
        alpha: tuple[np.ndarray, ...],
        dt: float,
        sigma: float,
    ) -> np.ndarray:
        """
        One Adjoint Semi-Lagrangian step for nD Fokker-Planck equation.

        Operator splitting:
            1. Forward advection with linear splatting (mass-conservative)
            2. Diffusion via ADI (Peaceman-Rachford)

        Args:
            m: Current density, shape grid_shape
            alpha: Velocity field tuple, each element shape grid_shape
            dt: Time step
            sigma: Diffusion coefficient

        Returns:
            Density at next time step, shape grid_shape
        """
        # Step 1: Forward Advection (Splatting)
        # =====================================
        # Compute destination positions for all grid points
        # x_dest[d] = x[d] + alpha[d] * dt

        # Create meshgrid of current positions
        meshes = np.meshgrid(*self.grid_coordinates, indexing="ij")

        # Compute destination positions
        x_dest = [meshes[d] + alpha[d] * dt for d in range(self.dimension)]

        # Apply boundary conditions (per dimension, vectorized)
        # For tensor product grids, each dimension is independent
        bc_op = self._get_bc_operation_type()
        for d in range(self.dimension):
            xmin_d, xmax_d = self.bounds[d]
            # Vectorized boundary conditions using numpy operations
            if bc_op == "periodic":
                length = xmax_d - xmin_d
                x_dest[d] = xmin_d + np.mod(x_dest[d] - xmin_d, length)
            elif bc_op == "reflect":
                # Reflect about boundaries using triangle wave
                # Maps any point to [xmin, xmax] via reflections
                length = xmax_d - xmin_d
                # Normalize: x_norm in [0, 1] for x in [xmin, xmax]
                x_norm = (x_dest[d] - xmin_d) / length
                # Triangle wave: 0→1→0→1... (period 2)
                # mod(x, 2) gives [0, 2), then |. - 1| gives [1, 0, 1)
                # finally 1 - |.| gives [0, 1, 0)
                x_fold = 1 - np.abs(np.mod(x_norm, 2) - 1)
                # Map back to domain
                x_dest[d] = xmin_d + x_fold * length
            else:
                # Clamp (dirichlet / absorbing)
                x_dest[d] = np.clip(x_dest[d], xmin_d, xmax_d)

        # Stack into (N_total, dimension) array for splatting
        x_dest_array = np.stack([xd.ravel() for xd in x_dest], axis=-1)

        # Linear splatting (mass-conservative)
        m_star = splat_nd(
            m=m.ravel(),
            x_dest=x_dest_array,
            grid_coordinates=self.grid_coordinates,
            grid_shape=self.grid_shape,
            bounds=self.bounds,
            method="linear",
        )
        m_star = m_star.reshape(self.grid_shape)

        # Ensure non-negativity
        m_star = self._clip_nonneg(m_star)

        # Step 2: Diffusion via ADI
        # =========================
        # Reuse the ADI diffusion from HJB-SL module.
        # Issue #1257, 2026-06-10 audit: pass bc_type so that a periodic domain
        # uses periodic ADI (Sherman-Morrison per axis) instead of defaulting to
        # 'neumann', which would impose a zero-flux seam mis-matched to the
        # periodic advection step above.  Mirrors HJB-SL _adi_diffusion_step
        # (hjb_semi_lagrangian.py:2263).
        m_new = adi_diffusion_step(
            U_star=m_star,
            dt=dt,
            sigma=sigma,
            spacing=self.spacing,
            grid_shape=self.grid_shape,
            bc_type=self._get_diffusion_bc_type(),
        )

        # Ensure non-negativity
        m_new = self._clip_nonneg(m_new)

        return m_new

    def _clip_nonneg(self, m: np.ndarray) -> np.ndarray:
        """Clip negative density to zero, warning once per solve if the clip injects
        non-trivial mass.

        Cubic/quintic splatting and the CN/ADI diffusion step are not monotone, so the
        density can undershoot below zero; deleting those undershoots injects positive
        mass and violates conservation. Surface it once per ``solve_fp_system`` call
        rather than failing silently (kernel fail-fast). Mirrors the
        ``WeakFormFPSolver`` positivity-clip diagnostic (Issue #1147).

        The injected/total ratio is grid-quadrature-invariant on a uniform grid, so raw
        sums (no ``dx`` weighting) give the correct fraction.
        """
        if not self._clip_warned:
            injected = -float(np.minimum(m, 0.0).sum())
            total = float(np.maximum(m, 0.0).sum())
            if injected > 1e-6 * max(total, 1e-300):
                logger.warning(
                    "FP-SL positivity clip injected mass %.2e (%.1f%% of total): cubic/quintic "
                    "splatting or CN/ADI diffusion is not monotone; consider linear interpolation "
                    "or a finer grid.",
                    injected,
                    100.0 * injected / max(total, 1e-300),
                )
                self._clip_warned = True
        return np.maximum(m, 0.0)

    def _get_solver_type_id(self) -> str | None:
        """Get solver type identifier for compatibility checking."""
        # Use semi_lagrangian for compatibility (adjoint is a variant)
        return "semi_lagrangian"


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPSLAdjointSolver...")
    print("=" * 60)

    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    # Test parameters
    X_MIN, X_MAX = -0.5, 0.5
    SIGMA = 0.2
    N = 100
    T = 10.0
    Nt = 1000

    L = X_MAX - X_MIN
    dx = L / N
    x = np.linspace(X_MIN, X_MAX, N + 1)

    print(f"Grid: N={N}, dx={dx:.4f}")
    print(f"Time: T={T}, Nt={Nt}, dt={T / Nt:.4f}")
    print(f"Diffusion: sigma={SIGMA}")

    # Create problem with Neumann BC
    left_bc = BCSegment(name="left", bc_type=BCType.NEUMANN, value=0.0, boundary="x_min")
    right_bc = BCSegment(name="right", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max")
    bc = BoundaryConditions(
        segments=[left_bc, right_bc], default_bc=BCType.NO_FLUX
    )  # Issue #1100: explicit (was implicit PERIODIC)

    domain = TensorProductGrid(bounds=[(X_MIN, X_MAX)], Nx_points=[N + 1], boundary_conditions=bc)

    # Create Hamiltonian and components
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0,
        coupling_dm=lambda m: 0.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )

    problem = MFGProblem(
        geometry=domain,
        T=T,
        Nt=Nt,
        sigma=SIGMA,
        components=components,
    )

    # Test 1: Solver initialization
    print("\n1. Testing solver initialization...")
    solver = FPSLSolver(problem)
    assert solver.dimension == 1
    assert solver.fp_method_name == "Adjoint Semi-Lagrangian"
    print("   Initialization: OK")

    # Test 2: Solve with drift toward center (confining potential)
    print("\n2. Testing with confining potential U = x^2...")

    U_well = np.tile(x**2, (Nt + 1, 1))

    # Start from uniform
    m_uniform = np.ones(N + 1) / L

    # Expected Gibbs
    m_gibbs = np.exp(-2 * x**2 / SIGMA**2)
    m_gibbs /= np.trapezoid(m_gibbs, x)

    print(f"   Initial: uniform, peak = {m_uniform.max():.4f}")
    print(f"   Target Gibbs: peak = {m_gibbs.max():.4f}")

    M = solver.solve_fp_system(M_initial=m_uniform.copy(), drift_field=U_well, show_progress=False)

    # Check evolution
    for t_idx in [0, 100, 500, 1000]:
        m_t = M[t_idx, :]
        peak_idx = np.argmax(m_t)
        peak_x = x[peak_idx]
        mass = np.trapezoid(m_t, x)
        print(f"   t={t_idx * T / Nt:5.2f}: peak={m_t.max():.4f} at x={peak_x:.3f}, mass={mass:.4f}")

    # Check final result
    m_final = M[-1, :]
    m_final_norm = m_final / np.trapezoid(m_final, x)
    l2_to_gibbs = np.sqrt(np.trapezoid((m_final_norm - m_gibbs) ** 2, x))
    print(f"\n   Final L2 to Gibbs: {l2_to_gibbs:.4e}")

    # Test 3: Mass conservation (Issue #708)
    print("\n3. Testing mass conservation (Issue #708 fix)...")

    # sum(m) is the conserved quantity for SL adjoint
    sum_m_initial = np.sum(M[0])
    sum_m_final = np.sum(M[-1])
    sum_m_error = abs(sum_m_final - sum_m_initial) / sum_m_initial

    print(f"   sum(m) initial: {sum_m_initial:.6f}")
    print(f"   sum(m) final:   {sum_m_final:.6f}")
    print(f"   sum(m) error:   {sum_m_error:.2e}")

    assert sum_m_error < 1e-10, f"Mass conservation failed: error={sum_m_error:.2e}"
    print("   Mass conservation: OK (error < 1e-10)")

    # Test 4: Compare with Backward SL
    print("\n4. Comparing with Backward SL (deprecated FPSLJacobianSolver)...")
    import warnings

    from mfgarchon.alg.numerical.fp_solvers import FPSLJacobianSolver

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        backward_solver = FPSLJacobianSolver(problem)
    M_backward = backward_solver.solve_fp_system(M_initial=m_uniform.copy(), drift_field=U_well, show_progress=False)

    m_backward_final = M_backward[-1, :]
    m_backward_norm = m_backward_final / np.trapezoid(m_backward_final, x)
    l2_backward = np.sqrt(np.trapezoid((m_backward_norm - m_gibbs) ** 2, x))

    print(f"   Adjoint SL peak: {m_final.max():.4f}, L2 to Gibbs: {l2_to_gibbs:.4e}")
    print(f"   Backward SL peak: {m_backward_final.max():.4f}, L2 to Gibbs: {l2_backward:.4e}")

    # Test 5: 2D solver test
    print("\n5. Testing 2D FP SL Adjoint solver...")

    from mfgarchon.geometry.boundary import no_flux_bc

    # 2D setup (smaller grid for speed)
    N2D = 30
    Nt2D = 100
    T2D = 1.0
    SIGMA2D = 0.3

    # 2D domain with no-flux BC
    bc_2d = no_flux_bc(dimension=2)
    domain_2d = TensorProductGrid(
        bounds=[(-0.5, 0.5), (-0.5, 0.5)],
        Nx_points=[N2D + 1, N2D + 1],
        boundary_conditions=bc_2d,
    )

    # 2D problem
    components_2d = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    problem_2d = MFGProblem(
        geometry=domain_2d,
        T=T2D,
        Nt=Nt2D,
        sigma=SIGMA2D,
        components=components_2d,
    )

    solver_2d = FPSLSolver(problem_2d)
    print(f"   Dimension: {solver_2d.dimension}")
    print(f"   Grid shape: {solver_2d.grid_shape}")
    print(f"   Spacing: {solver_2d.spacing}")
    assert solver_2d.dimension == 2

    # Create 2D confining potential U(x,y) = x^2 + y^2
    x2d = domain_2d.coordinates[0]
    y2d = domain_2d.coordinates[1]
    XX, YY = np.meshgrid(x2d, y2d, indexing="ij")
    U_2d_static = XX**2 + YY**2

    # Stack U for each time step
    U_2d = np.zeros((Nt2D + 1, N2D + 1, N2D + 1))
    for t in range(Nt2D + 1):
        U_2d[t] = U_2d_static

    # Initial uniform density
    domain_volume_2d = np.prod([ub - lb for lb, ub in domain_2d.bounds])
    m_2d_initial = np.ones((N2D + 1, N2D + 1)) / domain_volume_2d

    # Solve
    M_2d = solver_2d.solve_fp_system(M_initial=m_2d_initial, drift_field=U_2d, show_progress=False)

    print(f"   Output shape: {M_2d.shape}")

    # Check mass conservation (sum(m) invariant)
    sum_m_2d_initial = np.sum(M_2d[0])
    sum_m_2d_final = np.sum(M_2d[-1])
    sum_m_2d_error = abs(sum_m_2d_final - sum_m_2d_initial) / sum_m_2d_initial
    print(f"   sum(m) initial: {sum_m_2d_initial:.6f}")
    print(f"   sum(m) final:   {sum_m_2d_final:.6f}")
    print(f"   sum(m) error:   {sum_m_2d_error:.2e}")

    # ADI may have slightly worse mass conservation
    assert sum_m_2d_error < 1e-6, f"2D mass conservation failed: error={sum_m_2d_error:.2e}"
    print("   2D mass conservation: OK (error < 1e-6)")

    # Check that density concentrates at center
    m_2d_final = M_2d[-1]
    center_idx = N2D // 2
    center_density = m_2d_final[center_idx, center_idx]
    corner_density = m_2d_final[0, 0]
    print(f"   Final center density: {center_density:.4f}")
    print(f"   Final corner density: {corner_density:.6f}")
    assert center_density > corner_density * 10, "Density should concentrate at center"
    print("   Concentration: OK (center > 10x corner)")

    print("\n" + "=" * 60)
    print("All smoke tests passed!")


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================


# Backward compatibility: FPSLAdjointSolver -> FPSLSolver
# Uses subclass pattern (not deprecated_alias) to preserve isinstance checks
# and class attribute inheritance (_scheme_family trait for duality validation).
class FPSLAdjointSolver(FPSLSolver):
    """
    DEPRECATED: Use :class:`FPSLSolver` instead.

    .. deprecated:: 0.17.6
        Renamed to FPSLSolver. Will be removed in v1.0.0.
    """

    _deprecation_meta: ClassVar[dict[str, Any]] = {
        "since": "v0.17.6",
        "replacement": "FPSLSolver",
        "reason": "Renamed to FPSLSolver",
        "removal": "v1.0.0",
        "removal_blockers": ["internal_usage", "equivalence_test"],
        "symbol": "FPSLAdjointSolver",
        "alias_for": "FPSLSolver",
    }

    @deprecated(
        since="v0.17.6",
        replacement="FPSLSolver",
        reason="FPSLAdjointSolver was renamed to FPSLSolver",
    )
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

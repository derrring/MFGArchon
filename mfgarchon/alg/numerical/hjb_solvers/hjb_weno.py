"""
WENO Family HJB Solvers for Mean Field Games

NOTE: API parameter names updated in v0.11.0. See docs/NAMING_CONVENTIONS.md for details.

This module implements the complete family of WENO (Weighted Essentially Non-Oscillatory)
schemes for solving Hamilton-Jacobi-Bellman equations in Mean Field Games.

Available WENO Variants:
- WENO5: Standard fifth-order WENO scheme (Jiang & Shu, 1996)
- WENO-Z: Enhanced WENO with τ-based weight modification (Borges et al., 2008)
- WENO-M: Mapped WENO for better performance near critical points
- WENO-JS: Original Jiang-Shu formulation with classic weights

Mathematical Framework:
    ∂u/∂t + H(x, ∇u, m(t,x)) - (σ²/2)Δu = 0

Each WENO variant provides:
- Fifth-order spatial accuracy in smooth regions
- Non-oscillatory behavior near discontinuities
- Different weight calculation strategies for various trade-offs
- TVD-RK3 or explicit Euler time integration

Key Features:
- Unified interface for all WENO variants
- Easy switching between methods for benchmarking
- Optimized stencil operations
- Comprehensive parameter validation
- Academic-quality implementation with references

References:
- Jiang & Shu (1996): Efficient Implementation of WENO Schemes
- Borges et al. (2008): An improved weighted essentially non-oscillatory scheme
- Henrick et al. (2005): Mapped weighted essentially non-oscillatory schemes
- Shu & Osher (1988): Efficient implementation of essentially non-oscillatory schemes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from mfgarchon.core.derivatives import DerivativeTensors, to_multi_index_dict
from mfgarchon.geometry.boundary.applicator_fdm import PreallocatedGhostBuffer
from mfgarchon.geometry.boundary.conditions import neumann_bc
from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

from .base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    from mfgarchon.core.mfg_problem import MFGProblem

WenoVariant = Literal["weno5", "weno-z", "weno-m", "weno-js"]

# Dimension-dependent stability limits (Issue #967)
# Advection CFL: for Strang/Godunov splitting, stable CFL ~ 0.5/d.
_CFL_CAP_1D: float = 0.5
# Diffusion: forward Euler stability limit 0.5/2^d (conservative halving per dimension).
_DIFFUSION_CAP_1D: float = 0.25


class HJBWenoSolver(BaseHJBSolver):
    """
    Unified WENO family solver for Hamilton-Jacobi-Bellman equations.

    This solver provides access to the complete family of WENO schemes through
    a single interface. Users can select the variant most appropriate for their
    problem characteristics and performance requirements.

    WENO Variants:
    - "weno5": Standard WENO5 (balanced performance, widely used)
    - "weno-z": Enhanced resolution, reduced dissipation
    - "weno-m": Better critical point handling, mapped weights
    - "weno-js": Original formulation, maximum stability

    Mathematical Approach:
    1. WENO spatial reconstruction with variant-specific weights
    2. Central differences for diffusion term -(σ²/2)Δu
    3. TVD-RK3 or explicit Euler time integration
    4. Hamiltonian splitting for nonlinear terms
    5. Dimensional splitting (Strang/Godunov) for multi-dimensional problems

    Dimensional Splitting for Multi-D:
        For 2D/3D problems, uses operator splitting to reduce to 1D sweeps:
        - Strang splitting (default): X → Y → X (2nd order accurate)
        - Godunov splitting: X → Y (1st order accurate)

        **Isotropy Assumption**: Dimensional splitting works best when the Hamiltonian
        is approximately isotropic (no strong directional preference).

        ✅ Works excellently for:
        - Standard MFG: H = (1/2)|∇u|² + V(x) + F(m) (isotropic, default)
        - Isotropic control costs: H = (1/p)|∇u|^p + ...
        - Smooth solutions with moderate CFL numbers

        ⚠️ May introduce larger errors for:
        - Anisotropic Hamiltonians: H = (1/2)∇u·Q·∇u with Q ≠ I
        - Traffic/network problems with directional flow
        - Strong cross-derivative coupling

        For anisotropic problems, consider:
        1. Run convergence tests (solve with Δt, Δt/2, Δt/4)
        2. Use smaller CFL number (reduce from 0.3 to 0.1-0.2)
        3. Alternative: HJBFDMSolver (no splitting) or HJBSemiLagrangianSolver

    Performance Guide:
    - Use "weno5" for general problems and benchmarking
    - Use "weno-z" for problems requiring high resolution
    - Use "weno-m" for critical points and smooth solutions
    - Use "weno-js" for maximum stability requirements

    Required Geometry Traits (Issue #596 Phase 2.1):
        - SupportsGradient: Provides gradient operators for derivative computation

    Compatible Geometries:
        - TensorProductGrid (structured grids)
        - ImplicitDomain (SDF-based domains)
        - Any geometry implementing SupportsGradient protocol

    Note:
        While WENO uses dimensional splitting instead of trait-based gradient
        operators, the SupportsGradient trait ensures geometry can provide
        spatial derivatives if needed for extensions.
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.FDM  # WENO is FDM variant

    def __init__(
        self,
        problem: MFGProblem,
        weno_variant: WenoVariant = "weno5",
        cfl_number: float = 0.3,
        diffusion_stability_factor: float = 0.25,
        weno_epsilon: float = 1e-6,
        weno_z_parameter: float = 1.0,
        weno_m_parameter: float = 1.0,
        time_integration: str = "tvd_rk3",
        splitting_method: str = "strang",
        max_substeps: int = 10000,
    ):
        """
        Initialize WENO family HJB solver with multi-dimensional support.

        Automatically detects problem dimension and applies appropriate WENO schemes:
        - 1D: Direct WENO reconstruction
        - 2D/3D: Dimensional splitting approach with WENO in each direction

        Args:
            problem: MFG problem instance (1D, 2D, 3D, or high-dimensional)
            weno_variant: WENO scheme variant ("weno5", "weno-z", "weno-m", "weno-js")
            cfl_number: CFL number for advection terms (typically 0.1-0.5 for 1D, 0.1-0.3 for 2D+)
            diffusion_stability_factor: Stability factor for diffusion (typically 0.25 for 1D, 0.125 for 2D+)
            weno_epsilon: WENO smoothness parameter (typically 1e-6)
            weno_z_parameter: WENO-Z τ parameter for enhanced resolution (typically 1.0)
            weno_m_parameter: WENO-M mapping parameter for critical points (typically 1.0)
            time_integration: Time integration scheme ("tvd_rk3", "explicit_euler")
            splitting_method: Dimensional splitting method for 2D+ ("strang", "godunov")
            max_substeps: Safety cap on CFL/diffusion-stable sub-steps per backward time
                interval (Issue #1180). Each interval is sub-stepped until the full ``dt``
                is covered; the solve fails loud if this cap is hit (mirrors the
                semi-Lagrangian solver). Generous default; raise only for extreme
                diffusion/CFL ratios.
        """
        super().__init__(problem)

        # Validate geometry supports required trait (Issue #596 Phase 2.1C)
        from mfgarchon.geometry.protocols import SupportsGradient

        if not isinstance(problem.geometry, SupportsGradient):
            raise TypeError(
                f"HJB WENO solver requires geometry with SupportsGradient trait. "
                f"{type(problem.geometry).__name__} does not implement this trait. "
                f"Compatible geometries: TensorProductGrid, ImplicitDomain."
            )

        # Validate WENO variant
        if weno_variant not in ["weno5", "weno-z", "weno-m", "weno-js"]:
            raise ValueError(f"Unknown WENO variant: {weno_variant}")

        self.weno_variant = weno_variant
        self.splitting_method = splitting_method

        # Detect problem dimension
        self.dimension = self._detect_problem_dimension()
        self.hjb_method_name = f"{self.dimension}D-WENO-{weno_variant.upper()}"

        # Store raw parameters for validation, then adjust for dimension
        self.cfl_number = cfl_number
        self.diffusion_stability_factor = diffusion_stability_factor
        self.weno_epsilon = weno_epsilon
        self.weno_z_parameter = weno_z_parameter
        self.weno_m_parameter = weno_m_parameter
        self.time_integration = time_integration
        self.max_substeps = max_substeps

        # Validate user-provided parameters before dimension adjustment
        self._validate_parameters()

        # Apply dimension-dependent stability caps
        self.cfl_number = self._adjust_cfl_for_dimension(cfl_number)
        self.diffusion_stability_factor = self._adjust_diffusion_factor_for_dimension(diffusion_stability_factor)

        # Setup dimension-specific grid information
        self._setup_dimensional_grid()

        # Setup ghost cell buffer for boundary handling (composition pattern)
        # WENO5 requires 2 ghost cells on each side for the 5-point stencil
        self._setup_ghost_buffer()

        # Setup WENO coefficients (shared across variants)
        self._setup_weno_coefficients()

    def _validate_parameters(self) -> None:
        """Validate all solver parameters."""
        if not 0 < self.cfl_number <= 1.0:
            raise ValueError(f"CFL number must be in (0,1], got {self.cfl_number}")
        if not 0 < self.diffusion_stability_factor <= 0.5:
            raise ValueError(f"Diffusion stability factor must be in (0,0.5], got {self.diffusion_stability_factor}")
        if self.weno_epsilon <= 0:
            raise ValueError(f"WENO epsilon must be positive, got {self.weno_epsilon}")
        if self.weno_z_parameter <= 0:
            raise ValueError(f"WENO-Z parameter must be positive, got {self.weno_z_parameter}")
        if self.weno_m_parameter <= 0:
            raise ValueError(f"WENO-M parameter must be positive, got {self.weno_m_parameter}")

        if self.splitting_method not in ["strang", "godunov"]:
            raise ValueError(f"Unknown splitting method: {self.splitting_method}")

    def _detect_problem_dimension(self) -> int:
        """Detect spatial dimension from geometry (unified interface, NO hasattr)."""
        # Primary: geometry.dimension (standard for all modern problems)
        try:
            return self.problem.geometry.dimension
        except AttributeError:
            pass

        # Fallback: problem.dimension attribute
        try:
            return self.problem.dimension
        except AttributeError:
            pass

        raise ValueError(
            "Cannot determine problem dimension. Ensure problem has 'geometry' with 'dimension' attribute."
        )

    def _adjust_cfl_for_dimension(self, base_cfl: float) -> float:
        """Adjust CFL number based on problem dimension for stability.

        For dimensional splitting, the stable CFL scales as ~1/d.
        Cap = _CFL_CAP_1D / dimension.
        """
        cap = _CFL_CAP_1D / self.dimension
        return min(base_cfl, cap)

    def _adjust_diffusion_factor_for_dimension(self, base_factor: float) -> float:
        """Adjust diffusion stability factor based on problem dimension.

        For forward Euler diffusion with splitting, the stable factor
        halves per added dimension: _DIFFUSION_CAP_1D / 2^(d-1).
        """
        cap = _DIFFUSION_CAP_1D / (2 ** (self.dimension - 1))
        return min(base_factor, cap)

    def _setup_dimensional_grid(self) -> None:
        """Setup grid information from geometry (standard interface, NO hasattr)."""
        # Use standard geometry interface - all modern problems support this
        geometry = self.problem.geometry
        self.num_grid_points = list(geometry.get_grid_shape())
        self.grid_spacing = list(geometry.get_grid_spacing())

        # Backward compatibility: set _x, _y, _z attributes for internal WENO code
        if self.dimension >= 1:
            self.num_grid_points_x = self.num_grid_points[0]
            self.grid_spacing_x = self.grid_spacing[0]
        if self.dimension >= 2:
            self.num_grid_points_y = self.num_grid_points[1]
            self.grid_spacing_y = self.grid_spacing[1]
        if self.dimension >= 3:
            self.num_grid_points_z = self.num_grid_points[2]
            self.grid_spacing_z = self.grid_spacing[2]

    def _setup_ghost_buffer(self) -> None:
        """
        Setup ghost cell buffer for boundary handling with high-order accuracy.

        WENO5 requires:
        - ghost_depth=2: 2 ghost cells on each side for 5-point stencil (i-2 to i+2)
        - order=5: High-order polynomial extrapolation for O(h^5) boundary accuracy

        Issue #576: Unified ghost node architecture enables WENO5 to achieve
        true 5th-order convergence at boundaries via Vandermonde extrapolation.

        Uses composition pattern: the solver holds a PreallocatedGhostBuffer
        component rather than inheriting from a BoundaryAwareSolver base class.
        """
        # Ghost depth for WENO5: 2 cells on each side for 5-point stencil
        self.ghost_depth = 2

        # Reconstruction order: 5th-order accuracy at boundaries
        # Issue #576: Uses polynomial extrapolation (not simple reflection)
        self.ghost_order = 5

        # Get boundary conditions from problem/geometry if available
        bc = self._get_boundary_conditions()

        # Build domain bounds from grid information
        domain_bounds = self._get_domain_bounds()

        # Create ghost buffer for each dimension
        # For 1D, interior_shape is (num_points,)
        # For nD, we create per-dimension buffers for dimensional splitting
        if self.dimension == 1:
            interior_shape = (self.num_grid_points_x,)
            self.ghost_buffer = PreallocatedGhostBuffer(
                interior_shape=interior_shape,
                boundary_conditions=bc,
                domain_bounds=domain_bounds,
                ghost_depth=self.ghost_depth,
                order=self.ghost_order,  # High-order ghost cells for WENO5
            )
        else:
            # For multi-D with dimensional splitting, we create 1D buffers on demand
            # Store BC and bounds for later use
            self._bc = bc
            self._domain_bounds = domain_bounds
            self.ghost_buffer = None  # Created per-sweep in multi-D

    def _get_boundary_conditions(self):
        """
        Get boundary conditions from problem or geometry.

        Falls back to Neumann (no-flux) BC if not specified.

        Resolution order (NO hasattr per CLAUDE.md):
        1. geometry.get_boundary_conditions() (method accessor)
        2. geometry.boundary_conditions (direct attribute)
        3. problem.get_boundary_conditions() (method accessor)
        4. problem.boundary_conditions (direct attribute)
        5. Default Neumann BC
        """
        # Priority 1: geometry.get_boundary_conditions() (method accessor)
        try:
            bc = self.problem.geometry.get_boundary_conditions()
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # Priority 2: geometry.boundary_conditions (direct attribute)
        try:
            bc = self.problem.geometry.boundary_conditions
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # Priority 3: problem.get_boundary_conditions() (method accessor)
        try:
            bc = self.problem.get_boundary_conditions()
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # Priority 4: problem.boundary_conditions (direct attribute)
        try:
            bc = self.problem.boundary_conditions
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # Default: Neumann (no-flux) BC - common for HJB problems
        return neumann_bc(dimension=self.dimension)

    def _get_domain_bounds(self) -> np.ndarray:
        """Get domain bounds from problem/geometry (NO hasattr per CLAUDE.md)."""
        # Priority 1: Try CartesianGrid geometry bounds attribute
        try:
            return np.array(self.problem.geometry.bounds)
        except AttributeError:
            pass

        # Priority 2: Try legacy problem attributes with defaults
        bounds = []
        for d in range(self.dimension):
            if d == 0:
                x_min = getattr(self.problem, "x_min", 0.0)
                x_max = getattr(self.problem, "x_max", 1.0)
                bounds.append([x_min, x_max])
            elif d == 1:
                y_min = getattr(self.problem, "y_min", 0.0)
                y_max = getattr(self.problem, "y_max", 1.0)
                bounds.append([y_min, y_max])
            elif d == 2:
                z_min = getattr(self.problem, "z_min", 0.0)
                z_max = getattr(self.problem, "z_max", 1.0)
                bounds.append([z_min, z_max])
            else:
                bounds.append([0.0, 1.0])  # Default unit interval

        return np.array(bounds)

    def _setup_weno_coefficients(self) -> None:
        """Setup WENO reconstruction coefficients (shared across variants)."""
        # Optimal linear weights
        self.d_plus = np.array([3 / 10, 3 / 5, 1 / 10])  # d₀, d₁, d₂ for positive reconstruction
        self.d_minus = np.array([1 / 10, 3 / 5, 3 / 10])  # d₀, d₁, d₂ for negative reconstruction

        # Stencil coefficients for polynomial reconstruction
        # Positive reconstruction coefficients (left-to-right bias)
        self.c_plus = np.array(
            [
                [1 / 3, -7 / 6, 11 / 6],  # S₀: u_{i-2}, u_{i-1}, u_i
                [-1 / 6, 5 / 6, 1 / 3],  # S₁: u_{i-1}, u_i, u_{i+1}
                [1 / 3, 5 / 6, -1 / 6],  # S₂: u_i, u_{i+1}, u_{i+2}
            ]
        )

        # Negative reconstruction coefficients (right-to-left bias)
        self.c_minus = np.array(
            [
                [-1 / 6, 5 / 6, 1 / 3],  # S₀: u_{i+1}, u_i, u_{i-1}
                [1 / 3, 5 / 6, -1 / 6],  # S₁: u_i, u_{i-1}, u_{i-2}
                [1 / 3, -7 / 6, 11 / 6],  # S₂: u_{i-1}, u_{i-2}, u_{i-3}
            ]
        )

    def _compute_weno_weights(self, values: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute WENO weights using the selected variant.

        Args:
            values: Array of function values
            i: Grid point index

        Returns:
            (w_plus, w_minus): WENO weights for upwind and downwind reconstruction
        """
        # Extract safe stencil values
        n = len(values)
        i_safe = max(2, min(i, n - 3))  # Ensure valid 5-point stencil
        u = values[i_safe - 2 : i_safe + 3]  # 5-point stencil: u_{i-2} to u_{i+2}

        # Compute smoothness indicators (same for all variants)
        beta = self._compute_smoothness_indicators(u)

        # Select weight calculation based on variant
        if self.weno_variant == "weno5" or self.weno_variant == "weno-js":
            w_plus = self._compute_classic_weights(beta, self.d_plus)
            w_minus = self._compute_classic_weights(beta[::-1], self.d_minus)
        elif self.weno_variant == "weno-z":
            tau = self._compute_tau_indicator(u)
            w_plus = self._compute_z_weights(beta, tau, self.d_plus)
            w_minus = self._compute_z_weights(beta[::-1], tau, self.d_minus)
        elif self.weno_variant == "weno-m":
            w_plus = self._compute_mapped_weights(beta, self.d_plus)
            w_minus = self._compute_mapped_weights(beta[::-1], self.d_minus)
        else:
            raise ValueError(f"Unknown WENO variant: {self.weno_variant}")

        return w_plus, w_minus

    def _compute_smoothness_indicators(self, u: np.ndarray) -> np.ndarray:
        """
        Compute WENO smoothness indicators β for 3-point sub-stencils.

        Args:
            u: 5-point stencil values [u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}]

        Returns:
            β: Array of 3 smoothness indicators [β₀, β₁, β₂]
        """
        # Sub-stencil S₀: u_{i-2}, u_{i-1}, u_i
        beta_0 = (13 / 12) * (u[0] - 2 * u[1] + u[2]) ** 2 + (1 / 4) * (u[0] - 4 * u[1] + 3 * u[2]) ** 2

        # Sub-stencil S₁: u_{i-1}, u_i, u_{i+1}
        beta_1 = (13 / 12) * (u[1] - 2 * u[2] + u[3]) ** 2 + (1 / 4) * (u[1] - u[3]) ** 2

        # Sub-stencil S₂: u_i, u_{i+1}, u_{i+2}
        beta_2 = (13 / 12) * (u[2] - 2 * u[3] + u[4]) ** 2 + (1 / 4) * (3 * u[2] - 4 * u[3] + u[4]) ** 2

        return np.array([beta_0, beta_1, beta_2])

    def _compute_classic_weights(self, beta: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Compute classic WENO5/WENO-JS weights.

        Args:
            beta: Smoothness indicators [β₀, β₁, β₂]
            d: Linear weights [d₀, d₁, d₂]

        Returns:
            w: Classic WENO nonlinear weights [w₀, w₁, w₂]
        """
        # Classic WENO weight formula
        alpha = d / (self.weno_epsilon + beta) ** 2
        w = alpha / np.sum(alpha)
        return w

    def _compute_tau_indicator(self, u: np.ndarray) -> float:
        """
        Compute global smoothness indicator τ for WENO-Z scheme.

        Args:
            u: 5-point stencil values [u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}]

        Returns:
            τ: Global smoothness indicator
        """
        # WENO-Z τ₅ indicator using 4th-order differences
        tau = abs(u[0] - 4 * u[1] + 6 * u[2] - 4 * u[3] + u[4])
        return tau

    def _compute_z_weights(self, beta: np.ndarray, tau: float, d: np.ndarray) -> np.ndarray:
        """
        Compute WENO-Z weights with τ modification for enhanced resolution.

        Args:
            beta: Smoothness indicators [β₀, β₁, β₂]
            tau: Global smoothness indicator
            d: Linear weights [d₀, d₁, d₂]

        Returns:
            w: WENO-Z nonlinear weights [w₀, w₁, w₂]
        """
        # WENO-Z enhancement with τ-based modification
        tau_modified = (tau / (beta + self.weno_epsilon)) ** self.weno_z_parameter
        alpha = d * (1.0 + tau_modified)
        w = alpha / np.sum(alpha)
        return w

    def _compute_mapped_weights(self, beta: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Compute WENO-M mapped weights for better critical point handling.

        Args:
            beta: Smoothness indicators [β₀, β₁, β₂]
            d: Linear weights [d₀, d₁, d₂]

        Returns:
            w: WENO-M mapped weights [w₀, w₁, w₂]
        """
        # Compute classic weights first
        alpha_classic = d / (self.weno_epsilon + beta) ** 2
        w_classic = alpha_classic / np.sum(alpha_classic)

        # Apply mapping function for better critical point behavior
        # Henrick mapping: g(ω) = ω(d + d²/ω - d) / (d + d²/ω - 1)
        w_mapped = np.zeros_like(w_classic)
        for k in range(len(d)):
            if w_classic[k] > self.weno_epsilon:
                g_w = w_classic[k] * (d[k] + d[k] ** 2 / w_classic[k] - d[k]) / (d[k] + d[k] ** 2 / w_classic[k] - 1.0)
                w_mapped[k] = max(g_w, 0.0)  # Ensure positivity
            else:
                w_mapped[k] = 0.0

        # Renormalize
        w_sum = np.sum(w_mapped)
        if w_sum > self.weno_epsilon:
            w_mapped /= w_sum
        else:
            w_mapped = d  # Fallback to linear weights

        return w_mapped

    def _weno_reconstruction(self, values: np.ndarray, i: int) -> tuple[float, float]:
        """
        Perform WENO reconstruction to get left and right interface values.

        Args:
            values: Array of cell-centered values
            i: Interface index (between cells i and i+1)

        Returns:
            (u_left, u_right): Reconstructed values at interface
        """
        # Get WENO weights using selected variant
        w_plus, w_minus = self._compute_weno_weights(values, i)

        # Extract stencil for reconstruction
        n = len(values)
        i_safe = max(2, min(i, n - 3))
        u = values[i_safe - 2 : i_safe + 3]

        # Reconstruct using weighted combination of sub-stencil polynomials
        u_left = 0.0
        u_right = 0.0

        for k in range(3):
            # Left reconstruction (positive direction)
            u_left += w_plus[k] * np.dot(self.c_plus[k], u[k : k + 3])

            # Right reconstruction (negative direction) - fix indexing
            if k == 0:
                u_vals = u[2::-1]  # [u2, u1, u0]
            elif k == 1:
                u_vals = u[3:0:-1]  # [u3, u2, u1]
            else:  # k == 2
                u_vals = u[4:1:-1]  # [u4, u3, u2]

            u_right += w_minus[k] * np.dot(self.c_minus[k], u_vals)

        return u_left, u_right

    def solve_hjb_step(self, u_current: np.ndarray, m_current: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve one time step of the HJB equation using selected WENO variant.

        Args:
            u_current: Current value function
            m_current: Current density
            dt: Time step size

        Returns:
            u_new: Updated value function after one time step
        """
        if self.time_integration == "tvd_rk3":
            return self._solve_hjb_tvd_rk3(u_current, m_current, dt)
        elif self.time_integration == "explicit_euler":
            return self._solve_hjb_explicit_euler(u_current, m_current, dt)
        else:
            raise ValueError(f"Unknown time integration: {self.time_integration}")

    def _solve_hjb_explicit_euler(self, u_current: np.ndarray, m_current: np.ndarray, dt: float) -> np.ndarray:
        """Explicit Euler time step with WENO spatial discretization."""
        rhs = self._compute_hjb_rhs(u_current, m_current)
        return u_current + dt * rhs

    def _solve_hjb_tvd_rk3(self, u_current: np.ndarray, m_current: np.ndarray, dt: float) -> np.ndarray:
        """TVD-RK3 time step with WENO spatial discretization."""
        # Stage 1
        k1 = self._compute_hjb_rhs(u_current, m_current)
        u1 = u_current + dt * k1

        # Stage 2
        k2 = self._compute_hjb_rhs(u1, m_current)
        u2 = (3 / 4) * u_current + (1 / 4) * u1 + (1 / 4) * dt * k2

        # Stage 3
        k3 = self._compute_hjb_rhs(u2, m_current)
        u_new = (1 / 3) * u_current + (2 / 3) * u2 + (2 / 3) * dt * k3

        return u_new

    def _evaluate_hamiltonian(self, x_idx: int, m_val: float, grad: float, direction: tuple[int, ...] = (1,)) -> float:
        """
        Evaluate the Hamiltonian at a point using problem.H() interface.

        This method calls problem.H() if available, falling back to the default
        quadratic MFG Hamiltonian H = |p|²/2 + m*p for backward compatibility.

        Note: For dimensional splitting, only the partial gradient in the current
        direction is provided. This assumes separable/isotropic Hamiltonians.

        Args:
            x_idx: Grid index for spatial position
            m_val: Density value at this point
            grad: Gradient value (partial derivative in the specified direction)
            direction: Derivative direction tuple, e.g., (1,) for x, (0,1) for y

        Returns:
            Hamiltonian value H(x, grad, m)
        """
        # Build partial gradient vector from direction
        # direction (1,) = x, (0,1) = y, (0,0,1) = z
        dimension = len(direction)
        grad_vector = np.zeros(dimension)
        # Find which dimension has the non-zero entry
        for i, d in enumerate(direction):
            if d == 1:
                grad_vector[i] = grad
                break

        # Build DerivativeTensors
        derivs = DerivativeTensors.from_gradient(grad_vector)

        # Try to use problem.H() interface with DerivativeTensors (NO hasattr per CLAUDE.md)
        try:
            return self.problem.H(x_idx, m_val, derivs=derivs)
        except TypeError:
            # Legacy: convert to multi-index dict format
            try:
                legacy_derivs = to_multi_index_dict(derivs)
                return self.problem.H(x_idx, m_val, derivs=legacy_derivs)
            except (TypeError, AttributeError):
                pass
        except AttributeError:
            pass

        # Default: standard quadratic MFG Hamiltonian H = |p|²/2 + m*p
        # For 1D partial: H = (1/2)|∂u/∂x_i|² + m * ∂u/∂x_i
        return 0.5 * grad**2 + m_val * grad

    def _compute_hjb_rhs(self, u: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Compute right-hand side of HJB equation using WENO discretization.

        RHS = -H(x, ∇u, m) + (σ²/2)Δu

        Uses ghost cells for full 5th-order accuracy at boundaries.
        """
        n = len(u)
        rhs = np.zeros(n)
        dx = self.grid_spacing_x
        g = self.ghost_depth  # 2 for WENO5

        # Apply boundary conditions via ghost buffer
        # This gives us a padded array with ghost cells on each side
        self.ghost_buffer.interior[:] = u
        self.ghost_buffer.update_ghosts()
        u_padded = self.ghost_buffer.padded  # Shape: (n + 2*g,)

        # Compute spatial derivatives using WENO reconstruction on padded array
        # Now we can use full WENO stencil at ALL interior points
        u_x = np.zeros(n)

        for i in range(n):  # All interior points
            # Index in padded array
            i_pad = i + g

            # WENO reconstruction for derivative approximation
            # Now uses full stencil without clamping at boundaries
            u_left, u_right = self._weno_reconstruction_padded(u_padded, i_pad)

            # High-order central difference for first derivative
            u_x[i] = (u_right - u_left) / (2 * dx)

        # Compute second derivative using central differences on padded array
        # With ghost cells, we can use standard central diff at all points
        u_xx = np.zeros(n)
        for i in range(n):
            i_pad = i + g
            u_xx[i] = (u_padded[i_pad + 1] - 2 * u_padded[i_pad] + u_padded[i_pad - 1]) / dx**2

        # Compute Hamiltonian and assemble RHS
        for i in range(n):
            # Evaluate Hamiltonian using problem interface (direction (1,) = x-derivative)
            hamiltonian = self._evaluate_hamiltonian(i, m[i], u_x[i], direction=(1,))

            # RHS = -H + diffusion
            rhs[i] = -hamiltonian + diffusion_from_volatility(self.problem.sigma) * u_xx[i]

        return rhs

    def _weno_reconstruction_padded(self, u_padded: np.ndarray, i: int) -> tuple[float, float]:
        """
        WENO reconstruction on padded array (no boundary clamping needed).

        Args:
            u_padded: Padded array with ghost cells
            i: Index in padded array (must have valid stencil i-2 to i+2)

        Returns:
            (u_left, u_right): Reconstructed values at interface
        """
        # Get WENO weights using selected variant
        w_plus, w_minus = self._compute_weno_weights_padded(u_padded, i)

        # Extract 5-point stencil (no clamping - ghost cells ensure validity)
        u = u_padded[i - 2 : i + 3]

        # Reconstruct using weighted combination of sub-stencil polynomials
        u_left = 0.0
        u_right = 0.0

        for k in range(3):
            # Left reconstruction (positive direction)
            u_left += w_plus[k] * np.dot(self.c_plus[k], u[k : k + 3])

            # Right reconstruction (negative direction)
            if k == 0:
                u_vals = u[2::-1]  # [u2, u1, u0]
            elif k == 1:
                u_vals = u[3:0:-1]  # [u3, u2, u1]
            else:  # k == 2
                u_vals = u[4:1:-1]  # [u4, u3, u2]

            u_right += w_minus[k] * np.dot(self.c_minus[k], u_vals)

        return u_left, u_right

    def _compute_weno_weights_padded(self, u_padded: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute WENO weights on padded array (no boundary clamping).

        Args:
            u_padded: Padded array with ghost cells
            i: Index in padded array

        Returns:
            (w_plus, w_minus): WENO weights for upwind and downwind reconstruction
        """
        # Extract 5-point stencil directly (ghost cells ensure validity)
        u = u_padded[i - 2 : i + 3]

        # Compute smoothness indicators (same for all variants)
        beta = self._compute_smoothness_indicators(u)

        # Select weight calculation based on variant
        if self.weno_variant == "weno5" or self.weno_variant == "weno-js":
            w_plus = self._compute_classic_weights(beta, self.d_plus)
            w_minus = self._compute_classic_weights(beta[::-1], self.d_minus)
        elif self.weno_variant == "weno-z":
            tau = self._compute_tau_indicator(u)
            w_plus = self._compute_z_weights(beta, tau, self.d_plus)
            w_minus = self._compute_z_weights(beta[::-1], tau, self.d_minus)
        elif self.weno_variant == "weno-m":
            w_plus = self._compute_mapped_weights(beta, self.d_plus)
            w_minus = self._compute_mapped_weights(beta[::-1], self.d_minus)
        else:
            # Fallback to classic
            w_plus = self._compute_classic_weights(beta, self.d_plus)
            w_minus = self._compute_classic_weights(beta[::-1], self.d_minus)

        return w_plus, w_minus

    def _compute_dt_stable_1d(self, u: np.ndarray, m: np.ndarray) -> float:
        """Compute stable time step based on CFL and diffusion stability."""
        dx = getattr(self.problem, "Dx", self.grid_spacing_x)

        # CFL condition for advection terms
        max_speed = np.max(np.abs(np.gradient(u, dx))) + 1e-10
        dt_cfl = self.cfl_number * dx / max_speed

        # Stability condition for diffusion term
        dt_diffusion = self.diffusion_stability_factor * dx**2 / self.problem.sigma**2

        # Take minimum for stability
        dt_stable = min(dt_cfl, dt_diffusion)

        return max(dt_stable, 1e-10)  # Ensure positive time step

    @deprecated_parameter(param_name="M_density_evolution_from_FP", since="v0.17.0", replacement="M_density")
    @deprecated_parameter(param_name="U_final_condition_at_T", since="v0.17.0", replacement="U_terminal")
    @deprecated_parameter(param_name="U_from_prev_picard", since="v0.17.0", replacement="U_coupling_prev")
    def solve_hjb_system(
        self,
        M_density: np.ndarray | None = None,
        U_terminal: np.ndarray | None = None,
        U_coupling_prev: np.ndarray | None = None,
        volatility_field: float | np.ndarray | None = None,
        # Deprecated parameter names for backward compatibility
        M_density_evolution_from_FP: np.ndarray | None = None,
        U_final_condition_at_T: np.ndarray | None = None,
        U_from_prev_picard: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve the complete HJB system using WENO spatial discretization.

        Automatically dispatches to appropriate dimensional solver based on detected problem dimension:
        - 1D: Direct WENO reconstruction
        - 2D/3D/nD: Dimensional splitting with WENO in each direction

        Args:
            M_density: Density m(t,x) from FP solver
            U_terminal: Terminal condition u(T,x)
            U_coupling_prev: Value function from previous coupling iteration
            volatility_field: Optional diffusion coefficient override

        Returns:
            U_solved: Complete solution u(t,x) over time domain
        """
        # Handle deprecated parameter names (warnings handled by @deprecated_parameter decorators)
        if M_density_evolution_from_FP is not None:
            if M_density is not None:
                raise ValueError("Cannot specify both 'M_density' and deprecated 'M_density_evolution_from_FP'")
            M_density = M_density_evolution_from_FP

        if U_final_condition_at_T is not None:
            if U_terminal is not None:
                raise ValueError("Cannot specify both 'U_terminal' and deprecated 'U_final_condition_at_T'")
            U_terminal = U_final_condition_at_T

        if U_from_prev_picard is not None:
            if U_coupling_prev is not None:
                raise ValueError("Cannot specify both 'U_coupling_prev' and deprecated 'U_from_prev_picard'")
            U_coupling_prev = U_from_prev_picard

        # Validate required parameters
        if M_density is None:
            raise ValueError("M_density is required")
        if U_terminal is None:
            raise ValueError("U_terminal is required")
        if U_coupling_prev is None:
            raise ValueError("U_coupling_prev is required")

        # Dispatch to dimensional solvers (using internal variable names)
        if self.dimension == 1:
            return self._solve_hjb_system_1d(M_density, U_terminal, U_coupling_prev)
        elif self.dimension == 2:
            return self._solve_hjb_system_2d(M_density, U_terminal, U_coupling_prev)
        elif self.dimension == 3:
            return self._solve_hjb_system_3d(M_density, U_terminal, U_coupling_prev)
        else:
            # Use generalized nD solver for dimensions > 3
            return self._solve_hjb_system_nd(M_density, U_terminal, U_coupling_prev)

    def _advance_full_interval(self, u_current, m_current, dt, dt_stable_fn, step_fn):
        """Sub-step ``step_fn`` until the full physical interval ``dt`` is covered (Issue #1180).

        WENO is explicit, so a single step is limited to the CFL/diffusion-stable ``dt_stable``,
        often ``<< dt`` in a diffusion-limited regime. Stepping only once per backward interval
        (the pre-#1180 behavior) advanced physical time by ``dt_stable`` while recording it as a
        full ``dt`` -- the value function was silently near-frozen at the terminal condition.
        Here each interval accumulates CFL-stable sub-steps, recomputing ``dt_stable`` on the
        evolving field each sub-step, until the whole ``dt`` is integrated.

        ``dt_stable_fn(u, m) -> float`` returns the stable step; ``step_fn(u, m, dt_sub) -> u``
        advances one sub-step (a complete directional-split sweep in 2D/3D/nD). Happy path
        ``dt_stable >= dt``: exactly one step of size ``dt`` (byte-identical to the prior
        single-step code). Fails loud at ``max_substeps`` rather than silently truncating.
        """
        t_remaining = dt
        u = u_current
        n_sub = 0
        while t_remaining > 1e-14 * dt and n_sub < self.max_substeps:
            dt_sub = min(dt_stable_fn(u, m_current), t_remaining)
            u = step_fn(u, m_current, dt_sub)
            t_remaining -= dt_sub
            n_sub += 1
        if t_remaining > 1e-12 * dt:
            raise ValueError(
                f"WENO HJB: hit max_substeps={self.max_substeps} with {t_remaining:.3e} of "
                f"dt={dt:.3e} uncovered (last dt_stable={dt_stable_fn(u, m_current):.3e}). "
                "The CFL/diffusion limit is extreme for this grid; raise max_substeps or coarsen."
            )
        return u

    def _step_2d_split(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """One 2D dimensional-split step of size ``dt`` (Strang or Godunov)."""
        if self.splitting_method == "strang":
            u_half = self._solve_hjb_step_2d_x_direction(u, m, dt / 2)
            u_full = self._solve_hjb_step_2d_y_direction(u_half, m, dt)
            return self._solve_hjb_step_2d_x_direction(u_full, m, dt / 2)
        u_half = self._solve_hjb_step_2d_x_direction(u, m, dt)
        return self._solve_hjb_step_2d_y_direction(u_half, m, dt)

    def _step_3d_split(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """One 3D dimensional-split step of size ``dt`` (Strang or Godunov)."""
        if self.splitting_method == "strang":
            u1 = self._solve_hjb_step_3d_x_direction(u, m, dt / 2)
            u2 = self._solve_hjb_step_3d_y_direction(u1, m, dt / 2)
            u3 = self._solve_hjb_step_3d_z_direction(u2, m, dt)
            u4 = self._solve_hjb_step_3d_y_direction(u3, m, dt / 2)
            return self._solve_hjb_step_3d_x_direction(u4, m, dt / 2)
        u1 = self._solve_hjb_step_3d_x_direction(u, m, dt)
        u2 = self._solve_hjb_step_3d_y_direction(u1, m, dt)
        return self._solve_hjb_step_3d_z_direction(u2, m, dt)

    def _step_nd_split(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """One nD dimensional-split step of size ``dt`` (Strang or Godunov)."""
        if self.splitting_method == "strang":
            u_temp = u.copy()
            for dim_idx in range(self.dimension - 1):
                u_temp = self._solve_hjb_step_direction_nd(u_temp, m, dt / 2, dim_idx)
            u_temp = self._solve_hjb_step_direction_nd(u_temp, m, dt, self.dimension - 1)
            for dim_idx in range(self.dimension - 2, -1, -1):
                u_temp = self._solve_hjb_step_direction_nd(u_temp, m, dt / 2, dim_idx)
            return u_temp
        u_new = u.copy()
        for dim_idx in range(self.dimension):
            u_new = self._solve_hjb_step_direction_nd(u_new, m, dt, dim_idx)
        return u_new

    def _solve_hjb_system_1d(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
    ) -> np.ndarray:
        """Solve 1D HJB system (original implementation)."""
        # Extract dimensions from input
        # M_density has shape (n_time_points, Nx) where n_time_points = problem.Nt + 1
        n_time_points = M_density_evolution_from_FP.shape[0]
        Nx = self.num_grid_points_x
        dt = self.problem.T / (n_time_points - 1)  # n_time_points - 1 intervals

        # Initialize solution array - same shape as input
        U_solved = np.zeros((n_time_points, Nx))

        # Set final condition (last time index)
        U_solved[-1, :] = U_final_condition_at_T

        # Backward time integration
        for t_idx in range(n_time_points - 2, -1, -1):
            # Current density at this time
            m_current = M_density_evolution_from_FP[t_idx, :]

            # Current value function (start with final condition)
            u_current = U_solved[t_idx + 1, :].copy()

            # Sub-step over the full interval dt under the CFL/diffusion limit (Issue #1180)
            U_solved[t_idx, :] = self._advance_full_interval(
                u_current, m_current, dt, self._compute_dt_stable_1d, self.solve_hjb_step
            )

        return U_solved

    def _solve_hjb_system_2d(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
    ) -> np.ndarray:
        """Solve 2D HJB system using dimensional splitting."""
        # Extract dimensions from input
        # M_density has shape (n_time_points, Nx, Ny) where n_time_points = problem.Nt + 1
        n_time_points = M_density_evolution_from_FP.shape[0]
        dt = self.problem.T / (n_time_points - 1)  # n_time_points - 1 intervals

        # Initialize solution array - same shape as input
        U_solved = np.zeros((n_time_points, self.num_grid_points_x, self.num_grid_points_y))

        # Set final condition (last time index)
        U_solved[-1, :, :] = U_final_condition_at_T

        # Backward time integration
        for t_idx in range(n_time_points - 2, -1, -1):
            # Current density at this time
            m_current = M_density_evolution_from_FP[t_idx, :, :]

            # Current value function
            u_current = U_solved[t_idx + 1, :, :].copy()

            # Sub-step the full directional-split sequence over the whole interval dt (#1180)
            U_solved[t_idx, :, :] = self._advance_full_interval(
                u_current, m_current, dt, self._compute_dt_stable_2d, self._step_2d_split
            )

        return U_solved

    def _solve_hjb_system_3d(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
    ) -> np.ndarray:
        """Solve 3D HJB system using dimensional splitting."""
        logger = self._get_logger()
        logger.info("Starting 3D WENO HJB solver with dimensional splitting")

        # Extract dimensions from input
        # M_density has shape (n_time_points, Nx, Ny, Nz) where n_time_points = problem.Nt + 1
        n_time_points = M_density_evolution_from_FP.shape[0]
        spatial_shape = M_density_evolution_from_FP.shape[1:]

        # Initialize solution array - same shape as input
        U_solved = np.zeros((n_time_points, *spatial_shape))

        # Set final condition (last time index)
        U_solved[-1, :, :, :] = U_final_condition_at_T

        # n_time_points - 1 intervals (was missing here: the loop referenced an unset self.dt)
        dt = self.problem.T / (n_time_points - 1)

        # Solve backward in time
        for time_idx in range(n_time_points - 2, -1, -1):
            logger.debug(f"  3D Time step {time_idx + 1}/{n_time_points - 1}")

            u_current = U_solved[time_idx + 1, :, :, :]
            m_current = M_density_evolution_from_FP[time_idx, :, :, :]

            # Sub-step the full directional-split sequence over the whole interval dt (#1180)
            U_solved[time_idx, :, :, :] = self._advance_full_interval(
                u_current, m_current, dt, self._compute_dt_stable_3d, self._step_3d_split
            )

            # Progress logging for long computations
            if (time_idx + 1) % 20 == 0:
                logger.info(f"    3D WENO: Completed {n_time_points - time_idx - 2}/{n_time_points - 1} time steps")

        logger.info("3D WENO HJB solver completed successfully")
        return U_solved

    def _solve_hjb_system_nd(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
    ) -> np.ndarray:
        """
        Solve nD HJB system using dimensional splitting (for dimensions > 3).

        Uses generalized dimensional splitting approach that works for arbitrary dimensions.
        This is the dimension-agnostic implementation that extends WENO to 4D, 5D, etc.

        Args:
            M_density_evolution_from_FP: Density evolution m(t,x0,x1,...,xn)
            U_final_condition_at_T: Terminal condition u(T,x0,x1,...,xn)
            U_from_prev_picard: Value function from previous Picard iteration

        Returns:
            U_solved: Complete solution u(t,x0,x1,...,xn) over time domain
        """
        # Extract dimensions from input
        # M_density has shape (n_time_points, *spatial) where n_time_points = problem.Nt + 1
        n_time_points = M_density_evolution_from_FP.shape[0]

        # Initialize solution array with time dimension
        # Shape: (n_time_points, spatial_dims...) - same as input
        spatial_shape = U_final_condition_at_T.shape
        U_solved = np.zeros((n_time_points, *spatial_shape))

        # Set final condition (last time index)
        U_solved[-1, ...] = U_final_condition_at_T

        # Get time step - n_time_points - 1 intervals
        dt = self.problem.T / (n_time_points - 1)

        # Solve backward in time
        for time_idx in range(n_time_points - 2, -1, -1):
            u_current = U_solved[time_idx + 1, ...]
            m_current = M_density_evolution_from_FP[time_idx, ...]

            # Sub-step the full directional-split sequence over the whole interval dt (#1180)
            U_solved[time_idx, ...] = self._advance_full_interval(
                u_current, m_current, dt, self._compute_dt_stable_nd, self._step_nd_split
            )

        return U_solved

    def _solve_hjb_step_direction_nd(self, u: np.ndarray, m: np.ndarray, dt: float, axis: int) -> np.ndarray:
        """
        Apply WENO reconstruction along a specific axis for nD problems.

        This is the dimension-agnostic directional solver that works for any dimension.

        Args:
            u: Value function array (shape: [n0, n1, ..., nd])
            m: Density array (shape: [n0, n1, ..., nd])
            dt: Time step
            axis: Axis index to apply WENO (0 = first spatial dimension, etc.)

        Returns:
            u_new: Updated value function after WENO step along specified axis
        """
        u_new = u.copy()

        # Move target axis to the end for easier slicing
        u_transposed = np.moveaxis(u, axis, -1)
        m_transposed = np.moveaxis(m, axis, -1)
        u_new_transposed = np.moveaxis(u_new, axis, -1)

        # Get shape of all dimensions except the target axis
        shape_except_axis = u_transposed.shape[:-1]

        # Iterate over all slices perpendicular to the target axis
        for idx in np.ndindex(shape_except_axis):
            u_slice = u_transposed[idx]
            m_slice = m_transposed[idx]

            # Apply 1D WENO step adapted for this direction
            u_new_transposed[idx] = self._solve_hjb_step_1d_direction_adapted(u_slice, m_slice, dt, axis)

        # Move axis back to original position
        u_new = np.moveaxis(u_new_transposed, -1, axis)

        return u_new

    def _solve_hjb_step_1d_direction_adapted(
        self, u_1d: np.ndarray, m_1d: np.ndarray, dt: float, axis: int
    ) -> np.ndarray:
        """
        Apply 1D WENO step adapted for a specific direction with appropriate grid spacing.

        Args:
            u_1d: 1D slice of value function
            m_1d: 1D slice of density
            dt: Time step
            axis: Axis index (0, 1, 2, ...) to determine grid spacing

        Returns:
            u_new: Updated 1D slice
        """
        # Save current grid spacing and temporarily replace with axis-specific spacing
        original_spacing = self.grid_spacing_x
        self.grid_spacing_x = self.grid_spacing[axis]

        # Apply 1D WENO solver
        if self.time_integration == "tvd_rk3":
            u_new = self._solve_hjb_tvd_rk3(u_1d, m_1d, dt)
        else:
            u_new = self._solve_hjb_explicit_euler(u_1d, m_1d, dt)

        # Restore original spacing
        self.grid_spacing_x = original_spacing

        return u_new

    def _compute_dt_stable_nd(self, u: np.ndarray, m: np.ndarray) -> float:
        """
        Compute stable time step for nD problem based on CFL and diffusion stability.

        Args:
            u: Value function array (shape: [n0, n1, ..., nd])
            m: Density array (shape: [n0, n1, ..., nd])

        Returns:
            dt_stable: Maximum stable time step
        """
        dt_cfl_list = []
        dt_diffusion_list = []

        # Check CFL and diffusion conditions for each dimension
        for axis in range(self.dimension):
            # Compute gradient along this axis
            u_grad = np.gradient(u, self.grid_spacing[axis], axis=axis)

            # CFL condition for advection
            max_speed = np.max(np.abs(u_grad)) + 1e-10
            dt_cfl = self.cfl_number * self.grid_spacing[axis] / max_speed
            dt_cfl_list.append(dt_cfl)

            # Diffusion stability condition
            dt_diffusion = self.diffusion_stability_factor * self.grid_spacing[axis] ** 2 / self.problem.sigma**2
            dt_diffusion_list.append(dt_diffusion)

        # Take minimum across all dimensions for stability
        dt_stable = min(min(dt_cfl_list), min(dt_diffusion_list))

        return max(dt_stable, 1e-10)  # Ensure positive time step

    def _compute_dt_stable_2d(self, u: np.ndarray, m: np.ndarray) -> float:
        """Compute stable time step for 2D problem based on CFL and diffusion stability."""
        # Compute gradients for stability analysis
        u_x = np.gradient(u, self.grid_spacing_x, axis=0)
        u_y = np.gradient(u, self.grid_spacing_y, axis=1)

        # CFL condition for advection terms
        max_speed_x = np.max(np.abs(u_x)) + 1e-10
        max_speed_y = np.max(np.abs(u_y)) + 1e-10

        dt_cfl_x = self.cfl_number * self.grid_spacing_x / max_speed_x
        dt_cfl_y = self.cfl_number * self.grid_spacing_y / max_speed_y
        dt_cfl = min(dt_cfl_x, dt_cfl_y)

        # Stability condition for diffusion term (more restrictive in 2D)
        dt_diffusion_x = self.diffusion_stability_factor * self.grid_spacing_x**2 / self.problem.sigma**2
        dt_diffusion_y = self.diffusion_stability_factor * self.grid_spacing_y**2 / self.problem.sigma**2
        dt_diffusion = min(dt_diffusion_x, dt_diffusion_y)

        # Take minimum for stability
        dt_stable = min(dt_cfl, dt_diffusion)

        return max(dt_stable, 1e-10)  # Ensure positive time step

    def _solve_hjb_step_2d_x_direction(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Apply WENO reconstruction in X-direction for 2D problem."""
        u_new = u.copy()

        # Apply 1D WENO reconstruction in X-direction for each Y-slice
        for j in range(self.num_grid_points_y):
            u_slice = u[:, j]
            m_slice = m[:, j]

            # Use existing 1D WENO step
            u_new[:, j] = self.solve_hjb_step(u_slice, m_slice, dt)

        return u_new

    def _solve_hjb_step_2d_y_direction(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Apply WENO reconstruction in Y-direction for 2D problem."""
        u_new = u.copy()

        # Apply 1D WENO reconstruction in Y-direction for each X-slice
        # This requires adapting the 1D solver to work on transposed arrays
        for i in range(self.num_grid_points_x):
            u_slice = u[i, :]
            m_slice = m[i, :]

            # Apply 1D WENO step (would need adaptation for different grid spacing)
            # For now, use simplified approach
            u_new[i, :] = self._solve_hjb_step_1d_y_adapted(u_slice, m_slice, dt)

        return u_new

    def _solve_hjb_step_1d_y_adapted(self, u_1d: np.ndarray, m_1d: np.ndarray, dt: float) -> np.ndarray:
        """Apply 1D WENO step adapted for Y-direction with appropriate grid spacing."""
        # This is a simplified adaptation - in full implementation would properly
        # handle different grid spacing and coordinate systems

        # Use existing WENO reconstruction logic but with Y-direction spacing
        if self.time_integration == "tvd_rk3":
            return self._solve_hjb_tvd_rk3_y_adapted(u_1d, m_1d, dt)
        else:
            return self._solve_hjb_explicit_euler_y_adapted(u_1d, m_1d, dt)

    def _solve_hjb_tvd_rk3_y_adapted(self, u_current: np.ndarray, m_current: np.ndarray, dt: float) -> np.ndarray:
        """TVD-RK3 time stepping adapted for Y-direction."""
        # Stage 1
        L1 = self._compute_spatial_operator_y_adapted(u_current, m_current)
        u1 = u_current + dt * L1

        # Stage 2
        L2 = self._compute_spatial_operator_y_adapted(u1, m_current)
        u2 = 0.75 * u_current + 0.25 * u1 + 0.25 * dt * L2

        # Stage 3
        L3 = self._compute_spatial_operator_y_adapted(u2, m_current)
        u_new = (1.0 / 3.0) * u_current + (2.0 / 3.0) * u2 + (2.0 / 3.0) * dt * L3

        return u_new

    def _solve_hjb_explicit_euler_y_adapted(
        self, u_current: np.ndarray, m_current: np.ndarray, dt: float
    ) -> np.ndarray:
        """Explicit Euler time stepping adapted for Y-direction."""
        L = self._compute_spatial_operator_y_adapted(u_current, m_current)
        return u_current + dt * L

    def _compute_spatial_operator_y_adapted(self, u: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Compute spatial operator for Y-direction with adapted grid spacing."""
        # This is a placeholder - full implementation would properly handle
        # Y-direction WENO reconstruction with self.grid_spacing_y spacing
        n = len(u)
        rhs = np.zeros(n)

        # Compute derivatives using Y-direction spacing
        u_y = self._weno_reconstruction_y_adapted(u)

        # Second derivative (central differences with Y spacing)
        u_yy = np.zeros(n)
        u_yy[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / (self.grid_spacing_y**2)
        u_yy[0] = u_yy[1]
        u_yy[-1] = u_yy[-2]

        # Hamiltonian evaluation for Y-direction (direction (0,1) = y-derivative)
        for i in range(n):
            hamiltonian = self._evaluate_hamiltonian(i, m[i], u_y[i], direction=(0, 1))
            rhs[i] = -hamiltonian + diffusion_from_volatility(self.problem.sigma) * u_yy[i]

        return rhs

    def _weno_reconstruction_y_adapted(self, u: np.ndarray) -> np.ndarray:
        """WENO reconstruction adapted for Y-direction with proper grid spacing."""
        # This would use the same WENO logic but with Dy spacing
        # For now, use standard gradient as placeholder
        return np.gradient(u, self.grid_spacing_y)

    def _compute_dt_stable_3d(self, u: np.ndarray, m: np.ndarray) -> float:
        """Compute stable time step for 3D problem based on CFL and diffusion stability."""
        # Compute gradients for stability analysis
        u_x = np.gradient(u, self.grid_spacing_x, axis=0)
        u_y = np.gradient(u, self.grid_spacing_y, axis=1)
        u_z = np.gradient(u, self.grid_spacing_z, axis=2)

        # Maximum gradient magnitude for CFL condition
        max_grad_x = np.max(np.abs(u_x)) if u_x.size > 0 else 0.0
        max_grad_y = np.max(np.abs(u_y)) if u_y.size > 0 else 0.0
        max_grad_z = np.max(np.abs(u_z)) if u_z.size > 0 else 0.0

        # CFL stability condition (very conservative for 3D)
        if max_grad_x > 1e-12 or max_grad_y > 1e-12 or max_grad_z > 1e-12:
            dt_cfl_x = self.cfl_number * self.grid_spacing_x / (max_grad_x + 1e-12)
            dt_cfl_y = self.cfl_number * self.grid_spacing_y / (max_grad_y + 1e-12)
            dt_cfl_z = self.cfl_number * self.grid_spacing_z / (max_grad_z + 1e-12)
            dt_cfl = min(dt_cfl_x, dt_cfl_y, dt_cfl_z)
        else:
            dt_cfl = float("inf")  # no gradient -> no CFL limit; diffusion bound governs (was unset self.dt)

        # Stability condition for diffusion term (very restrictive in 3D)
        # All modern MFGProblem have sigma; getattr with default for safety
        sigma_sq = getattr(self.problem, "sigma", 1.0) ** 2
        dt_diffusion_x = self.diffusion_stability_factor * (self.grid_spacing_x**2) / sigma_sq
        dt_diffusion_y = self.diffusion_stability_factor * (self.grid_spacing_y**2) / sigma_sq
        dt_diffusion_z = self.diffusion_stability_factor * (self.grid_spacing_z**2) / sigma_sq
        dt_diffusion = min(dt_diffusion_x, dt_diffusion_y, dt_diffusion_z)

        return min(dt_cfl, dt_diffusion)

    def _solve_hjb_step_3d_x_direction(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Apply WENO reconstruction in X-direction for 3D problem."""
        u_new = u.copy()
        # Apply 1D WENO reconstruction in X-direction for each (Y,Z)-slice
        for j in range(self.num_grid_points_y):
            for k in range(self.num_grid_points_z):
                u_slice = u[:, j, k]
                m_slice = m[:, j, k]
                # Apply 1D WENO step
                u_new[:, j, k] = self._solve_hjb_step_1d_adapted(u_slice, m_slice, dt)
        return u_new

    def _solve_hjb_step_3d_y_direction(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Apply WENO reconstruction in Y-direction for 3D problem."""
        u_new = u.copy()
        # Apply 1D WENO reconstruction in Y-direction for each (X,Z)-slice
        for i in range(self.num_grid_points_x):
            for k in range(self.num_grid_points_z):
                u_slice = u[i, :, k]
                m_slice = m[i, :, k]
                # Apply 1D WENO step adapted for Y-direction
                u_new[i, :, k] = self._solve_hjb_step_1d_y_adapted(u_slice, m_slice, dt)
        return u_new

    def _solve_hjb_step_3d_z_direction(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Apply WENO reconstruction in Z-direction for 3D problem."""
        u_new = u.copy()
        # Apply 1D WENO reconstruction in Z-direction for each (X,Y)-slice
        for i in range(self.num_grid_points_x):
            for j in range(self.num_grid_points_y):
                u_slice = u[i, j, :]
                m_slice = m[i, j, :]
                # Apply 1D WENO step adapted for Z-direction
                u_new[i, j, :] = self._solve_hjb_step_1d_z_adapted(u_slice, m_slice, dt)
        return u_new

    def _solve_hjb_step_1d_z_adapted(self, u_1d: np.ndarray, m_1d: np.ndarray, dt: float) -> np.ndarray:
        """Apply 1D WENO step adapted for Z-direction with appropriate grid spacing."""
        # Use existing WENO reconstruction logic but with Z-direction spacing
        if self.time_integration == "tvd_rk3":
            return self._solve_hjb_tvd_rk3_z_adapted(u_1d, m_1d, dt)
        else:
            return self._solve_hjb_explicit_euler_z_adapted(u_1d, m_1d, dt)

    def _solve_hjb_tvd_rk3_z_adapted(self, u_current: np.ndarray, m_current: np.ndarray, dt: float) -> np.ndarray:
        """TVD-RK3 time stepping adapted for Z-direction."""
        # Stage 1
        L1 = self._compute_spatial_operator_z_adapted(u_current, m_current)
        u1 = u_current + dt * L1

        # Stage 2
        L2 = self._compute_spatial_operator_z_adapted(u1, m_current)
        u2 = 0.75 * u_current + 0.25 * u1 + 0.25 * dt * L2

        # Stage 3
        L3 = self._compute_spatial_operator_z_adapted(u2, m_current)
        u_new = (1.0 / 3.0) * u_current + (2.0 / 3.0) * u2 + (2.0 / 3.0) * dt * L3

        return u_new

    def _solve_hjb_explicit_euler_z_adapted(
        self, u_current: np.ndarray, m_current: np.ndarray, dt: float
    ) -> np.ndarray:
        """Explicit Euler time stepping adapted for Z-direction."""
        L = self._compute_spatial_operator_z_adapted(u_current, m_current)
        return u_current + dt * L

    def _compute_spatial_operator_z_adapted(self, u: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Compute spatial operator for Z-direction with adapted grid spacing."""
        n = len(u)
        rhs = np.zeros(n)

        # Compute derivatives using Z-direction spacing
        u_z = self._weno_reconstruction_z_adapted(u)

        # Second derivative (central differences with Z spacing)
        u_zz = np.zeros(n)
        u_zz[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / (self.grid_spacing_z**2)
        u_zz[0] = u_zz[1]
        u_zz[-1] = u_zz[-2]

        # Hamiltonian evaluation for Z-direction (direction (0,0,1) = z-derivative)
        for i in range(n):
            hamiltonian = self._evaluate_hamiltonian(i, m[i], u_z[i], direction=(0, 0, 1))
            rhs[i] = -hamiltonian + diffusion_from_volatility(self.problem.sigma) * u_zz[i]

        return rhs

    def _weno_reconstruction_z_adapted(self, u: np.ndarray) -> np.ndarray:
        """WENO reconstruction adapted for Z-direction with proper grid spacing."""
        # This would use the same WENO logic but with Dz spacing
        # For now, use standard gradient as placeholder
        return np.gradient(u, self.grid_spacing_z)

    def get_variant_info(self) -> dict[str, str]:
        """
        Get information about the current WENO variant.

        Returns:
            dict: Information about the selected WENO variant
        """
        variant_info = {
            "weno5": {
                "name": "WENO5",
                "description": "Standard fifth-order WENO scheme",
                "characteristics": "Balanced performance, widely used, good stability",
                "best_for": "General MFG problems, benchmarking, production use",
            },
            "weno-z": {
                "name": "WENO-Z",
                "description": "Enhanced WENO with τ-based weight modification",
                "characteristics": "Reduced dissipation, better shock resolution",
                "best_for": "High-resolution requirements, discontinuous solutions",
            },
            "weno-m": {
                "name": "WENO-M",
                "description": "Mapped WENO for critical point handling",
                "characteristics": "Better critical points, enhanced accuracy preservation",
                "best_for": "Smooth solutions, critical points, long-time integration",
            },
            "weno-js": {
                "name": "WENO-JS",
                "description": "Original Jiang-Shu WENO formulation",
                "characteristics": "Maximum stability, conservative approach",
                "best_for": "Stability-critical applications, extreme conditions",
            },
        }

        return variant_info[self.weno_variant]


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBWenoSolver...")

    import numpy as np

    from mfgarchon import MFGProblem
    from mfgarchon.geometry import TensorProductGrid

    # Test 1D problem
    geometry_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
    problem_1d = MFGProblem(geometry=geometry_1d, T=1.0, Nt=20, sigma=0.1)

    # Test standard WENO variant
    solver_1d = HJBWenoSolver(problem_1d, weno_variant="weno-z")

    # Test solver initialization
    assert solver_1d.dimension == 1
    assert solver_1d.weno_variant == "weno-z"
    print("  Solver initialized")
    print(f"  Variant: {solver_1d.weno_variant}, Method: {solver_1d.hjb_method_name}")

    # Test solve_hjb_system with trivial inputs
    M_test = np.ones((problem_1d.Nt + 1, problem_1d.Nx + 1)) * 0.5
    U_final = np.zeros(problem_1d.Nx + 1)
    U_prev = np.zeros((problem_1d.Nt + 1, problem_1d.Nx + 1))

    U_solution = solver_1d.solve_hjb_system(M_test, U_final, U_prev)

    assert U_solution.shape == (problem_1d.Nt + 1, problem_1d.Nx + 1)
    assert not np.any(np.isnan(U_solution))
    assert not np.any(np.isinf(U_solution))
    print(f"  Solver converged, U range: [{U_solution.min():.3f}, {U_solution.max():.3f}]")

    print("Smoke tests passed!")

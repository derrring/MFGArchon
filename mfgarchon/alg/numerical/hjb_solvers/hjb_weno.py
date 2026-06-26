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
from mfgarchon.utils.deprecation import deprecated_alias
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


class HJBWENOSolver(BaseHJBSolver):
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
        # Issue #1426: weno_m_parameter is validated and stored but never applied in the WENO-M
        # mapping. Fail loud on a non-default value rather than silently ignoring it.
        if self.weno_m_parameter != 1.0:
            raise NotImplementedError(
                f"weno_m_parameter={self.weno_m_parameter} is not implemented (Issue #1426): it is stored "
                f"but never applied in the WENO-M mapping. Only the default 1.0 is supported."
            )

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

        HJ-WENO5 requires:
        - ghost_depth=3: 3 ghost cells on each side for the one-sided derivative
          stencils p_minus / p_plus (undivided differences over i-3 to i+3, #1200)
        - order=5: High-order polynomial extrapolation for O(h^5) boundary accuracy

        Issue #576: Unified ghost node architecture enables WENO5 to achieve
        true 5th-order convergence at boundaries via Vandermonde extrapolation.

        Uses composition pattern: the solver holds a PreallocatedGhostBuffer
        component rather than inheriting from a BoundaryAwareSolver base class.
        """
        # Ghost depth for the Osher-Shu HJ-WENO5 one-sided derivative stencil.
        # Reconstructing p_minus / p_plus at an interior node needs the undivided
        # differences spanning u_{i-3} .. u_{i+3} (Issue #1200), i.e. 3 ghost cells
        # per side -- not 2. (2 sufficed only for the previous, incorrect, value-
        # interface reconstruction.)
        self.ghost_depth = 3

        # Reconstruction order: 5th-order accuracy at boundaries
        # Issue #576: Uses polynomial extrapolation (not simple reflection)
        self.ghost_order = 5

        # Get boundary conditions from problem/geometry if available
        bc = self._get_boundary_conditions()

        # Build domain bounds from grid information
        domain_bounds = self._get_domain_bounds()

        # Single nD ghost buffer for every dimension. update_ghosts() fills ghosts
        # on all axes; each directional sweep reads the BC-correct line along its
        # axis from the padded buffer (Issue #1200 -- replaces the former
        # per-dimension special-casing and the multi-D None placeholder).
        self.ghost_buffer = PreallocatedGhostBuffer(
            interior_shape=tuple(self.num_grid_points),
            boundary_conditions=bc,
            domain_bounds=domain_bounds,
            ghost_depth=self.ghost_depth,
            order=self.ghost_order,  # High-order ghost cells for WENO5
        )

    def _get_boundary_conditions(self):
        """Resolve boundary conditions through the single-source inherited
        ``BaseMFGSolver.get_boundary_conditions()`` (Issue #634 pattern), falling back to Neumann
        (no-flux) when none is configured.

        Issue #1429 (S0-21): this replaces a private 4-accessor copy of the resolution chain that
        also diverged at the terminal (private ``neumann_bc`` vs inherited ``None`` vs the
        ConditionsMixin ``periodic_bc``). The inherited chain resolves to the same stored BC for a
        real ``MFGProblem``: ``self._boundary_conditions`` is never assigned, so its Priority-1 is a
        no-op, and the geometry/problem attribute and method accessors return the same object. WENO
        keeps a concrete-BC requirement — the ghost buffer cannot take ``None`` — so a no-flux
        Neumann fallback is applied here when the single source yields ``None``.
        """
        bc = self.get_boundary_conditions()
        return bc if bc is not None else neumann_bc(dimension=self.dimension)

    def _get_domain_bounds(self) -> np.ndarray:
        """Get domain bounds from problem/geometry (NO hasattr per CLAUDE.md)."""
        # Priority 1: uniform get_bounds() accessor (Issue #1056). Returns (mins, maxs); rebuild
        # the (d, 2) ndarray contract this method promises. np.array(geometry.bounds) gave the
        # same (d, 2) for grid/rectangle but raised for geometries lacking the ad-hoc .bounds.
        try:
            mins, maxs = self.problem.geometry.get_bounds()
            return np.column_stack([mins, maxs])
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

        # Negative reconstruction coefficients (right-to-left bias).
        # Each row is dotted (in _weno_reconstruction) against the reversed
        # sub-stencil slice, so the value ordering below matches that dot:
        # the k-th comment lists values in the same order as the coefficients.
        self.c_minus = np.array(
            [
                [-1 / 6, 5 / 6, 1 / 3],  # S₀: u_i, u_{i-1}, u_{i-2}
                [1 / 3, 5 / 6, -1 / 6],  # S₁: u_{i+1}, u_i, u_{i-1}
                [1 / 3, -7 / 6, 11 / 6],  # S₂: u_{i+2}, u_{i+1}, u_i
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
        Solve one 1D backward time step of the HJB equation (axis 0).

        Multi-dimensional sweeps call ``_solve_hjb_step_axis`` directly with the
        target axis; this is the public 1D entry point.

        Args:
            u_current: Current value function
            m_current: Current density
            dt: Time step size

        Returns:
            u_new: Updated value function after one time step
        """
        return self._solve_hjb_step_axis(u_current, m_current, dt, axis=0)

    def _solve_hjb_step_axis(self, u: np.ndarray, m: np.ndarray, dt: float, axis: int) -> np.ndarray:
        """One backward time step of the 1D HJB operator along ``axis``.

        The spatial discretisation (HJ-WENO5 derivatives + Lax-Friedrichs
        numerical Hamiltonian + central diffusion) is supplied by
        ``_compute_hjb_rhs_axis``; this method only advances it in time. Used for
        the 1D solve (axis 0) and for every direction of the multi-D
        dimensional split.
        """
        if self.time_integration == "tvd_rk3":
            # Stage 1
            k1 = self._compute_hjb_rhs_axis(u, m, axis)
            u1 = u + dt * k1
            # Stage 2
            k2 = self._compute_hjb_rhs_axis(u1, m, axis)
            u2 = (3 / 4) * u + (1 / 4) * u1 + (1 / 4) * dt * k2
            # Stage 3
            k3 = self._compute_hjb_rhs_axis(u2, m, axis)
            return (1 / 3) * u + (2 / 3) * u2 + (2 / 3) * dt * k3
        elif self.time_integration == "explicit_euler":
            return u + dt * self._compute_hjb_rhs_axis(u, m, axis)
        else:
            raise ValueError(f"Unknown time integration: {self.time_integration}")

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

        # Issue #1071 / fail-fast: do NOT silently fall back to a hardcoded Hamiltonian
        # H = 0.5*|p|^2 + m*p. That substitutes the WRONG physics (a specific LQ-congestion
        # model) for whatever the caller actually defined, returning a plausible-but-incorrect
        # solution with no error — the silent-fallback class this codebase forbids.
        raise ValueError(
            "HJB WENO: no Hamiltonian available (problem.H is absent / raised). Provide a "
            "problem exposing H, e.g. MFGProblem(components=MFGComponents(hamiltonian=...)). "
            "The solver will not silently substitute the LQ default H=0.5*|p|^2 + m*p "
            "(Issue #1071, fail-fast)."
        )

    def _compute_hjb_rhs_axis(self, u: np.ndarray, m: np.ndarray, axis: int) -> np.ndarray:
        """Right-hand side of the HJB equation along a single ``axis``.

        Implements the monotone Hamilton-Jacobi discretisation (Issue #1200):

            RHS = -Hhat(p_minus, p_plus) + D * d^2u/dx_axis^2

        where ``p_minus`` / ``p_plus`` are the Osher-Shu HJ-WENO5 one-sided nodal
        derivatives along ``axis`` and ``Hhat`` is the global Lax-Friedrichs
        numerical Hamiltonian

            Hhat = H((p_minus + p_plus)/2) - (alpha/2) * (p_plus - p_minus),
            alpha = max |dH/dp|.

        The LF viscosity (proportional to ``p_plus - p_minus ~ dx * u_xx``) damps
        the high-frequency modes the previous central-difference scheme amplified,
        while staying O(h^5) where the field is smooth. ``D = sigma^2/2``. Works in
        any dimension: the field is padded on every axis and differentiated along
        ``axis`` only, so this single operator serves both the 1D solve and each
        direction of the multi-D dimensional split.
        """
        dx = self.grid_spacing[axis]
        g = self.ghost_depth

        # BC-aware ghost padding on every axis; the line along `axis` is then exact.
        self.ghost_buffer.interior[:] = u
        self.ghost_buffer.update_ghosts()
        u_padded = self.ghost_buffer.padded

        # One-sided HJ-WENO5 derivatives along `axis`.
        p_minus, p_plus = self._weno5_hj_derivatives(u_padded, axis, dx)

        # Central second derivative along `axis` (ghost cells make it valid everywhere).
        # Restrict non-swept axes to interior so u_aa is interior-shaped on all axes.
        interior = [slice(g, -g)] * u_padded.ndim
        interior[axis] = slice(None)
        ua = np.moveaxis(u_padded[tuple(interior)], axis, -1)
        n = ua.shape[-1] - 2 * g
        u_aa = (ua[..., g + 1 : g + 1 + n] - 2.0 * ua[..., g : g + n] + ua[..., g - 1 : g - 1 + n]) / dx**2
        u_aa = np.moveaxis(u_aa, -1, axis)

        # Per-direction Hamiltonian value at the averaged momentum + LF dissipation.
        p_mid = 0.5 * (p_minus + p_plus)
        direction = tuple(1 if d == axis else 0 for d in range(self.dimension))
        h_mid, alpha = self._directional_hamiltonian_and_speed(m, p_mid, direction, axis)
        h_hat = h_mid - 0.5 * alpha * (p_plus - p_minus)

        diffusion = diffusion_from_volatility(self.problem.sigma)
        return -h_hat + diffusion * u_aa

    def _weno5_hj_derivatives(self, u_padded: np.ndarray, axis: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
        """Osher-Shu HJ-WENO5 one-sided nodal derivatives along ``axis``.

        Returns ``(p_minus, p_plus)`` -- the left- and right-biased fifth-order
        WENO reconstructions of ``du/dx_axis`` at every interior node, shaped like
        the interior field. Both are O(h^5) where the field is smooth and degrade
        gracefully (no Gibbs growth) near kinks. Requires ``ghost_depth >= 3``: the
        stencils span the undivided differences over ``u_{i-3} .. u_{i+3}``.
        """
        g = self.ghost_depth
        # Keep ghosts on the swept axis (the derivative needs them); restrict every
        # other axis to its interior so the result is interior-shaped on all axes.
        interior = [slice(g, -g)] * u_padded.ndim
        interior[axis] = slice(None)
        ua = np.moveaxis(u_padded[tuple(interior)], axis, -1)
        # Undivided differences D[k] = (u[k+1] - u[k]) / dx, living at k+1/2.
        diffs = np.diff(ua, axis=-1) / dx
        n = ua.shape[-1] - 2 * g

        # Left-biased stencil:  v_minus = [D[i],   D[i+1], D[i+2], D[i+3], D[i+4]]
        vm = np.stack([diffs[..., k : k + n] for k in range(5)], axis=-1)
        # Right-biased stencil: v_plus  = [D[i+5], D[i+4], D[i+3], D[i+2], D[i+1]]
        vp = np.stack([diffs[..., (5 - k) : (5 - k) + n] for k in range(5)], axis=-1)

        p_minus = self._weno5_reconstruct_derivative(vm)
        p_plus = self._weno5_reconstruct_derivative(vp)
        return np.moveaxis(p_minus, -1, axis), np.moveaxis(p_plus, -1, axis)

    def _weno5_reconstruct_derivative(self, v: np.ndarray) -> np.ndarray:
        """Fifth-order WENO combination of three candidate derivative stencils.

        ``v`` has shape ``(..., 5)`` and holds the five undivided differences,
        ordered so the smooth (optimal-weight) combination approximates the nodal
        derivative. Honours the selected WENO variant via the nonlinear weights.
        Each candidate's coefficients sum to one, so on a constant-gradient field
        the reconstruction is exact.
        """
        beta = self._weno5_smoothness(v)
        w = self._weno5_nonlinear_weights(beta)
        v1, v2, v3, v4, v5 = (v[..., k] for k in range(5))
        # Candidate third-order derivative reconstructions (Osher-Shu / Jiang-Peng).
        q0 = v1 / 3.0 - 7.0 * v2 / 6.0 + 11.0 * v3 / 6.0
        q1 = -v2 / 6.0 + 5.0 * v3 / 6.0 + v4 / 3.0
        q2 = v3 / 3.0 + 5.0 * v4 / 6.0 - v5 / 6.0
        return w[..., 0] * q0 + w[..., 1] * q1 + w[..., 2] * q2

    @staticmethod
    def _weno5_smoothness(v: np.ndarray) -> np.ndarray:
        """Vectorised WENO5 smoothness indicators from stacked differences ``v`` (..., 5)."""
        v1, v2, v3, v4, v5 = (v[..., k] for k in range(5))
        b0 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3) ** 2
        b1 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
        b2 = (13.0 / 12.0) * (v3 - 2.0 * v4 + v5) ** 2 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5) ** 2
        return np.stack([b0, b1, b2], axis=-1)

    def _weno5_nonlinear_weights(self, beta: np.ndarray) -> np.ndarray:
        """Vectorised nonlinear weights for the HJ derivative reconstruction.

        Optimal linear weights for the one-sided derivative are ``(0.1, 0.6, 0.3)``.
        ``weno5`` / ``weno-js`` use the classic Jiang-Shu weights; ``weno-z`` adds
        the global smoothness measure ``tau_5 = |beta_0 - beta_2|``; ``weno-m``
        applies the Henrick mapping toward the optimal weights.
        """
        eps = self.weno_epsilon
        d = np.array([0.1, 0.6, 0.3])

        if self.weno_variant == "weno-z":
            tau5 = np.abs(beta[..., 0:1] - beta[..., 2:3])
            alpha = d * (1.0 + (tau5 / (eps + beta)) ** self.weno_z_parameter)
        elif self.weno_variant == "weno-m":
            alpha0 = d / (eps + beta) ** 2
            w0 = alpha0 / np.sum(alpha0, axis=-1, keepdims=True)
            # Henrick et al. (2005) mapping toward the optimal linear weights.
            alpha = w0 * (d + d * d - 3.0 * d * w0 + w0 * w0) / (d * d + w0 * (1.0 - 2.0 * d))
        else:  # weno5 / weno-js (classic Jiang-Shu)
            alpha = d / (eps + beta) ** 2

        return alpha / np.sum(alpha, axis=-1, keepdims=True)

    def _directional_hamiltonian_and_speed(
        self, m: np.ndarray, p_mid: np.ndarray, direction: tuple[int, ...], axis: int
    ) -> tuple[np.ndarray, float]:
        """Per-node Hamiltonian ``H(p_mid)`` and global LF speed ``alpha = max|dH/dp|``.

        Reuses ``_evaluate_hamiltonian`` (dimensional-splitting convention: only the
        derivative along ``axis`` is non-zero). ``alpha`` is estimated by a central
        finite difference of ``H`` in ``p``; over-estimating it only adds
        dissipation, so the bound is safe. ``x_idx`` is the index along the swept
        axis, matching the pre-#1200 per-direction operators.
        """
        h_mid = np.empty_like(p_mid)
        speed = np.empty_like(p_mid)
        eps = 1e-7
        for idx in np.ndindex(p_mid.shape):
            x_idx = idx[axis]
            m_val = float(m[idx])
            p = float(p_mid[idx])
            h_mid[idx] = self._evaluate_hamiltonian(x_idx, m_val, p, direction=direction)
            h_plus = self._evaluate_hamiltonian(x_idx, m_val, p + eps, direction=direction)
            h_minus = self._evaluate_hamiltonian(x_idx, m_val, p - eps, direction=direction)
            speed[idx] = abs((h_plus - h_minus) / (2.0 * eps))
        alpha = float(np.max(speed)) if speed.size else 0.0
        return h_mid, alpha

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

    def solve_hjb_system(
        self,
        M_density: np.ndarray | None = None,
        U_terminal: np.ndarray | None = None,
        U_coupling_prev: np.ndarray | None = None,
        volatility_field: float | np.ndarray | None = None,
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
        # Issue #1316: the WENO solver reads diffusion from problem.sigma at multiple
        # scattered sites (the diffusion CFL bound and the diffusion update in each
        # dimensional sweep), with no single sigma chokepoint to redirect. Honoring a
        # volatility_field that differs from problem.sigma would require threading it
        # through all of them; doing nothing would silently solve HJB with problem.sigma
        # while FP uses the field, breaking the Picard correspondence. Fail loud instead
        # of accept-and-ignore. A scalar field equal to problem.sigma is the iterator's
        # redundant forwarding of problem.volatility_field (Issue #1248) and is a no-op.
        if volatility_field is not None and not (
            np.isscalar(volatility_field) and float(volatility_field) == float(self.problem.sigma)
        ):
            raise NotImplementedError(
                "HJBWENOSolver cannot honor a volatility_field that differs from problem.sigma: "
                "it reads diffusion from problem.sigma at multiple sites with no single chokepoint "
                "(Issue #1316). A spatially-varying or mismatched field would make HJB solve a "
                "different diffusion than FP, breaking the Picard fixed point. Use HJBGFDMSolver "
                "(which consumes volatility_field) or set problem.sigma to match."
            )

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
        return self._step_nd_split(u, m, dt)

    def _step_3d_split(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """One 3D dimensional-split step of size ``dt`` (Strang or Godunov)."""
        return self._step_nd_split(u, m, dt)

    def _step_nd_split(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """One dimensional-split step of size ``dt`` (Strang or Godunov), any dim.

        Each axis is advanced by the single unified operator ``_solve_hjb_step_axis``
        (Issue #1200). Strang splitting sweeps axes ``0..d-2`` at half step, the last
        axis ``d-1`` at full step, then ``0..d-2`` again at half step (second-order in
        time); Godunov sweeps each axis once at full step (first-order).
        """
        if self.splitting_method == "strang":
            u_temp = u.copy()
            for dim_idx in range(self.dimension - 1):
                u_temp = self._solve_hjb_step_axis(u_temp, m, dt / 2, dim_idx)
            u_temp = self._solve_hjb_step_axis(u_temp, m, dt, self.dimension - 1)
            for dim_idx in range(self.dimension - 2, -1, -1):
                u_temp = self._solve_hjb_step_axis(u_temp, m, dt / 2, dim_idx)
            return u_temp
        u_new = u.copy()
        for dim_idx in range(self.dimension):
            u_new = self._solve_hjb_step_axis(u_new, m, dt, dim_idx)
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


# Issue #1426: renamed HJBWenoSolver -> HJBWENOSolver (WENO is an acronym; matches the
# all-caps HJBGFDMSolver / HJBFDMSolver siblings and PEP 8 acronym capitalization).
# Deprecated alias kept for backward compatibility (removal per deprecation policy).
HJBWenoSolver = deprecated_alias("HJBWenoSolver", HJBWENOSolver, "v0.20.5")


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBWENOSolver...")

    import numpy as np

    from mfgarchon import MFGProblem
    from mfgarchon.geometry import TensorProductGrid

    # Test 1D problem
    geometry_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
    problem_1d = MFGProblem(geometry=geometry_1d, T=1.0, Nt=20, sigma=0.1)
    n_pts = problem_1d.geometry.num_spatial_points  # number of spatial grid points (Nx + 1)

    # Test standard WENO variant
    solver_1d = HJBWENOSolver(problem_1d, weno_variant="weno-z")

    # Test solver initialization
    assert solver_1d.dimension == 1
    assert solver_1d.weno_variant == "weno-z"
    print("  Solver initialized")
    print(f"  Variant: {solver_1d.weno_variant}, Method: {solver_1d.hjb_method_name}")

    # Test solve_hjb_system with trivial inputs
    M_test = np.ones((problem_1d.Nt + 1, n_pts)) * 0.5
    U_final = np.zeros(n_pts)
    U_prev = np.zeros((problem_1d.Nt + 1, n_pts))

    U_solution = solver_1d.solve_hjb_system(M_test, U_final, U_prev)

    assert U_solution.shape == (problem_1d.Nt + 1, n_pts)
    assert not np.any(np.isnan(U_solution))
    assert not np.any(np.isinf(U_solution))
    print(f"  Solver converged, U range: [{U_solution.min():.3f}, {U_solution.max():.3f}]")

    print("Smoke tests passed!")

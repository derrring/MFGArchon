"""
GFDM-based Fokker-Planck Solver for Meshfree Density Evolution.

This module provides a Fokker-Planck solver using Generalized Finite Difference
Method (GFDM) for spatial derivatives on scattered collocation points.

The solver is suitable for:
- Unstructured/scattered point distributions
- Complex domain geometries
- Meshfree discretizations

Mathematical Formulation:
    Continuity equation: dm/dt + div(m * alpha) = sigma^2/2 * Laplacian(m)

    where:
    - m(t,x): density at collocation points
    - alpha(t,x) = -grad U: drift from value function
    - sigma: volatility coefficient (sigma); diffusion D = sigma^2/2

Author: MFGarchon Development Team
Created: 2025-12-12
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver, DriftConvention
from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import TaylorOperator
from mfgarchon.geometry.boundary.types import BCType
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.boundary import BoundaryConditions


class FPGFDMSolver(BaseFPSolver):
    """
    Fokker-Planck solver using GFDM on collocation points.

    Solves the continuity equation:
        dm/dt + div(m * alpha) = D * Laplacian(m)  where D = sigma^2/2

    using Generalized Finite Difference Method for spatial derivatives
    and forward Euler for time stepping.

    Attributes:
        collocation_points: Scattered points for density evolution, shape (N, d)
        gfdm_operator: Precomputed GFDM operator for spatial derivatives
        delta: Neighborhood radius for GFDM

    Example:
        >>> from mfgarchon import MFGProblem
        >>> from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver
        >>> from mfgarchon.geometry import TensorProductGrid
        >>>
        >>> grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
        >>> problem = MFGProblem(geometry=grid, Nt=20, T=1.0, sigma=0.1)
        >>> points = np.random.rand(100, 1)  # Scattered 1D points
        >>> solver = FPGFDMSolver(problem, collocation_points=points)
        >>>
        >>> m_init = np.exp(-10 * (points[:, 0] - 0.5)**2)
        >>> U_drift = np.zeros((21, 100))  # Zero drift
        >>> M = solver.solve_fp_system(m_init, drift_field=U_drift)
    """

    # Issue #1420 (G-017 V2): this meshfree solver consumes the precomputed velocity α* directly
    # (drift_field), never the value function U — it has no potential_field path. Declared explicit
    # so the coupling layer (resolve_fp_drift_kwargs) does not auto-route U as a velocity.
    _drift_convention = DriftConvention.VELOCITY

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.GFDM

    # BoundaryCapable protocol (Issue #1456): _resolve_boundary_type handles no-flux / Neumann /
    # periodic and returns None (silently) for everything else; declare exactly those so
    # Dirichlet / Robin / Reflecting / Extrapolation fail loud instead.
    _SUPPORTED_BC_TYPES: frozenset = frozenset({BCType.NO_FLUX, BCType.NEUMANN, BCType.PERIODIC})

    @property
    def supported_bc_types(self) -> frozenset:
        """BC types this solver supports (BoundaryCapable protocol)."""
        return self._SUPPORTED_BC_TYPES

    def __init__(
        self,
        problem: MFGProblem,
        collocation_points: np.ndarray,
        delta: float | None = None,
        taylor_order: int = 2,
        weight_function: str = "wendland",
        boundary_indices: set[int] | np.ndarray | None = None,
        domain_bounds: list[tuple[float, float]] | None = None,
        boundary_type: str | None = None,
        boundary_conditions: BoundaryConditions | None = None,
        upwind_scheme: str = "none",
        upwind_strength: float = 0.5,
        obstacle_sdf: object | None = None,
        visibility_samples: int = 10,
        visibility_margin: float = 0.0,
    ):
        """
        Initialize GFDM-based FP solver.

        Args:
            problem: MFG problem definition
            collocation_points: Scattered points for density evolution, shape (N, d)
            delta: Neighborhood radius for GFDM. If None, computed adaptively
                   as 2x median nearest neighbor distance.
            taylor_order: Order of Taylor expansion (1 or 2)
            weight_function: Weight function type ("wendland", "gaussian", "uniform")
            boundary_indices: Set/array of indices of boundary points (optional)
            domain_bounds: List of (min, max) tuples for each dimension (optional)
            boundary_type: Type of boundary condition ("no_flux" or None).
                Deprecated: Use boundary_conditions parameter instead.
            boundary_conditions: BoundaryConditions object from geometry.boundary
                infrastructure. Takes precedence over boundary_type string.
            upwind_scheme: Upwind stabilization scheme ("none", "exponential", "linear")
            upwind_strength: Upwind bias parameter β (typically 0.3-1.0)

        BC Resolution Order:
            1. Explicit boundary_conditions parameter
            2. problem.geometry.boundary_conditions
            3. problem.geometry.get_boundary_conditions()
            4. boundary_type string (legacy, deprecated)
            5. None (no BC enforcement)
        """
        super().__init__(problem)
        self.fp_method_name = "GFDM"

        # Store collocation points
        self.collocation_points = np.asarray(collocation_points)
        if self.collocation_points.ndim == 1:
            self.collocation_points = self.collocation_points.reshape(-1, 1)

        self.n_points = self.collocation_points.shape[0]
        self.dimension = self.collocation_points.shape[1]

        # Compute adaptive delta if not provided
        if delta is None:
            delta = self._compute_adaptive_delta()
        self.delta = delta

        # Resolve boundary conditions using unified infrastructure
        # Issue #527: Integrate with geometry.boundary infrastructure
        resolved_bc_type = self._resolve_boundary_type(
            boundary_conditions=boundary_conditions,
            boundary_type_str=boundary_type,
        )
        # Issue #1456: fail loud if the resolved BC requests a type GFDM cannot honor —
        # _resolve_boundary_type returns None (silently) for Dirichlet/Robin/Reflecting/Extrapolation.
        _bc_for_gate = boundary_conditions if boundary_conditions is not None else self.get_boundary_conditions()
        self._validate_bc_support(_bc_for_gate)

        # Create GFDM operator using TaylorOperator (Strategy Pattern, Issue #844)
        # TaylorOperator handles neighborhoods and derivative weights
        # BC is handled externally (no ghost particles — cleaner separation)
        # Issue #1556: thread the obstacle SDF into the operator so the FP density derivatives
        # (D_lap / D_grad) respect obstacle connectivity, exactly like the HJB-GFDM side (#1124).
        # Without it, in a coupled obstacle-cloud solve the FP stencils couple through walls while
        # the HJB stencils are visibility-filtered — asymmetric physics.
        self.gfdm_operator = TaylorOperator(
            points=self.collocation_points,
            delta=self.delta,
            taylor_order=taylor_order,
            weight_function=weight_function,
            obstacle_sdf=obstacle_sdf,
            visibility_samples=visibility_samples,
            visibility_margin=visibility_margin,
        )
        # Store boundary info for solver-level BC enforcement
        self._boundary_indices = boundary_indices
        self._domain_bounds = domain_bounds
        # Issue #1426 (S0-26): boundary_indices / domain_bounds are accepted and stored but never
        # read — boundary handling uses the problem geometry's boundary conditions. Fail loud on a
        # non-default (non-None) value rather than silently ignoring it.
        if boundary_indices is not None:
            raise NotImplementedError(
                "FPGFDMSolver(boundary_indices=...) is accepted but never applied (Issue #1426): "
                "boundary handling uses the problem geometry's boundary conditions. Remove the "
                "argument or set boundary conditions on problem.geometry."
            )
        if domain_bounds is not None:
            raise NotImplementedError(
                "FPGFDMSolver(domain_bounds=...) is accepted but never applied (Issue #1426): the "
                "domain is taken from problem.geometry. Remove the argument."
            )
        self._boundary_type = resolved_bc_type

        # Store upwind parameters
        self.upwind_scheme = upwind_scheme
        self.upwind_strength = upwind_strength

    def _resolve_boundary_type(
        self,
        boundary_conditions: BoundaryConditions | None,
        boundary_type_str: str | None,
    ) -> str | None:
        """
        Resolve boundary type from unified BC infrastructure.

        Maps BCType enum to string that GFDMOperator expects.

        Args:
            boundary_conditions: BoundaryConditions object (preferred)
            boundary_type_str: Legacy string parameter (deprecated)

        Returns:
            String boundary type for GFDMOperator ("no_flux" or None)

        Note:
            Uses inherited get_boundary_conditions() from BaseNumericalSolver
            for unified BC resolution. Falls back to legacy boundary_type_str.
        """
        from mfgarchon.geometry.boundary import BCType

        # Priority 1: Explicit boundary_conditions parameter
        # Priority 2-5: Use inherited unified BC resolution
        bc = boundary_conditions
        if bc is None:
            bc = self.get_boundary_conditions()

        # Map BoundaryConditions to string for GFDMOperator
        if bc is not None:
            try:
                # Fails loud (ValueError) if default_bc unset; Issue #1100.
                # AttributeError only for legacy objects lacking the unified interface.
                default_bc = bc._resolve_default_bc("FPGFDMSolver._resolve_boundary_type")
            except AttributeError:
                default_bc = None
            if default_bc is not None:
                if default_bc == BCType.NO_FLUX:
                    return "no_flux"
                elif default_bc == BCType.NEUMANN:
                    return "no_flux"  # Neumann with value=0 is equivalent
                elif default_bc == BCType.PERIODIC:
                    return "periodic"  # GFDMOperator may support this in future
                # Other BC types not yet supported by GFDMOperator
                return None

        # Priority: Legacy boundary_type string
        return boundary_type_str

    def _compute_adaptive_delta(self) -> float:
        """Compute adaptive delta based on point spacing."""
        if self.n_points <= 1:
            return 0.1

        from scipy.spatial import cKDTree

        tree = cKDTree(self.collocation_points)
        # Query 2 nearest neighbors (first is self)
        distances, _ = tree.query(self.collocation_points, k=2)
        # Use 2x median nearest neighbor distance
        return 2.0 * float(np.median(distances[:, 1]))

    def _compute_upwind_divergence(
        self,
        drift_field: np.ndarray,
        density: np.ndarray,
    ) -> np.ndarray:
        """
        Compute div(m * α) with streamline-biased weighted finite differences.

        ACTUAL Implementation (径向有限差分 + 流线加权):
        =========================================================
        This is NOT full GFDM Taylor expansion - it's a simplified approach:

        1. Use radial finite differences: div F ≈ (F_j - F_i) · r_ij / ||r_ij||²
        2. Weight each neighbor by:
           - Distance: w_dist = 1 / ||r_ij||  (inverse distance)
           - Streamline bias: w_upwind = exp(β cos θ)  where cos θ = (α · r) / (||α|| · ||r||)
        3. Compute: div F = Σ_j [w_dist * w_upwind * (F_j - F_i) · r_ij / ||r_ij||²] / Σ_j w_total

        Why NOT Full GFDM Taylor Expansion:
        ------------------------------------
        Full GFDM would require:
        1. Build weighted LS matrix: A_ij = [1, r_x, r_y, r_x², r_xy, r_y², ...]
        2. Solve: (A^T W A) c = A^T W b  → get derivative coefficients
        3. Use coefficients to compute ∇·F

        Problem: Modifying weights requires rebuilding operator for EACH point
        at EACH time step → computationally expensive.

        Current Approach (Simplified):
        -------------------------------
        - Uses SPH-style radial gradient: (F_j - F_i) · r_ij / ||r_ij||²
        - Faster than full GFDM rebuild
        - Still maintains key property: streamline-biased weighting
        - Accuracy: lower than full GFDM Taylor (O(h) vs O(h²)), but stable

        Weight Formula:
        ---------------
        w_combined = w_distance * w_upwind

        where:
        - w_distance = 1 / ||r_ij||  (closer neighbors → higher weight)
        - w_upwind = exp(β cos θ)  (downstream → exp(+β·cos), upstream → exp(-β·|cos|))
        - cos θ = (α · r) / (||α|| · ||r||)  (streamline alignment)
        - β = upwind_strength  (typically 0.5-1.0)

        Key Properties:
        ---------------
        ✓ No dimension splitting (single unified computation)
        ✓ Rotation invariant (no grid orientation effect)
        ✓ All neighbors kept (no hard selection → stable)
        ✓ Streamline-aware (not axis-aware)

        ✗ Lower order accuracy than full GFDM (O(h) radial FD vs O(h²) Taylor)
        ✗ Not using precomputed GFDM operator (rebuilds each call)

        Args:
            drift_field: Drift α = -∇U at each point, shape (N, d)
            density: Density m at each point, shape (N,)

        Returns:
            divergence: div(m*α) with streamline-biased weighting, shape (N,)

        References:
            - SPH radial gradient: Monaghan (2005), Rep. Prog. Phys.
            - Streamline upwinding: Oñate et al. (1996), FPM
        """
        N = len(self.collocation_points)
        divergence = np.zeros(N)

        # Compute flux: F = m * α
        flux = density[:, np.newaxis] * drift_field  # Shape: (N, d)

        # For each point, compute div(F) with streamline-biased weights
        for i in range(N):
            neighborhood = self.gfdm_operator.neighborhoods[i]
            neighbors = neighborhood["indices"]

            # Filter ghost particles
            real_neighbors = neighbors[neighbors >= 0]
            if len(real_neighbors) == 0:
                divergence[i] = 0.0
                continue

            drift_i = drift_field[i]
            drift_norm = np.linalg.norm(drift_i)

            # Compute streamline-biased weights for neighbors
            upwind_weights = []
            neighbor_points = []

            for j in real_neighbors:
                r_ij = self.collocation_points[j] - self.collocation_points[i]
                r_norm = np.linalg.norm(r_ij)

                if r_norm < 1e-12:
                    continue  # Skip coincident points

                # Issue #1286, 2026-06-11 survey: guard zero-drift — when ||α||≈0
                # the streamline direction is undefined; dividing by drift_norm
                # produces nan.  Fall back to symmetric (unbiased) weight=1.
                if drift_norm < 1e-12:
                    weight_factor = 1.0
                else:
                    # Streamline alignment: cos θ = (α · r) / (||α|| · ||r||)
                    cos_theta = np.dot(drift_i, r_ij) / (drift_norm * r_norm)

                    # Upwind weight modification
                    if self.upwind_scheme == "exponential":
                        # Scharfetter-Gummel style: exp(β cos θ)
                        # Upstream (cos<0): weight < 1
                        # Downstream (cos>0): weight > 1
                        weight_factor = np.exp(self.upwind_strength * cos_theta)
                    elif self.upwind_scheme == "linear":
                        # Linear bias: 1 + β*cos θ
                        weight_factor = 1.0 + self.upwind_strength * cos_theta
                    else:
                        weight_factor = 1.0

                upwind_weights.append(weight_factor)
                neighbor_points.append(j)

            if len(neighbor_points) == 0:
                divergence[i] = 0.0
                continue

            upwind_weights = np.array(upwind_weights)
            neighbor_points = np.array(neighbor_points)

            # Build local GFDM operator with modified weights
            # We need to manually compute divergence using upwind-weighted Taylor expansion

            # Get neighbor positions and flux values
            X_neighbors = self.collocation_points[neighbor_points]  # Shape: (K, d)
            F_neighbors = flux[neighbor_points]  # Shape: (K, d)
            F_i = flux[i]  # Shape: (d,)

            # Relative positions
            dX = X_neighbors - self.collocation_points[i]  # Shape: (K, d)

            # Build weighted least-squares system for divergence
            # We want: div(F) ≈ Σ_k w_k * ∇·F|_k
            # Using finite differences: ∇·F ≈ (F_k - F_i) · r_k / ||r_k||²

            div_i = 0.0
            total_weight = 0.0

            for idx, _j in enumerate(neighbor_points):
                r_ij = dX[idx]
                r_norm_sq = np.dot(r_ij, r_ij)

                if r_norm_sq < 1e-12:
                    continue

                # Flux difference
                dF = F_neighbors[idx] - F_i

                # Divergence approximation: (dF · r) / ||r||²
                div_contrib = np.dot(dF, r_ij) / r_norm_sq

                # Weight by distance (GFDM-style) AND upwind bias
                dist_weight = 1.0 / (np.sqrt(r_norm_sq) + 1e-12)
                combined_weight = dist_weight * upwind_weights[idx]

                div_i += combined_weight * div_contrib
                total_weight += combined_weight

            if total_weight > 1e-12:
                divergence[i] = div_i / total_weight
            else:
                divergence[i] = 0.0

        return divergence

    def solve_fp_system(
        self,
        m_initial_condition: np.ndarray,
        drift_field: np.ndarray | Callable | None = None,
        volatility_field: float | np.ndarray | Callable | None = None,
        show_progress: bool | None = None,
    ) -> np.ndarray:
        """
        Solve FP system on collocation points using GFDM.

        Solves: dm/dt + div(m * alpha) = D * Laplacian(m)

        where alpha is the drift velocity.

        Args:
            m_initial_condition: Initial density at collocation points, shape (N,)
            drift_field: Drift velocity specification (Issue #573):
                - None: Zero drift (pure diffusion)
                - np.ndarray: Drift velocity field α*(t,x), shape (Nt+1, N, d)
                  Caller computes α* = -∂_p H(x, ∇U, m) for their Hamiltonian:
                    * Quadratic H = (1/2)|p|²: compute grad(U), then α* = -grad(U)
                    * L1 control H = |p|: α* = -sign(grad(U))
                    * Quartic H = (1/4)|p|⁴: α* = -sign(grad(U)) |grad(U)|^(1/3)
                    * Custom H: Any function of grad(U)
                - Callable: Custom drift function α(t, x, m) -> drift_vector
                Default: None
            volatility_field: Volatility coefficient σ (SDE noise). If None, uses problem.sigma.
                            Currently only scalar volatility supported.
                            Note: Internally converted to diffusion D = σ²/2 for FP equation.
            show_progress: Display progress bar (not yet implemented)

        Returns:
            Density evolution M(t,x) at collocation points, shape (Nt+1, N)

        Examples:
            Pure diffusion (heat equation):
            >>> M = solver.solve_fp_system(m0)

            Quadratic control (H = (1/2)|p|²):
            >>> U_hjb = hjb_solver.solve(M_density)
            >>> grad_U_all = np.array([gfdm_op.gradient(U_hjb[t]) for t in range(Nt+1)])
            >>> alpha = -grad_U_all  # α* = -∇U, shape (Nt+1, N, d)
            >>> M = solver.solve_fp_system(m0, drift_field=alpha)

            L1 control cost (H = |p|, minimal fuel):
            >>> U_hjb = hjb_solver.solve_hjb_L1(M_density)
            >>> grad_U_all = np.array([gfdm_op.gradient(U_hjb[t]) for t in range(Nt+1)])
            >>> alpha_L1 = -np.sign(grad_U_all)  # α* = -sign(∇U), shape (Nt+1, N, d)
            >>> M = solver.solve_fp_system(m0, drift_field=alpha_L1)

            Quartic control cost (H = (1/4)|p|⁴):
            >>> U_hjb = hjb_solver.solve_hjb_quartic(M_density)
            >>> grad_U_all = np.array([gfdm_op.gradient(U_hjb[t]) for t in range(Nt+1)])
            >>> alpha_quartic = -np.sign(grad_U_all) * np.abs(grad_U_all) ** (1/3)
            >>> M = solver.solve_fp_system(m0, drift_field=alpha_quartic)
        """
        # Time discretization
        n_time_points = self.problem.Nt + 1
        dt = self.problem.T / self.problem.Nt

        # Volatility coefficient (Issue #717: unified API)
        if volatility_field is None:
            sigma = self.problem.sigma
        elif isinstance(volatility_field, (int, float)):
            sigma = float(volatility_field)
        else:
            raise NotImplementedError("Only scalar volatility currently supported")

        diffusion_coeff = diffusion_from_volatility(sigma)

        # Validate inputs
        m_init = np.asarray(m_initial_condition).ravel()
        if m_init.shape[0] != self.n_points:
            raise ValueError(f"m_initial_condition length {m_init.shape[0]} must match n_points {self.n_points}")

        # Handle drift field (Issue #573: accepts drift velocity α* for any H)
        N_colloc = self.n_points  # Number of collocation points

        if drift_field is None:
            # Zero drift - treat as if drift velocity is zero
            drift_velocity_array = np.zeros((n_time_points, N_colloc, self.dimension))
        elif isinstance(drift_field, np.ndarray):
            # Direct drift velocity array provided
            drift_velocity_array = np.asarray(drift_field)

            # Issue #618 fix. Validated in:
            # mfg-research/experiments/crowd_evacuation_2d/experiments/exp15_lq_benchmark/validation/validate_gfdm_shape_bug.py
            # Standard shape: (Nt+1, N_colloc) for 1D, (Nt+1, N_colloc, d) for d>1
            # Dimension inferred from self.dimension (set from collocation_points.shape[1] at init)

            if self.dimension == 1:
                # 1D: Accept scalar representation (Nt+1, N_colloc)
                expected_shape = (n_time_points, N_colloc)
                if drift_velocity_array.shape != expected_shape:
                    raise ValueError(
                        f"For 1D problems, drift_field shape {drift_velocity_array.shape} "
                        f"must be {expected_shape} (Nt+1={n_time_points}, N_colloc={N_colloc})"
                    )
                # Reshape to internal format (Nt+1, N_colloc, 1)
                drift_velocity_array = drift_velocity_array.reshape(n_time_points, N_colloc, 1)
            else:
                # nD: Require explicit dimension axis (Nt+1, N_colloc, d)
                expected_shape = (n_time_points, N_colloc, self.dimension)
                if drift_velocity_array.shape != expected_shape:
                    raise ValueError(
                        f"For {self.dimension}D problems, drift_field shape {drift_velocity_array.shape} "
                        f"must be {expected_shape} "
                        f"(Nt+1={n_time_points}, N_colloc={N_colloc}, d={self.dimension})"
                    )
        else:
            # Callable drift function (advanced use)
            raise NotImplementedError("Callable drift_field not yet supported for GFDM")

        # Storage for density evolution
        M_solution = np.zeros((n_time_points, self.n_points))
        M_solution[0, :] = m_init.copy()
        mass_initial = np.sum(m_init)

        # Issue #886: track raw mass drift before normalization
        from mfgarchon.utils.mfg_logging import get_logger

        _logger = get_logger(__name__)
        max_mass_drift = 0.0

        # Time stepping loop (forward Euler)
        for t_idx in range(n_time_points - 1):
            m_current = M_solution[t_idx, :]

            # Use drift velocity directly (Issue #573: drift_field is α*, not U)
            drift = drift_velocity_array[t_idx, :, :]  # Shape: (N, d)

            # Advection term: div(m * alpha)
            if self.upwind_scheme != "none":
                # Use upwind-stabilized divergence
                advection = self._compute_upwind_divergence(drift, m_current)
            else:
                # Conservative continuity form: advection = div(m*α) = sum_d d/dx_d (m * α_d).
                # The previous code computed only `np.sum(drift * grad_m)` = α·∇m (transport form),
                # silently dropping the m·div(α) term. For any α with nonzero divergence (e.g.
                # α = -∇U with ΔU != 0, the standard MFG case) that term is O(1), so the default
                # path returned a wrong-SHAPE density on every call; the renormalization below masks
                # total-mass drift but not the shape error. Take the divergence of the flux
                # F = m*α directly via the GFDM gradient — the same conservative flux form the
                # upwind path (_compute_upwind_divergence) uses. (#1279, 2026-06-11 survey)
                flux = m_current[:, np.newaxis] * drift  # (N, d): F_d = m * alpha_d
                advection = sum(self.gfdm_operator.gradient(flux[:, d])[:, d] for d in range(drift.shape[1]))

            # Diffusion term: D * Laplacian(m)
            laplacian = self.gfdm_operator.laplacian(m_current)
            diffusion = diffusion_coeff * laplacian

            # Forward Euler update: dm/dt = -div(m*alpha) + D*Laplacian(m)
            dm_dt = -advection + diffusion
            M_solution[t_idx + 1, :] = m_current + dt * dm_dt

            # Physical constraints
            M_solution[t_idx + 1, :] = np.maximum(M_solution[t_idx + 1, :], 0.0)

            # Issue #886: log raw mass drift before normalization
            mass_current = np.sum(M_solution[t_idx + 1, :])
            if mass_initial > 0:
                mass_drift = abs(mass_current - mass_initial) / mass_initial
                max_mass_drift = max(max_mass_drift, mass_drift)
                if mass_drift > 0.01:
                    _logger.warning("GFDM mass drift %.2e at t_idx=%d (before renorm)", mass_drift, t_idx + 1)

            # Renormalize to conserve mass
            if mass_current > 0:
                M_solution[t_idx + 1, :] *= mass_initial / mass_current

        if max_mass_drift > 1e-6:
            _logger.debug("GFDM max mass drift: %.2e (renormalized each step)", max_mass_drift)

        return M_solution


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Smoke test for FPGFDMSolver."""
    print("Testing FPGFDMSolver...")

    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import neumann_bc

    # Minimal components for FP-only testing
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

    # Test 1D problem
    print("\n[1D] Testing 1D GFDM FP solver...")
    geometry_1d = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[31],
        boundary_conditions=neumann_bc(dimension=1),
    )
    problem = MFGProblem(geometry=geometry_1d, T=1.0, Nt=20, sigma=0.1, components=components)

    # Create 1D collocation points
    points_1d = np.linspace(0, 1, 50).reshape(-1, 1)
    solver = FPGFDMSolver(problem, collocation_points=points_1d)

    assert solver.fp_method_name == "GFDM"
    assert solver.n_points == 50
    assert solver.dimension == 1
    print(f"     Delta: {solver.delta:.4f}")

    # Initial Gaussian density
    m_init = np.exp(-50 * (points_1d[:, 0] - 0.5) ** 2)
    m_init = m_init / np.sum(m_init)  # Normalize

    # Zero drift (pure diffusion)
    U_drift = np.zeros((problem.Nt + 1, 50))

    M_solution = solver.solve_fp_system(m_init, drift_field=U_drift)

    assert M_solution.shape == (problem.Nt + 1, 50)
    assert not np.any(np.isnan(M_solution))
    assert not np.any(np.isinf(M_solution))
    assert np.all(M_solution >= 0)
    print(f"     M range: [{M_solution.min():.4f}, {M_solution.max():.4f}]")
    print("     1D test passed!")

    # Test 2D problem
    print("\n[2D] Testing 2D GFDM FP solver...")

    geometry_2d = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[10, 10],
        boundary_conditions=neumann_bc(dimension=2),
    )
    problem_2d = MFGProblem(geometry=geometry_2d, Nt=10, T=0.5, sigma=0.1, components=components)

    # Create 2D scattered points
    np.random.seed(42)
    points_2d = np.random.rand(100, 2)
    solver_2d = FPGFDMSolver(problem_2d, collocation_points=points_2d)

    assert solver_2d.n_points == 100
    assert solver_2d.dimension == 2
    print(f"     Delta: {solver_2d.delta:.4f}")

    # Initial Gaussian density
    r2 = (points_2d[:, 0] - 0.5) ** 2 + (points_2d[:, 1] - 0.5) ** 2
    m_init_2d = np.exp(-20 * r2)
    m_init_2d = m_init_2d / np.sum(m_init_2d)

    # Zero drift (2D: shape must be (Nt+1, N_colloc, 2))
    U_drift_2d = np.zeros((problem_2d.Nt + 1, 100, 2))

    M_solution_2d = solver_2d.solve_fp_system(m_init_2d, drift_field=U_drift_2d)

    assert M_solution_2d.shape == (problem_2d.Nt + 1, 100)
    assert not np.any(np.isnan(M_solution_2d))
    assert not np.any(np.isinf(M_solution_2d))
    assert np.all(M_solution_2d >= 0)
    print(f"     M range: [{M_solution_2d.min():.4f}, {M_solution_2d.max():.4f}]")
    print("     2D test passed!")

    print("\nFPGFDMSolver smoke tests passed!")

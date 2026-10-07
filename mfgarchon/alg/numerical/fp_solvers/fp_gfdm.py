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
from mfgarchon.types.callable_protocols import evaluate_solver_source
from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.numerical import (
    GROSS_MASS_CHANGE_BAND,
    MAX_CONSERVED_MASS_DRIFT,
    clip_nonnegative_or_raise,
    gross_mass_excursion,
    stop_on_gross_mass_change,
    stop_on_mass_drift,
)
from mfgarchon.utils.pde_coefficients import (
    diffusion_from_volatility,
    resolve_volatility_override,
    retired_volatility_keywords,
)

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
        >>> problem = MFGProblem(geometry=grid, Nt=20, T=1.0, volatility=0.1)
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

    # BoundaryCapable protocol (Issue #1456): _resolve_boundary_type resolves no-flux / Neumann /
    # periodic and returns None (silently) for everything else, so Dirichlet / Robin / Reflecting /
    # Extrapolation fail loud at construction.
    #
    # PERIODIC is NOT declared (#1822), and the reason is structural rather than behavioural:
    # this solver builds its TaylorOperator with no geometry=, so no periodic wrap is ever
    # constructed and the "periodic" its resolver returns is read by nothing downstream. There is
    # no path to honour, at any grid size. Declaring it bought a mid-solve ValueError -- the
    # density goes negative, sooner on finer grids -- instead of a refusal a caller can act on.
    # Use TensorProductGrid + FDM/FVM for periodic geometries.
    _SUPPORTED_BC_TYPES: frozenset = frozenset({BCType.NO_FLUX, BCType.NEUMANN})

    #: Issue #1686: this family reads a NEUMANN segment's type and drops its value.
    #: On the FP side a Neumann value is a prescribed flux J.n = g, and no FP solver
    #: implements an inhomogeneous flux wall, so a non-zero g is refused rather than
    #: silently discarded. Flip this to True in the same commit that implements it.
    honors_inhomogeneous_neumann: bool = False

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
        mass_drift: str = "raise",
        mass_drift_tolerance: float = MAX_CONSERVED_MASS_DRIFT,
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
            mass_drift: ``"raise"`` (default) stops the solve at the first step whose mass leaves its
                budget by more than ``mass_drift_tolerance``; ``"warn"`` keeps the solve going and logs
                the worst excursion at the end (#2512 S5, #1752). With this solver's declared BCs
                (no-flux, homogeneous Neumann) only the source S moves the mass (S = 0 when none is
                passed): the budget is the initial mass plus the net source the solve added, by its own
                forward-Euler rule, and the drift is |mass - budget| / (initial mass + |source| added).
                The mass is the geometry's own measure, so the budget is checked only when the
                collocation points are the grid's nodes (C order). On other points there is no measure,
                and only a gross check on sum|m| runs (``GROSS_MASS_CHANGE_BAND``): it stops the solve
                above 10 x (sum|m_0| + the positive source added) or below 0.1 x (sum|m_0| - the most
                the source can drain), the lower bound waived only once that reference is <= 0. A
                non-finite density raises always.
            mass_drift_tolerance: Tolerance on |mass - budget| / (initial mass + |source| added).

        BC Resolution Order:
            1. Explicit boundary_conditions parameter
            2. problem.geometry.boundary_conditions
            3. problem.geometry.get_boundary_conditions()
            4. boundary_type string (legacy, deprecated)
            5. None (no BC enforcement)
        """
        super().__init__(problem)
        self.fp_method_name = "GFDM"
        if mass_drift not in ("raise", "warn"):
            raise ValueError(f"FPGFDMSolver: mass_drift must be 'raise' or 'warn', got {mass_drift!r}")
        if not (np.isfinite(mass_drift_tolerance) and mass_drift_tolerance > 0):
            raise ValueError(
                f"FPGFDMSolver: mass_drift_tolerance must be finite and positive, got {mass_drift_tolerance!r}"
            )
        self.mass_drift = mass_drift
        self.mass_drift_tolerance = float(mass_drift_tolerance)

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
        # Called for its validation side effect only. `_resolve_default_bc` (#1100) raises on a
        # segments-only BoundaryConditions carrying no default, and this is the ONLY call on this
        # construction path that does -- `_validate_bc_support` below does not invoke it. Measured
        # 2026-08-17: replacing this call with `None` leaves 755 gfdm/boundary tests green, so the
        # refusal is now pinned by test_issue_1456_bc_capability_gate.py.
        #
        # The RETURN value is discarded. It used to be stored as `self._boundary_type`, which
        # nothing in the package ever read (removed 2026-08-17); `TaylorOperator` takes no boundary
        # argument at all, so the resolved string reached nothing.
        self._resolve_boundary_type(
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

        # Store upwind parameters
        self.upwind_scheme = upwind_scheme
        self.upwind_strength = upwind_strength

        # Decided once: the measure depends only on the collocation points and the geometry.
        self._mass_measure = self._grid_measure()
        if self._mass_measure is None:
            from mfgarchon.utils.mfg_logging import get_logger

            get_logger(__name__).warning(
                "FPGFDMSolver: the collocation points are not the grid's nodes in C (ij) order, so there is "
                "no measure to check the mass with; only a gross check on sum|m| runs (band %s, against "
                "sum|m_0| + the positive source added above and sum|m_0| - the most the source can drain "
                "below). This operator does not conserve mass (Issue #1752).",
                GROSS_MASS_CHANGE_BAND,
            )

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

    def _grid_measure(self) -> Callable[[np.ndarray], float] | None:
        """``m -> integral of m`` with the geometry's own measure, when the collocation points are exactly
        the grid's nodes (C order); ``None`` otherwise -- scattered points carry no measure here."""
        geometry = getattr(self.problem, "geometry", None)
        integrate = getattr(geometry, "integrate", None)
        coords = getattr(geometry, "coordinates", None)
        if not callable(integrate) or not isinstance(coords, (list, tuple)) or len(coords) != self.dimension:
            return None
        nodes = np.stack(np.meshgrid(*[np.asarray(c, dtype=float) for c in coords], indexing="ij"), axis=-1)
        nodes = nodes.reshape(-1, self.dimension)
        if nodes.shape != self.collocation_points.shape or not np.allclose(nodes, self.collocation_points):
            return None
        shape = tuple(len(c) for c in coords)
        return lambda m: float(integrate(np.asarray(m, dtype=float).reshape(shape)))

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

    @retired_volatility_keywords
    @deprecated_parameter(param_name="m_initial_condition", since="v0.22.0", replacement="M_initial")
    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        drift_field: np.ndarray | Callable | None = None,
        volatility: float | np.ndarray | Callable | None = None,
        source_term: Callable | None = None,
        show_progress: bool | None = None,
        m_initial_condition: np.ndarray | None = None,  # deprecated alias for M_initial (#2377)
        volatility_kind: str | None = None,
    ) -> np.ndarray:
        """
        Solve FP system on collocation points using GFDM.

        Solves: dm/dt + div(m * alpha) = D * Laplacian(m)

        where alpha is the drift velocity.

        Args:
            M_initial: Initial density at collocation points, shape (N,)
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
            volatility: Volatility coefficient σ (SDE noise). If None, uses problem.volatility.
                A scalar only; an array or callable is refused (#2376).
            volatility_kind: "field" or "tensor" for an array (#2378); arrays are refused here.
                            Currently only scalar volatility supported.
                            Note: Internally converted to diffusion D = σ²/2 for FP equation.
            source_term: Manufactured forcing S(t, x) -> (N,), with x the collocation points of
                shape (N, d) (Issue #2020). It enters the equation as
                ``dm/dt + div(m α) = D Δm + S``, i.e. it is ADDED to ``dm_dt``, the same sign and
                the same package-wide callable convention as the FDM path
                (``fp_fdm_time_stepping.py``: ``M_next += dt * source_term``). Evaluated at t_k, the level
                of every other term of this explicit update (#2020; the implicit FDM path uses t_{k+1},
                the level of its operator). This is what
                lets a manufactured solution reach this solver; before #2020 the parameter was
                absent from the signature, so passing one raised ``TypeError`` and the family's
                convergence order had never been measured.
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

        # Volatility coefficient (Issue #717: unified API). None is the problem's own (#2376).
        volatility, volatility_kind = resolve_volatility_override(
            volatility, volatility_kind, problem=self.problem, consumer=f"{type(self).__name__}.solve_fp_system"
        )
        if isinstance(volatility, (int, float)):
            sigma = float(volatility)
        else:
            raise NotImplementedError(
                f"FPGFDMSolver supports only a scalar volatility, got {type(volatility).__name__} (#2376)."
            )

        diffusion_coeff = diffusion_from_volatility(sigma)

        if m_initial_condition is not None:
            if M_initial is not None:
                raise ValueError("Cannot specify both M_initial and m_initial_condition; use M_initial (#2377).")
            M_initial = m_initial_condition
        if M_initial is None:
            raise ValueError("M_initial is required")

        # Validate inputs
        m_init = np.asarray(M_initial).ravel()
        if m_init.shape[0] != self.n_points:
            raise ValueError(f"M_initial length {m_init.shape[0]} must match n_points {self.n_points}")

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
        from mfgarchon.utils.mfg_logging import get_logger

        _logger = get_logger(__name__)
        # The conserved quantity is the integral of m, so the drift is measured with the geometry's own
        # measure. An unweighted sum is a different functional: on the exact mode 1 + 0.5 cos(2 pi x)
        # e^{-D (2 pi)^2 t} it drifts 1.37% at 21 uniform points while the trapezoid holds to 3e-16 (#2512
        # S5). Only the source moves the mass under the declared BCs: d/dt (integral of m) = integral of S,
        # with S = 0 when none is passed -- one law, so the gate has no source/no-source case. Each case
        # boundary it once had silenced a check where main warned (#2526 reviews 3 and 4). The references
        # sum S_k at the level the update adds it (t_k, forward Euler).
        measure = self._mass_measure
        mass_initial = measure(m_init) if measure is not None else 0.0
        gross_initial = float(np.sum(np.abs(m_init))) if measure is None else 0.0
        source_added = 0.0  # sum dt * measure(S_k): the budget is mass_initial + this
        source_scale = 0.0  # sum dt * measure(|S_k|): the drift is measured against mass_initial + this
        gross_added = 0.0  # sum dt * sum(max(S_k, 0)): the most the source can have added to sum|m|
        gross_drained = 0.0  # sum dt * sum(max(-S_k, 0)): the most it can have removed
        max_mass_drift = 0.0
        max_mass_drift_t_idx = 0
        max_mass_drift_scale = 0.0
        worst_gross = (1.0, 0)
        if self.upwind_scheme == "none":
            lever = "with a drift field, upwind_scheme='linear' or 'exponential' reduces the leak"
        else:
            lever = f"with a drift field, upwind_scheme={self.upwind_scheme!r} reduces the leak without removing it"
        drift_remedy = (
            f"This operator does not conserve mass (Issue #1752: it diverges under refinement when a drift "
            f"field drives it, and leaks at the boundary stencil even without one, which neither upwinding nor "
            f"refining dt removes); {lever}."
        )
        raise_suffix = " Pass mass_drift='warn' to keep the density, or raise mass_drift_tolerance."

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

            # Forward Euler update: dm/dt = -div(m*alpha) + D*Laplacian(m) + S
            dm_dt = -advection + diffusion
            if source_term is None:
                s_values = np.zeros(self.n_points)
            else:
                # Evaluated at t_n, the level every OTHER term in this expression is evaluated at.
                # This update is explicit forward Euler -- `m + dt * (L(m^n) + S)` -- so pairing an
                # operator at t_n with a source at t_{n+1} would mix two levels inside one
                # expression. `fp_fdm.py` uses `source_term(t_next, x_grid)` because that path is
                # IMPLICIT, where t_{n+1} is the level of its operator; the package convention is
                # the level the operator is evaluated at, not the literal t_next (#2020).
                #
                # Measured, nx=41, both choices consistent and the gap clean O(dt):
                #   nt=200/400/800/1600, |S(t_n+1) - S(t_n)| = 1.13e-05, 5.63e-06, 2.81e-06, 1.41e-06
                # S(t_n) sits on the space-limited floor at once (1.535e-04 -> 1.539e-04, flat)
                # while S(t_{n+1}) descends toward it from above (1.648e-04 -> 1.553e-04).
                #
                # A sign error here is not silent: flipping it puts the error at 1.041e-01, about
                # twice the no-source 5.25e-02, against 1.65e-04 with the sign right.
                s_values = np.asarray(
                    evaluate_solver_source(source_term, t=t_idx * dt, x=self.collocation_points), dtype=float
                ).ravel()
                if s_values.size != self.n_points:
                    raise ValueError(
                        f"FPGFDMSolver: source_term returned {s_values.size} values at "
                        f"t={t_idx * dt:.6g}, expected {self.n_points} (one per collocation "
                        f"point). The package convention is source_term(t, x) -> (N,) with x of "
                        f"shape (N, d) taken from the collocation points."
                    )
            dm_dt = dm_dt + s_values
            if measure is not None:
                source_added += dt * measure(s_values)
                source_scale += dt * measure(np.abs(s_values))
            else:
                gross_added += dt * float(np.sum(np.maximum(s_values, 0.0)))
                gross_drained += dt * float(np.sum(np.maximum(-s_values, 0.0)))
            M_solution[t_idx + 1, :] = m_current + dt * dm_dt
            if not np.all(np.isfinite(M_solution[t_idx + 1, :])):
                raise ValueError(
                    f"GFDM FP solve: the density is not finite at step {t_idx + 1}. That is a failed solve, "
                    f"not a mass drift, so it stops whatever mass_drift is set to."
                )
            if gross_initial > 0:
                gross = float(np.sum(np.abs(M_solution[t_idx + 1, :])))
                if self.mass_drift == "raise":
                    stop_on_gross_mass_change(
                        gross,
                        gross_initial,
                        step=t_idx + 1,
                        context="GFDM FP solve",
                        remedy=drift_remedy + " Pass mass_drift='warn' to keep the density.",
                        added=gross_added,
                        drained=gross_drained,
                    )
                excursion = gross_mass_excursion(gross, gross_initial, added=gross_added, drained=gross_drained)
                # Farthest from the band in either direction: a vanishing density is as bad as a growing one.
                if excursion is not None and abs(np.log(excursion)) > abs(np.log(worst_gross[0])):
                    worst_gross = (excursion, t_idx + 1)

            # Issue #1683: this clipped, then renormalised to the initial mass, and warned
            # only above 1% drift. Every configuration therefore returned a final mass of
            # exactly 1.0000 -- including one measured to clip **61%** of the present mass
            # at a single step. Reporting perfect conservation over that is the defect.
            #
            # Issue #1752: the mechanism is the unstabilised central flux divergence below
            # with `upwind_scheme="none"` (the default), NOT the explicit time stepping. The
            # first version of this comment blamed forward Euler; refining dt at fixed h
            # makes the drift monotonically WORSE, converging upward to a semi-discrete
            # limit (2.79, 4.47, 6.08, 7.26, 7.99, 8.62, 8.73 at Nt = 10, 20, 40, 80,
            # 160, 640, 1280 with sigma=0.3), which rules time discretisation out. Refining h with the drift on
            # also diverges (4.31 -> 8.62 -> 12.18 -> 17.88 for N = 11..81), while pure
            # diffusion converges cleanly at ~O(h). So this is not merely non-conservative;
            # it is divergent under refinement, and `dt` is not a lever on it.
            #
            # The clip gate's ratio is fabricated mass over present mass, a ratio of two sums over
            # the same points, so it is unweighted on purpose and `weights=` is omitted. The mass-
            # drift check below is a different question and uses the geometry's measure (#2512 S5).
            M_solution[t_idx + 1, :] = clip_nonnegative_or_raise(
                M_solution[t_idx + 1, :],
                context=f"GFDM FP solve: at t_idx={t_idx + 1}",
                remedy=(
                    (
                        "upwind_scheme is 'none', which leaves the flux divergence "
                        "unstabilised -- that is what drives this (Issue #1752). 'linear' or "
                        "'exponential' measurably reduce it: on a 21-point grid at sigma=0.3, "
                        "final mass 2.795 -> 1.437 -> 1.418 at Nt=10, and 8.62 -> 2.34 -> 2.25 "
                        "at Nt=640."
                        if self.upwind_scheme == "none"
                        else f"upwind_scheme is already {self.upwind_scheme!r}, so stabilisation "
                        "is on and is not enough here -- it reduces the drift without removing "
                        "it (Issue #1752: this operator diverges under refinement). Coarsen the "
                        "value-function gradient driving the flux, or use a different FP scheme."
                    )
                    + " Do NOT reduce dt to silence this: refining the timestep drives the "
                    "per-step clip below the threshold while the final mass climbs from "
                    "8.4e+02 to 2.5e+09, so it removes the message and not the defect. If "
                    "dt*D/dx^2 < 0.5 is also violated, fix that on its own merits; it is not "
                    "what binds here."
                ),
            )
            if measure is not None:
                # One owner of the drift: under "warn" the same function measures it and never raises.
                step_drift = stop_on_mass_drift(
                    measure(M_solution[t_idx + 1, :]),
                    mass_initial,
                    step=t_idx + 1,
                    tolerance=self.mass_drift_tolerance if self.mass_drift == "raise" else np.inf,
                    context="GFDM FP solve",
                    remedy=drift_remedy + raise_suffix,
                    source_added=source_added,
                    source_scale=source_scale,
                )
                if step_drift > max_mass_drift:
                    max_mass_drift, max_mass_drift_t_idx = step_drift, t_idx + 1
                    max_mass_drift_scale = mass_initial + source_scale

        # Issue #1752: the clip gate cannot catch a divergence that stays positive -- it measures
        # fabricated mass as a RATIO of the mass present, which is scale-invariant. Measured:
        # sigma=0.1/drift=25 at Nt=2560 returns a finite, non-negative density whose mass grows
        # 2.11e+09 x under the measure (2.55e+09 x in the unweighted sum) with no negatives. The
        # per-step check above stops it; under mass_drift="warn" this warning is the only signal,
        # so it cannot stay at DEBUG, where it once hid a 144% drift on a run that clips nothing.
        if max_mass_drift > 1e-6:
            _logger.warning(
                "GFDM FP mass drift %.2e (worst at t_idx=%d): |mass - budget| / %.6g, with the geometry's "
                "measure. The budget is the initial mass plus the net source added, the only thing that moves "
                "the mass under the declared BCs; the scale is the initial mass plus the |source| added. %s "
                "The returned density carries the drift rather than being rescaled to hide it.",
                max_mass_drift,
                max_mass_drift_t_idx,
                max_mass_drift_scale,
                drift_remedy,
            )
        if worst_gross[1]:
            _logger.warning(
                "GFDM FP gross blow-up check: at t_idx=%d sum|m| was %.6g times its reference, outside the band "
                "%s. The references are sum|m_0| plus the positive source added (above) and sum|m_0| minus the "
                "most the source can drain (below). Not a conservation check -- these points carry no measure. %s",
                worst_gross[1],
                worst_gross[0],
                GROSS_MASS_CHANGE_BAND,
                drift_remedy,
            )

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
    problem = MFGProblem(geometry=geometry_1d, T=1.0, Nt=20, volatility=0.1, components=components)

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
    problem_2d = MFGProblem(geometry=geometry_2d, Nt=10, T=0.5, volatility=0.1, components=components)

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

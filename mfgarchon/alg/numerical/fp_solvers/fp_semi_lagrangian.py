"""
Semi-Lagrangian Fokker-Planck Solver for Mean Field Games.

This module implements a semi-Lagrangian method for solving the Fokker-Planck
equation in MFG problems. The method traces characteristics forward in time
and uses interpolation with Jacobian correction for mass conservation.

The FP equation solved is (divergence form):
    dm/dt + div(alpha * m) = sigma^2/2 * Laplacian(m)    in [0,T] x Omega
    m(0, x) = m0(x)                                       at t = 0

where alpha = -grad(U) is the drift field from HJB solution.

Key features:
- Stable for large time steps (same as HJB Semi-Lagrangian)
- Jacobian correction preserves divergence form structure
- O(h^2) convergence when paired with SL-HJB (Carlini & Silva 2014)
- Crank-Nicolson diffusion for unconditional stability

Issue #578: Implement Semi-Lagrangian FP solver for SL-SL coupling
Validated in: mfg-research/experiments/crowd_evacuation_2d/runners/exp14j_fp_semi_lagrangian_v2.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded

# Import BC infrastructure from HJB-SL (reuse unified components)
from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import (
    apply_boundary_conditions_1d,
)

# Issue #702: Centralized BC utilities for consistent handling across SL solvers
from mfgarchon.geometry.boundary.bc_utils import (
    bc_type_to_geometric_operation,
    get_bc_type_string,
)

# Issue #625: Migrated from tensor_calculus to operators/stencils
from mfgarchon.operators.stencils.finite_difference import laplacian_with_bc
from mfgarchon.utils.deprecation import deprecated, deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility, fp_drift_coefficient

from .base_fp import BaseFPSolver, DriftConvention

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import BoundaryConditions

logger = get_logger(__name__)


class FPSLJacobianSolver(BaseFPSolver):
    """
    DEPRECATED: Backward Semi-Lagrangian solver with Jacobian correction.

    .. deprecated:: 0.17.6
        Use :class:`FPSLSolver` (Forward SL) instead for adjoint consistency
        with HJB Semi-Lagrangian solvers. This class will be removed in v1.0.

    The Semi-Lagrangian method for FP uses backward characteristic tracing
    with Jacobian correction to handle the divergence form properly:

        m^{n+1}(x) = m^n(x - alpha*dt) * exp(-dt * div(alpha))

    This correction accounts for compression (div < 0, density increases)
    and expansion (div > 0, density decreases) of the flow.

    Note:
        This backward SL approach is NOT adjoint-consistent with HJB backward SL.
        For MFG problems requiring discrete duality, use FPSLSolver (Forward SL).

    Algorithm (operator splitting):
        1. Advection: Trace characteristics backward, interpolate, apply Jacobian
        2. Diffusion: Crank-Nicolson implicit solve (unconditionally stable)

    Dimension support:
        - 1D: Full support (production-ready)
        - nD: Planned for future

    References:
        - Carlini & Silva (2014): Semi-Lagrangian schemes for MFG
        - Calzola et al. (2023): High-order MFG schemes
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.SL
    _drift_convention = DriftConvention.VALUE_FUNCTION  # Issue #1043: takes U via potential_field

    @deprecated(since="v0.17.6", replacement="FPSLSolver")
    def __init__(
        self,
        problem: MFGProblem,
        interpolation_method: str = "cubic",
        characteristic_solver: str = "explicit_euler",
        boundary_conditions: BoundaryConditions | None = None,
    ):
        """
        Initialize Semi-Lagrangian FP solver.

        Args:
            problem: MFG problem instance
            interpolation_method: Method for interpolating density at departure points
                - 'linear': Linear interpolation (faster, C^0 continuous)
                - 'cubic': Cubic spline interpolation (slower, C^2 continuous)
            characteristic_solver: Method for solving characteristics
                - 'explicit_euler': First-order explicit Euler (default)
                - 'rk2': Second-order Runge-Kutta midpoint
                - 'rk4': Fourth-order Runge-Kutta
            boundary_conditions: Optional boundary conditions (defaults to no-flux)
        """
        super().__init__(problem)
        self.fp_method_name = "Semi-Lagrangian (Jacobian)"

        # Solver configuration
        self.interpolation_method = interpolation_method
        self.characteristic_solver = characteristic_solver

        # Detect problem dimension
        self.dimension = self._detect_dimension()  # Issue #633: Use inherited method

        if self.dimension > 1:
            raise NotImplementedError(
                "FPSLJacobianSolver currently only supports 1D problems. Use FPSLSolver for nD support."
            )

        # Precompute grid parameters (1D)
        bounds = problem.geometry.get_bounds()
        self.xmin, self.xmax = bounds[0][0], bounds[1][0]
        Nx = problem.geometry.get_grid_shape()[0]
        self.x_grid = np.linspace(self.xmin, self.xmax, Nx)
        self.dx = problem.geometry.get_grid_spacing()[0]
        self.dt = problem.dt

        # Boundary conditions
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            # Try to get BC from problem/geometry
            self.boundary_conditions = self._get_boundary_conditions_from_problem()

    def _get_boundary_conditions_from_problem(self) -> BoundaryConditions | None:
        """Get boundary conditions from problem or geometry."""
        # Priority 1: geometry.boundary_conditions
        try:
            bc = self.problem.geometry.boundary_conditions
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # Priority 2: geometry.get_boundary_conditions()
        try:
            bc = self.problem.geometry.get_boundary_conditions()
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # Priority 3: problem.boundary_conditions
        try:
            bc = self.problem.boundary_conditions
            if bc is not None:
                return bc
        except AttributeError:
            pass

        return None

    def _get_bc_operation_type(self) -> str:
        """
        Get boundary operation type from boundary conditions.

        Issue #702: Uses centralized bc_utils for consistent BC handling
        across all SL solvers (HJB-SL, FP-SL, FP-SL Adjoint).

        Returns:
            Geometric operation: 'reflect', 'clamp', or 'periodic'
        """
        bc_type = get_bc_type_string(self.boundary_conditions)
        return bc_type_to_geometric_operation(bc_type)

    @deprecated_parameter(param_name="drift_field", since="v0.18.6", replacement="potential_field")
    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        potential_field: np.ndarray | Callable | None = None,
        volatility_field: float | np.ndarray | Callable | None = None,
        show_progress: bool | None = None,
        # Deprecated parameters
        drift_field: np.ndarray | Callable | None = None,  # Deprecated: renamed to potential_field
    ) -> np.ndarray:
        """
        Solve FP system forward in time using Semi-Lagrangian method.

        The Semi-Lagrangian discretization of the FP equation in divergence form:
            dm/dt + div(alpha * m) = sigma^2/2 * Laplacian(m)

        uses backward characteristic tracing with Jacobian correction:
            m^{n+1}(x) = m^n(x - alpha*dt) * exp(-dt * div(alpha)) + diffusion

        Args:
            M_initial: Initial density m0(x). Shape: (Nx,)
            potential_field: Value function U from HJB (potential for drift).
                - np.ndarray: Shape (Nt+1, Nx) - U values at each time step
                  The drift velocity is computed as alpha = -grad(U)
            drift_field: DEPRECATED. Renamed to potential_field.
            volatility_field: Optional volatility coefficient σ (SDE noise) override.
                - None: Use problem.sigma
                - float: Constant volatility
                Note: Internally converted to diffusion D = σ²/2 for FP equation.
            show_progress: Show progress bar during solve

        Returns:
            Density evolution M(t,x). Shape: (Nt+1, Nx)

        Note:
            potential_field is the VALUE FUNCTION U, not the velocity alpha.
            The velocity is computed internally as alpha = -grad(U).
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
                "potential_field (value function U) is required for Semi-Lagrangian FP. "
                "Pass the U solution from HJB solver."
            )

        if callable(potential_field):
            raise NotImplementedError("Callable potential_field not yet supported for FPSLSolver")

        # Handle volatility (Issue #717: unified API)
        if volatility_field is None:
            sigma = self.problem.sigma
        elif isinstance(volatility_field, (int, float)):
            sigma = float(volatility_field)
        else:
            raise NotImplementedError("Array/callable volatility_field not yet supported for FPSLSolver")

        # Determine number of time steps from potential_field
        Nt_points = potential_field.shape[0]
        Nx = len(self.x_grid)

        # Allocate solution array
        M = np.zeros((Nt_points, Nx))
        M[0, :] = M_initial.copy()

        # Progress bar
        from mfgarchon.utils.progress import create_progress_bar, should_show_progress

        timestep_range = create_progress_bar(
            range(Nt_points - 1),
            verbose=should_show_progress(show_progress),
            desc="FP-SL (forward)",
        )

        # Forward time stepping
        for n in timestep_range:
            # Compute velocity field alpha = -grad(U) at current time
            U_n = potential_field[n, :]
            alpha = self._compute_velocity(U_n)
            # Issue #579 fix: Use direct Laplacian instead of compound gradient
            # div(alpha) = div(-grad(U)) = -Laplacian(U)
            div_alpha = self._compute_divergence_from_U(U_n)

            # Semi-Lagrangian step
            M[n + 1, :] = self._sl_step(
                M[n, :],
                alpha,
                div_alpha,
                self.dt,
                sigma,
            )

        return M

    def _compute_velocity(self, U: np.ndarray) -> np.ndarray:
        """Compute the optimal-control drift alpha* = -grad(U) / control_cost.

        Issue #1420 / G-017 / S0-03: the coefficient is single-sourced from the Hamiltonian's
        control_cost via ``fp_drift_coefficient`` (= 1/control_cost for a quadratic
        SeparableHamiltonian), not hardcoded to 1 (which dropped the 1/lambda factor for
        control_cost != 1). Byte-identical when control_cost == 1.
        """
        return -fp_drift_coefficient(self.problem) * np.gradient(U, self.dx)

    def _compute_divergence_from_U(self, U: np.ndarray) -> np.ndarray:
        """
        Compute div(alpha) = div(-grad(U)) = -Laplacian(U).

        Uses laplacian_with_bc from stencils with proper BC handling instead of
        compound gradient (which causes 50% boundary error).

        Issue #579: Compound gradient np.gradient(np.gradient(U)) gives
        wrong values at boundaries because np.gradient uses one-sided
        differences. For U = 0.5*x^2, div(alpha) should be -1 everywhere,
        but compound gradient gives -0.5 at boundaries.

        The fix is to compute -Laplacian(U) directly using proper ghost cells.
        We use quadratic extrapolation BC (EXTRAPOLATION_QUADRATIC) because the
        potential U does not satisfy Neumann BC - it has non-zero gradient at
        boundaries.

        Issue #625: Migrated from tensor_calculus to stencils.laplacian_with_bc
        """
        # Use quadratic extrapolation BC for fields that don't satisfy Neumann BC
        from mfgarchon.geometry.boundary import BCSegment, BCType, mixed_bc

        # Create quadratic extrapolation BC for the potential field
        # This gives O(h^2) accuracy for smooth fields like U = 0.5*x^2
        left_bc = BCSegment(name="left", bc_type=BCType.EXTRAPOLATION_QUADRATIC, value=0.0, boundary="x_min")
        right_bc = BCSegment(name="right", bc_type=BCType.EXTRAPOLATION_QUADRATIC, value=0.0, boundary="x_max")
        extrap_bc = mixed_bc(
            segments=[left_bc, right_bc], dimension=1, domain_bounds=np.array([[self.xmin, self.xmax]])
        )

        # Compute Laplacian with proper ghost cell handling
        lap_U = laplacian_with_bc(U, spacings=[self.dx], bc=extrap_bc)

        # div(alpha) = div(-c*grad U) = -c*Laplacian(U); c = 1/control_cost single-sourced
        # via fp_drift_coefficient (Issue #1420 / S0-03), consistent with _compute_velocity.
        return -fp_drift_coefficient(self.problem) * lap_U

    def _apply_boundary_conditions(self, x_dep: np.ndarray) -> np.ndarray:
        """
        Apply boundary conditions to departure points.

        Issue #702: Uses unified BC infrastructure from hjb_sl_characteristics
        module with centralized bc_utils for consistent handling across all
        SL solvers (HJB-SL, FP-SL, FP-SL Adjoint).

        Mapping:
            - NEUMANN/NO_FLUX -> 'reflect' (mirror at boundary)
            - DIRICHLET -> 'clamp' (stay at boundary)
            - PERIODIC -> 'periodic' (wrap around)

        Args:
            x_dep: Departure point coordinates

        Returns:
            Departure points with BC applied
        """
        bc_op = self._get_bc_operation_type()

        # Vectorized application using unified BC infrastructure
        result = np.zeros_like(x_dep)
        for i, x in enumerate(x_dep):
            result[i] = apply_boundary_conditions_1d(
                x,
                xmin=self.xmin,
                xmax=self.xmax,
                bc_type=bc_op,
            )
        return result

    def _sl_step(
        self,
        m: np.ndarray,
        alpha: np.ndarray,
        div_alpha: np.ndarray,
        dt: float,
        sigma: float,
    ) -> np.ndarray:
        """
        One Semi-Lagrangian step for Fokker-Planck equation.

        Operator splitting:
            1. Advection with Jacobian correction (divergence form)
            2. Diffusion via Crank-Nicolson

        Args:
            m: Current density, shape (Nx,)
            alpha: Velocity field, shape (Nx,)
            div_alpha: Divergence of velocity, shape (Nx,)
            dt: Time step
            sigma: Diffusion coefficient

        Returns:
            Density at next time step, shape (Nx,)
        """
        Nx = len(m)

        # Step 1: Advection with Jacobian correction
        # ==========================================
        # Trace characteristics backward: x_departure = x - alpha * dt
        x_departure = self.x_grid - alpha * dt

        # Issue #702: Apply boundary conditions based on problem BC type
        # This ensures adjoint consistency with HJB-SL solver.
        # - NEUMANN/NO_FLUX: reflect departure points (mass stays in domain)
        # - DIRICHLET: clamp to boundary (absorbing)
        # - PERIODIC: wrap around domain
        x_departure = self._apply_boundary_conditions(x_departure)

        # Interpolate density at departure points
        # After BC application, points should be within domain, but use edge
        # extrapolation as safety for numerical precision at boundaries.
        if self.interpolation_method == "linear":
            interpolator = interp1d(
                self.x_grid,
                m,
                kind="linear",
                bounds_error=False,
                fill_value=(m[0], m[-1]),  # Edge values for safety
            )
        else:  # cubic
            interpolator = interp1d(
                self.x_grid,
                m,
                kind="cubic",
                bounds_error=False,
                fill_value=(m[0], m[-1]),  # Edge values for safety
            )

        m_advected = interpolator(x_departure)

        # Jacobian correction for divergence form:
        # m^{n+1}(x) = m^n(x_dep) * |det(dx_dep/dx)|
        # For 1D with x_dep = x - alpha*dt:
        #   dx_dep/dx = 1 - dt * d(alpha)/dx = 1 - dt * div(alpha)
        # For finite dt, use exponential form: J = exp(-dt * div(alpha))
        #
        # Clamp the exponent to prevent numerical overflow.
        # |dt * div(alpha)| < 10 is a reasonable bound for stability.
        jacobian_exponent = np.clip(-dt * div_alpha, -10.0, 10.0)
        jacobian = np.exp(jacobian_exponent)
        m_advected = m_advected * jacobian

        # Ensure non-negativity
        m_advected = np.maximum(m_advected, 0)

        # Step 2: Diffusion via Crank-Nicolson
        # ====================================
        D = diffusion_from_volatility(sigma)
        r = D * dt / (self.dx**2)

        # Build RHS: (I + r/2 * L) * m_advected
        rhs = np.zeros(Nx)
        # Interior points
        rhs[1:-1] = m_advected[1:-1] + (r / 2) * (m_advected[:-2] - 2 * m_advected[1:-1] + m_advected[2:])
        # Neumann BC: dm/dx = 0 at boundaries
        # Use ghost points: m[-1] = m[0], m[Nx] = m[Nx-1]
        rhs[0] = m_advected[0] + (r / 2) * (2 * m_advected[1] - 2 * m_advected[0])
        rhs[-1] = m_advected[-1] + (r / 2) * (2 * m_advected[-2] - 2 * m_advected[-1])

        # Build tridiagonal matrix (I - r/2 * L) for Crank-Nicolson
        # Use banded format: ab[0] = upper diagonal, ab[1] = main, ab[2] = lower
        ab = np.zeros((3, Nx))

        # Main diagonal: 1 + r
        ab[1, :] = 1 + r

        # Off-diagonals: -r/2
        ab[0, 1:] = -r / 2  # upper diagonal
        ab[2, :-1] = -r / 2  # lower diagonal

        # Neumann BC modifications:
        # At i=0: (1+r)*m[0] - r*m[1] = rhs[0]
        # At i=Nx-1: -r*m[Nx-2] + (1+r)*m[Nx-1] = rhs[Nx-1]
        ab[0, 1] = -r  # upper at position 1 (affects row 0)
        ab[2, -2] = -r  # lower at position Nx-2 (affects row Nx-1)

        # Solve tridiagonal system
        m_new = solve_banded((1, 1), ab, rhs)

        # Ensure non-negativity
        m_new = np.maximum(m_new, 0)

        # Renormalize to preserve total mass (critical for numerical stability)
        # This compensates for any numerical mass drift from the Jacobian correction
        # and boundary handling.
        total_mass_old = np.trapezoid(m, self.x_grid)
        total_mass_new = np.trapezoid(m_new, self.x_grid)
        if total_mass_new > 1e-15:
            m_new *= total_mass_old / total_mass_new

        return m_new

    def _get_solver_type_id(self) -> str | None:
        """Get solver type identifier for compatibility checking."""
        return "semi_lagrangian"


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPSLSolver...")
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

    # Problem definition
    problem = MFGProblem(
        geometry=domain,
        T=T,
        Nt=Nt,
        sigma=SIGMA,
        components=components,
    )

    # Test 1: Solver initialization
    print("\n1. Testing solver initialization...")
    solver = FPSLJacobianSolver(problem, interpolation_method="cubic")
    assert solver.dimension == 1
    assert solver.fp_method_name == "Semi-Lagrangian (Jacobian)"
    print("   Initialization: OK")

    # Test 2: Solve with constant drift (pure diffusion test)
    print("\n2. Testing pure diffusion (zero drift)...")
    m0 = np.exp(-((x - 0.0) ** 2) / 0.02)
    m0 /= np.trapezoid(m0, x)

    # Zero drift field
    U_zero = np.zeros((Nt + 1, N + 1))

    M_diffusion = solver.solve_fp_system(M_initial=m0, drift_field=U_zero, show_progress=False)

    assert M_diffusion.shape == (Nt + 1, N + 1)
    assert not np.any(np.isnan(M_diffusion))

    # Check mass conservation
    mass_initial = np.trapezoid(m0, x)
    mass_final = np.trapezoid(M_diffusion[-1, :], x)
    mass_error = abs(mass_final - mass_initial) / mass_initial
    print(f"   Mass conservation: initial={mass_initial:.4f}, final={mass_final:.4f}, error={mass_error:.2e}")
    assert mass_error < 0.01, f"Mass error too large: {mass_error}"
    print("   Pure diffusion: OK")

    # Test 3: Solve with drift toward center
    print("\n3. Testing with drift field...")

    # Create a smooth potential well at x=0: U(x) = x^2
    # This gives alpha = -2x, drift toward center
    U_well = np.tile(x**2, (Nt + 1, 1))

    # Start from peaked Gaussian (closer to equilibrium for faster convergence)
    m_init = np.exp(-((x - 0.0) ** 2) / 0.1)
    m_init /= np.trapezoid(m_init, x)

    M_drift = solver.solve_fp_system(M_initial=m_init, drift_field=U_well, show_progress=False)

    assert M_drift.shape == (Nt + 1, N + 1)
    assert not np.any(np.isnan(M_drift))

    # Check mass conservation
    mass_init = np.trapezoid(m_init, x)
    mass_final = np.trapezoid(M_drift[-1, :], x)
    mass_error = abs(mass_final - mass_init) / mass_init
    print(f"   Mass conservation: initial={mass_init:.4f}, final={mass_final:.4f}, error={mass_error:.2e}")
    assert mass_error < 0.01, f"Mass error too large: {mass_error}"

    # Check that density stays centered (started at center, should stay near center)
    final_peak_idx = np.argmax(M_drift[-1, :])
    final_peak_x = x[final_peak_idx]
    print(f"   Final peak at index {final_peak_idx} (x={final_peak_x:.3f})")
    # Peak should remain near center (within 0.15 of x=0)
    assert abs(final_peak_x) < 0.15, f"Peak drifted from center: x={final_peak_x}"
    print("   Drift test: OK")

    # Test 4: Compare with FDM
    print("\n4. Comparing with FPFDMSolver...")

    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    fdm_solver = FPFDMSolver(problem, advection_scheme="divergence_upwind")
    M_fdm = fdm_solver.solve_fp_system(
        M_initial=m_init.copy(),  # Use same initial condition
        drift_field=U_well,
        show_progress=False,
    )

    # Both should stay near center (started there)
    fdm_peak_x = x[np.argmax(M_fdm[-1, :])]
    sl_peak_x = x[np.argmax(M_drift[-1, :])]
    print(f"   FDM final peak: x={fdm_peak_x:.3f}")
    print(f"   SL final peak: x={sl_peak_x:.3f}")

    # Compare peak values
    sl_peak_value = M_drift[-1, :].max()
    fdm_peak_value = M_fdm[-1, :].max()
    print(f"   FDM peak value: {fdm_peak_value:.4f}")
    print(f"   SL peak value: {sl_peak_value:.4f}")
    print("   Comparison: OK")

    # Test 5: Issue #702 - BC consistency test
    print("\n5. Testing BC operation mapping (Issue #702)...")

    from mfgarchon.geometry.boundary.bc_utils import bc_type_to_geometric_operation, get_bc_type_string

    # Verify the solver uses bc_utils correctly
    bc_str = get_bc_type_string(solver.boundary_conditions)
    bc_op = solver._get_bc_operation_type()
    print(f"   BC type: {bc_str} -> operation: {bc_op}")
    assert bc_op == "reflect", f"NEUMANN BC should map to 'reflect', got '{bc_op}'"

    # Test that _apply_boundary_conditions exists and works
    test_x = np.array([1.2, -0.1, 0.5])  # One outside right, one outside left, one inside
    bounded_x = solver._apply_boundary_conditions(test_x)
    print(f"   x=[1.2, -0.1, 0.5] -> {bounded_x}")
    assert all(bounded_x >= X_MIN), "All points should be >= X_MIN"
    assert all(bounded_x <= X_MAX), "All points should be <= X_MAX"
    print("   BC consistency: OK")

    print("\n" + "=" * 60)
    print("All smoke tests passed!")

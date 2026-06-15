#!/usr/bin/env python3
"""
Variational Mean Field Games Problem Formulation

This module implements the variational (direct optimization) perspective of Mean Field Games,
providing a complementary formulation to the traditional Hamilton-Jacobi-Bellman approach.

Mathematical Framework:
- Individual agents solve: min∫[L(x,ẋ,m,t) + V(x,t)]dt + g(x(T))
- Collective behavior: consistency between individual optimization and population flow
- Variational principle: Minimize total social cost over all admissible flows

The variational formulation provides:
1. Direct optimization perspective
2. Natural constraint handling
3. Economic interpretation via cost functionals
4. Primal-dual solution methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    # For type checking only - these imports may not be available at runtime
    import contextlib
    from collections.abc import Callable

    from numpy.typing import NDArray

    try:
        import jax.numpy as jnp
        from jax import jit
    except ImportError:
        pass  # JAX not available for type checking

    with contextlib.suppress(ImportError):
        from scipy.optimize import minimize

logger = get_logger(__name__)
try:
    import jax.numpy as jnp  # noqa: F401
    from jax import grad, jit, vmap  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    from scipy.optimize import minimize  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class VariationalMFGComponents:
    """
    Container for all components defining a Lagrangian MFG problem.

    The Lagrangian formulation focuses on the cost functional that agents minimize:
    J[x,m] = ∫₀ᵀ L(t,x(t),ẋ(t),m(t)) dt + g(x(T))

    where:
    - L(t,x,v,m) is the running cost (Lagrangian)
    - g(x) is the terminal cost
    - m(t) is the population density
    """

    # Core Lagrangian function
    lagrangian_func: Callable | None = None  # L(t, x, v, m) -> float

    # Derivatives for optimization (can be provided or computed numerically)
    lagrangian_dx_func: Callable | None = None  # ∂L/∂x
    lagrangian_dv_func: Callable | None = None  # ∂L/∂v (velocity)
    lagrangian_dm_func: Callable | None = None  # ∂L/∂m (population coupling)

    # Terminal cost
    terminal_cost_func: Callable | None = None  # g(x) -> float
    terminal_cost_dx_func: Callable | None = None  # ∂g/∂x

    # Running potential (time-varying external forces)
    potential_func: Callable | None = None  # V(t, x) -> float
    potential_dx_func: Callable | None = None  # ∂V/∂x

    # Initial conditions
    m_initial: Callable | None = None  # m₀(x) -> float

    # Constraints (optional)
    state_constraints: list[Callable] | None = None  # c(t, x) ≤ 0
    velocity_constraints: list[Callable] | None = None  # h(t, x, v) ≤ 0
    integral_constraints: list[Callable] | None = None  # ∫ψ(x,m)dx = constant

    # Problem parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Noise/stochasticity
    noise_intensity: float = 0.0  # sigma for stochastic differential equations

    # Metadata
    description: str = "Lagrangian MFG Problem"
    problem_type: str = "lagrangian_mfg"


class VariationalMFGProblem:
    """
    Lagrangian formulation of Mean Field Games.

    This class represents MFG problems from the perspective of cost minimization
    and variational principles, complementing the Hamiltonian HJB-FP approach.

    Key features:
    - Direct specification of cost functionals
    - Natural handling of constraints
    - Variational solution methods
    - Conversion to/from Hamiltonian formulation
    """

    def __init__(
        self,
        # Domain parameters
        xmin: float = 0.0,
        xmax: float = 1.0,
        Nx: int = 51,
        T: float = 1.0,
        Nt: int = 51,
        # Lagrangian components
        components: VariationalMFGComponents | None = None,
        # Standard MFG parameters (for compatibility)
        sigma: float = 1.0,
        **kwargs: Any,
    ):
        """
        Initialize Lagrangian MFG problem.

        Args:
            xmin, xmax, Nx: Spatial domain discretization
            T, Nt: Time domain discretization
            components: Lagrangian problem specification
            sigma: Noise intensity (volatility coefficient)
            **kwargs: Additional parameters
        """
        # Domain setup
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        self.Lx = xmax - xmin
        self.dx = self.Lx / Nx

        self.T = T
        self.Nt = Nt  # Number of time intervals (NOT points)
        # dt = T / Nt: Nt intervals means Nt+1 points from t=0 to t=T
        self.dt = T / Nt if Nt > 0 else T

        # Grid points: Nt+1 time points, Nx+1 space points
        self.x = np.linspace(xmin, xmax, Nx + 1)
        self.t = np.linspace(0, T, Nt + 1)

        # Noise/diffusion
        self.sigma = sigma

        # Lagrangian components
        self.components = components or self._create_default_components()

        # Additional parameters
        self.parameters = kwargs

        # Setup JAX functions if available
        self.use_jax = JAX_AVAILABLE
        if self.use_jax:
            self._setup_jax_functions()

        logger.info(f"Created Lagrangian MFG problem: domain=[{xmin},{xmax}], T={T}")
        logger.info(f"  Grid: Nx={Nx}, Nt={Nt}, σ={sigma}")

    def _create_default_components(self) -> VariationalMFGComponents:
        """Create default Lagrangian components for standard MFG problem."""

        def default_lagrangian(t: float, x: float, v: float, m: float) -> float:
            """
            Default quadratic Lagrangian: L(t,x,v,m) = |v|²/2 + α*m

            This corresponds to:
            - Kinetic cost: |v|²/2 (control effort)
            - Congestion cost: α*m (population interaction)
            """
            kinetic_cost = 0.5 * v**2
            congestion_cost = 0.5 * m  # Default congestion coefficient
            return kinetic_cost + congestion_cost

        def default_lagrangian_dv(t: float, x: float, v: float, m: float) -> float:
            """∂L/∂v = v (momentum)"""
            return v

        def default_lagrangian_dm(t: float, x: float, v: float, m: float) -> float:
            """∂L/∂m = α (marginal congestion cost)"""
            return 0.5  # Default congestion coefficient

        def default_terminal_cost(x: float) -> float:
            """Default terminal cost: g(x) = 0"""
            return 0.0

        def default_initial_density(x: float) -> float:
            """Default initial density: uniform distribution"""
            return 1.0 / (self.xmax - self.xmin)

        return VariationalMFGComponents(
            lagrangian_func=default_lagrangian,
            lagrangian_dv_func=default_lagrangian_dv,
            lagrangian_dm_func=default_lagrangian_dm,
            terminal_cost_func=default_terminal_cost,
            m_initial=default_initial_density,
            noise_intensity=self.sigma,
            parameters={"congestion_coefficient": 0.5},
        )

    def _setup_jax_functions(self):
        """Setup JAX-accelerated functions for performance."""
        if not self.use_jax:
            return

        # JAX-compiled versions of key functions
        if self.components.lagrangian_func is not None:

            @jit
            def jax_lagrangian(t, x, v, m):
                # Safe to call since we checked for None above
                assert self.components.lagrangian_func is not None
                return self.components.lagrangian_func(t, x, v, m)

            self._jax_lagrangian = jax_lagrangian

        if self.components.terminal_cost_func is not None:

            @jit
            def jax_terminal_cost(x):
                # Safe to call since we checked for None above
                assert self.components.terminal_cost_func is not None
                return self.components.terminal_cost_func(x)

            self._jax_terminal_cost = jax_terminal_cost

    def evaluate_lagrangian(
        self,
        t: float,
        x: float | NDArray,
        v: float | NDArray,
        m: float | NDArray,
    ) -> float | NDArray:
        """
        Evaluate Lagrangian function L(t,x,v,m).

        Args:
            t: Time
            x: Position (scalar or array)
            v: Velocity (scalar or array)
            m: Density (scalar or array)

        Returns:
            Lagrangian value(s)
        """
        if self.use_jax and hasattr(self, "_jax_lagrangian"):
            return self._jax_lagrangian(t, x, v, m)
        elif self.components.lagrangian_func:
            return self.components.lagrangian_func(t, x, v, m)
        else:
            raise ValueError("No Lagrangian function defined")

    def evaluate_terminal_cost(self, x: float | NDArray) -> float | NDArray:
        """
        Evaluate terminal cost g(x).

        Args:
            x: Final position (scalar or array)

        Returns:
            Terminal cost value(s)
        """
        if self.use_jax and hasattr(self, "_jax_terminal_cost"):
            return self._jax_terminal_cost(x)
        elif self.components.terminal_cost_func:
            return self.components.terminal_cost_func(x)
        else:
            return 0.0  # Default: no terminal cost

    def compute_lagrangian_derivatives(self, t: float, x: float, v: float, m: float) -> dict[str, float]:
        """
        Compute all partial derivatives of Lagrangian.

        Args:
            t: Time
            x: Position
            v: Velocity
            m: Density

        Returns:
            Dictionary with derivative values
        """
        derivatives = {}

        # ∂L/∂x (force term)
        if self.components.lagrangian_dx_func:
            derivatives["dx"] = self.components.lagrangian_dx_func(t, x, v, m)
        else:
            # Numerical derivative
            eps = 1e-8
            L_plus = self.evaluate_lagrangian(t, x + eps, v, m)
            L_minus = self.evaluate_lagrangian(t, x - eps, v, m)
            derivatives["dx"] = (L_plus - L_minus) / (2 * eps)

        # ∂L/∂v (momentum)
        if self.components.lagrangian_dv_func:
            derivatives["dv"] = self.components.lagrangian_dv_func(t, x, v, m)
        else:
            # Numerical derivative
            eps = 1e-8
            L_plus = self.evaluate_lagrangian(t, x, v + eps, m)
            L_minus = self.evaluate_lagrangian(t, x, v - eps, m)
            derivatives["dv"] = (L_plus - L_minus) / (2 * eps)

        # ∂L/∂m (population interaction)
        if self.components.lagrangian_dm_func:
            derivatives["dm"] = self.components.lagrangian_dm_func(t, x, v, m)
        else:
            # Numerical derivative
            eps = 1e-8
            L_plus = self.evaluate_lagrangian(t, x, v, m + eps)
            L_minus = self.evaluate_lagrangian(t, x, v, m - eps)
            derivatives["dm"] = (L_plus - L_minus) / (2 * eps)

        return derivatives

    def convert_to_hamiltonian(self) -> dict[str, Callable]:
        """
        Convert Lagrangian formulation to Hamiltonian via Legendre transform.

        The Hamiltonian is obtained via: H(x,p,m) = max_v [p·v - L(x,v,m)]
        For quadratic Lagrangians: H(x,p,m) = |p|²/2 + interaction_terms

        Returns:
            Dictionary with Hamiltonian functions
        """

        def hamiltonian(x: float, p: float, m: float, t: float = 0.0) -> float:
            """
            Hamiltonian from Legendre transform of Lagrangian.

            For quadratic kinetic energy L = |v|²/2 + I(x,m):
            H(x,p,m) = |p|²/2 + I(x,m)
            """
            # For standard quadratic Lagrangian
            kinetic_hamiltonian = 0.5 * p**2

            # Interaction terms (independent of velocity)
            interaction_term = 0.0
            if self.components.lagrangian_dm_func:
                # Extract interaction part by evaluating at v=0
                total_cost_at_zero_velocity = self.evaluate_lagrangian(t, x, 0.0, m)
                kinetic_at_zero = 0.0  # No kinetic energy at v=0
                interaction_term = float(total_cost_at_zero_velocity - kinetic_at_zero)
                # Ensure scalar return for scalar inputs

                if hasattr(interaction_term, "ndim") and getattr(interaction_term, "ndim", None) == 0:
                    interaction_term = float(interaction_term)

            result = kinetic_hamiltonian + interaction_term
            # Ensure scalar return for scalar inputs
            if isinstance(result, np.ndarray) and result.ndim == 0:
                return float(result.item())
            return float(result)

        def hamiltonian_dp(x: float, p: float, m: float, t: float = 0.0) -> float:
            """∂H/∂p = p (for quadratic Hamiltonian)"""
            return p

        def hamiltonian_dm(x: float, p: float, m: float, t: float = 0.0) -> float:
            """∂H/∂m from Lagrangian coupling"""
            if self.components.lagrangian_dm_func:
                # For velocity-independent coupling, ∂H/∂m = ∂L/∂m
                return self.components.lagrangian_dm_func(t, x, p, m)
            else:
                return 0.0

        return {
            "hamiltonian": hamiltonian,
            "hamiltonian_dp": hamiltonian_dp,
            "hamiltonian_dm": hamiltonian_dm,
        }

    def compute_action_functional(self, trajectory: NDArray, velocity: NDArray, density_evolution: NDArray) -> float:
        """
        Compute action functional for given trajectory and density evolution.

        S[x,m] = ∫₀ᵀ L(t,x(t),ẋ(t),m(t)) dt + g(x(T))

        Args:
            trajectory: x(t) trajectory shape (Nt+1,) - one value per time point
            velocity: ẋ(t) velocity shape (Nt+1,) - one value per time point
            density_evolution: m(t,x) density shape (Nt+1, Nx+1)

        Returns:
            Total action value
        """
        n_time_points = self.Nt + 1
        if len(trajectory) != n_time_points or len(velocity) != n_time_points:
            raise ValueError(f"Trajectory and velocity must have length Nt+1={n_time_points}")

        # Integrate running cost
        running_cost = 0.0
        for i, (t, x, v) in enumerate(zip(self.t, trajectory, velocity, strict=False)):
            # Interpolate density at current position
            m_at_x = self._interpolate_density(density_evolution[i], x)

            # Add running cost
            cost_contribution = self.evaluate_lagrangian(t, x, v, m_at_x)
            running_cost += float(cost_contribution * self.dt)

        # Add terminal cost
        terminal_cost = self.evaluate_terminal_cost(trajectory[-1])

        total_action = running_cost + terminal_cost

        # Ensure scalar return
        if isinstance(total_action, np.ndarray) and total_action.ndim == 0:
            return float(total_action.item())
        return float(total_action)

    def _interpolate_density(self, density_field: NDArray, x: float) -> float:
        """Interpolate density at position x from density field."""
        if x <= self.xmin:
            return density_field[0]
        elif x >= self.xmax:
            return density_field[-1]
        else:
            # Linear interpolation
            idx = (x - self.xmin) / self.dx
            i = int(idx)
            alpha = idx - i

            if i >= len(density_field) - 1:
                return density_field[-1]

            return (1 - alpha) * density_field[i] + alpha * density_field[i + 1]

    def compute_euler_lagrange_residual(
        self,
        trajectory: NDArray,
        velocity: NDArray,
        acceleration: NDArray,
        density_evolution: NDArray,
    ) -> NDArray:
        """
        Compute Euler-Lagrange equation residual.

        The Euler-Lagrange equation for individual trajectories:
        d/dt(∂L/∂v) - ∂L/∂x = 0

        Args:
            trajectory: x(t) shape (Nt+1,) - one value per time point
            velocity: ẋ(t) shape (Nt+1,) - one value per time point
            acceleration: ẍ(t) shape (Nt+1,) - one value per time point
            density_evolution: m(t,x) shape (Nt+1, Nx+1)

        Returns:
            Residual array shape (Nt+1,) - one value per time point
        """
        n_time_points = self.Nt + 1
        residual = np.zeros(n_time_points)

        for i, (t, x, v, a) in enumerate(zip(self.t, trajectory, velocity, acceleration, strict=False)):
            # Interpolate density
            m_at_x = self._interpolate_density(density_evolution[i], x)

            # Compute Lagrangian derivatives
            derivatives = self.compute_lagrangian_derivatives(t, x, v, m_at_x)

            # Euler-Lagrange equation: d/dt(∂L/∂v) - ∂L/∂x = 0
            # For L = |v|²/2 + I(x,m): d/dt(v) - ∂I/∂x = a - ∂L/∂x = 0
            residual[i] = a - derivatives["dx"]

        return residual

    def create_compatible_mfg_problem(self):
        """
        Create compatible MFGProblem for use with existing HJB-FP solvers.

        Returns:
            MFGProblem instance with Hamiltonian derived from this Lagrangian
        """
        from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem

        # Convert to Hamiltonian formulation
        hamiltonian_funcs = self.convert_to_hamiltonian()

        # Create MFG components with modern signature (Issue #670)
        # hamiltonian_func signature: (x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem)
        def _hamiltonian_adapter(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem):
            """Adapter converting modern derivs format to converted Hamiltonian."""
            p = derivs.get((1,), 0.0) if isinstance(derivs, dict) else 0.0
            t = current_time if current_time is not None else self.t[t_idx] if t_idx < len(self.t) else 0.0
            return hamiltonian_funcs["hamiltonian"](x_position, p, m_at_x, t)

        def _hamiltonian_dm_adapter(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem):
            """Adapter converting modern derivs format to converted Hamiltonian dm."""
            p = derivs.get((1,), 0.0) if isinstance(derivs, dict) else 0.0
            t = current_time if current_time is not None else self.t[t_idx] if t_idx < len(self.t) else 0.0
            return hamiltonian_funcs["hamiltonian_dm"](x_position, p, m_at_x, t)

        mfg_components = MFGComponents(
            hamiltonian_func=_hamiltonian_adapter,
            hamiltonian_dm_func=_hamiltonian_dm_adapter,
            m_initial=self.components.m_initial,
            u_terminal=lambda x: -self.evaluate_terminal_cost(x),  # Value = -cost
            description=f"Hamiltonian formulation of {self.components.description}",
        )

        # Create MFG problem (Issue #1363: geometry-first API)
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc

        geometry = TensorProductGrid(
            bounds=[(self.xmin, self.xmax)],
            Nx_points=[self.Nx + 1],  # Nx intervals -> Nx + 1 grid points
            boundary_conditions=no_flux_bc(dimension=1),
        )
        return MFGProblem(
            geometry=geometry,
            T=self.T,
            Nt=self.Nt,
            sigma=self.sigma,
            components=mfg_components,
        )

    def get_problem_info(self) -> dict[str, Any]:
        """Get comprehensive problem information."""
        return {
            "formulation": "Lagrangian",
            "domain": {
                "spatial": [self.xmin, self.xmax],
                "temporal": [0.0, self.T],
                "discretization": {"Nx": self.Nx, "Nt": self.Nt},
            },
            "parameters": {
                "noise_intensity": self.sigma,
                "lagrangian_parameters": self.components.parameters,
            },
            "components": {
                "has_lagrangian": self.components.lagrangian_func is not None,
                "has_terminal_cost": self.components.terminal_cost_func is not None,
                "has_constraints": (
                    self.components.state_constraints is not None or self.components.velocity_constraints is not None
                ),
                "has_jax_acceleration": self.use_jax,
            },
            "description": self.components.description,
        }


# Utility functions for common Lagrangian formulations


def create_quadratic_variational_mfg(
    xmin: float = 0.0,
    xmax: float = 1.0,
    Nx: int = 51,
    T: float = 1.0,
    Nt: int = 51,
    kinetic_coefficient: float = 0.5,
    congestion_coefficient: float = 0.5,
    sigma: float = 1.0,
) -> VariationalMFGProblem:
    """
    Create standard quadratic Lagrangian MFG problem.

    L(t,x,v,m) = α|v|²/2 + β*m

    Args:
        Domain and discretization parameters
        kinetic_coefficient: α (control cost)
        congestion_coefficient: β (congestion cost)
        sigma: Noise intensity

    Returns:
        VariationalMFGProblem with quadratic structure
    """

    def quadratic_lagrangian(t: float, x: float, v: float, m: float) -> float:
        return kinetic_coefficient * v**2 / 2 + congestion_coefficient * m

    def lagrangian_dv(t: float, x: float, v: float, m: float) -> float:
        return kinetic_coefficient * v

    def lagrangian_dm(t: float, x: float, v: float, m: float) -> float:
        return congestion_coefficient

    components = VariationalMFGComponents(
        lagrangian_func=quadratic_lagrangian,
        lagrangian_dv_func=lagrangian_dv,
        lagrangian_dm_func=lagrangian_dm,
        terminal_cost_func=lambda x: 0.0,
        m_initial=lambda x: 1.0 / (xmax - xmin),
        noise_intensity=sigma,
        parameters={
            "kinetic_coefficient": kinetic_coefficient,
            "congestion_coefficient": congestion_coefficient,
        },
        description="Quadratic Lagrangian MFG",
    )

    return VariationalMFGProblem(xmin=xmin, xmax=xmax, Nx=Nx, T=T, Nt=Nt, sigma=sigma, components=components)


def create_obstacle_variational_mfg(
    xmin: float = 0.0,
    xmax: float = 1.0,
    Nx: int = 51,
    T: float = 1.0,
    Nt: int = 51,
    obstacle_center: float = 0.5,
    obstacle_radius: float = 0.1,
    obstacle_penalty: float = 100.0,
    sigma: float = 1.0,
) -> VariationalMFGProblem:
    """
    Create Lagrangian MFG with obstacle avoidance.

    L(t,x,v,m) = |v|²/2 + congestion*m + obstacle_penalty*I_obstacle(x)

    Args:
        Domain parameters
        obstacle_center: Center of obstacle
        obstacle_radius: Radius of obstacle region
        obstacle_penalty: Penalty coefficient for obstacle
        sigma: Noise intensity

    Returns:
        VariationalMFGProblem with obstacle constraints
    """

    def obstacle_lagrangian(t: float, x: float, v: float, m: float) -> float:
        kinetic = 0.5 * v**2
        congestion = 0.5 * m

        # Obstacle penalty
        obstacle_distance = abs(x - obstacle_center)
        if obstacle_distance < obstacle_radius:
            obstacle_cost = obstacle_penalty * (obstacle_radius - obstacle_distance) ** 2
        else:
            obstacle_cost = 0.0

        return kinetic + congestion + obstacle_cost

    # State constraint: avoid obstacle
    def obstacle_constraint(t: float, x: float) -> float:
        """Constraint: distance from obstacle center >= obstacle_radius"""
        return abs(x - obstacle_center) - obstacle_radius

    components = VariationalMFGComponents(
        lagrangian_func=obstacle_lagrangian,
        lagrangian_dv_func=lambda t, x, v, m: v,
        lagrangian_dm_func=lambda t, x, v, m: 0.5,
        terminal_cost_func=lambda x: 0.0,
        m_initial=lambda x: 1.0 / (xmax - xmin),
        state_constraints=[obstacle_constraint],
        noise_intensity=sigma,
        parameters={
            "obstacle_center": obstacle_center,
            "obstacle_radius": obstacle_radius,
            "obstacle_penalty": obstacle_penalty,
        },
        description="Obstacle Avoidance Lagrangian MFG",
    )

    return VariationalMFGProblem(xmin=xmin, xmax=xmax, Nx=Nx, T=T, Nt=Nt, sigma=sigma, components=components)

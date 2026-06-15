"""
Stochastic MFG problem definitions.

This module extends the base MFG problem class to support stochastic formulations
with common noise processes and conditional dynamics.

Mathematical Background:
    Stochastic MFG with common noise involves:

    HJB Equation (Conditional on noise θ_t):
        ∂u/∂t + H(x, ∇u, m^θ, θ_t) + σ²/2 Δu = 0
        u(T, x, θ_T) = g(x, θ_T)

    Fokker-Planck Equation (Conditional):
        ∂m^θ/∂t - div(m^θ ∇_p H(x, ∇u, m^θ, θ)) - σ²/2 Δm^θ = 0
        m^θ(0, x) = m_0(x)

    Common Noise Process:
        dθ_t = μ(θ_t, t) dt + σ_θ(θ_t, t) dB_t
        θ_0 given

References:
    - Carmona & Delarue (2018): Probabilistic Theory of Mean Field Games
    - Carmona, Fouque, & Sun (2015): Mean Field Games and Systemic Risk
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from mfgarchon.core.stochastic.noise_processes import NoiseProcess


class StochasticMFGProblem(MFGProblem):
    """
    Stochastic MFG problem with common noise process.

    Extends the base MFG problem to support:
    - Common noise process θ_t affecting all agents
    - Conditional Hamiltonians H(x, p, m^θ, θ)
    - Noise-dependent terminal conditions g(x, θ_T)
    - Stochastic coupling terms

    The problem is solved by:
    1. Sampling K paths of the common noise process θ_t
    2. Solving conditional MFG for each noise realization
    3. Aggregating solutions via Monte Carlo averaging

    Attributes:
        noise_process: Common noise process θ_t
        conditional_hamiltonian: H(x, p, m, θ) depending on noise
        noise_coupling: Additional coupling through noise
        theta_initial: Initial value of noise process

    Example:
        >>> from mfgarchon.core.stochastic import (
        ...     StochasticMFGProblem,
        ...     OrnsteinUhlenbeckProcess
        ... )
        >>>
        >>> # Define common noise (market volatility)
        >>> vix_process = OrnsteinUhlenbeckProcess(
        ...     kappa=2.0, mu=20.0, sigma=8.0
        ... )
        >>>
        >>> # Define conditional Hamiltonian
        >>> def market_hamiltonian(x, p, m, theta):
        ...     # Control cost adjusted by market volatility
        ...     risk_premium = 0.5 * (theta / 20.0) * p**2
        ...     congestion = 0.1 * m
        ...     return risk_premium + congestion
        >>>
        >>> # Create stochastic MFG problem
        >>> problem = StochasticMFGProblem(
        ...     xmin=0.0, xmax=10.0, Nx=100,
        ...     T=1.0, Nt=100,
        ...     noise_process=vix_process,
        ...     conditional_hamiltonian=market_hamiltonian,
        ... )
    """

    def __init__(
        self,
        xmin: float = 0.0,
        xmax: float = 1.0,
        Nx: int = 51,
        T: float = 1.0,
        Nt: int = 51,
        sigma: float = 1.0,
        noise_process: NoiseProcess | None = None,
        conditional_hamiltonian: Callable | None = None,
        conditional_terminal_cost: Callable | None = None,
        theta_initial: float | None = None,
        components: MFGComponents | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize stochastic MFG problem.

        Args:
            xmin, xmax, Nx: Spatial domain [xmin, xmax] with Nx grid points
            T, Nt: Time horizon [0, T] with Nt time steps
            sigma: Diffusion coefficient in spatial dynamics
            noise_process: Common noise process θ_t (e.g., OrnsteinUhlenbeck)
            conditional_hamiltonian: H(x, p, m, theta) - Hamiltonian depending on noise
            conditional_terminal_cost: g(x, theta_T) - Terminal cost depending on final noise
            theta_initial: Initial value θ_0 of noise process (default: process-dependent)
            components: Optional MFGComponents for additional customization
            **kwargs: Additional problem parameters

        Raises:
            ValueError: If noise_process provided but conditional_hamiltonian is None
        """
        # Initialize base MFG problem
        # If geometry is provided in kwargs, don't pass legacy xmin/xmax/Nx
        # to avoid ambiguous initialization (Issue #543 compliance)
        if "geometry" in kwargs:
            super().__init__(
                T=T,
                Nt=Nt,
                sigma=sigma,
                components=components,
                **kwargs,
            )
        else:
            super().__init__(
                xmin=xmin,
                xmax=xmax,
                Nx=Nx,
                T=T,
                Nt=Nt,
                sigma=sigma,
                components=components,
                **kwargs,
            )

        # Stochastic-specific attributes
        self.noise_process = noise_process
        self.conditional_hamiltonian = conditional_hamiltonian
        self.conditional_terminal_cost = conditional_terminal_cost
        self.theta_initial = theta_initial

        # Normalize terminal cost attribute names (Issue #543 - eliminate hasattr)
        # Support both 'g' (MFGProblem standard) and 'terminal_cost' (simplified API)
        self._terminal_cost_normalized = getattr(self, "terminal_cost", None) or getattr(self, "g", None)

        # Validate configuration
        if noise_process is not None and conditional_hamiltonian is None:
            raise ValueError("If noise_process is provided, conditional_hamiltonian must also be specified")

        # Store stochastic problem type
        self.problem_type = "stochastic_mfg"

    def has_common_noise(self) -> bool:
        """
        Check if problem has common noise component.

        Returns:
            True if noise process is defined, False otherwise
        """
        return self.noise_process is not None

    def sample_noise_path(self, seed: int | None = None) -> np.ndarray:
        """
        Sample a path of the common noise process.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Array of shape (Nt+1,) with noise values θ_0, ..., θ_T

        Raises:
            ValueError: If no noise process is defined
        """
        if self.noise_process is None:
            raise ValueError("Cannot sample noise path: noise_process is None")

        return self.noise_process.sample_path(
            T=self.T,
            Nt=self.Nt,
            theta0=self.theta_initial,
            seed=seed,
        )

    def H_conditional(
        self,
        x: float | np.ndarray,
        p: float | np.ndarray,
        m: float | np.ndarray,
        theta: float | np.ndarray,
        t: float,
    ) -> float | np.ndarray:
        """
        Evaluate conditional Hamiltonian H(x, p, m, θ).

        Args:
            x: Spatial position
            p: Momentum (∇u)
            m: Density value
            theta: Current noise value θ_t
            t: Current time

        Returns:
            Hamiltonian value H(x, p, m, θ)

        Raises:
            ValueError: If conditional_hamiltonian not defined
        """
        if self.conditional_hamiltonian is None:
            raise ValueError("Conditional Hamiltonian not defined for this problem")

        # Check if user's function accepts time parameter using inspect
        import inspect

        sig = inspect.signature(self.conditional_hamiltonian)
        num_params = len(sig.parameters)

        if num_params >= 5:
            # Function accepts time parameter
            return self.conditional_hamiltonian(x, p, m, theta, t)
        else:
            # Function does not accept time (standard case)
            return self.conditional_hamiltonian(x, p, m, theta)

    def g_conditional(self, x: float | np.ndarray, theta_T: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate conditional terminal cost g(x, θ_T).

        Args:
            x: Spatial position
            theta_T: Terminal noise value

        Returns:
            Terminal cost value g(x, θ_T)
        """
        if self.conditional_terminal_cost is None:
            # Default: no dependence on terminal noise
            # Use normalized terminal cost attribute (set in __init__)
            if self._terminal_cost_normalized is not None:
                return self._terminal_cost_normalized(x)
            else:
                # No terminal cost defined - use zero
                return 0.0

        return self.conditional_terminal_cost(x, theta_T)

    def create_conditional_problem(self, noise_path: np.ndarray) -> MFGProblem:
        """
        Create conditional MFG problem for given noise realization.

        Given a sample path θ_0, ..., θ_T of the noise process,
        create a deterministic MFG problem with noise-dependent coefficients.

        Args:
            noise_path: Array of shape (Nt+1,) with noise realization

        Returns:
            Deterministic MFGProblem with frozen noise path

        Example:
            >>> problem = StochasticMFGProblem(...)
            >>> noise_path = problem.sample_noise_path(seed=42)
            >>> conditional_problem = problem.create_conditional_problem(noise_path)
            >>> # Now solve conditional_problem as standard MFG
        """
        # Create components for conditional problem
        conditional_components = MFGComponents(
            description="Conditional MFG with noise path (seed dependent)",
            problem_type="conditional_mfg",
        )

        # Spatial grid (geometry-first API); captured by the closures below.
        x_grid = self.geometry.get_spatial_grid()

        # Create wrapper functions that incorporate noise path
        # These must match the MFGComponents API signature
        def conditional_H(x_idx, m_at_x, p_values, t_idx, x_position=None, current_time=None, problem=None):
            """Hamiltonian with frozen noise path."""
            import numpy as np

            # Get scalar values from grid-based inputs
            x = x_position if x_position is not None else x_grid[x_idx]
            t = current_time if current_time is not None else t_idx * self.dt
            m = m_at_x

            # Extract gradient from p_values dict (use average of forward/backward)
            if isinstance(p_values, dict):
                p_fwd = p_values.get("forward", 0.0)
                p_bwd = p_values.get("backward", 0.0)
                p = 0.5 * (p_fwd + p_bwd) if not (np.isnan(p_fwd) or np.isnan(p_bwd)) else 0.0
            else:
                p = p_values  # Assume scalar

            # Get noise value at this time
            theta_t = noise_path[min(t_idx, len(noise_path) - 1)]

            # Call simplified API Hamiltonian
            return self.H_conditional(x, p, m, theta_t, t)

        def conditional_H_dm(x_idx, m_at_x, p_values, t_idx, x_position=None, current_time=None, problem=None):
            """Hamiltonian derivative w.r.t. m using finite differences."""
            import numpy as np

            # Get scalar values
            x = x_position if x_position is not None else x_grid[x_idx]
            t = current_time if current_time is not None else t_idx * self.dt
            m = m_at_x

            # Extract gradient
            if isinstance(p_values, dict):
                p_fwd = p_values.get("forward", 0.0)
                p_bwd = p_values.get("backward", 0.0)
                p = 0.5 * (p_fwd + p_bwd) if not (np.isnan(p_fwd) or np.isnan(p_bwd)) else 0.0
            else:
                p = p_values

            # Get noise value
            theta_t = noise_path[min(t_idx, len(noise_path) - 1)]

            # Compute ∂H/∂m using central finite difference
            eps = 1e-6
            H_plus = self.H_conditional(x, p, m + eps, theta_t, t)
            H_minus = self.H_conditional(x, p, m - eps, theta_t, t)
            return (H_plus - H_minus) / (2 * eps)

        def conditional_g(x):
            """Terminal cost with final noise value."""
            theta_T = noise_path[-1]
            return self.g_conditional(x, theta_T)

        conditional_components.hamiltonian_func = conditional_H
        conditional_components.hamiltonian_dm_func = conditional_H_dm
        conditional_components.u_terminal = conditional_g

        # Preserve other problem components
        if self.components is not None:
            conditional_components.m_initial = self.components.m_initial
            conditional_components.boundary_conditions = self.components.boundary_conditions
            conditional_components.parameters = self.components.parameters.copy()

        # Create conditional problem
        conditional_problem = MFGProblem(
            geometry=self.geometry,
            T=self.T,
            Nt=self.Nt,
            sigma=self.sigma,
            components=conditional_components,
        )

        return conditional_problem

    def __repr__(self) -> str:
        """String representation of stochastic MFG problem."""
        noise_info = "None" if self.noise_process is None else str(self.noise_process)
        b = self.geometry.get_bounds()
        return (
            f"StochasticMFGProblem(\n"
            f"  domain=[{b[0][0]}, {b[1][0]}], Nx={self.geometry.num_spatial_points - 1},\n"
            f"  time=[0, {self.T}], Nt={self.Nt},\n"
            f"  sigma={self.sigma},\n"
            f"  noise_process={noise_info}\n"
            f")"
        )

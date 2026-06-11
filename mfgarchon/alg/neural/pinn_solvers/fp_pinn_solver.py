"""
Physics-Informed Neural Network solver for Fokker-Planck equations.

This module implements a PINN solver specifically designed for the Fokker-Planck equation
in Mean Field Games:

∂m/∂t - div(m∇H_p(∇u, x, m)) - (σ²/2)Δm = 0

where m(t,x) is the population density, u(t,x) is the value function from HJB,
H_p is the derivative of the Hamiltonian with respect to momentum, and σ is the
diffusion coefficient.

Key Features:
- Automatic differentiation for computing gradients and divergences
- Mass conservation constraints and monitoring
- Flexible drift term specification via Hamiltonian derivatives
- Support for various boundary and initial conditions
- Integration with HJB coupling through given value function
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

from mfgarchon.alg.neural.nn import create_mfg_networks
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility_torch

from .base_pinn import PINNBase, PINNConfig

if TYPE_CHECKING:
    from mfgarchon.core.mfg_problem import MFGProblem

# PyTorch imports with fallback
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class FPPINNSolver(PINNBase):
    """
    Physics-Informed Neural Network solver for Fokker-Planck equations.

    This solver focuses on the FP equation in MFG systems:
    ∂m/∂t - div(m∇H_p(∇u, x, m)) - (σ²/2)Δm = 0

    The neural network approximates m(t,x) and uses automatic differentiation
    to compute the required derivatives for the PDE residual.
    """

    def __init__(
        self,
        problem: MFGProblem,
        config: PINNConfig | None = None,
        value_function_net: nn.Module | None = None,
        drift_func: Callable | None = None,
        networks: dict[str, nn.Module] | None = None,
    ):
        """
        Initialize FP PINN solver.

        Args:
            problem: MFG problem containing FP equation specification
            config: PINN configuration
            value_function_net: Pre-trained network for u(t,x) or None
            drift_func: Custom drift function b(∇u, x, m)
            networks: Pre-defined neural networks (optional)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PINN functionality")

        # Store FP-specific parameters
        self.value_function_net = value_function_net
        self.drift_func = drift_func

        # Extract problem parameters
        self.sigma = getattr(problem, "sigma", 0.1)  # Diffusion coefficient
        self.initial_density = getattr(problem, "initial_density", None)

        # Mass conservation parameters
        self.enforce_mass_conservation = True
        self.target_total_mass = 1.0  # Typical normalization

        # Initialize base PINN
        super().__init__(problem, config, networks)

        bounds = problem.geometry.get_bounds()
        print("Initialized FP PINN solver")
        print(f"  Problem domain: t ∈ [0, {problem.T}], x ∈ [{bounds[0][0]}, {bounds[1][0]}]")
        print(f"  Diffusion coefficient: σ = {self.sigma}")

    def _initialize_networks(self) -> None:
        """Initialize neural network for density function m(t,x)."""
        # Issue #1314 (Refs #1290): prior code used architecture_type/separate_networks
        # which are not parameters of create_mfg_networks (the function takes network_type
        # and returns a single nn.Module, not a dict).  Build the {"m_net": ...} dict
        # explicitly — FP needs m_net, not u_net.
        m_net = create_mfg_networks(
            network_type="feedforward",
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation
            if self.config.activation in ("tanh", "relu", "sigmoid", "elu")
            else "tanh",
            input_dim=2,  # (t, x)
            output_dim=1,  # scalar density function
            problem_type="fp",
        )
        self.networks = {"m_net": m_net}
        self.m_net = self.networks["m_net"]

    def get_value_function_gradient(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Get gradient of value function ∇u(t,x).

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Value function gradient [N, spatial_dim]
        """
        if self.value_function_net is not None:
            # Use provided value function network
            x_input = x.clone().detach().requires_grad_(True)
            inputs = torch.cat([t, x_input], dim=-1)
            u = self.value_function_net(inputs)

            u_x = torch.autograd.grad(
                outputs=u, inputs=x_input, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
            )[0]

            return u_x
        else:
            # Use analytical approximation or default
            # This is a placeholder - in practice, you'd have a coupled solver
            # or analytical solution for simple cases
            return torch.zeros_like(x)

    def default_drift_function(self, u_x: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Default drift function: b(∇u, x, m) = -∇u (optimal control)

        For the standard MFG Hamiltonian H(p, x, m) = (1/2)|p|² + V(x, m),
        we have ∇H_p = p = ∇u, so the drift is b = -∇u.

        Args:
            u_x: Gradient of value function [N, spatial_dim]
            x: Spatial coordinates [N, spatial_dim]
            m: Density values [N, 1]

        Returns:
            Drift term [N, spatial_dim]
        """
        return -u_x  # Standard optimal control for quadratic Hamiltonian

    def compute_drift(self, u_x: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Compute drift term b(∇u, x, m).

        Args:
            u_x: Value function gradient
            x: Spatial coordinates
            m: Density values

        Returns:
            Drift term
        """
        if self.drift_func is not None:
            return self.drift_func(u_x, x, m)
        else:
            return self.default_drift_function(u_x, x, m)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute m(t,x).

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Density function values [N, 1] (enforced to be positive)
        """
        # Concatenate inputs
        inputs = torch.cat([t, x], dim=-1)

        # Forward pass through network
        m_raw = self.m_net(inputs)

        # Enforce positivity constraint: m(t,x) ≥ 0
        # Use softplus or exponential to ensure positivity
        m = torch.exp(m_raw)  # Always positive, can handle large/small values
        # Alternative: m = functional.softplus(m_raw) + 1e-8

        return m

    def compute_derivatives(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute derivatives of m(t,x) using automatic differentiation.

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Tuple of (∂m/∂t, ∂m/∂x, ∂²m/∂x²)
        """
        # Enable gradient computation
        t_input = t.clone().detach().requires_grad_(True)
        x_input = x.clone().detach().requires_grad_(True)

        # Forward pass
        m = self.forward(t_input, x_input)

        # First derivatives
        m_t = torch.autograd.grad(
            outputs=m, inputs=t_input, grad_outputs=torch.ones_like(m), create_graph=True, retain_graph=True
        )[0]

        m_x = torch.autograd.grad(
            outputs=m, inputs=x_input, grad_outputs=torch.ones_like(m), create_graph=True, retain_graph=True
        )[0]

        # Second derivative (for diffusion term)
        m_xx = torch.autograd.grad(
            outputs=m_x, inputs=x_input, grad_outputs=torch.ones_like(m_x), create_graph=True, retain_graph=True
        )[0]

        return m_t, m_x, m_xx

    def compute_pde_residual(self, t: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute FP equation residual: ∂m/∂t - div(mb) - (σ²/2)Δm = 0

        The full form is: ∂m/∂t - div(m·b(∇u,x,m)) - (σ²/2)Δm = 0
        where b is the drift term related to the Hamiltonian.

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Dictionary with FP residual
        """
        # Get current density and its derivatives
        m = self.forward(t, x)
        m_t, m_x, m_xx = self.compute_derivatives(t, x)

        # Get value function gradient
        u_x = self.get_value_function_gradient(t, x)

        # Compute drift term
        drift = self.compute_drift(u_x, x, m)

        # Compute divergence of (m * drift) = m * div(drift) + drift · ∇m
        # For 1D: div(m * b) = ∂(m * b)/∂x = m * ∂b/∂x + b * ∂m/∂x

        # Since drift typically depends on u_x which depends on x,
        # we need to compute ∂b/∂x carefully
        # For simplicity in 1D case: div(m * b) ≈ drift * m_x + m * drift_x

        # Compute drift derivative (this is approximate for general case)
        x_input = x.clone().detach().requires_grad_(True)
        u_x_at_x = self.get_value_function_gradient(t, x_input)
        drift_at_x = self.compute_drift(u_x_at_x, x_input, m.detach())

        if drift_at_x.requires_grad:
            drift_x = torch.autograd.grad(
                outputs=drift_at_x,
                inputs=x_input,
                grad_outputs=torch.ones_like(drift_at_x),
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            drift_x = torch.zeros_like(drift)

        # Divergence term: div(m * b) = b * ∂m/∂x + m * ∂b/∂x
        divergence_term = drift * m_x + m * drift_x

        # Diffusion term: D * Δm  where D = σ²/2 (Issue #1189: route through single source)
        diffusion_term = diffusion_from_volatility_torch(self.sigma) * m_xx

        # FP equation residual: ∂m/∂t - div(m*b) - (σ²/2)Δm = 0
        fp_residual = m_t - divergence_term - diffusion_term

        return {"fp": fp_residual}

    def compute_mass_conservation_loss(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mass conservation constraint: ∫ m(t,x) dx = constant

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Mass conservation loss
        """
        # Evaluate density at current points
        m = self.forward(t, x)

        # Approximate integral using trapezoidal rule
        # Sort by x coordinate for proper integration
        x_flat = x.flatten()
        m_flat = m.flatten()

        if len(x_flat) > 1:
            # Sort by spatial coordinate
            sorted_indices = torch.argsort(x_flat)
            x_sorted = x_flat[sorted_indices]
            m_sorted = m_flat[sorted_indices]

            # Trapezoidal integration
            dx = x_sorted[1:] - x_sorted[:-1]
            m_avg = (m_sorted[1:] + m_sorted[:-1]) / 2
            total_mass = torch.sum(dx * m_avg)
        else:
            # Single point approximation
            bounds = self.problem.geometry.get_bounds()
            domain_size = bounds[1][0] - bounds[0][0]
            total_mass = m_flat[0] * domain_size

        # Mass conservation loss: |∫m dx - target_mass|²
        mass_loss = (total_mass - self.target_total_mass) ** 2

        return mass_loss

    def compute_initial_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition loss: m(0, x) = m₀(x)

        Args:
            x: Spatial coordinates at initial time [N, 1]

        Returns:
            Initial condition loss
        """
        # Create initial time tensor (t = 0)
        t_initial = torch.zeros_like(x)

        # Evaluate network at initial time
        m_initial = self.forward(t_initial, x)

        if self.initial_density is not None:
            # Evaluate initial density
            if callable(self.initial_density):
                x_np = x.detach().cpu().numpy()
                m0_target = self.initial_density(x_np)
                m0_target = torch.from_numpy(m0_target).to(self.device, dtype=self.dtype).reshape(-1, 1)
            else:
                # Assume initial_density is a constant or tensor
                m0_target = torch.full_like(m_initial, self.initial_density)

            # Initial condition loss: |m(0,x) - m₀(x)|²
            initial_loss = torch.mean((m_initial - m0_target) ** 2)
        else:
            # If no initial condition specified, assume uniform density
            bounds = self.problem.geometry.get_bounds()
            uniform_density = 1.0 / (bounds[1][0] - bounds[0][0])
            m0_target = torch.full_like(m_initial, uniform_density)
            initial_loss = torch.mean((m_initial - m0_target) ** 2)

        return initial_loss

    def compute_boundary_loss(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss.

        Args:
            t: Time coordinates on boundary [N, 1]
            x: Spatial coordinates on boundary [N, 1]

        Returns:
            Boundary condition loss
        """
        # Default: No-flux boundary conditions
        # This means: m(t,x)*b(t,x) + (σ²/2)*∂m/∂x = 0 at boundaries

        # Get density and its spatial derivative at boundary
        m = self.forward(t, x)
        _, m_x, _ = self.compute_derivatives(t, x)

        # Get drift at boundary
        u_x = self.get_value_function_gradient(t, x)
        drift = self.compute_drift(u_x, x, m)

        # No-flux condition: j = m*b + D*∂m/∂x = 0  where D = σ²/2 (Issue #1189)
        flux = m * drift + diffusion_from_volatility_torch(self.sigma) * m_x

        boundary_loss = torch.mean(flux**2)

        return boundary_loss

    def compute_total_loss(
        self,
        interior_points: tuple[torch.Tensor, torch.Tensor],
        boundary_points: tuple[torch.Tensor, torch.Tensor],
        initial_points: torch.Tensor,
        data_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute total loss including mass conservation constraint.

        Args:
            interior_points: Interior domain points
            boundary_points: Boundary points
            initial_points: Initial condition points
            data_points: Optional supervised data

        Returns:
            Dictionary of loss components
        """
        # Get base losses
        losses = super().compute_total_loss(interior_points, boundary_points, initial_points, data_points)

        # Add mass conservation constraint
        if self.enforce_mass_conservation:
            t_interior, x_interior = interior_points
            mass_loss = self.compute_mass_conservation_loss(t_interior, x_interior)
            losses["mass_conservation"] = mass_loss * 10.0  # Weight for mass conservation

            # Update total loss
            losses["total_loss"] = losses["total_loss"] + losses["mass_conservation"]

        return losses

    def solve(self, **kwargs: Any) -> dict:
        """
        Solve FP equation using PINN approach.

        Args:
            **kwargs: Additional arguments (verbose, etc.)

        Returns:
            Dictionary with solution results
        """
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Starting FP PINN training...")

        # Train the network
        training_history = self.train(verbose=verbose)

        # Generate solution on evaluation grid
        solution = self._generate_fp_solution()

        # Compute mass conservation metrics
        mass_metrics = self._compute_mass_metrics(solution)

        results = {
            "m": solution["m"],
            "t_grid": solution["t_grid"],
            "x_grid": solution["x_grid"],
            "training_history": training_history,
            "mass_metrics": mass_metrics,
            "network_state": self.m_net.state_dict(),
            "config": self.config,
        }

        if verbose:
            print(f"FP solution completed. Final loss: {self.best_loss:.6f}")
            print(f"Mass conservation error: {mass_metrics['max_mass_error']:.6f}")

        return results

    def _generate_fp_solution(self) -> dict[str, np.ndarray]:
        """Generate FP solution on evaluation grid."""
        # Create evaluation grid
        nt, nx = 100, 100
        bounds = self.problem.geometry.get_bounds()
        t_eval = np.linspace(0, self.problem.T, nt)
        x_eval = np.linspace(bounds[0][0], bounds[1][0], nx)

        T_grid, X_grid = np.meshgrid(t_eval, x_eval)
        t_flat = torch.from_numpy(T_grid.flatten().reshape(-1, 1)).to(self.device, dtype=self.dtype)
        x_flat = torch.from_numpy(X_grid.flatten().reshape(-1, 1)).to(self.device, dtype=self.dtype)

        # Generate solution
        with torch.no_grad():
            m_pred = self.forward(t_flat, x_flat)
            m_solution = m_pred.cpu().numpy().reshape(nx, nt)

        return {
            "m": m_solution,
            "t_grid": t_eval,
            "x_grid": x_eval,
            "T_grid": T_grid,
            "X_grid": X_grid,
        }

    def _compute_mass_metrics(self, solution: dict[str, np.ndarray]) -> dict[str, float]:
        """Compute mass conservation metrics."""
        m = solution["m"]
        x_grid = solution["x_grid"]
        dx = x_grid[1] - x_grid[0]

        # Compute total mass at each time
        total_mass = np.trapezoid(m, axis=0, dx=dx)

        metrics = {
            "initial_mass": total_mass[0],
            "final_mass": total_mass[-1],
            "mass_change": abs(total_mass[-1] - total_mass[0]),
            "max_mass_error": np.max(np.abs(total_mass - self.target_total_mass)),
            "mean_mass_error": np.mean(np.abs(total_mass - self.target_total_mass)),
        }

        return metrics

    def evaluate_solution_quality(self) -> dict[str, float]:
        """
        Evaluate quality of the learned density solution.

        Returns:
            Dictionary of solution quality metrics
        """
        metrics = {}

        # Sample test points
        n_test = 1000
        bounds = self.problem.geometry.get_bounds()
        t_test = torch.rand(n_test, 1, device=self.device, dtype=self.dtype) * self.problem.T
        x_test = (
            torch.rand(n_test, 1, device=self.device, dtype=self.dtype) * (bounds[1][0] - bounds[0][0]) + bounds[0][0]
        )

        with torch.no_grad():
            # PDE residual
            pde_residuals = self.compute_pde_residual(t_test, x_test)
            metrics["pde_residual_mean"] = torch.mean(torch.abs(pde_residuals["fp"])).item()
            metrics["pde_residual_max"] = torch.max(torch.abs(pde_residuals["fp"])).item()

            # Initial condition error
            bounds = self.problem.geometry.get_bounds()
            x_initial = torch.linspace(bounds[0][0], bounds[1][0], 100, device=self.device, dtype=self.dtype).reshape(
                -1, 1
            )
            initial_loss = self.compute_initial_loss(x_initial)
            metrics["initial_condition_error"] = initial_loss.item()

            # Mass conservation error
            mass_loss = self.compute_mass_conservation_loss(t_test, x_test)
            metrics["mass_conservation_error"] = mass_loss.item()

            # Solution statistics
            m_values = self.forward(t_test, x_test)
            metrics["density_mean"] = torch.mean(m_values).item()
            metrics["density_std"] = torch.std(m_values).item()
            metrics["density_min"] = torch.min(m_values).item()
            metrics["density_max"] = torch.max(m_values).item()

        return metrics

    def plot_solution(self, save_path: str | None = None) -> None:
        """
        Plot the FP solution.

        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib not available. Cannot plot solution.")
            return

        # Generate solution
        solution = self._generate_fp_solution()
        mass_metrics = self._compute_mass_metrics(solution)

        # Create plot
        _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Surface plot
        im1 = ax1.contourf(solution["T_grid"], solution["X_grid"], solution["m"], levels=20, cmap="plasma")
        ax1.set_xlabel("Time t")
        ax1.set_ylabel("Space x")
        ax1.set_title("FP Solution m(t,x)")
        plt.colorbar(im1, ax=ax1)

        # Time evolution at center
        center_idx = len(solution["x_grid"]) // 2
        ax2.plot(solution["t_grid"], solution["m"][center_idx, :])
        ax2.set_xlabel("Time t")
        ax2.set_ylabel("m(t, x_center)")
        ax2.set_title(f"Time Evolution at x = {solution['x_grid'][center_idx]:.2f}")
        ax2.grid(True)

        # Mass conservation
        dx = solution["x_grid"][1] - solution["x_grid"][0]
        total_mass = np.trapezoid(solution["m"], axis=0, dx=dx)
        ax3.plot(solution["t_grid"], total_mass)
        ax3.axhline(y=self.target_total_mass, color="r", linestyle="--", label="Target")
        ax3.set_xlabel("Time t")
        ax3.set_ylabel("Total Mass")
        ax3.set_title("Mass Conservation")
        ax3.legend()
        ax3.grid(True)

        # Density profile at final time
        ax4.plot(solution["x_grid"], solution["m"][:, -1])
        ax4.set_xlabel("Space x")
        ax4.set_ylabel("m(T, x)")
        ax4.set_title(f"Final Density Profile (t = {self.problem.T})")
        ax4.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

        # Print mass conservation metrics
        print("Mass Conservation Metrics:")
        for key, value in mass_metrics.items():
            print(f"  {key}: {value:.6f}")

    def get_results(self) -> dict:
        """
        Get solver results in standard MFGarchon format.

        Returns:
            Dictionary containing solution data, convergence info, and metadata
        """
        # State validation: Check if solve() was called
        # training_history is initialized in __init__, populated during solve()
        if not self.training_history["total_loss"]:
            raise RuntimeError("No training results available. Run solve() first.")

        # Generate solution for return
        solution = self._generate_fp_solution()
        mass_metrics = self._compute_mass_metrics(solution)

        results = {
            # Solution data
            "m": solution["m"],
            "x_grid": solution["x_grid"],
            "t_grid": solution["t_grid"],
            # Training information
            "training_history": self.training_history,
            "converged": len(self.training_history.get("total_loss", [])) > 0,
            "final_loss": self.training_history.get("total_loss", [float("inf")])[-1]
            if self.training_history.get("total_loss")
            else float("inf"),
            "epochs_trained": len(self.training_history.get("total_loss", [])),
            # Mass conservation metrics
            "mass_metrics": mass_metrics,
            # Solver metadata
            "solver_type": "FP_PINN",
            "device": str(self.device),
            "config": self.config,
        }

        return results

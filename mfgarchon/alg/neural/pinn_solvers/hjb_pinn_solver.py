"""
Physics-Informed Neural Network solver for Hamilton-Jacobi-Bellman equations.

This module implements a PINN solver specifically designed for the HJB equation
in Mean Field Games:

∂u/∂t + H(∇u, x, m(t,x)) = 0

where H is the Hamiltonian, u(t,x) is the value function, and m(t,x) is the
population density (which can be given or solved coupled with Fokker-Planck).

Key Features:
- Automatic differentiation for computing ∇u and ∂u/∂t
- Flexible Hamiltonian specification
- Support for various boundary and terminal conditions
- Integration with MFG coupling terms
- Advanced training strategies for nonlinear PDEs
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


class HJBPINNSolver(PINNBase):
    """
    Physics-Informed Neural Network solver for Hamilton-Jacobi-Bellman equations.

    This solver focuses on the HJB equation in MFG systems:
    ∂u/∂t + H(∇u, x, m) = 0

    The neural network approximates u(t,x) and uses automatic differentiation
    to compute the required derivatives for the PDE residual.
    """

    def __init__(
        self,
        problem: MFGProblem,
        config: PINNConfig | None = None,
        hamiltonian_func: Callable | None = None,
        density_func: Callable | None = None,
        networks: dict[str, nn.Module] | None = None,
    ):
        """
        Initialize HJB PINN solver.

        Args:
            problem: MFG problem containing HJB equation specification
            config: PINN configuration
            hamiltonian_func: Custom Hamiltonian function H(p, x, m)
            density_func: Given density function m(t,x) or None for coupling
            networks: Pre-defined neural networks (optional)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PINN functionality")

        # Store HJB-specific parameters
        self.hamiltonian_func = hamiltonian_func
        self.density_func = density_func

        # Extract problem parameters
        self.sigma = getattr(problem, "sigma", 0.1)  # Diffusion coefficient
        self.terminal_condition = getattr(problem, "terminal_condition", None)
        self.terminal_time = problem.T

        # Initialize base PINN
        super().__init__(problem, config, networks)

        bounds = problem.geometry.get_bounds()
        print("Initialized HJB PINN solver")
        print(f"  Problem domain: t ∈ [0, {self.terminal_time}], x ∈ [{bounds[0][0]}, {bounds[1][0]}]")
        print(f"  Diffusion coefficient: σ = {self.sigma}")

    def _initialize_networks(self) -> None:
        """Initialize neural network for value function u(t,x)."""
        # Issue #1290: prior code used architecture_type/separate_networks which
        # are not parameters of create_mfg_networks (the function takes network_type
        # and returns a single nn.Module, not a dict).  Build the dict explicitly.
        u_net = create_mfg_networks(
            network_type="feedforward",
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation
            if self.config.activation in ("tanh", "relu", "sigmoid", "elu")
            else "tanh",
            input_dim=2,  # (t, x)
            output_dim=1,  # scalar value function
            problem_type="hjb",
        )
        self.networks = {"u_net": u_net}
        self.u_net = self.networks["u_net"]

    def default_hamiltonian(self, u_x: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Default Hamiltonian for MFG problems: H(p, x, m) = (1/2)|p|² + V(x, m)

        Args:
            u_x: Gradient of value function ∇u [N, spatial_dim]
            x: Spatial coordinates [N, spatial_dim]
            m: Population density [N, 1]

        Returns:
            Hamiltonian values [N, 1]
        """
        # Kinetic term: (1/2)|∇u|²
        kinetic = 0.5 * torch.sum(u_x**2, dim=-1, keepdim=True)

        # Interaction potential: V(x, m) = log(m + ε) (logarithmic interaction)
        # This is a common choice in MFG literature
        epsilon = 1e-8  # Regularization to avoid log(0)
        interaction = torch.log(m + epsilon)

        return kinetic + interaction

    def compute_hamiltonian(self, u_x: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Compute Hamiltonian H(∇u, x, m).

        Args:
            u_x: Gradient of value function
            x: Spatial coordinates
            m: Population density

        Returns:
            Hamiltonian values
        """
        if self.hamiltonian_func is not None:
            return self.hamiltonian_func(u_x, x, m)
        else:
            return self.default_hamiltonian(u_x, x, m)

    def get_density(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Get population density m(t,x).

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Population density [N, 1]
        """
        if self.density_func is not None:
            # Use provided density function
            return self.density_func(t, x)
        else:
            # Use initial density or constant density as approximation
            # This is a placeholder - in coupled MFG, this would come from FP solver
            initial_density = getattr(self.problem, "initial_density", None)
            if initial_density is not None:
                # Evaluate initial density at current spatial points
                m0_np = initial_density(x.detach().cpu().numpy())
                return torch.from_numpy(m0_np).to(self.device, dtype=self.dtype).reshape(-1, 1)
            else:
                # Default to uniform density
                bounds = self.problem.geometry.get_bounds()
                return torch.ones_like(x) / (bounds[1][0] - bounds[0][0])

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute u(t,x).

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Value function values [N, 1]
        """
        # Concatenate inputs
        inputs = torch.cat([t, x], dim=-1)

        # Forward pass through network
        u = self.u_net(inputs)

        return u

    def compute_derivatives(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute derivatives of u(t,x) using automatic differentiation.

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Tuple of (du/dt, du/dx, d^2u/dx^2) — u_xx required for viscous HJB term.

        Note:
            Issue #1281 fix (2026-06-11 survey): u_xx was missing; HJB residual
            lacked the -(sigma^2/2)*u_xx viscous term, making sigma>0 incorrect.
        """
        # Enable gradient computation
        t_input = t.clone().detach().requires_grad_(True)
        x_input = x.clone().detach().requires_grad_(True)

        # Forward pass
        u = self.forward(t_input, x_input)

        # Compute gradients
        u_t = torch.autograd.grad(
            outputs=u, inputs=t_input, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]

        u_x = torch.autograd.grad(
            outputs=u, inputs=x_input, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]

        # Second spatial derivative (for viscous term in HJB)
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=x_input, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True
        )[0]

        return u_t, u_x, u_xx

    def compute_pde_residual(self, t: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute backward-HJB equation residual:
        -du/dt + H(grad_u, x, m) - (sigma^2/2)*u_xx = 0

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Dictionary with HJB residual

        Note:
            Issue #1281 fix (2026-06-11 survey): added -(sigma^2/2)*u_xx viscous term.
            Issue #1571 fix: the time-derivative sign is -du/dt, not +du/dt. The HJB is
            BACKWARD (terminal condition u(T,x)=g at t=T, t fed un-reversed), so the
            canonical residual is -du/dt + H - (sigma^2/2)*u_xx = 0 (base_hjb residual
            convention). The earlier +du/dt mirrored the FORWARD FP sign (+m_t) onto the
            backward HJB and fit the wrong PDE (e.g. H=const c, D=0 gave u(0)=g+cT instead
            of the correct g-cT). Only the diffusion sign mirrors FP, not the time-derivative.
        """
        # Compute derivatives (u_t, u_x, u_xx)
        u_t, u_x, u_xx = self.compute_derivatives(t, x)

        # Get population density
        m = self.get_density(t, x)

        # Compute Hamiltonian
        H = self.compute_hamiltonian(u_x, x, m)

        # Viscous term: (sigma^2/2) * u_xx  (matches FP diffusion convention)
        # Issue #1193: route sigma->D through the single-source converter
        # (D = sigma^2/2) instead of an inline literal, mirroring the sibling
        # PINN solvers (fp_pinn_solver.py, mfg_pinn_solver.py).
        viscous_term = diffusion_from_volatility_torch(self.sigma) * u_xx

        # Backward-HJB residual: -du/dt + H(grad_u, x, m) - (sigma^2/2)*u_xx = 0 (Issue #1571)
        hjb_residual = -u_t + H - viscous_term

        return {"hjb": hjb_residual}

    def compute_terminal_condition_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute terminal condition loss: u(T, x) = g(x)

        Args:
            x: Spatial coordinates at terminal time [N, 1]

        Returns:
            Terminal condition loss
        """
        # Create terminal time tensor
        t_terminal = torch.full_like(x, self.terminal_time, device=self.device)

        # Evaluate network at terminal time
        u_terminal = self.forward(t_terminal, x)

        if self.terminal_condition is not None:
            # Evaluate terminal condition
            if callable(self.terminal_condition):
                x_np = x.detach().cpu().numpy()
                g_target = self.terminal_condition(x_np)
                g_target = torch.from_numpy(g_target).to(self.device, dtype=self.dtype).reshape(-1, 1)
            else:
                # Assume terminal_condition is a scalar or tensor
                g_target = torch.full_like(u_terminal, self.terminal_condition)

            # Terminal condition loss: |u(T,x) - g(x)|²
            terminal_loss = torch.mean((u_terminal - g_target) ** 2)
        else:
            # If no terminal condition specified, assume u(T,x) = 0
            terminal_loss = torch.mean(u_terminal**2)

        return terminal_loss

    def compute_boundary_loss(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss.

        Args:
            t: Time coordinates on boundary [N, 1]
            x: Spatial coordinates on boundary [N, 1]

        Returns:
            Boundary condition loss
        """
        # Default: Homogeneous Neumann boundary conditions (∂u/∂x = 0)
        # This is common for MFG problems on bounded domains

        # Compute spatial derivative at boundary (u_xx not needed here)
        _, u_x, _u_xx = self.compute_derivatives(t, x)

        # Neumann BC: ∂u/∂n = 0 (normal derivative = 0)
        boundary_loss = torch.mean(u_x**2)

        return boundary_loss

    def compute_initial_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition loss (actually terminal condition for HJB).

        The HJB equation is typically solved backward in time from the terminal condition.

        Args:
            x: Spatial coordinates at initial time [N, 1]

        Returns:
            Initial condition loss
        """
        # For HJB, we typically have terminal conditions, not initial conditions
        # But if initial conditions are provided, handle them here
        initial_condition_u = getattr(self.problem, "initial_condition_u", None)
        if initial_condition_u is not None:
            t_initial = torch.zeros_like(x)
            u_initial = self.forward(t_initial, x)

            if callable(initial_condition_u):
                x_np = x.detach().cpu().numpy()
                u0_target = initial_condition_u(x_np)
                u0_target = torch.from_numpy(u0_target).to(self.device, dtype=self.dtype).reshape(-1, 1)
            else:
                u0_target = torch.full_like(u_initial, initial_condition_u)

            initial_loss = torch.mean((u_initial - u0_target) ** 2)
        else:
            # If no initial condition, use terminal condition loss instead
            initial_loss = self.compute_terminal_condition_loss(x)

        return initial_loss

    def solve(self, **kwargs: Any) -> dict:
        """
        Solve HJB equation using PINN approach.

        Args:
            **kwargs: Additional arguments (verbose, etc.)

        Returns:
            Dictionary with solution results
        """
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Starting HJB PINN training...")

        # Train the network
        training_history = self.train(verbose=verbose)

        # Generate solution on evaluation grid
        solution = self._generate_hjb_solution()

        results = {
            "u": solution["u"],
            "t_grid": solution["t_grid"],
            "x_grid": solution["x_grid"],
            "training_history": training_history,
            "network_state": self.u_net.state_dict(),
            "config": self.config,
        }

        if verbose:
            print(f"HJB solution completed. Final loss: {self.best_loss:.6f}")

        return results

    def _generate_hjb_solution(self) -> dict[str, np.ndarray]:
        """Generate HJB solution on evaluation grid."""
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
            u_pred = self.forward(t_flat, x_flat)
            u_solution = u_pred.cpu().numpy().reshape(nx, nt)

        return {
            "u": u_solution,
            "t_grid": t_eval,
            "x_grid": x_eval,
            "T_grid": T_grid,
            "X_grid": X_grid,
        }

    def compute_optimal_control(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal control α*(t,x) = -∇u(t,x).

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Optimal control [N, spatial_dim]
        """
        _, u_x, _u_xx = self.compute_derivatives(t, x)

        # For quadratic cost, optimal control is α* = -∇u
        optimal_control = -u_x

        return optimal_control

    def evaluate_solution_quality(self) -> dict[str, float]:
        """
        Evaluate quality of the learned solution.

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
            metrics["pde_residual_mean"] = torch.mean(torch.abs(pde_residuals["hjb"])).item()
            metrics["pde_residual_max"] = torch.max(torch.abs(pde_residuals["hjb"])).item()

            # Terminal condition error
            bounds = self.problem.geometry.get_bounds()
            x_terminal = torch.linspace(bounds[0][0], bounds[1][0], 100, device=self.device, dtype=self.dtype).reshape(
                -1, 1
            )
            terminal_loss = self.compute_terminal_condition_loss(x_terminal)
            metrics["terminal_condition_error"] = terminal_loss.item()

            # Boundary condition error
            bounds = self.problem.geometry.get_bounds()
            t_boundary = torch.rand(100, 1, device=self.device, dtype=self.dtype) * self.problem.T
            x_boundary = torch.cat(
                [
                    torch.full((50, 1), bounds[0][0], device=self.device, dtype=self.dtype),
                    torch.full((50, 1), bounds[1][0], device=self.device, dtype=self.dtype),
                ]
            )
            boundary_loss = self.compute_boundary_loss(t_boundary, x_boundary)
            metrics["boundary_condition_error"] = boundary_loss.item()

            # Solution statistics
            u_values = self.forward(t_test, x_test)
            metrics["solution_mean"] = torch.mean(u_values).item()
            metrics["solution_std"] = torch.std(u_values).item()
            metrics["solution_min"] = torch.min(u_values).item()
            metrics["solution_max"] = torch.max(u_values).item()

        return metrics

    def plot_solution(self, save_path: str | None = None) -> None:
        """
        Plot the HJB solution.

        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib not available. Cannot plot solution.")
            return

        # Generate solution
        solution = self._generate_hjb_solution()

        # Create plot
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Surface plot
        im1 = ax1.contourf(solution["T_grid"], solution["X_grid"], solution["u"], levels=20, cmap="viridis")
        ax1.set_xlabel("Time t")
        ax1.set_ylabel("Space x")
        ax1.set_title("HJB Solution u(t,x)")
        plt.colorbar(im1, ax=ax1)

        # Time evolution at center
        center_idx = len(solution["x_grid"]) // 2
        ax2.plot(solution["t_grid"], solution["u"][center_idx, :])
        ax2.set_xlabel("Time t")
        ax2.set_ylabel("u(t, x_center)")
        ax2.set_title(f"Time Evolution at x = {solution['x_grid'][center_idx]:.2f}")
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

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
        solution = self._generate_hjb_solution()

        results = {
            # Solution data
            "u": solution["u"],
            "x_grid": solution["x_grid"],
            "t_grid": solution["t_grid"],
            # Training information
            "training_history": self.training_history,
            "converged": self.converged,
            "final_loss": self.training_history.get("total_loss", [float("inf")])[-1]
            if self.training_history.get("total_loss")
            else float("inf"),
            "epochs_trained": len(self.training_history.get("total_loss", [])),
            # Solver metadata
            "solver_type": "HJB_PINN",
            "device": str(self.device),
            "config": self.config,
        }

        return results

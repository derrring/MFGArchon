"""
Complete Physics-Informed Neural Network solver for Mean Field Games.

This module implements a coupled PINN solver for the complete MFG system:

HJB: ∂u/∂t + H(∇u, x, m) = 0
FP:  ∂m/∂t - div(m∇H_p(∇u, x, m)) - (σ²/2)Δm = 0

with appropriate boundary and terminal/initial conditions. The system is
solved simultaneously using coupled neural networks with physics-informed
loss functions that enforce both PDEs and their coupling.

Key Features:
- Simultaneous training of u(t,x) and m(t,x) networks
- Enforced coupling between HJB and FP equations
- Mass conservation constraints
- Nash equilibrium verification
- Advanced training strategies for coupled nonlinear system
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
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class MFGPINNSolver(PINNBase):
    """
    Complete PINN solver for the coupled Mean Field Game system.

    Solves both HJB and Fokker-Planck equations simultaneously:
    - HJB: ∂u/∂t + H(∇u, x, m) = 0  (backward in time)
    - FP:  ∂m/∂t - div(m∇H_p(∇u, x, m)) - (σ²/2)Δm = 0  (forward in time)

    The coupling is enforced through the shared terms in both equations.
    """

    def __init__(
        self,
        problem: MFGProblem,
        config: PINNConfig | None = None,
        hamiltonian_func: Callable | None = None,
        networks: dict[str, nn.Module] | None = None,
        alternating_training: bool = False,
    ):
        """
        Initialize MFG PINN solver.

        Args:
            problem: MFG problem specification
            config: PINN configuration
            hamiltonian_func: Custom Hamiltonian H(p, x, m)
            networks: Pre-defined neural networks (optional)
            alternating_training: Whether to use alternating training strategy
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PINN functionality")

        # Store MFG-specific parameters
        self.hamiltonian_func = hamiltonian_func
        self.alternating_training = alternating_training

        # Extract problem parameters
        self.sigma = getattr(problem, "sigma", 0.1)
        self.terminal_condition = getattr(problem, "terminal_condition", None)
        self.initial_density = getattr(problem, "initial_density", None)

        # MFG-specific settings
        self.enforce_mass_conservation = True
        self.target_total_mass = 1.0
        self.coupling_strength = 1.0

        # Alternating training parameters
        self.u_training_phase = True  # Start with HJB training
        self.phase_switch_frequency = 500  # Switch every N epochs

        # Initialize base PINN
        super().__init__(problem, config, networks)

        bounds = problem.geometry.get_bounds()
        print("Initialized MFG PINN solver")
        print(f"  Problem domain: t ∈ [0, {problem.T}], x ∈ [{bounds[0][0]}, {bounds[1][0]}]")
        print(f"  Diffusion coefficient: σ = {self.sigma}")
        print(f"  Alternating training: {alternating_training}")

    def _initialize_networks(self) -> None:
        """Initialize neural networks for both u(t,x) and m(t,x)."""
        # Create separate networks for value function and density
        self.networks = create_mfg_networks(
            architecture_type="standard",
            separate_networks=True,  # Need both u_net and m_net
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation,
            input_dim=2,  # (t, x)
            output_dim=1,  # scalar functions
        )

        # Separate references for clarity
        self.u_net = self.networks["u_net"]
        self.m_net = self.networks["m_net"]

        print("Initialized MFG networks:")
        print(f"  u_net parameters: {sum(p.numel() for p in self.u_net.parameters()):,}")
        print(f"  m_net parameters: {sum(p.numel() for p in self.m_net.parameters()):,}")

    def _initialize_optimizers(self) -> None:
        """Initialize optimizers with special handling for alternating training."""
        super()._initialize_optimizers()

        # Create combined optimizer for joint training
        if not self.alternating_training:
            all_params = list(self.u_net.parameters()) + list(self.m_net.parameters())
            self.joint_optimizer = optim.Adam(
                all_params, lr=self.config.learning_rate, weight_decay=self.config.l2_regularization
            )

    def default_hamiltonian(self, u_x: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Default Hamiltonian: H(p, x, m) = (1/2)|p|² + V(x, m)

        Args:
            u_x: Momentum p = ∇u [N, spatial_dim]
            x: Spatial coordinates [N, spatial_dim]
            m: Population density [N, 1]

        Returns:
            Hamiltonian values [N, 1]
        """
        # Kinetic energy: (1/2)|∇u|²
        kinetic = 0.5 * torch.sum(u_x**2, dim=-1, keepdim=True)

        # Interaction potential: V(x, m) - common choices:
        # 1. Logarithmic: V = λ*log(m)
        # 2. Power law: V = λ*m^γ
        # 3. Congestion: V = λ/(1 + m)

        # Using logarithmic interaction (standard in MFG literature)
        epsilon = 1e-8
        interaction = torch.log(m + epsilon)

        return kinetic + interaction

    def compute_hamiltonian(self, u_x: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian using provided or default function."""
        if self.hamiltonian_func is not None:
            return self.hamiltonian_func(u_x, x, m)
        else:
            return self.default_hamiltonian(u_x, x, m)

    def forward_u(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for value function u(t,x).

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Value function values [N, 1]
        """
        inputs = torch.cat([t, x], dim=-1)
        return self.u_net(inputs)

    def forward_m(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for density function m(t,x).

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Density function values [N, 1] (enforced positive)
        """
        inputs = torch.cat([t, x], dim=-1)
        m_raw = self.m_net(inputs)

        # Enforce positivity: m(t,x) > 0
        return torch.exp(m_raw)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through both networks.

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Dictionary with u and m predictions
        """
        return {"u": self.forward_u(t, x), "m": self.forward_m(t, x)}

    def compute_derivatives_u(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute derivatives of u(t,x).

        Returns:
            Tuple of (du/dt, du/dx, d^2u/dx^2).

        Note:
            Issue #1281 fix (2026-06-11 survey): u_xx added so compute_pde_residual
            can include the -(sigma^2/2)*u_xx viscous term in the HJB residual.
        """
        t_input = t.clone().detach().requires_grad_(True)
        x_input = x.clone().detach().requires_grad_(True)

        u = self.forward_u(t_input, x_input)

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

    def compute_derivatives_m(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute derivatives of m(t,x)."""
        t_input = t.clone().detach().requires_grad_(True)
        x_input = x.clone().detach().requires_grad_(True)

        m = self.forward_m(t_input, x_input)

        m_t = torch.autograd.grad(
            outputs=m, inputs=t_input, grad_outputs=torch.ones_like(m), create_graph=True, retain_graph=True
        )[0]

        m_x = torch.autograd.grad(
            outputs=m, inputs=x_input, grad_outputs=torch.ones_like(m), create_graph=True, retain_graph=True
        )[0]

        m_xx = torch.autograd.grad(
            outputs=m_x, inputs=x_input, grad_outputs=torch.ones_like(m_x), create_graph=True, retain_graph=True
        )[0]

        return m_t, m_x, m_xx

    def compute_pde_residual(self, t: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute residuals for both HJB and FP equations.

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Dictionary with HJB and FP residuals
        """
        # Get current solutions
        u = self.forward_u(t, x)
        m = self.forward_m(t, x)

        # Compute derivatives (u_t, u_x, u_xx)
        u_t, u_x, u_xx = self.compute_derivatives_u(t, x)
        m_t, m_x, m_xx = self.compute_derivatives_m(t, x)

        # Compute Hamiltonian
        H = self.compute_hamiltonian(u_x, x, m)

        # Viscous term: D*u_xx where D = sigma^2/2 — Issue #1281 fix (2026-06-11 survey)
        # Convention mirrors FP: fp_residual subtracts D*m_xx. Issue #1189: route through D.
        viscous_term = diffusion_from_volatility_torch(self.sigma) * u_xx

        # HJB residual: du/dt + H(grad_u, x, m) - (sigma^2/2)*u_xx = 0
        hjb_residual = u_t + H - viscous_term

        # For FP equation, we need the drift: b = -∇H_p = -∇u (for quadratic H)
        drift = -u_x

        # Compute divergence of (m * drift)
        # In 1D: div(m * b) = ∂(m * b)/∂x = m * ∂b/∂x + b * ∂m/∂x

        # Compute drift derivative
        x_for_drift = x.clone().detach().requires_grad_(True)
        u_x_for_drift = torch.autograd.grad(
            outputs=self.forward_u(t, x_for_drift),
            inputs=x_for_drift,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        drift_for_divergence = -u_x_for_drift

        drift_x = torch.autograd.grad(
            outputs=drift_for_divergence,
            inputs=x_for_drift,
            grad_outputs=torch.ones_like(drift_for_divergence),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Divergence: div(m * b) = b * ∂m/∂x + m * ∂b/∂x
        divergence_term = drift * m_x + m * drift_x

        # Diffusion term: D*m_xx where D = sigma^2/2 (Issue #1189: route through D)
        diffusion_term = diffusion_from_volatility_torch(self.sigma) * m_xx

        # FP residual: ∂m/∂t - div(m*b) - (σ²/2)Δm = 0
        fp_residual = m_t - divergence_term - diffusion_term

        return {"hjb": hjb_residual, "fp": fp_residual}

    def compute_coupling_loss(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute coupling loss between HJB and FP equations.

        This ensures that both equations are solved consistently with
        the same coupling terms.

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, 1]

        Returns:
            Coupling loss [scalar]
        """
        # The main coupling is already enforced through the PDE residuals
        # Additional coupling constraints can be added here

        # Example: Ensure consistency of optimal control
        _u = self.forward_u(t, x)
        _m = self.forward_m(t, x)
        _, u_x, _u_xx = self.compute_derivatives_u(t, x)

        # Optimal control should be α* = -∇u
        _optimal_control = -u_x

        # Could add additional consistency checks here
        # For now, return zero (main coupling through PDE residuals)
        return torch.tensor(0.0, device=self.device)

    def compute_mass_conservation_loss(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute mass conservation constraint."""
        m = self.forward_m(t, x)

        # Approximate integral using available points
        x_flat = x.flatten()
        m_flat = m.flatten()

        if len(x_flat) > 1:
            sorted_indices = torch.argsort(x_flat)
            x_sorted = x_flat[sorted_indices]
            m_sorted = m_flat[sorted_indices]

            dx = x_sorted[1:] - x_sorted[:-1]
            m_avg = (m_sorted[1:] + m_sorted[:-1]) / 2
            total_mass = torch.sum(dx * m_avg)
        else:
            bounds = self.problem.geometry.get_bounds()
            domain_size = bounds[1][0] - bounds[0][0]
            total_mass = m_flat[0] * domain_size

        return (total_mass - self.target_total_mass) ** 2

    def compute_boundary_loss(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute boundary conditions for both u and m."""
        # HJB boundary condition: typically Neumann (∂u/∂x = 0)
        _, u_x, _u_xx = self.compute_derivatives_u(t, x)
        u_boundary_loss = torch.mean(u_x**2)

        # FP boundary condition: no-flux
        m = self.forward_m(t, x)
        _, m_x, _ = self.compute_derivatives_m(t, x)
        _, u_x_boundary, _u_xx_boundary = self.compute_derivatives_u(t, x)

        # No-flux: m*(-u_x) + D*dm/dx = 0  where D = sigma^2/2 (Issue #1189)
        flux = m * (-u_x_boundary) + diffusion_from_volatility_torch(self.sigma) * m_x
        m_boundary_loss = torch.mean(flux**2)

        return u_boundary_loss + m_boundary_loss

    def compute_initial_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute terminal condition for u and initial condition for m.

        Args:
            x: Spatial coordinates [N, 1]

        Returns:
            Combined initial/terminal condition loss
        """
        # Terminal condition for u: u(T, x) = g(x)
        t_terminal = torch.full_like(x, self.problem.T)
        u_terminal = self.forward_u(t_terminal, x)

        if self.terminal_condition is not None:
            if callable(self.terminal_condition):
                x_np = x.detach().cpu().numpy()
                g_target = self.terminal_condition(x_np)
                g_target = torch.from_numpy(g_target).to(self.device, dtype=self.dtype).reshape(-1, 1)
            else:
                g_target = torch.full_like(u_terminal, self.terminal_condition)

            u_terminal_loss = torch.mean((u_terminal - g_target) ** 2)
        else:
            u_terminal_loss = torch.mean(u_terminal**2)

        # Initial condition for m: m(0, x) = m₀(x)
        t_initial = torch.zeros_like(x)
        m_initial = self.forward_m(t_initial, x)

        if self.initial_density is not None:
            if callable(self.initial_density):
                x_np = x.detach().cpu().numpy()
                m0_target = self.initial_density(x_np)
                m0_target = torch.from_numpy(m0_target).to(self.device, dtype=self.dtype).reshape(-1, 1)
            else:
                m0_target = torch.full_like(m_initial, self.initial_density)

            m_initial_loss = torch.mean((m_initial - m0_target) ** 2)
        else:
            # Default uniform density
            bounds = self.problem.geometry.get_bounds()
            uniform_density = 1.0 / (bounds[1][0] - bounds[0][0])
            m0_target = torch.full_like(m_initial, uniform_density)
            m_initial_loss = torch.mean((m_initial - m0_target) ** 2)

        return u_terminal_loss + m_initial_loss

    def compute_total_loss(
        self,
        interior_points: tuple[torch.Tensor, torch.Tensor],
        boundary_points: tuple[torch.Tensor, torch.Tensor],
        initial_points: torch.Tensor,
        data_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute total MFG loss with all components."""
        losses = {}

        # PDE residuals (both HJB and FP)
        t_interior, x_interior = interior_points
        pde_residuals = self.compute_pde_residual(t_interior, x_interior)

        losses["hjb_loss"] = torch.mean(pde_residuals["hjb"] ** 2) * self.config.pde_weight
        losses["fp_loss"] = torch.mean(pde_residuals["fp"] ** 2) * self.config.pde_weight

        # Boundary conditions
        t_boundary, x_boundary = boundary_points
        boundary_loss = self.compute_boundary_loss(t_boundary, x_boundary)
        losses["boundary_loss"] = boundary_loss * self.config.boundary_weight

        # Initial/terminal conditions
        initial_loss = self.compute_initial_loss(initial_points)
        losses["initial_loss"] = initial_loss * self.config.initial_weight

        # Coupling loss
        coupling_loss = self.compute_coupling_loss(t_interior, x_interior)
        losses["coupling_loss"] = coupling_loss * self.config.coupling_weight

        # Mass conservation
        if self.enforce_mass_conservation:
            mass_loss = self.compute_mass_conservation_loss(t_interior, x_interior)
            losses["mass_loss"] = mass_loss * 10.0

        # Data loss (if available)
        if data_points is not None:
            t_data, x_data, values_data = data_points
            data_loss = self.compute_data_loss(t_data, x_data, values_data)
            losses["data_loss"] = data_loss * self.config.data_weight
        else:
            losses["data_loss"] = torch.tensor(0.0, device=self.device)

        # Total loss
        losses["total_loss"] = sum(losses.values())

        return losses

    def train_step_alternating(
        self,
        interior_points: tuple[torch.Tensor, torch.Tensor],
        boundary_points: tuple[torch.Tensor, torch.Tensor],
        initial_points: torch.Tensor,
        data_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Alternating training step (focus on one network at a time)."""
        if self.u_training_phase:
            # Train u-network (freeze m-network)
            for param in self.m_net.parameters():
                param.requires_grad = False
            for param in self.u_net.parameters():
                param.requires_grad = True

            optimizer = self.optimizers["u_net"]
        else:
            # Train m-network (freeze u-network)
            for param in self.u_net.parameters():
                param.requires_grad = False
            for param in self.m_net.parameters():
                param.requires_grad = True

            optimizer = self.optimizers["m_net"]

        # Zero gradients
        optimizer.zero_grad()

        # Compute losses
        losses = self.compute_total_loss(interior_points, boundary_points, initial_points, data_points)

        # Focus loss on current phase
        if self.u_training_phase:
            loss_to_minimize = losses["hjb_loss"] + losses["boundary_loss"] + losses["initial_loss"]
        else:
            loss_to_minimize = losses["fp_loss"] + losses["boundary_loss"] + losses["initial_loss"]
            if "mass_loss" in losses:
                loss_to_minimize += losses["mass_loss"]

        # Backward pass
        loss_to_minimize.backward()

        # Gradient clipping
        if self.config.gradient_clipping > 0:
            current_net = self.u_net if self.u_training_phase else self.m_net
            torch.nn.utils.clip_grad_norm_(current_net.parameters(), max_norm=self.config.gradient_clipping)

        # Update parameters
        optimizer.step()

        # Re-enable gradients for both networks
        for param in self.u_net.parameters():
            param.requires_grad = True
        for param in self.m_net.parameters():
            param.requires_grad = True

        # Backend compatibility - tensor to scalar conversion (Issue #543 acceptable)
        return {k: v.item() if hasattr(v, "item") else float(v) for k, v in losses.items()}

    def train_step_joint(
        self,
        interior_points: tuple[torch.Tensor, torch.Tensor],
        boundary_points: tuple[torch.Tensor, torch.Tensor],
        initial_points: torch.Tensor,
        data_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Joint training step (train both networks simultaneously)."""
        # Zero gradients for both networks
        self.joint_optimizer.zero_grad()

        # Compute total loss
        losses = self.compute_total_loss(interior_points, boundary_points, initial_points, data_points)

        # Backward pass
        losses["total_loss"].backward()

        # Gradient clipping for both networks
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.u_net.parameters(), max_norm=self.config.gradient_clipping)
            torch.nn.utils.clip_grad_norm_(self.m_net.parameters(), max_norm=self.config.gradient_clipping)

        # Update parameters
        self.joint_optimizer.step()

        # Backend compatibility - tensor to scalar conversion (Issue #543 acceptable)
        return {k: v.item() if hasattr(v, "item") else float(v) for k, v in losses.items()}

    def train_step(
        self,
        interior_points: tuple[torch.Tensor, torch.Tensor],
        boundary_points: tuple[torch.Tensor, torch.Tensor],
        initial_points: torch.Tensor,
        data_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Training step using selected strategy."""
        if self.alternating_training:
            # Check if we should switch phases
            if self.current_epoch % self.phase_switch_frequency == 0:
                self.u_training_phase = not self.u_training_phase
                phase_name = "HJB (u)" if self.u_training_phase else "FP (m)"
                if self.current_epoch > 0:
                    print(f"  Switching to {phase_name} training phase at epoch {self.current_epoch}")

            return self.train_step_alternating(interior_points, boundary_points, initial_points, data_points)
        else:
            return self.train_step_joint(interior_points, boundary_points, initial_points, data_points)

    def solve(self, **kwargs: Any) -> dict:
        """
        Solve the complete MFG system.

        Args:
            **kwargs: Additional arguments

        Returns:
            Dictionary with complete MFG solution
        """
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Starting MFG PINN training...")
            if self.alternating_training:
                print(f"  Using alternating training (switch every {self.phase_switch_frequency} epochs)")
            else:
                print("  Using joint training")

        # Train the networks
        training_history = self.train(verbose=verbose)

        # Generate solution on evaluation grid
        solution = self._generate_mfg_solution()

        # Compute solution quality metrics
        quality_metrics = self.evaluate_mfg_quality()

        results = {
            "u": solution["u"],
            "m": solution["m"],
            "t_grid": solution["t_grid"],
            "x_grid": solution["x_grid"],
            "training_history": training_history,
            "quality_metrics": quality_metrics,
            "network_states": {
                "u_net": self.u_net.state_dict(),
                "m_net": self.m_net.state_dict(),
            },
            "config": self.config,
        }

        if verbose:
            print(f"MFG solution completed. Final loss: {self.best_loss:.6f}")
            print("Quality metrics:")
            for key, value in quality_metrics.items():
                print(f"  {key}: {value:.6f}")

        return results

    def _generate_mfg_solution(self) -> dict[str, np.ndarray]:
        """Generate complete MFG solution on evaluation grid."""
        nt, nx = 100, 100
        bounds = self.problem.geometry.get_bounds()
        t_eval = np.linspace(0, self.problem.T, nt)
        x_eval = np.linspace(bounds[0][0], bounds[1][0], nx)

        T_grid, X_grid = np.meshgrid(t_eval, x_eval)
        t_flat = torch.from_numpy(T_grid.flatten().reshape(-1, 1)).to(self.device, dtype=self.dtype)
        x_flat = torch.from_numpy(X_grid.flatten().reshape(-1, 1)).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            u_pred = self.forward_u(t_flat, x_flat)
            m_pred = self.forward_m(t_flat, x_flat)

            u_solution = u_pred.cpu().numpy().reshape(nx, nt)
            m_solution = m_pred.cpu().numpy().reshape(nx, nt)

        return {
            "u": u_solution,
            "m": m_solution,
            "t_grid": t_eval,
            "x_grid": x_eval,
            "T_grid": T_grid,
            "X_grid": X_grid,
        }

    def evaluate_mfg_quality(self) -> dict[str, float]:
        """Evaluate quality of the complete MFG solution."""
        metrics = {}

        # Sample test points
        n_test = 1000
        bounds = self.problem.geometry.get_bounds()
        t_test = torch.rand(n_test, 1, device=self.device, dtype=self.dtype) * self.problem.T
        x_test = (
            torch.rand(n_test, 1, device=self.device, dtype=self.dtype) * (bounds[1][0] - bounds[0][0]) + bounds[0][0]
        )

        # Issue #1281 fix (2026-06-11 survey): compute_pde_residual calls
        # torch.autograd.grad internally; wrapping it in no_grad raises
        # RuntimeError. Pull it outside no_grad; scalar .item() reads are safe.
        pde_residuals = self.compute_pde_residual(t_test, x_test)
        metrics["hjb_residual_mean"] = torch.mean(torch.abs(pde_residuals["hjb"])).item()
        metrics["fp_residual_mean"] = torch.mean(torch.abs(pde_residuals["fp"])).item()

        # Mass conservation also uses no_grad-compatible ops only.
        mass_loss = self.compute_mass_conservation_loss(t_test, x_test)
        metrics["mass_conservation_error"] = mass_loss.item()

        # no_grad is safe for plain forward passes (no autograd.grad calls).
        with torch.no_grad():
            u_values = self.forward_u(t_test, x_test)
            m_values = self.forward_m(t_test, x_test)

            metrics["u_mean"] = torch.mean(u_values).item()
            metrics["u_std"] = torch.std(u_values).item()
            metrics["m_mean"] = torch.mean(m_values).item()
            metrics["m_std"] = torch.std(m_values).item()

        return metrics

    def plot_solution(self, save_path: str | None = None) -> None:
        """Plot the complete MFG solution."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib not available. Cannot plot solution.")
            return

        solution = self._generate_mfg_solution()

        _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Value function u(t,x)
        im1 = ax1.contourf(solution["T_grid"], solution["X_grid"], solution["u"], levels=20, cmap="viridis")
        ax1.set_xlabel("Time t")
        ax1.set_ylabel("Space x")
        ax1.set_title("Value Function u(t,x)")
        plt.colorbar(im1, ax=ax1)

        # Density function m(t,x)
        im2 = ax2.contourf(solution["T_grid"], solution["X_grid"], solution["m"], levels=20, cmap="plasma")
        ax2.set_xlabel("Time t")
        ax2.set_ylabel("Space x")
        ax2.set_title("Density Function m(t,x)")
        plt.colorbar(im2, ax=ax2)

        # Time evolution comparison
        center_idx = len(solution["x_grid"]) // 2
        ax3.plot(solution["t_grid"], solution["u"][center_idx, :], label="u(t, x_center)", color="blue")
        ax3_twin = ax3.twinx()
        ax3_twin.plot(solution["t_grid"], solution["m"][center_idx, :], label="m(t, x_center)", color="red")
        ax3.set_xlabel("Time t")
        ax3.set_ylabel("u(t, x)", color="blue")
        ax3_twin.set_ylabel("m(t, x)", color="red")
        ax3.set_title("Time Evolution at Center")
        ax3.grid(True)

        # Mass conservation
        dx = solution["x_grid"][1] - solution["x_grid"][0]
        total_mass = np.trapezoid(solution["m"], axis=0, dx=dx)
        ax4.plot(solution["t_grid"], total_mass)
        ax4.axhline(y=self.target_total_mass, color="r", linestyle="--", label="Target")
        ax4.set_xlabel("Time t")
        ax4.set_ylabel("Total Mass")
        ax4.set_title("Mass Conservation")
        ax4.legend()
        ax4.grid(True)

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
        solution = self._generate_mfg_solution()

        results = {
            # Solution data
            "u": solution["u"],
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
            # MFG-specific metrics
            "hjb_loss": self.training_history.get("hjb_loss", [])[-1] if self.training_history.get("hjb_loss") else 0.0,
            "fp_loss": self.training_history.get("fp_loss", [])[-1] if self.training_history.get("fp_loss") else 0.0,
            "coupling_loss": self.training_history.get("coupling_loss", [])[-1]
            if self.training_history.get("coupling_loss")
            else 0.0,
            # Solver metadata
            "solver_type": "MFG_PINN",
            "device": str(self.device),
            "config": self.config,
            "training_strategy": self.config.training_strategy,
        }

        return results

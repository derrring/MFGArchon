"""
Base Physics-Informed Neural Network (PINN) implementation for MFGarchon.

This module provides the abstract base class for all PINN implementations,
establishing the core interface and common functionality for physics-informed
neural networks in Mean Field Games.

The PINN approach solves PDEs by:
1. Parameterizing solutions u(t,x), m(t,x) as neural networks
2. Computing PDE residuals using automatic differentiation
3. Minimizing physics-informed loss: L = L_PDE + L_BC + L_IC + L_data
4. Supporting complex domains and high-dimensional problems

Key Features:
- Abstract base class with standardized PINN interface
- Automatic differentiation utilities for PDE residual computation
- Flexible network architecture support
- Advanced training strategies with adaptive sampling
- Device management (CPU/GPU) with automatic detection
- Comprehensive loss function framework
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.alg.base_solver import BaseNeuralSolver
from mfgarchon.utils.deprecation import deprecated_parameter

if TYPE_CHECKING:
    from mfgarchon.core.mfg_problem import MFGProblem

# PyTorch imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True

    # Import adaptive training components conditionally
    try:
        from .adaptive_training import AdaptiveTrainingConfig, AdaptiveTrainingStrategy

        ADAPTIVE_AVAILABLE = True
    except ImportError:
        ADAPTIVE_AVAILABLE = False
except ImportError:
    TORCH_AVAILABLE = False

    # Mock classes for type hints when torch isn't available
    class MockNN:
        class Module:
            pass

    nn = MockNN

    torch = None
    optim = None
    DataLoader = None
    TensorDataset = None


class NormalizationType(str, Enum):
    """
    Normalization layer type for PINN neural networks.

    Determines which normalization technique is applied to network layers
    to stabilize training and improve convergence.

    Attributes:
        NONE: No normalization (standard feed-forward network)
        BATCH: Batch normalization after each hidden layer
        LAYER: Layer normalization after each hidden layer
    """

    NONE = "none"
    BATCH = "batch"
    LAYER = "layer"


@dataclass
class PINNConfig:
    """
    Configuration for Physics-Informed Neural Networks.

    This configuration class provides comprehensive settings for PINN training,
    including network architecture, optimization parameters, sampling strategies,
    and advanced training techniques.
    """

    # Network architecture
    hidden_layers: list[int] = field(default_factory=lambda: [50, 50, 50])
    activation: str = "tanh"  # "tanh", "relu", "swish", "sine", "gelu"
    initialization: str = "xavier_normal"  # "xavier_normal", "xavier_uniform", "kaiming"

    # Normalization strategy (replaces use_batch_norm, use_layer_norm)
    normalization: NormalizationType = NormalizationType.NONE

    # Deprecated parameters (kept for backward compatibility)
    use_batch_norm: bool | None = None  # Deprecated: use normalization
    use_layer_norm: bool | None = None  # Deprecated: use normalization

    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 1000
    max_epochs: int = 10000
    convergence_tolerance: float = 1e-6

    # Loss function weights (adaptive weights can be implemented)
    pde_weight: float = 1.0
    boundary_weight: float = 10.0
    initial_weight: float = 10.0
    data_weight: float = 1.0
    coupling_weight: float = 1.0  # For MFG coupling terms

    # Sampling parameters
    n_interior_points: int = 5000
    n_boundary_points: int = 500
    n_initial_points: int = 500
    adaptive_sampling: bool = True
    adaptive_refinement_threshold: float = 0.1

    # Regularization
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0
    dropout_rate: float = 0.0
    gradient_clipping: float = 1.0

    # Device and precision
    device: str = "auto"  # "auto", "cpu", "cuda", "mps", "cuda:0", etc.
    dtype: str = "float32"  # "float32", "float64"

    # Advanced training strategies
    use_lr_scheduler: bool = True
    scheduler_type: str = "plateau"  # "plateau", "cosine", "exponential"
    scheduler_patience: int = 100
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 500

    # Curriculum learning
    use_curriculum: bool = False
    curriculum_start_weight: float = 0.1
    curriculum_end_epoch: int = 1000

    # Advanced adaptive training
    enable_advanced_adaptive: bool = False  # Enable sophisticated adaptive strategies
    adaptive_config: dict = field(default_factory=dict)  # Configuration for AdaptiveTrainingStrategy

    # Basic adaptive sampling (legacy)
    resample_frequency: int = 100
    residual_threshold: float = 0.01

    # Checkpointing and monitoring
    save_checkpoints: bool = True
    checkpoint_interval: int = 1000
    log_interval: int = 100

    def __post_init__(self):
        """Validate configuration parameters."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PINN functionality. Install with: pip install torch torchvision")

        # Handle deprecated normalization parameters (warnings issued by deprecated_parameter decorator on __init__)
        if self.use_batch_norm is not None or self.use_layer_norm is not None:
            # Map deprecated booleans to normalization enum
            if self.normalization == NormalizationType.NONE:  # Only if not explicitly set
                if self.use_batch_norm:
                    self.normalization = NormalizationType.BATCH
                elif self.use_layer_norm:
                    self.normalization = NormalizationType.LAYER
                # If both False or both True, keep NONE (invalid config will be caught by validation)

        # Validation checks
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.convergence_tolerance <= 0:
            raise ValueError("Convergence tolerance must be positive")

        if len(self.hidden_layers) == 0:
            raise ValueError("At least one hidden layer is required")

        if self.activation not in ["tanh", "relu", "swish", "sine", "gelu", "elu"]:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Warning for potential issues
        if self.batch_size > self.n_interior_points:
            warnings.warn("Batch size larger than number of interior points")


# Apply deprecated_parameter decorators to auto-generated __init__
PINNConfig.__init__ = deprecated_parameter(  # type: ignore[misc]
    param_name="use_batch_norm", since="v0.17.0", replacement="normalization"
)(deprecated_parameter(param_name="use_layer_norm", since="v0.17.0", replacement="normalization")(PINNConfig.__init__))


class PINNBase(BaseNeuralSolver, ABC):
    """
    Abstract base class for Physics-Informed Neural Networks.

    This class provides the common interface and functionality for all PINN
    implementations in the MFGarchon framework, inheriting from BaseMFGSolver
    to ensure compatibility with the existing solver ecosystem.
    """

    def __init__(
        self,
        problem: MFGProblem,
        config: PINNConfig | None = None,
        networks: dict[str, nn.Module] | None = None,
    ):
        """
        Initialize PINN solver.

        Args:
            problem: MFG problem to solve
            config: PINN configuration
            networks: Pre-defined neural networks (optional)

        Raises:
            ImportError: If PyTorch is not available
            ValueError: If configuration is invalid
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PINN functionality. Install with: pip install torch torchvision")

        # Issue #1290: resolve config before super().__init__ so that
        # BaseMFGSolver.__init__(problem, config) receives the config argument.
        # Prior code called super().__init__(problem), omitting config entirely,
        # which caused TypeError because BaseMFGSolver requires both positional args.
        _pinn_config = config or PINNConfig()

        # Initialize base solver — passes both required positional args
        super().__init__(problem, _pinn_config)

        self.config = _pinn_config

        # Set up computational device
        self.device = self._setup_device()
        self.dtype = getattr(torch, self.config.dtype)

        # Initialize neural networks
        self.networks = networks or {}
        if not self.networks:
            self._initialize_networks()

        # Move networks to device
        for network in self.networks.values():
            network.to(self.device)
            network.train()  # Set to training mode

        # Initialize optimizers and schedulers
        self._initialize_optimizers()

        # Training state tracking
        self.training_history = {
            "total_loss": [],
            "pde_loss": [],
            "boundary_loss": [],
            "initial_loss": [],
            "data_loss": [],
            "coupling_loss": [],
            "learning_rate": [],
        }

        self.best_loss = float("inf")
        self.epochs_without_improvement = 0
        self.current_epoch = 0

        # Adaptive sampling state
        self.current_points = None
        self.residual_history: list[float] = []

        # Initialize advanced adaptive training if enabled
        self.adaptive_strategy = None
        if self.config.enable_advanced_adaptive and ADAPTIVE_AVAILABLE:
            adaptive_config = AdaptiveTrainingConfig(**self.config.adaptive_config)
            self.adaptive_strategy = AdaptiveTrainingStrategy(adaptive_config)
            print(f"Initialized PINN solver on {self.device} with advanced adaptive training")
        else:
            print(f"Initialized PINN solver on {self.device}")

    def _setup_device(self) -> torch.device:
        """
        Set up computation device based on configuration and availability.

        Returns:
            Configured torch device

        Raises:
            ValueError: If specified device is not available
        """
        if self.config.device == "auto":
            # Automatic device selection with priority: CUDA > MPS > CPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name()
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"Auto-selected CUDA GPU: {gpu_name} ({memory_gb:.1f} GB)")
            # Backend compatibility - Apple Silicon MPS support (Issue #543 acceptable)
            # hasattr check required: torch.backends.mps added in PyTorch 1.12+
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Auto-selected Apple Silicon MPS (Metal Performance Shaders)")
            else:
                device = torch.device("cpu")
                print("Auto-selected CPU (no GPU acceleration available)")
        else:
            try:
                device = torch.device(self.config.device)

                # Validate device availability
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise ValueError("CUDA device requested but CUDA is not available")
                # Backend compatibility - Apple Silicon MPS support (Issue #543 acceptable)
                elif device.type == "mps" and not (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    raise ValueError("MPS device requested but MPS is not available")

                # Device-specific messaging
                if device.type == "cuda":
                    gpu_name = torch.cuda.get_device_name(device)
                    memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
                    print(f"Using CUDA GPU: {gpu_name} ({memory_gb:.1f} GB)")
                elif device.type == "mps":
                    print("Using Apple Silicon MPS acceleration")
                else:
                    print(f"Using device: {device}")

            except Exception as e:
                raise ValueError(f"Invalid device specification: {self.config.device}") from e

        return device

    @abstractmethod
    def _initialize_networks(self) -> None:
        """
        Initialize neural networks for the specific PINN implementation.

        This method must be implemented by subclasses to define the
        network architecture appropriate for their specific PDE system.
        """

    def _initialize_optimizers(self) -> None:
        """Initialize optimizers and learning rate schedulers for all networks."""
        self.optimizers = {}
        self.schedulers = {}

        for name, network in self.networks.items():
            # Create optimizer
            optimizer = optim.Adam(
                network.parameters(), lr=self.config.learning_rate, weight_decay=self.config.l2_regularization
            )
            self.optimizers[name] = optimizer

            # Create learning rate scheduler
            if self.config.use_lr_scheduler:
                if self.config.scheduler_type == "plateau":
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        patience=self.config.scheduler_patience,
                        factor=self.config.scheduler_factor,
                    )
                elif self.config.scheduler_type == "cosine":
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
                elif self.config.scheduler_type == "exponential":
                    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
                else:
                    raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

                self.schedulers[name] = scheduler

    # ------------------------------------------------------------------
    # BaseNeuralSolver / BaseMFGSolver abstract method implementations
    # Issue #1290: PINNBase left these unimplemented, making every
    # PINNBase subclass abstract and non-instantiable.
    # ------------------------------------------------------------------

    def build_networks(self) -> None:
        """
        Implement BaseNeuralSolver interface; delegates to _initialize_networks.

        PINNBase uses _initialize_networks() as its primary network-setup hook
        (called during __init__).  This method exists for BaseNeuralSolver
        compatibility; calling it re-initialises all networks.
        """
        self._initialize_networks()

    def compute_loss(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """
        Implement BaseNeuralSolver interface; delegates to compute_total_loss.

        Accepts the same positional and keyword arguments as compute_total_loss
        and converts the torch.Tensor values to Python floats so the return
        type matches the BaseNeuralSolver contract (dict[str, float]).

        For full tensor results (needed during training), call
        compute_total_loss() directly.
        """
        tensor_losses = self.compute_total_loss(*args, **kwargs)
        return {k: float(v) for k, v in tensor_losses.items()}

    def validate_solution(self) -> dict[str, float]:
        """
        Implement BaseMFGSolver interface; delegates to evaluate_convergence.

        Returns PDE residual and loss metrics computed on a fresh sample of
        test points drawn from the domain.
        """
        return self.evaluate_convergence()

    # ------------------------------------------------------------------

    @abstractmethod
    def compute_pde_residual(self, t: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute PDE residual at given points using automatic differentiation.

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, d] where d is spatial dimension

        Returns:
            Dictionary of PDE residuals for each equation in the system

        Note:
            This is the core of the PINN method - computing the residual
            of the PDE when the neural network solution is substituted.
        """

    @abstractmethod
    def compute_boundary_loss(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss.

        Args:
            t: Time coordinates on boundary [N, 1]
            x: Spatial coordinates on boundary [N, d]

        Returns:
            Boundary condition loss tensor [scalar]
        """

    @abstractmethod
    def compute_initial_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition loss.

        Args:
            x: Spatial coordinates at t=0 [N, d]

        Returns:
            Initial condition loss tensor [scalar]
        """

    def compute_coupling_loss(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute coupling loss for MFG systems (u-m coupling).

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, d]

        Returns:
            Coupling loss tensor [scalar]

        Note:
            Default implementation returns zero. Override for specific
            MFG coupling requirements.
        """
        return torch.tensor(0.0, device=self.device)

    def compute_total_loss(
        self,
        interior_points: tuple[torch.Tensor, torch.Tensor],
        boundary_points: tuple[torch.Tensor, torch.Tensor],
        initial_points: torch.Tensor,
        data_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute total weighted loss function.

        Args:
            interior_points: (t, x) coordinates in domain interior
            boundary_points: (t, x) coordinates on domain boundary
            initial_points: x coordinates at initial time t=0
            data_points: Optional (t, x, values) for supervised data

        Returns:
            Dictionary containing all loss components and total loss
        """
        losses = {}

        # PDE residual loss (physics-informed component)
        t_interior, x_interior = interior_points
        pde_residuals = self.compute_pde_residual(t_interior, x_interior)
        pde_loss = sum(torch.mean(residual**2) for residual in pde_residuals.values())
        losses["pde_loss"] = pde_loss * self.config.pde_weight

        # Boundary condition loss
        t_boundary, x_boundary = boundary_points
        boundary_loss = self.compute_boundary_loss(t_boundary, x_boundary)
        losses["boundary_loss"] = boundary_loss * self.config.boundary_weight

        # Initial condition loss
        initial_loss = self.compute_initial_loss(initial_points)
        losses["initial_loss"] = initial_loss * self.config.initial_weight

        # Coupling loss for MFG systems
        coupling_loss = self.compute_coupling_loss(t_interior, x_interior)
        losses["coupling_loss"] = coupling_loss * self.config.coupling_weight

        # Data loss (if supervised data available)
        if data_points is not None:
            t_data, x_data, values_data = data_points
            data_loss = self.compute_data_loss(t_data, x_data, values_data)
            losses["data_loss"] = data_loss * self.config.data_weight
        else:
            losses["data_loss"] = torch.tensor(0.0, device=self.device)

        # Total loss with dynamic weighting (can be overridden)
        losses["total_loss"] = self._compute_weighted_total_loss(losses)

        return losses

    def _compute_weighted_total_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted total loss (can implement adaptive weighting).

        Args:
            losses: Dictionary of individual loss components

        Returns:
            Weighted total loss
        """
        # Simple sum - can be enhanced with adaptive weighting
        total = (
            losses["pde_loss"]
            + losses["boundary_loss"]
            + losses["initial_loss"]
            + losses["coupling_loss"]
            + losses["data_loss"]
        )

        # Add regularization terms
        if self.config.l1_regularization > 0:
            l1_reg = sum(
                torch.sum(torch.abs(param)) for network in self.networks.values() for param in network.parameters()
            )
            total += self.config.l1_regularization * l1_reg

        return total

    def compute_data_loss(self, t: torch.Tensor, x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised data loss (if available).

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, d]
            values: Target values [N, output_dim]

        Returns:
            Mean squared error data loss [scalar]
        """
        predictions = self.forward(t, x)

        # Handle multiple outputs
        if isinstance(predictions, dict):
            # Assume values correspond to first network output
            prediction = next(iter(predictions.values()))
        else:
            prediction = predictions

        return torch.mean((prediction - values) ** 2)

    @abstractmethod
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass through neural networks.

        Args:
            t: Time coordinates [N, 1]
            x: Spatial coordinates [N, d]

        Returns:
            Network predictions (single tensor or dict of predictions)
        """

    def sample_points(self) -> dict[str, tuple[torch.Tensor, ...]]:
        """
        Sample training points from domain using specified sampling strategy.

        Returns:
            Dictionary of sampled point sets with keys:
            - 'interior': Interior domain points
            - 'boundary': Boundary points
            - 'initial': Initial condition points

        Note:
            This implementation assumes 1D spatial domain. Override for
            higher-dimensional problems.
        """
        points = {}

        # Interior points (random sampling in space-time)
        t_interior = torch.rand(self.config.n_interior_points, 1, device=self.device, dtype=self.dtype) * self.problem.T

        # Try geometry-aware sampling, fall back to uniform sampling if not available
        try:
            x_interior = self.problem.geometry.sample_interior(self.config.n_interior_points)
            x_interior = torch.from_numpy(x_interior).to(self.device, dtype=self.dtype)
        except AttributeError:
            # Fallback: uniform sampling using geometry bounds
            bounds = self.problem.geometry.get_bounds()
            x_interior = (
                torch.rand(self.config.n_interior_points, 1, device=self.device, dtype=self.dtype)
                * (bounds[1][0] - bounds[0][0])
                + bounds[0][0]
            )

        points["interior"] = (t_interior, x_interior)

        # Boundary points
        t_boundary = torch.rand(self.config.n_boundary_points, 1, device=self.device, dtype=self.dtype) * self.problem.T

        # Try geometry-aware boundary sampling, fall back to default 1D boundaries
        try:
            x_boundary = self.problem.geometry.sample_boundary(self.config.n_boundary_points)
            x_boundary = torch.from_numpy(x_boundary).to(self.device, dtype=self.dtype)
        except AttributeError:
            # Fallback: 1D boundary points (left and right boundaries)
            n_left = self.config.n_boundary_points // 2
            n_right = self.config.n_boundary_points - n_left

            bounds = self.problem.geometry.get_bounds()
            x_boundary = torch.cat(
                [
                    torch.full((n_left, 1), bounds[0][0], device=self.device, dtype=self.dtype),
                    torch.full((n_right, 1), bounds[1][0], device=self.device, dtype=self.dtype),
                ]
            )

        points["boundary"] = (t_boundary, x_boundary)

        # Initial condition points
        try:
            x_initial = self.problem.geometry.sample_interior(self.config.n_initial_points)
            x_initial = torch.from_numpy(x_initial).to(self.device, dtype=self.dtype)
        except AttributeError:
            # Fallback: uniform sampling
            bounds = self.problem.geometry.get_bounds()
            x_initial = (
                torch.rand(self.config.n_initial_points, 1, device=self.device, dtype=self.dtype)
                * (bounds[1][0] - bounds[0][0])
                + bounds[0][0]
            )

        points["initial"] = x_initial

        return points

    def train_step(
        self,
        interior_points: tuple[torch.Tensor, torch.Tensor],
        boundary_points: tuple[torch.Tensor, torch.Tensor],
        initial_points: torch.Tensor,
        data_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """
        Perform single training step.

        Args:
            interior_points: Interior domain points (t, x)
            boundary_points: Boundary points (t, x)
            initial_points: Initial condition points (x)
            data_points: Optional supervised data (t, x, values)

        Returns:
            Dictionary of loss values (all converted to float)
        """
        # Zero gradients for all networks
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        # Compute all loss components
        losses = self.compute_total_loss(interior_points, boundary_points, initial_points, data_points)

        # Backward pass
        losses["total_loss"].backward()

        # Gradient clipping for stability
        if self.config.gradient_clipping > 0:
            for network in self.networks.values():
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=self.config.gradient_clipping)

        # Update parameters
        for optimizer in self.optimizers.values():
            optimizer.step()

        # Backend compatibility - tensor to scalar conversion (Issue #543 acceptable)
        # PyTorch tensors have .item() method, plain Python scalars don't
        return {k: v.item() if hasattr(v, "item") else float(v) for k, v in losses.items()}

    def solve(self, **kwargs: Any) -> dict:
        """
        Solve the MFG system using PINN approach.

        This method implements the BaseMFGSolver interface and provides
        the main entry point for solving MFG problems with PINNs.

        Args:
            **kwargs: Additional arguments (data_points, verbose, etc.)

        Returns:
            Dictionary with solution results and training history
        """
        data_points = kwargs.get("data_points")
        verbose = kwargs.get("verbose", True)

        # Train the networks
        training_history = self.train(data_points=data_points, verbose=verbose)

        # Generate solution on evaluation grid
        solution = self._generate_solution_grid()

        return {
            "solution": solution,
            "training_history": training_history,
            "best_loss": self.best_loss,
            "total_epochs": self.current_epoch,
            "networks": {name: net.state_dict() for name, net in self.networks.items()},
        }

    def train(
        self, data_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None, verbose: bool = True
    ) -> dict[str, list[float]]:
        """
        Train PINN networks using physics-informed loss.

        Args:
            data_points: Optional supervised training data (t, x, values)
            verbose: Whether to print training progress

        Returns:
            Training history with loss evolution
        """
        if verbose:
            print(f"Starting PINN training on {self.device}")
            print(
                f"Configuration: max_epochs={self.config.max_epochs}, "
                f"lr={self.config.learning_rate}, "
                f"n_interior={self.config.n_interior_points}"
            )

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch

            # Sample training points (with optional adaptive resampling)
            if epoch % self.config.resample_frequency == 0 or self.current_points is None:
                self.current_points = self.sample_points()

            # Apply adaptive training strategy if enabled
            adaptive_updates = {}
            if self.adaptive_strategy is not None:
                # Prepare data for adaptive strategy
                physics_points = self.current_points["interior"]
                physics_residuals = self.compute_physics_residual(physics_points)

                domain_bounds = self.get_domain_bounds()

                adaptive_updates = self.adaptive_strategy.step(
                    epoch=epoch, current_points=physics_points, residuals=physics_residuals, domain_bounds=domain_bounds
                )

                # Update sampling points if provided
                if "physics_points" in adaptive_updates:
                    self.current_points["interior"] = adaptive_updates["physics_points"]

            # Perform training step
            losses = self.train_step(
                self.current_points["interior"],
                self.current_points["boundary"],
                self.current_points["initial"],
                data_points,
            )

            # Update adaptive training with loss information
            if self.adaptive_strategy is not None:
                # Provide additional loss information for adaptive strategy
                additional_updates = self.adaptive_strategy.step(
                    epoch=epoch,
                    physics_loss=torch.tensor(losses.get("pde_loss", 0.0)),
                    boundary_loss=torch.tensor(losses.get("boundary_loss", 0.0)),
                    initial_loss=torch.tensor(losses.get("initial_loss", 0.0)),
                    total_loss=losses["total_loss"],
                )

                # Update loss weights if provided
                if "loss_weights" in additional_updates:
                    physics_weight, boundary_weight, initial_weight = additional_updates["loss_weights"]
                    self.config.pde_weight = physics_weight
                    self.config.boundary_weight = boundary_weight
                    self.config.initial_weight = initial_weight

            # Record training history
            for loss_name, loss_value in losses.items():
                self.training_history[loss_name].append(loss_value)

            # Record current learning rate
            current_lr = self.optimizers[next(iter(self.optimizers.keys()))].param_groups[0]["lr"]
            self.training_history["learning_rate"].append(current_lr)

            # Update learning rate schedulers
            if self.config.use_lr_scheduler:
                for scheduler in self.schedulers.values():
                    if self.config.scheduler_type == "plateau":
                        scheduler.step(losses["total_loss"])
                    else:
                        scheduler.step()

            # Check for improvement and early stopping
            if losses["total_loss"] < self.best_loss:
                self.best_loss = losses["total_loss"]
                self.epochs_without_improvement = 0

                # Save best model
                if self.config.save_checkpoints:
                    self._save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1

            # Early stopping check
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            # Periodic logging
            if verbose and (epoch + 1) % self.config.log_interval == 0:
                print(
                    f"Epoch {epoch + 1:5d}: Loss = {losses['total_loss']:.6f} "
                    f"(PDE: {losses['pde_loss']:.6f}, BC: {losses['boundary_loss']:.6f}, "
                    f"IC: {losses['initial_loss']:.6f})"
                )

            # Periodic checkpointing
            if self.config.save_checkpoints and (epoch + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            # Convergence check
            if losses["total_loss"] < self.config.convergence_tolerance:
                if verbose:
                    print(f"Converged at epoch {epoch + 1}")
                break

        if verbose:
            print(f"Training completed. Best loss: {self.best_loss:.6f}")

        return self.training_history

    def _generate_solution_grid(self) -> dict[str, np.ndarray]:
        """
        Generate solution on evaluation grid for visualization/analysis.

        Returns:
            Dictionary with solution arrays on computational grid
        """
        # Default grid generation (can be overridden)
        nt, nx = 100, 100
        bounds = self.problem.geometry.get_bounds()
        t_eval = np.linspace(0, self.problem.T, nt)
        x_eval = np.linspace(bounds[0][0], bounds[1][0], nx)

        T_eval, X_eval = np.meshgrid(t_eval, x_eval)
        t_flat = torch.from_numpy(T_eval.flatten().reshape(-1, 1)).to(self.device, dtype=self.dtype)
        x_flat = torch.from_numpy(X_eval.flatten().reshape(-1, 1)).to(self.device, dtype=self.dtype)

        # Generate predictions
        with torch.no_grad():
            predictions = self.predict(t_flat, x_flat)

        # Convert back to numpy and reshape
        solution = {}
        if isinstance(predictions, dict):
            for name, pred in predictions.items():
                pred_np = pred.cpu().numpy().reshape(nx, nt)
                solution[name] = pred_np
        else:
            solution["u"] = predictions.cpu().numpy().reshape(nx, nt)

        solution["t_grid"] = t_eval
        solution["x_grid"] = x_eval
        solution["T_grid"] = T_eval
        solution["X_grid"] = X_eval

        return solution

    def compute_physics_residual(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute physics residuals at given points for adaptive training.

        Args:
            points: Evaluation points [n_points, dim]

        Returns:
            Physics residuals [n_points]
        """
        # This is a simplified version - each PINN subclass should override
        # with their specific physics residual computation

        if points.shape[1] == 2:  # [t, x] format
            t, x = points[:, 0:1], points[:, 1:2]
        else:
            # For higher dimensions, assume first is time
            t, x = points[:, 0:1], points[:, 1:]

        # Get network predictions
        predictions = self.forward(t, x)

        # Compute simple residual (subclasses should override with proper physics)
        if isinstance(predictions, dict):
            residuals = []
            for pred in predictions.values():
                # Compute gradients for residual computation
                pred_sum = pred.sum()  # For gradient computation
                grad_t = (
                    torch.autograd.grad(pred_sum, t, create_graph=True)[0] if t.requires_grad else torch.zeros_like(t)
                )
                grad_x = (
                    torch.autograd.grad(pred_sum, x, create_graph=True)[0] if x.requires_grad else torch.zeros_like(x)
                )

                # Simple residual: du/dt + du/dx (placeholder)
                residual = torch.abs(grad_t + grad_x.sum(dim=1, keepdim=True))
                residuals.append(residual.squeeze())

            return torch.stack(residuals).mean(dim=0)
        else:
            # Single prediction
            pred_sum = predictions.sum()
            grad_t = torch.autograd.grad(pred_sum, t, create_graph=True)[0] if t.requires_grad else torch.zeros_like(t)
            grad_x = torch.autograd.grad(pred_sum, x, create_graph=True)[0] if x.requires_grad else torch.zeros_like(x)

            residual = torch.abs(grad_t + grad_x.sum(dim=1, keepdim=True))
            return residual.squeeze()

    def get_domain_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get domain bounds for adaptive sampling.

        Returns:
            (lower_bounds, upper_bounds) as torch tensors
        """
        # Default domain bounds - subclasses should override
        # Assuming 2D domain [t, x] with t ∈ [0, 1], x ∈ [0, 1]

        lower_bounds = torch.tensor([0.0, 0.0], device=self.device, dtype=self.dtype)
        upper_bounds = torch.tensor([1.0, 1.0], device=self.device, dtype=self.dtype)

        # Try to get bounds from problem if available
        # Issue #1056: use the uniform get_bounds() accessor on .geometry (every other site in
        # this class uses self.problem.geometry; .domain does not exist, so the prior code always
        # fell through to the default bounds).
        try:
            bounds = self.problem.geometry.get_bounds()
            if isinstance(bounds, list | tuple) and len(bounds) == 2:
                lower_bounds = torch.tensor(bounds[0], device=self.device, dtype=self.dtype)
                upper_bounds = torch.tensor(bounds[1], device=self.device, dtype=self.dtype)
        except (AttributeError, TypeError, ValueError):
            # Problem may not have domain attribute, or bounds may have wrong format
            pass  # Use default bounds

        return lower_bounds, upper_bounds

    def predict(
        self, t: torch.Tensor | np.ndarray, x: torch.Tensor | np.ndarray
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Make predictions using trained networks.

        Args:
            t: Time coordinates
            x: Spatial coordinates

        Returns:
            Network predictions (tensor or dict of tensors)
        """
        # Convert to tensors if needed
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).to(self.device, dtype=self.dtype)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device, dtype=self.dtype)

        # Ensure correct shapes
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Set networks to evaluation mode
        for network in self.networks.values():
            network.eval()

        try:
            with torch.no_grad():
                predictions = self.forward(t, x)
        finally:
            # Set networks back to training mode
            for network in self.networks.values():
                network.train()

        return predictions

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint to disk."""
        checkpoint = {
            "epoch": epoch,
            "config": self.config,
            "networks": {name: net.state_dict() for name, net in self.networks.items()},
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "schedulers": {name: sched.state_dict() for name, sched in self.schedulers.items()},
            "training_history": self.training_history,
            "best_loss": self.best_loss,
            "current_epoch": self.current_epoch,
        }

        filename = f"pinn_checkpoint_epoch_{epoch}.pt"
        if is_best:
            filename = "pinn_best_model.pt"

        torch.save(checkpoint, filename)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint from disk."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load network states
        for name, state_dict in checkpoint["networks"].items():
            if name in self.networks:
                self.networks[name].load_state_dict(state_dict)

        # Load optimizer states
        for name, state_dict in checkpoint["optimizers"].items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state_dict)

        # Load scheduler states
        for name, state_dict in checkpoint.get("schedulers", {}).items():
            if name in self.schedulers:
                self.schedulers[name].load_state_dict(state_dict)

        # Load training state
        self.training_history = checkpoint.get("training_history", self.training_history)
        self.best_loss = checkpoint.get("best_loss", self.best_loss)
        self.current_epoch = checkpoint.get("current_epoch", 0)

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def evaluate_convergence(self) -> dict[str, float]:
        """
        Evaluate convergence metrics on test data.

        Returns:
            Dictionary of convergence metrics
        """
        # Sample test points
        test_points = self.sample_points()

        with torch.no_grad():
            losses = self.compute_total_loss(test_points["interior"], test_points["boundary"], test_points["initial"])

        # Backend compatibility - tensor to scalar conversion (Issue #543 acceptable)
        metrics = {k: v.item() if hasattr(v, "item") else float(v) for k, v in losses.items()}

        # Additional convergence metrics can be added here
        # e.g., mass conservation, energy conservation, etc.

        return metrics

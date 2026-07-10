"""
Adaptive training strategies for Physics-Informed Neural Networks.

This module implements sophisticated adaptive training techniques that
improve PINN convergence and solution quality by dynamically adjusting
training strategies based on physics residuals and solution behavior.

Key Strategies:
- Physics-Guided Sampling: Adaptive point generation based on residual analysis
- Curriculum Learning: Progressive difficulty in training
- Multi-Scale Training: Hierarchical resolution progression
- Loss Balancing: Dynamic weight adjustment for loss components
- Residual-Based Refinement: Targeted sampling in high-error regions

Mathematical Framework:
- Residual Analysis: R(x) = |PDE_residual(x)| for adaptive sampling
- Importance Sampling: p(x) ∝ |R(x)|^α for weighted point selection
- Curriculum Function: C(epoch) defining training complexity progression
- Multi-Scale Decomposition: L = Σᵢ wᵢ(epoch) L_scale_i

Applications:
- High-dimensional MFG problems with complex physics
- Multi-scale phenomena in crowd dynamics
- Singular perturbation problems in finance
- Problems with boundary layers and sharp gradients
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Import with availability checking
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger(__name__)


class AdaptiveTrainingMode(str, Enum):
    """
    Adaptive training strategy for PINN solvers.

    Determines which adaptive training techniques are enabled during
    physics-informed neural network training.

    Attributes:
        BASIC: Standard training without adaptive features
        CURRICULUM: Progressive complexity training
        MULTISCALE: Multi-resolution hierarchical training
        FULL_ADAPTIVE: All adaptive features (curriculum + multiscale + refinement)
    """

    BASIC = "basic"
    CURRICULUM = "curriculum"
    MULTISCALE = "multiscale"
    FULL_ADAPTIVE = "full_adaptive"


@dataclass
class AdaptiveTrainingConfig:
    """Configuration for adaptive training strategies."""

    # Physics-guided sampling
    residual_threshold: float = 1e-3  # Threshold for high-residual regions
    adaptive_sampling_frequency: int = 100  # Update sampling every N epochs
    max_adaptive_points: int = 10000  # Maximum points to add adaptively
    importance_exponent: float = 0.5  # p(x) ∝ |R(x)|^α

    # Training strategy (replaces enable_curriculum, enable_multiscale, enable_refinement)
    training_mode: AdaptiveTrainingMode = AdaptiveTrainingMode.FULL_ADAPTIVE

    # Deprecated parameters (kept for backward compatibility)
    enable_curriculum: bool | None = None  # Deprecated: use training_mode
    enable_multiscale: bool | None = None  # Deprecated: use training_mode
    enable_refinement: bool | None = None  # Deprecated: use training_mode

    # Curriculum learning
    curriculum_epochs: int = 5000  # Epochs for curriculum progression
    initial_complexity: float = 0.1  # Start with 10% of full problem
    complexity_growth: str = "linear"  # "linear" | "exponential" | "sigmoid"

    # Multi-scale training
    num_scales: int = 3  # Number of resolution scales
    scale_transition_epochs: int = 2000  # Epochs between scale transitions
    coarse_to_fine: bool = True  # Start coarse and refine

    # Loss balancing
    adaptive_loss_weights: bool = True
    weight_update_frequency: int = 50  # Update weights every N epochs
    weight_momentum: float = 0.9  # Momentum for weight updates
    gradient_balance: bool = True  # Balance gradients across loss terms

    # Residual-based refinement
    refinement_frequency: int = 500  # Refine mesh every N epochs
    refinement_factor: float = 1.5  # Factor to increase points in high-residual regions
    max_refinement_levels: int = 5  # Maximum refinement levels

    # Performance monitoring
    monitor_convergence: bool = True
    stagnation_patience: int = 1000  # Epochs to wait before intervention
    stagnation_threshold: float = 1e-6  # Minimum improvement threshold

    def __post_init__(self):
        """Handle deprecated parameters with backward compatibility."""
        # Check if any deprecated boolean parameters were used
        deprecated_params_used = any(
            param is not None for param in [self.enable_curriculum, self.enable_multiscale, self.enable_refinement]
        )

        if deprecated_params_used:
            # Map deprecated booleans to training mode (only if training_mode not explicitly set)
            if self.training_mode == AdaptiveTrainingMode.FULL_ADAPTIVE:  # Default value
                # Determine mode from boolean combination
                curriculum = self.enable_curriculum if self.enable_curriculum is not None else True
                multiscale = self.enable_multiscale if self.enable_multiscale is not None else True
                refinement = self.enable_refinement if self.enable_refinement is not None else True

                if not (curriculum or multiscale or refinement):
                    self.training_mode = AdaptiveTrainingMode.BASIC
                elif curriculum and not multiscale and not refinement:
                    self.training_mode = AdaptiveTrainingMode.CURRICULUM
                elif multiscale and not curriculum and not refinement:
                    self.training_mode = AdaptiveTrainingMode.MULTISCALE
                else:
                    # Any combination with 2+ features enabled → FULL_ADAPTIVE
                    self.training_mode = AdaptiveTrainingMode.FULL_ADAPTIVE

    @property
    def uses_curriculum(self) -> bool:
        """Whether curriculum learning is enabled."""
        return self.training_mode in (AdaptiveTrainingMode.CURRICULUM, AdaptiveTrainingMode.FULL_ADAPTIVE)

    @property
    def uses_multiscale(self) -> bool:
        """Whether multiscale training is enabled."""
        return self.training_mode in (AdaptiveTrainingMode.MULTISCALE, AdaptiveTrainingMode.FULL_ADAPTIVE)

    @property
    def uses_refinement(self) -> bool:
        """Whether residual-based refinement is enabled."""
        return self.training_mode == AdaptiveTrainingMode.FULL_ADAPTIVE


# Apply deprecated_parameter decorators to auto-generated __init__
AdaptiveTrainingConfig.__init__ = deprecated_parameter(  # type: ignore[misc]
    param_name="enable_curriculum", since="v0.17.0", replacement="training_mode"
)(
    deprecated_parameter(param_name="enable_multiscale", since="v0.17.0", replacement="training_mode")(
        deprecated_parameter(param_name="enable_refinement", since="v0.17.0", replacement="training_mode")(
            AdaptiveTrainingConfig.__init__
        )
    )
)


@dataclass
class TrainingState:
    """Current state of adaptive training."""

    epoch: int = 0
    current_complexity: float = 0.1
    current_scale: int = 0
    num_physics_points: int = 1000
    num_boundary_points: int = 200
    num_initial_points: int = 200

    # Loss weights
    physics_weight: float = 1.0
    boundary_weight: float = 10.0
    initial_weight: float = 10.0

    # Sampling state
    importance_function: Callable | None = None
    high_residual_regions: list[NDArray] = field(default_factory=list)

    # Performance tracking
    loss_history: list[float] = field(default_factory=list)
    residual_history: list[NDArray] = field(default_factory=list)
    convergence_rate: float = 0.0
    is_stagnating: bool = False


if TORCH_AVAILABLE:

    class PhysicsGuidedSampler:
        """
        Physics-guided adaptive sampling for PINN training.

        This class implements intelligent point sampling based on physics
        residuals, focusing computational effort on regions with poor
        solution quality.
        """

        def __init__(self, config: AdaptiveTrainingConfig):
            """Initialize physics-guided sampler."""
            self.config = config
            self.logger = get_logger(self.__class__.__name__)

        def compute_importance_weights(self, residuals: torch.Tensor) -> torch.Tensor:
            """
            Compute importance weights based on residual magnitudes.

            Args:
                residuals: Physics residuals at sample points [n_points]

            Returns:
                Importance weights for sampling [n_points]
            """
            # Compute residual magnitudes
            residual_magnitude = torch.abs(residuals)

            # Apply importance exponent: p(x) ∝ |R(x)|^α
            importance = torch.pow(
                residual_magnitude + 1e-8,  # Add small epsilon for stability
                self.config.importance_exponent,
            )

            # Normalize to probability distribution
            importance = importance / torch.sum(importance)

            return importance

        def adaptive_point_generation(
            self,
            current_points: torch.Tensor,
            residuals: torch.Tensor,
            domain_bounds: tuple[torch.Tensor, torch.Tensor],
            n_new_points: int,
        ) -> torch.Tensor:
            """
            Generate new training points based on residual analysis.

            Args:
                current_points: Current sample points [n_points, dim]
                residuals: Physics residuals [n_points]
                domain_bounds: (lower_bounds, upper_bounds) for domain
                n_new_points: Number of new points to generate

            Returns:
                New sample points [n_new_points, dim]
            """
            device = current_points.device
            dim = current_points.shape[1]
            lower_bounds, upper_bounds = domain_bounds

            # Identify high-residual regions
            residual_magnitude = torch.abs(residuals)
            high_residual_mask = residual_magnitude > self.config.residual_threshold

            if torch.sum(high_residual_mask) > 0:
                # Focus on high-residual regions
                high_residual_points = current_points[high_residual_mask]
                high_residuals = residual_magnitude[high_residual_mask]

                # Create Gaussian mixture around high-residual points
                n_points_per_center = max(1, n_new_points // len(high_residual_points))
                new_points = []

                for i, center in enumerate(high_residual_points):
                    # Adaptive standard deviation based on residual magnitude
                    std = 0.1 * (high_residuals[i] / torch.max(high_residuals))
                    std = torch.clamp(std, 0.01, 0.5)  # Reasonable bounds

                    # Generate points around center
                    points = center.unsqueeze(0) + std * torch.randn(n_points_per_center, dim, device=device)

                    # Clamp to domain bounds
                    points = torch.clamp(points, lower_bounds, upper_bounds)
                    new_points.append(points)

                new_points = torch.cat(new_points, dim=0)

                # If we need more points, fill with uniform random
                if len(new_points) < n_new_points:
                    remaining = n_new_points - len(new_points)
                    uniform_points = torch.rand(remaining, dim, device=device)
                    uniform_points = lower_bounds + uniform_points * (upper_bounds - lower_bounds)
                    new_points = torch.cat([new_points, uniform_points], dim=0)

            else:
                # No high-residual regions, use uniform sampling
                new_points = torch.rand(n_new_points, dim, device=device)
                new_points = lower_bounds + new_points * (upper_bounds - lower_bounds)

            return new_points[:n_new_points]  # Ensure exact number

    class AdaptiveTrainingStrategy:
        """
        Comprehensive adaptive training strategy for PINNs.

        This class orchestrates multiple adaptive techniques including
        curriculum learning, multi-scale training, and dynamic loss weighting.
        """

        def __init__(self, config: AdaptiveTrainingConfig):
            """Initialize adaptive training strategy."""
            self.config = config
            self.state = TrainingState()
            self.sampler = PhysicsGuidedSampler(config)
            self.logger = get_logger(self.__class__.__name__)

        def update_curriculum(self) -> float:
            """
            Update curriculum complexity based on training progress.

            Returns:
                Current complexity factor [0, 1]
            """
            # Issue #1572: read the canonical training_mode-derived property, not the raw
            # deprecated boolean (which is None unless the old kwarg was passed).
            if not self.config.uses_curriculum:
                return 1.0

            progress = min(1.0, self.state.epoch / self.config.curriculum_epochs)

            if self.config.complexity_growth == "linear":
                complexity = self.config.initial_complexity + progress * (1.0 - self.config.initial_complexity)
            elif self.config.complexity_growth == "exponential":
                complexity = self.config.initial_complexity * np.exp(3 * progress)
                complexity = min(1.0, complexity)
            elif self.config.complexity_growth == "sigmoid":
                # Sigmoid curve: slow start, rapid middle, slow end
                x = 10 * (progress - 0.5)  # Map to [-5, 5]
                sigmoid = 1 / (1 + np.exp(-x))
                complexity = self.config.initial_complexity + sigmoid * (1.0 - self.config.initial_complexity)
            else:
                complexity = 1.0

            self.state.current_complexity = complexity
            return complexity

        def update_loss_weights(
            self, physics_loss: torch.Tensor, boundary_loss: torch.Tensor, initial_loss: torch.Tensor
        ) -> tuple[float, float, float]:
            """
            Update loss weights based on gradient balancing.

            Args:
                physics_loss: Current physics loss
                boundary_loss: Current boundary loss
                initial_loss: Current initial condition loss

            Returns:
                Updated (physics_weight, boundary_weight, initial_weight)
            """
            if not self.config.adaptive_loss_weights:
                return self.state.physics_weight, self.state.boundary_weight, self.state.initial_weight

            if self.state.epoch % self.config.weight_update_frequency == 0:
                # Simple gradient balancing approach
                losses = torch.stack([physics_loss, boundary_loss, initial_loss])

                # Compute relative loss magnitudes
                loss_magnitudes = torch.abs(losses)
                max_loss = torch.max(loss_magnitudes)

                if max_loss > 1e-8:  # Avoid division by zero
                    # Inverse weighting: higher weight for lower losses
                    new_weights = max_loss / (loss_magnitudes + 1e-8)
                    new_weights = new_weights / torch.sum(new_weights) * 3  # Normalize to sum to 3

                    # Apply momentum
                    momentum = self.config.weight_momentum
                    self.state.physics_weight = (
                        momentum * self.state.physics_weight + (1 - momentum) * new_weights[0].item()
                    )
                    self.state.boundary_weight = (
                        momentum * self.state.boundary_weight + (1 - momentum) * new_weights[1].item()
                    )
                    self.state.initial_weight = (
                        momentum * self.state.initial_weight + (1 - momentum) * new_weights[2].item()
                    )

            return self.state.physics_weight, self.state.boundary_weight, self.state.initial_weight

        def update_sampling_points(
            self,
            current_points: torch.Tensor,
            residuals: torch.Tensor,
            domain_bounds: tuple[torch.Tensor, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            """
            Update sampling points based on residual analysis.

            Args:
                current_points: Current sample points
                residuals: Physics residuals
                domain_bounds: Domain boundaries

            Returns:
                Dictionary with updated point sets
            """
            # Issue #1572: canonical property, not the raw deprecated boolean.
            if not self.config.uses_refinement:
                return {"physics_points": current_points}

            if self.state.epoch % self.config.adaptive_sampling_frequency == 0:
                # Determine number of new points to add
                complexity = self.state.current_complexity
                base_points = int(complexity * self.config.max_adaptive_points)

                # Generate adaptive points
                n_new_points = min(1000, max(100, base_points // 10))  # Reasonable bounds
                new_points = self.sampler.adaptive_point_generation(
                    current_points, residuals, domain_bounds, n_new_points
                )

                # Combine with existing points
                updated_points = torch.cat([current_points, new_points], dim=0)

                # Optionally limit total number of points
                max_total = self.config.max_adaptive_points
                if len(updated_points) > max_total:
                    # Keep points with highest residuals
                    all_residuals = torch.cat([residuals, torch.zeros(len(new_points), device=residuals.device)])
                    _, top_indices = torch.topk(torch.abs(all_residuals), max_total)
                    updated_points = updated_points[top_indices]

                return {"physics_points": updated_points}
            else:
                return {"physics_points": current_points}

        def check_stagnation(self, current_loss: float) -> bool:
            """
            Check if training is stagnating.

            Args:
                current_loss: Current total loss

            Returns:
                True if training is stagnating
            """
            self.state.loss_history.append(current_loss)

            if len(self.state.loss_history) >= self.config.stagnation_patience:
                recent_losses = self.state.loss_history[-self.config.stagnation_patience :]

                # Check if improvement is below threshold
                improvement = recent_losses[0] - recent_losses[-1]
                relative_improvement = improvement / (recent_losses[0] + 1e-8)

                self.state.is_stagnating = relative_improvement < self.config.stagnation_threshold

                if self.state.is_stagnating:
                    self.logger.warning(f"Training stagnation detected at epoch {self.state.epoch}")

            return self.state.is_stagnating

        def step(self, epoch: int, **kwargs: Any) -> dict[str, Any]:
            """
            Perform one step of adaptive training strategy.

            Args:
                epoch: Current epoch
                **kwargs: Additional training information

            Returns:
                Dictionary with updated training parameters
            """
            self.state.epoch = epoch

            updates = {}

            # Update curriculum complexity (Issue #1572: canonical property, not raw boolean)
            if self.config.uses_curriculum:
                complexity = self.update_curriculum()
                updates["complexity"] = complexity

            # Update loss weights if losses provided
            if "physics_loss" in kwargs and "boundary_loss" in kwargs and "initial_loss" in kwargs:
                weights = self.update_loss_weights(
                    kwargs["physics_loss"], kwargs["boundary_loss"], kwargs["initial_loss"]
                )
                updates["loss_weights"] = weights

            # Update sampling points if residuals provided
            if "residuals" in kwargs and "current_points" in kwargs and "domain_bounds" in kwargs:
                point_updates = self.update_sampling_points(
                    kwargs["current_points"], kwargs["residuals"], kwargs["domain_bounds"]
                )
                updates.update(point_updates)

            # Check for stagnation
            if "total_loss" in kwargs:
                stagnating = self.check_stagnation(kwargs["total_loss"])
                updates["is_stagnating"] = stagnating

            return updates

else:
    # Placeholder classes when PyTorch is not available
    class PhysicsGuidedSampler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("Adaptive training requires PyTorch")

    class AdaptiveTrainingStrategy:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("Adaptive training requires PyTorch")

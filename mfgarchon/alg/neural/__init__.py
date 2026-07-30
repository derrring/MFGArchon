"""
[FROZEN PROTOTYPE — not under development]

Neural paradigm for MFG problems.

⛔ This paradigm is a DESIGN PROTOTYPE / PLACEHOLDER. It is not under active development and
should not be extended, refactored, or given test coverage without an explicit instruction naming
it. See CLAUDE.md § "FROZEN: alg/neural/ and alg/reinforcement/".

Adding tests here is discouraged for a reason worth stating: coverage reads as a promise that the
behaviour is intended and load-bearing. On a placeholder that promise is false, and it makes later
readers preserve decisions nobody made. Record defects as issues instead — filing is free.


This module contains neural network-based approaches for solving Mean Field Games:
- nn: Neural network architectures (shared by all neural methods)
- pinn_solvers: Physics-Informed Neural Networks (HJB, FP, MFG solvers)
- operator_learning: Neural operator methods (FNO, DeepONet) for parameter-to-solution mapping
- core: Shared neural network infrastructure and utilities

Key Neural Methods:
- PINN Solvers: Individual HJBPINNSolver, FPPINNSolver, and coupled MFGPINNSolver
- Neural Operators: FourierNeuralOperator and DeepONet for fast parameter-to-solution mapping
- Training Infrastructure: Comprehensive training managers and data handling

The 'nn' module follows PyTorch convention and contains architectures suitable
for both PINN methods and operator learning approaches, avoiding confusion
with network-based (graph) MFG problems.

Note: Neural solvers require PyTorch installation.
"""

from mfgarchon.alg.base_solver import BaseNeuralSolver

# Conditional imports based on PyTorch availability
try:
    import importlib.util

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        TORCH_AVAILABLE = True
    else:
        raise ImportError("torch not available")
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    # Import neural network architectures (shared by all neural methods)
    from . import nn

    # Import core neural components
    from .core import (
        BoundaryLoss,
        DataLoss,
        FeedForwardNetwork,
        MFGLossFunction,
        ModifiedMLP,
        NetworkArchitecture,
        PhysicsLoss,
        PINNLossFunction,
        ResidualNetwork,
        TrainingManager,
        auto_differentiation,
        compute_gradients,
        create_mfg_networks,
        neural_network_utils,
        sample_points,
    )

    # Neural operator methods for parameter-to-solution mapping
    from .operator_learning import (
        BaseNeuralOperator,
        DeepONet,
        DeepONetConfig,
        FNOConfig,
        FourierNeuralOperator,
        OperatorConfig,
        OperatorDataset,
        OperatorTrainingManager,
        TrainingConfig,
        create_mfg_operator,
    )

    # PINN solvers: Individual equation and coupled system solvers
    try:
        from .pinn_solvers import (
            AdaptiveTrainingConfig,
            AdaptiveTrainingStrategy,
            FPPINNSolver,
            HJBPINNSolver,
            MFGPINNSolver,
            PhysicsGuidedSampler,
            PINNBase,
            PINNConfig,
        )
    except ImportError:
        # PINN solvers not available, set empty classes for compatibility
        FPPINNSolver = None  # type: ignore[misc]
        HJBPINNSolver = None  # type: ignore[misc]
        MFGPINNSolver = None  # type: ignore[misc]
        PINNBase = None  # type: ignore[misc]
        PINNConfig = None  # type: ignore[misc]

    pinn_exports = []
    if FPPINNSolver is not None:
        pinn_exports = [
            "FPPINNSolver",
            "HJBPINNSolver",
            "MFGPINNSolver",
            "PINNBase",
            "PINNConfig",
        ]

    __all__ = [
        # PINN Solvers
        "AdaptiveTrainingConfig",
        "AdaptiveTrainingStrategy",
        # Neural Operator Methods
        "BaseNeuralOperator",
        "BaseNeuralSolver",
        # Core Neural Components
        "BoundaryLoss",
        "DataLoss",
        "DeepONet",
        "DeepONetConfig",
        "FNOConfig",
        "FPPINNSolver",
        "FeedForwardNetwork",
        "FourierNeuralOperator",
        "HJBPINNSolver",
        "MFGLossFunction",
        "MFGPINNSolver",
        "ModifiedMLP",
        "NetworkArchitecture",
        "OperatorConfig",
        "OperatorDataset",
        "OperatorTrainingManager",
        "PINNBase",
        "PINNConfig",
        "PINNLossFunction",
        "PhysicsGuidedSampler",
        "PhysicsLoss",
        "ResidualNetwork",
        "TrainingConfig",
        "TrainingManager",
        "auto_differentiation",
        "compute_gradients",
        "create_mfg_networks",
        "create_mfg_operator",
        "neural_network_utils",
        # Neural Network Architectures Module
        "nn",
        "sample_points",
    ]
    __all__.extend(pinn_exports)
else:
    import warnings

    warnings.warn(
        "PyTorch is required for neural network solvers. "
        "Install with: pip install mfgarchon[neural] or pip install torch",
        ImportWarning,
    )

    __all__ = [
        "BaseNeuralSolver",
    ]

# Always export availability info
__all__.extend(["TORCH_AVAILABLE"])

"""
Minimal core infrastructure for MFG problems.

This module provides only the essential base classes and protocols.
Framework extensions (network, variational, multi-population) are in mfgarchon.extensions.
"""

from mfgarchon.geometry import BoundaryConditions

from .base_problem import CartesianGridMFGProtocol, MFGProblemProtocol
from .derivatives import (
    DerivativeTensors,
    from_multi_index_dict,
    to_multi_index_dict,
)
from .hamiltonian import (
    # Control cost base classes (original)
    BoundedControlCost,
    BoundedHamiltonian,
    # Non-separable Hamiltonian with density-dependent kinetic cost (Issue #782)
    CongestionHamiltonian,
    ControlCostBase,
    # Dual classes (Legendre transform)
    DualHamiltonian,
    DualLagrangian,
    # Full MFG Hamiltonian classes (Issue #673)
    HamiltonianBase,
    # Single-source contract (Issue #1071)
    HamiltonianValues,
    HEvalState,
    L1ControlCost,
    L1Hamiltonian,
    # Lagrangian classes (Issue #651)
    LagrangianBase,
    # Common base (Issue #651)
    MFGOperatorBase,
    OptimizationSense,
    QuadraticControlCost,
    QuadraticHamiltonian,
    QuadraticMFGHamiltonian,
    Regularizer,
    SeparableHamiltonian,
    create_hamiltonian,
)
from .mfg_components import (
    ConditionsMixin,
    HamiltonianMixin,
    MFGComponents,
)
from .mfg_problem import MFGProblem
from .model import Conditions, ErgodicConditions, Model

__all__ = [
    # Geometry
    "BoundaryConditions",
    # Core protocols
    "MFGProblemProtocol",
    "CartesianGridMFGProtocol",
    # Derivative tensors
    "DerivativeTensors",
    "from_multi_index_dict",
    "to_multi_index_dict",
    # Base MFG problem
    "MFGProblem",
    "MFGComponents",
    # API v1.0 classes (Issue #875)
    "Model",
    "Conditions",
    "ErgodicConditions",
    # Mixins
    "HamiltonianMixin",
    "ConditionsMixin",
    # Hamiltonian/Lagrangian (Issues #623, #651, #667, #673)
    "OptimizationSense",
    # Common base for H and L (Issue #651)
    "MFGOperatorBase",
    # Control cost classes (original, H(p) only)
    "ControlCostBase",
    "QuadraticControlCost",
    "QuadraticHamiltonian",
    "L1ControlCost",
    "L1Hamiltonian",
    "BoundedControlCost",
    "BoundedHamiltonian",
    # Full MFG Hamiltonian classes (Issue #673, #782)
    "HamiltonianBase",
    # Single-source contract (Issue #1071)
    "HEvalState",
    "HamiltonianValues",
    "Regularizer",
    "SeparableHamiltonian",
    "CongestionHamiltonian",
    "QuadraticMFGHamiltonian",
    "create_hamiltonian",
    # Lagrangian classes (Issue #651)
    "LagrangianBase",
    # Dual classes (Legendre transform)
    "DualHamiltonian",
    "DualLagrangian",
]

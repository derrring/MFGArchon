"""
Configuration management for MFGarchon solvers.

This module provides the unified solver configuration system using Pydantic.
Configurations specify HOW to solve problems (algorithmic choices), not WHAT
problems to solve (mathematical definitions - those are MFGProblem instances).

Quick Start
-----------
>>> from mfgarchon.config import FPConfig, HJBConfig, ParticleConfig, PicardConfig, SolverConfig
>>> config = SolverConfig(
...     hjb=HJBConfig(method="fdm", accuracy_order=2),
...     fp=FPConfig(method="particle", particle=ParticleConfig(num_particles=5000)),
...     picard=PicardConfig(max_iterations=100, tolerance=1e-8),
... )

>>> # Or load from YAML
>>> from mfgarchon.config import load_solver_config
>>> config = load_solver_config("experiments/baseline.yaml")

Key Principle
-------------
Configuration is for SOLVERS (how to solve), not PROBLEMS (what to solve):
- MFGProblem (Python): Mathematical definition (g, H, rho_0, geometry)
- SolverConfig (YAML/Python): Algorithmic choices (method, tolerance, backend)
"""

# =============================================================================
# CORE CONFIG CLASSES (Pydantic-based)
# =============================================================================

from .array_validation import (
    ArrayValidationConfig,
    CollocationConfig,
    ExperimentConfig,
    MFGArrays,
    MFGGridConfig,
)
from .core import (
    BackendConfig,
    BaseConfig,
    LoggingConfig,
    MFGSolverConfig,
    PicardConfig,
    SolverConfig,  # Backward compatibility alias for MFGSolverConfig
)

# YAML I/O
from .io import load_solver_config, save_solver_config, validate_yaml_config

# MFG method configurations (unified)
from .mfg_methods import (
    BoundaryAccuracyConfig,
    DerivativeConfig,
    FDMConfig,
    FEMConfig,
    FPConfig,
    GFDMConfig,
    HJBConfig,
    NeighborhoodConfig,
    NetworkConfig,
    NewtonConfig,
    ParticleConfig,
    QPConfig,
    SLConfig,
    WENOConfig,
)

# Config → solver kwargs translator (Issue #1155)
from .translator import (
    backend_config_to_kwargs,
    check_logging_config,
    fp_config_to_kwargs,
    hjb_config_to_kwargs,
    picard_config_to_iterator_kwargs,
    translate_solver_config,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Core classes
    "BaseConfig",
    "SolverConfig",
    "PicardConfig",
    "BackendConfig",
    "LoggingConfig",
    # Method configs (unified)
    "BoundaryAccuracyConfig",
    "DerivativeConfig",
    "FDMConfig",
    "FEMConfig",
    "GFDMConfig",
    "NeighborhoodConfig",
    "QPConfig",
    "SLConfig",
    "WENOConfig",
    "ParticleConfig",
    "NetworkConfig",
    "NewtonConfig",
    # Composite solver configs
    "HJBConfig",
    "FPConfig",
    # I/O
    "load_solver_config",
    "save_solver_config",
    "validate_yaml_config",
    # Translator (Issue #1155)
    "hjb_config_to_kwargs",
    "fp_config_to_kwargs",
    "picard_config_to_iterator_kwargs",
    "backend_config_to_kwargs",
    "check_logging_config",
    "translate_solver_config",
    # MFG solver config
    "MFGSolverConfig",
    # Array validation
    "ArrayValidationConfig",
    "CollocationConfig",
    "ExperimentConfig",
    "MFGArrays",
    "MFGGridConfig",
]

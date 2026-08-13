"""
Algorithm structure for MFGarchon.

Paradigm-based organization:
- numerical: Classical numerical analysis methods (FDM, FEM, GFDM, SL)
- optimization: Direct optimization approaches (variational, OT)

Iteration infrastructure (schedules, convergence) lives in utils/convergence/,
not here. See Issue #985.
"""

from __future__ import annotations

# Import paradigm modules
from . import numerical, optimization

# Import base types (Issue #580)
from .base_solver import SchemeFamily

__all__ = [
    # Paradigm modules
    "numerical",
    "optimization",
    # Base types
    "SchemeFamily",
]

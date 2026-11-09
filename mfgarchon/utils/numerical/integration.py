"""
Integration utilities for MFGarchon.

Provides consistent integration functions that work across NumPy versions,
with preference for NumPy 2.0+ standard but fallbacks for older versions.
"""

from __future__ import annotations

# Import from the comprehensive NumPy compatibility module
from mfgarchon.utils.numpy_compat import get_numpy_info, trapezoid


# Re-export for backward compatibility
def get_integration_info() -> dict[str, str]:
    """Get information about available integration methods."""
    return get_numpy_info()


# Export the main functions
__all__ = ["get_integration_info", "trapezoid"]

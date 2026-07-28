"""
Notebook utilities for MFGarchon.

This module provides Jupyter notebook integration and reporting tools:
- reporting: HTML report generation and notebook utilities
"""

from __future__ import annotations

# Re-export from submodules - these have optional dependencies
try:
    from .reporting import MFGNotebookReporter

    __all__ = ["MFGNotebookReporter"]
except ImportError:
    __all__ = []

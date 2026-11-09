"""
I/O utilities for MFGarchon solver data persistence.

This module provides file format support for saving and loading MFG solver results:
- HDF5: Primary format for solver solutions (recommended)
- Checkpoint/resume capability for long-running computations
"""

from __future__ import annotations

from .hdf5_utils import (
    HDF5_AVAILABLE,
    get_hdf5_info,
    load_checkpoint,
    load_hdf5,
    load_solution,
    save_checkpoint,
    save_hdf5,
    save_solution,
)

__all__ = [
    "HDF5_AVAILABLE",
    "get_hdf5_info",
    "load_checkpoint",
    "load_hdf5",
    "load_solution",
    "save_checkpoint",
    "save_hdf5",
    "save_solution",
]

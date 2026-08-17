"""
NumPy Backend for MFGarchon

Reference implementation using NumPy for CPU-based computations.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.utils.numerical.integration import trapezoid

from .base_backend import BaseBackend


class NumPyBackend(BaseBackend):
    """NumPy-based computational backend."""

    def _setup_backend(self):
        """Initialize NumPy backend."""
        # Set dtype
        if self.precision == "float32":
            self.dtype = np.float32
        else:
            self.dtype = np.float64

        # NumPy uses CPU only
        if self.device != "cpu" and self.device != "auto":
            import warnings

            warnings.warn(f"NumPy backend only supports CPU, ignoring device='{self.device}'")
        self.device = "cpu"

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def array_module(self):
        return np

    # Array Operations
    def array(self, data, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return np.array(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return np.ones(shape, dtype=dtype)

    def linspace(self, start, stop, num):
        return np.linspace(start, stop, num, dtype=self.dtype)

    def meshgrid(self, *arrays, indexing="xy"):
        from typing import Literal, cast

        valid_indexing = cast("Literal['xy', 'ij']", indexing)
        return np.meshgrid(*arrays, indexing=valid_indexing)

    # Mathematical Operations
    def trapezoid(self, y, x=None, dx=1.0, axis=-1):
        return trapezoid(y, x=x, dx=dx, axis=axis)

    def diff(self, a, n=1, axis=-1):
        return np.diff(a, n=n, axis=axis)

    def interp(self, x, xp, fp):
        return np.interp(x, xp, fp)

    # Linear Algebra
    def solve(self, A, b):
        return np.linalg.solve(A, b)

    def eig(self, a):
        return np.linalg.eig(a)

    # Statistics
    def mean(self, a, axis=None):
        return np.mean(a, axis=axis)

    def std(self, a, axis=None):
        return np.std(a, axis=axis)

    def max(self, a, axis=None):
        return np.max(a, axis=axis)

    def min(self, a, axis=None):
        return np.min(a, axis=axis)

    # MFG-Specific Operations
    def vectorize(self, func, signature=None):
        return np.vectorize(func, signature=signature)

    # Device Management (CPU only)
    def to_device(self, array):
        return array

    def from_device(self, array):
        return array

    def to_numpy(self, array) -> np.ndarray:
        return np.asarray(array)

    def from_numpy(self, array: np.ndarray):
        return array

    def get_device_info(self) -> dict:
        return {
            "backend": self.name,
            "device": "cpu",
            "precision": self.precision,
            "numpy_version": np.__version__,
        }

    def memory_usage(self) -> dict | None:
        """Get memory usage information."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,  # Resident Set Size
                "vms": memory_info.vms,  # Virtual Memory Size
                "percent": process.memory_percent(),
            }
        except ImportError:
            return None

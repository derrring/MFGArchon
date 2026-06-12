"""
NumPy Backend for MFGarchon

Reference implementation using NumPy for CPU-based computations.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.utils.numerical.integration import trapezoid
from mfgarchon.utils.pde_coefficients import resolve_volatility

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
    def grad(self, func, argnum=0):
        """
        Numerical gradient using finite differences.
        For NumPy backend, this is a simple finite difference approximation.
        """

        def gradient_func(*args):
            # Simple finite difference gradient
            # Get machine epsilon for the current dtype
            # Ensure dtype is properly converted for np.finfo
            dtype_for_finfo = np.dtype(self.dtype) if not isinstance(self.dtype, np.dtype) else self.dtype
            finfo = np.finfo(dtype_for_finfo)
            eps = float(np.sqrt(finfo.eps))
            args_list = list(args)
            x = args_list[argnum]

            # Forward difference
            args_list[argnum] = x + eps
            f_plus = func(*args_list)

            args_list[argnum] = x - eps
            f_minus = func(*args_list)

            return (f_plus - f_minus) / (2 * eps)

        return gradient_func

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
    def compute_hamiltonian(self, x, p, m, problem_params):
        """
        Default Hamiltonian: H(x, p, m) = 0.5 * p^2
        Override in specific problems.
        """
        return 0.5 * p**2

    def compute_optimal_control(self, x, p, m, problem_params):
        """
        Default optimal control: a*(x, p, m) = -p
        Override in specific problems.
        """
        return -p

    def hjb_step(self, U, M, dt, dx, problem_params):
        """
        Hamilton-Jacobi-Bellman time step using finite differences.

        ∂U/∂t + H(x, ∇U, M) = 0
        """
        # Compute spatial gradient of U using central differences
        dU_dx = np.zeros_like(U)
        dU_dx[1:-1] = (U[2:] - U[:-2]) / (2 * dx)
        dU_dx[0] = (U[1] - U[0]) / dx  # Forward difference at boundary
        dU_dx[-1] = (U[-1] - U[-2]) / dx  # Backward difference at boundary

        # Compute Hamiltonian
        x_grid = problem_params.get("x_grid", np.linspace(0, 1, len(U)))
        H = self.compute_hamiltonian(x_grid, dU_dx, M, problem_params)

        # Time step: U^{n+1} = U^n - dt * H
        U_new = U - dt * H

        return U_new

    def fpk_step(self, M, U, dt, dx, problem_params):
        """
        Fokker-Planck-Kolmogorov time step using finite differences.

        ∂M/∂t - ∇ · (M * a*(x, ∇U, M)) + (σ²/2) * ∇²M = 0
        """
        # Compute spatial gradient of U
        dU_dx = np.zeros_like(U)
        dU_dx[1:-1] = (U[2:] - U[:-2]) / (2 * dx)
        dU_dx[0] = (U[1] - U[0]) / dx
        dU_dx[-1] = (U[-1] - U[-2]) / dx

        # Compute optimal control
        x_grid = problem_params.get("x_grid", np.linspace(0, 1, len(M)))
        a_opt = self.compute_optimal_control(x_grid, dU_dx, M, problem_params)

        # Compute flux: J = M * a_opt
        flux = M * a_opt

        # Compute divergence of flux using central differences
        div_flux = np.zeros_like(M)
        div_flux[1:-1] = (flux[2:] - flux[:-2]) / (2 * dx)
        div_flux[0] = (flux[1] - flux[0]) / dx
        div_flux[-1] = (flux[-1] - flux[-2]) / dx

        # Diffusion term: σ²/2 * ∇²M.  Issue #1282: read the volatility through the
        # single-source resolver (canonical "sigma" key; legacy "sigma_sq" holds sigma**2,
        # default preserves the prior sqrt(0.01)=0.1 no-key behavior), then D = sigma**2/2.
        sigma = resolve_volatility(problem_params, legacy_key="sigma_sq", legacy_is_squared=True, default=0.1)
        d2M_dx2 = np.zeros_like(M)
        d2M_dx2[1:-1] = (M[2:] - 2 * M[1:-1] + M[:-2]) / (dx**2)
        # Zero Neumann boundary conditions for diffusion
        d2M_dx2[0] = d2M_dx2[1]
        d2M_dx2[-1] = d2M_dx2[-2]

        diffusion = 0.5 * sigma * sigma * d2M_dx2

        # Time step: M^{n+1} = M^n + dt * (-∇·J + diffusion)
        M_new = M + dt * (-div_flux + diffusion)

        # Ensure non-negativity and conservation
        M_new = np.maximum(M_new, 0)
        total_mass = self.trapezoid(M_new, dx=dx)
        if total_mass > 1e-12:  # Avoid division by zero
            M_new = M_new / total_mass

        return M_new

    # Performance and Compilation (no-ops for NumPy)
    def compile_function(self, func, *args, **kwargs):
        return func

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

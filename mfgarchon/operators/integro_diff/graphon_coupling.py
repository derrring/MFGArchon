"""
Graphon coupling operator for continuum-limit network MFG.

Computes the nonlocal coupling integral:

    W[f](x) = int_0^1 W(x, y) f(y) dy

where W(x, y) is a graphon kernel and f is a function on [0, 1]
(typically a density or value-function transformation).

This is structurally identical to the Levy integro-differential operator
but applied to the graphon kernel instead of a Levy measure. Both are
nonlocal integral operators computed by quadrature.

Inherits from scipy.sparse.linalg.LinearOperator for matrix-free usage.

Issue #963: GraphonCouplingOperator.

Mathematical background:
    In graphon MFG, the HJB and FP equations have nonlocal coupling:

    HJB: -du/dt + H(x, Du) + int W(x,y) f(m(y)) dy = 0
    FP:  dm/dt - L[m] + div(m * int W(x,y) g(v(y)) dy) = 0

    The operator W[f](x) = int W(x,y) f(y) dy is linear in f,
    hence a LinearOperator.

References:
    Caines & Huang (2021), "Graphon Mean Field Games and Their Equations"
    Carmona, Cooney, Graves, Lauriere (2022), "Stochastic Graphon Games"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .graphon_kernels import GraphonKernel


class GraphonCouplingOperator(LinearOperator):
    """Nonlocal coupling operator via graphon kernel W(x, y).

    Computes W[f](x_i) = int_0^1 W(x_i, y) f(y) dy using
    Gauss-Legendre quadrature (general) or FFT (translational kernels).

    Inherits LinearOperator: supports ``W @ f_values`` and ``W.T @ g_values``.

    Parameters
    ----------
    grid_points : NDArray
        1D spatial grid on [0, 1], shape (N,). Must be sorted.
    kernel : GraphonKernel
        Graphon kernel W(x, y). Must implement ``evaluate(x, y)``.
    quadrature_order : int
        Number of Gauss-Legendre quadrature points (default 32).
        Ignored for FFT path (uses grid points directly).
    use_fft : bool | None
        Use FFT convolution for translational kernels. Auto-detect if None.

    Example
    -------
    >>> from mfgarchon.operators.integro_diff.graphon_kernels import ConstantGraphon
    >>> x = np.linspace(0, 1, 101)
    >>> W_op = GraphonCouplingOperator(x, ConstantGraphon(p=0.5))
    >>> f = np.ones(101)
    >>> Wf = W_op @ f  # = 0.5 * integral of f over [0,1]
    """

    def __init__(
        self,
        grid_points: NDArray,
        kernel: GraphonKernel,
        quadrature_order: int = 32,
        use_fft: bool | None = None,
    ):
        self._grid = np.asarray(grid_points, dtype=float).ravel()
        N = len(self._grid)
        super().__init__(dtype=np.float64, shape=(N, N))

        self._kernel = kernel
        self._N = N

        # Auto-detect FFT eligibility
        self._use_fft = use_fft
        if use_fft is None:
            self._use_fft = getattr(kernel, "is_translational", False)

        if self._use_fft:
            self._setup_fft()
        else:
            self._setup_quadrature(quadrature_order)

    def _setup_quadrature(self, order: int) -> None:
        """Precompute quadrature weights and kernel matrix."""
        # Gauss-Legendre on [0, 1]
        nodes, weights = np.polynomial.legendre.leggauss(order)
        # Transform from [-1, 1] to [0, 1]
        self._quad_y = 0.5 * (nodes + 1)
        self._quad_w = 0.5 * weights

        # Precompute kernel matrix W(x_i, y_k) for all grid x quadrature
        # Shape: (N, order)
        self._W_matrix = self._kernel.evaluate(self._grid, self._quad_y)

        # Precompute interpolation from grid to quadrature points
        # For evaluating f(y_k) from f(x_i) via linear interpolation
        self._interp_indices = np.searchsorted(self._grid, self._quad_y, side="left").clip(1, self._N - 1)
        y = self._quad_y
        x = self._grid
        idx = self._interp_indices
        # Linear interpolation weights: f(y_k) = (1-t)*f(x_{idx-1}) + t*f(x_{idx})
        self._interp_t = (y - x[idx - 1]) / (x[idx] - x[idx - 1] + 1e-30)

    def _setup_fft(self) -> None:
        """Precompute FFT of translational kernel."""
        N = self._N
        dx = self._grid[1] - self._grid[0] if N > 1 else 1.0
        # Kernel evaluated at lag distances
        lags = self._grid - self._grid[0]
        # For circular convolution: need kernel at negative lags too
        full_lags = np.concatenate([lags, -lags[1:][::-1]])
        kernel_1d = self._kernel.evaluate(np.zeros(1), full_lags)
        self._kernel_fft = np.fft.fft(kernel_1d.ravel())
        self._dx = dx

    def _matvec(self, f: NDArray) -> NDArray:
        """Compute W[f](x_i) = int W(x_i, y) f(y) dy."""
        f = np.asarray(f).ravel()

        if self._use_fft:
            return self._matvec_fft(f)

        # Interpolate f to quadrature points
        idx = self._interp_indices
        t = self._interp_t
        f_quad = (1 - t) * f[idx - 1] + t * f[idx]

        # Quadrature: sum_k w_k * W(x_i, y_k) * f(y_k)
        return self._W_matrix @ (self._quad_w * f_quad)

    def _matvec_fft(self, f: NDArray) -> NDArray:
        """FFT-based convolution for translational kernels."""
        N = self._N
        # Zero-pad for linear (not circular) convolution
        f_padded = np.zeros(2 * N - 1)
        f_padded[:N] = f
        result = np.real(np.fft.ifft(self._kernel_fft * np.fft.fft(f_padded)))
        return result[:N] * self._dx

    def _rmatvec(self, g: NDArray) -> NDArray:
        """Adjoint W^T[g](y) = int W(x, y) g(x) dx.

        For symmetric kernels (W(x,y) = W(y,x)), this equals _matvec.
        """
        if getattr(self._kernel, "is_symmetric", False):
            return self._matvec(g)

        # General case: transpose of the kernel matrix
        g = np.asarray(g).ravel()
        if self._use_fft:
            # Translational kernels are symmetric by definition
            return self._matvec_fft(g)

        # Quadrature-based adjoint: W^T_{ki} = W(y_k, x_i) * w_k
        # This requires kernel evaluation at swapped arguments
        W_T = self._kernel.evaluate(self._quad_y, self._grid)  # (order, N)
        g_quad = np.zeros(len(self._quad_y))
        idx = self._interp_indices
        t = self._interp_t
        # Distribute g to quadrature points
        np.add.at(g_quad, np.arange(len(self._quad_y)), (1 - t) * g[idx - 1] + t * g[idx])
        return W_T.T @ (self._quad_w * g_quad)

    def as_dense(self) -> NDArray:
        """Return the full dense kernel matrix W_{ij} = int W(x_i, y) delta(y - x_j) dy.

        Approximation: W_{ij} ~ W(x_i, x_j) * dx_j for uniform grids.
        """
        dx = self._grid[1] - self._grid[0] if self._N > 1 else 1.0
        return self._kernel.evaluate(self._grid, self._grid) * dx

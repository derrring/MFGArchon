"""
Convolution coupling operator for spatial agent-agent interaction.

Computes the non-local game coupling

    F[m](x) = integral K(x - y) m(y) dy,

the first variation of the quadratic interaction energy
``F[m] = (1/2) integral integral K(x - y) m(x) m(y) dx dy``.

Two evaluation paths, selected from the geometry:

(a) **FFT path** — for a translation-invariant kernel ``K(x - y)`` on a regular
    Cartesian grid. The discrete convolution is evaluated with
    ``scipy.signal.fftconvolve`` in ``O(N log N)``.
(b) **Direct-quadrature path** — for irregular point sets (GFDM clouds) or when
    FFT is disabled. The dense matrix ``W_ij = K(|x_i - x_j|)`` is applied as
    ``F = W @ (m * w)`` with quadrature weights ``w`` (cell volumes).

On a regular grid the two paths agree to FFT round-off (the FFT path evaluates
the same discrete sum ``sum_j K(x_i - x_j) m_j * cell_volume``).

Inherits :class:`scipy.sparse.linalg.LinearOperator` for matrix-free use in
iterative solvers and operator algebra (``F @ m``, ``F.T @ g``).

Issue #1023: ``operators/interaction/`` subpackage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import fftconvolve
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from .kernels import RadialKernel


class ConvolutionCouplingOperator(LinearOperator):
    """Non-local interaction operator ``F[m](x) = integral K(x - y) m(y) dy``.

    Construct in one of two mutually exclusive geometry modes:

    - **Regular grid**: pass ``grid_shape`` and ``spacings``. The cell volume is
      ``prod(spacings)``. FFT is used when the kernel is translation-invariant
      (the default for radial kernels) unless ``use_fft=False`` forces the
      direct path.
    - **Irregular cloud**: pass ``points`` (shape ``(N,)`` or ``(N, d)``) and a
      ``cell_volume`` (scalar quadrature weight, or per-point array). Only the
      direct path is available; ``use_fft=True`` raises.

    Parameters
    ----------
    kernel : RadialKernel
        Radial interaction kernel ``K(r)``. Must provide ``matrix(points)`` and,
        for the FFT path, be translation-invariant (``is_translational``).
    points : NDArray | None
        Irregular point set, shape ``(N,)`` or ``(N, d)``. Mutually exclusive
        with ``grid_shape``.
    grid_shape : tuple[int, ...] | None
        Regular Cartesian grid shape ``(n_0, ..., n_{d-1})``. Mutually exclusive
        with ``points``.
    spacings : Sequence[float] | None
        Grid spacings ``(dx_0, ..., dx_{d-1})``; required with ``grid_shape``.
    cell_volume : float | NDArray | None
        Quadrature weight for the irregular path (scalar or shape ``(N,)``).
        Required with ``points``; ignored for regular grids (computed from
        ``spacings``).
    use_fft : bool | None
        Force/disable FFT. ``None`` auto-selects (FFT on a regular grid with a
        translation-invariant kernel).

    Example
    -------
    >>> import numpy as np
    >>> from mfgarchon.operators.interaction.kernels import GaussianKernel
    >>> K = GaussianKernel(amplitude=1.0, length_scale=0.1)
    >>> F = ConvolutionCouplingOperator(K, grid_shape=(128,), spacings=[1 / 127])
    >>> m = np.exp(-((np.linspace(0, 1, 128) - 0.5) ** 2) / 0.02)
    >>> Fm = F @ m  # = integral K(x - y) m(y) dy
    """

    def __init__(
        self,
        kernel: RadialKernel,
        *,
        points: NDArray | None = None,
        grid_shape: tuple[int, ...] | None = None,
        spacings: Sequence[float] | None = None,
        cell_volume: float | NDArray | None = None,
        use_fft: bool | None = None,
    ):
        regular = grid_shape is not None
        if regular and points is not None:
            raise ValueError("provide either grid_shape (regular) or points (irregular), not both")
        if not regular and points is None:
            raise ValueError("must provide grid_shape+spacings (regular) or points (irregular)")

        self._kernel = kernel
        self._W: NDArray | None = None  # dense matrix, built lazily

        if regular:
            if spacings is None:
                raise ValueError("regular grid requires spacings")
            if cell_volume is not None:
                raise ValueError("cell_volume is computed from spacings for a regular grid")
            self._grid_shape = tuple(int(n) for n in grid_shape)
            self._spacings = tuple(float(s) for s in spacings)
            if len(self._grid_shape) != len(self._spacings):
                raise ValueError("grid_shape and spacings must have the same length")
            N = int(np.prod(self._grid_shape))
            self._cell_volume = float(np.prod(self._spacings))
            self._w = np.full(N, self._cell_volume)
            self._points = self._build_grid_points()

            if use_fft is None:
                self._use_fft = bool(getattr(kernel, "is_translational", False))
            else:
                self._use_fft = bool(use_fft)
            if self._use_fft and not getattr(kernel, "is_translational", False):
                raise ValueError("FFT path requires a translation-invariant kernel")

            if self._use_fft:
                self._kernel_samples = self._build_kernel_samples()
            else:
                self._W = kernel.matrix(self._points)
        else:
            if cell_volume is None:
                raise ValueError("irregular point set requires cell_volume")
            if use_fft:
                raise ValueError("FFT path requires a regular grid; got irregular points")
            pts = np.asarray(points, dtype=float)
            self._points = pts
            N = pts.shape[0]
            self._grid_shape = None
            self._spacings = None
            self._use_fft = False
            w = np.asarray(cell_volume, dtype=float)
            if w.ndim == 0:
                self._cell_volume = float(w)
                self._w = np.full(N, self._cell_volume)
            else:
                if w.shape != (N,):
                    raise ValueError(f"cell_volume array must have shape ({N},), got {w.shape}")
                self._cell_volume = None
                self._w = w
            self._W = kernel.matrix(self._points)

        super().__init__(dtype=np.float64, shape=(N, N))

    def _build_grid_points(self) -> NDArray:
        """Construct the regular grid point set, shape ``(N, d)`` (origin at 0)."""
        axes = [np.arange(n) * s for n, s in zip(self._grid_shape, self._spacings, strict=True)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([g.ravel() for g in mesh], axis=-1)

    def _build_kernel_samples(self) -> NDArray:
        """Sample ``K`` on the centred lag grid for FFT convolution.

        The lag grid has shape ``(2 n_0 - 1, ..., 2 n_{d-1} - 1)`` with the
        centre element at lag zero, matching ``fftconvolve(..., mode="same")``.
        """
        lag_axes = [np.arange(-(n - 1), n) * s for n, s in zip(self._grid_shape, self._spacings, strict=True)]
        mesh = np.meshgrid(*lag_axes, indexing="ij")
        r = np.sqrt(sum(g**2 for g in mesh))
        return self._kernel(r)

    def _matvec(self, m: NDArray) -> NDArray:
        """Compute ``F[m](x_i) = sum_j K(x_i - x_j) m_j * w_j``."""
        m = np.asarray(m, dtype=float).ravel()
        if self._use_fft:
            m_field = m.reshape(self._grid_shape)
            out = fftconvolve(m_field, self._kernel_samples, mode="same") * self._cell_volume
            return out.ravel()
        return self._dense_matrix() @ (m * self._w)

    def _rmatvec(self, g: NDArray) -> NDArray:
        """Adjoint ``F^T[g]_j = w_j * sum_i K(x_i - x_j) g_i``.

        Radial kernels are symmetric (``K(x_i - x_j) = K(x_j - x_i)``). With
        uniform weights the operator matrix ``W @ diag(w)`` is symmetric, so the
        adjoint coincides with :meth:`_matvec`; with per-point weights it is
        ``w * (W @ g)``.
        """
        g = np.asarray(g, dtype=float).ravel()
        if self._use_fft:
            # Uniform weights + symmetric kernel: self-adjoint.
            return self._matvec(g)
        return self._w * (self._dense_matrix() @ g)

    def _dense_matrix(self) -> NDArray:
        """Return the cached kernel matrix ``W_ij = K(|x_i - x_j|)`` (build if needed)."""
        if self._W is None:
            self._W = self._kernel.matrix(self._points)
        return self._W

    def as_dense(self) -> NDArray:
        """Return the full operator matrix ``M = W @ diag(w)`` (shape ``(N, N)``).

        ``F[m] = M @ m`` exactly. For a regular grid with uniform cell volume,
        ``M = cell_volume * W`` is symmetric.
        """
        return self._dense_matrix() * self._w[None, :]

    @property
    def points(self) -> NDArray:
        """The point set backing the operator, shape ``(N, d)``."""
        return self._points

    @property
    def cell_volume(self) -> float | NDArray:
        """Scalar cell volume (uniform weights) or per-point weight array."""
        return self._cell_volume if self._cell_volume is not None else self._w

    @property
    def uses_fft(self) -> bool:
        """Whether the operator evaluates ``matvec`` via FFT."""
        return self._use_fft

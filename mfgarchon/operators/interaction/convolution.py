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

Both paths weight node ``j`` by ``w_j``: the FFT path convolves ``m * w``. On a regular grid
``w`` is the trapezoid weights of its endpoint-inclusive axes (``quadrature_weights_nd``),
the measure a problem on that grid integrates with, so the two paths agree to FFT round-off
and both are second-order. The rectangle rule ``prod(spacings)`` they replace gave each end
node a full cell reaching past the domain, an O(dx) error (#1642).

The convolution is Euclidean on both paths: it does not wrap, so it is not the interaction on
a periodic domain. :meth:`ConvolutionCouplingOperator.from_problem` refuses a periodic axis.

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

    from mfgarchon.core.mfg_problem import MFGProblem

    from .kernels import RadialKernel


class ConvolutionCouplingOperator(LinearOperator):
    """Non-local interaction operator ``F[m](x) = integral K(x - y) m(y) dy``.

    Construct in one of two mutually exclusive geometry modes:

    - **Regular grid**: pass ``grid_shape`` and ``spacings``. The weights are the
      trapezoid weights of the axes ``arange(n) * s`` (`quadrature_weights_nd`),
      unless a per-node ``cell_volume`` is given. FFT is used when the kernel is
      translation-invariant (the default for radial kernels) unless
      ``use_fft=False`` forces the direct path.
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
        Quadrature weights. Required with ``points`` (scalar or shape ``(N,)``).
        On a regular grid, optional and per node (shape ``(N,)`` or
        ``grid_shape``); without it the trapezoid weights are used. A scalar is
        refused there: one weight for every node is the rectangle rule.
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

    On a problem, :meth:`from_problem` reads the grid, the problem's measure and its topology.
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
            self._grid_shape = tuple(int(n) for n in grid_shape)
            self._spacings = tuple(float(s) for s in spacings)
            if len(self._grid_shape) != len(self._spacings):
                raise ValueError("grid_shape and spacings must have the same length")
            N = int(np.prod(self._grid_shape))
            self._w = self._grid_weights(cell_volume, N)
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
            pts = np.array(points, dtype=float)  # a copy: the kernel matrix below is built from these points
            self._points = pts
            N = pts.shape[0]
            self._grid_shape = None
            self._spacings = None
            self._use_fft = False
            w = np.array(cell_volume, dtype=float)  # a copy, as on a regular grid
            if w.ndim == 0:
                self._w = np.full(N, float(w))
            else:
                if w.shape != (N,):
                    raise ValueError(f"cell_volume array must have shape ({N},), got {w.shape}")
                self._w = w
            self._W = kernel.matrix(self._points)

        super().__init__(dtype=np.float64, shape=(N, N))

    @classmethod
    def from_problem(
        cls, kernel: RadialKernel, problem: MFGProblem, *, use_fft: bool | None = None
    ) -> ConvolutionCouplingOperator:
        """The interaction on ``problem``'s grid, weighted by the problem's own measure (#1642).

        Reads three things the problem owns: its grid, a `TensorProductGrid`; its measure,
        :meth:`MFGProblem.spatial_measure`, whose ``weights`` become the operator's; and its topology,
        :meth:`MFGProblem.get_boundary_conditions` read face by face (`periodic_faces`). A uniform grid gets
        the regular mode, where the FFT path is available. A graded grid gets the point mode with the same
        weights, which is direct only.

        Refused by name:

        - a periodic axis. The convolution is Euclidean and does not wrap, so near one wall it would miss
          the mass just beyond the other;
        - a geometry that is not a `TensorProductGrid`. A point cloud's measure is a point average, not a
          quadrature, and its quadrature weights have no owner yet.
        """
        from mfgarchon.geometry.boundary.conditions import periodic_faces
        from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

        geometry = problem.geometry
        if not isinstance(geometry, TensorProductGrid):
            raise TypeError(
                f"ConvolutionCouplingOperator.from_problem: the problem's geometry is a {type(geometry).__name__}, "
                "not a TensorProductGrid. Only a grid's measure is a quadrature here; for a point set, construct the "
                "operator with points= and the cell_volume that is its quadrature (#1642)."
            )
        shape = tuple(geometry.get_grid_shape())
        periodic = periodic_faces(problem.get_boundary_conditions(), len(shape), geometry)
        wrapped = sorted({face // 2 for face, is_periodic in enumerate(periodic) if is_periodic})
        if wrapped:
            raise ValueError(
                f"ConvolutionCouplingOperator.from_problem: axis {wrapped} of the problem's grid is periodic, and "
                "this convolution is Euclidean: it does not wrap, so near one wall it would miss the mass just beyond "
                "the other. A periodic interaction is #2725; pose the problem on a non-periodic grid (#1642)."
            )
        weights = problem.spatial_measure().weights
        if geometry.is_uniform:
            return cls(
                kernel, grid_shape=shape, spacings=geometry.get_grid_spacing(), cell_volume=weights, use_fft=use_fft
            )
        mesh = np.meshgrid(*geometry.coordinates[: len(shape)], indexing="ij")
        points = np.stack([axis.ravel() for axis in mesh], axis=-1)
        return cls(kernel, points=points, cell_volume=np.asarray(weights, dtype=float).ravel(), use_fft=use_fft)

    def _grid_axes(self) -> list[NDArray]:
        """The regular grid's axes, ``arange(n) * s`` (origin at 0, endpoint-inclusive)."""
        return [np.arange(n) * s for n, s in zip(self._grid_shape, self._spacings, strict=True)]

    def _grid_weights(self, cell_volume: float | NDArray | None, N: int) -> NDArray:
        """The regular grid's per-node weights, flattened: ``cell_volume`` when given per node, else the trapezoid
        weights of its axes (#1642)."""
        if cell_volume is None:
            from mfgarchon.utils.numerical.quadrature import quadrature_weights_nd

            return quadrature_weights_nd(self._grid_axes()).ravel()
        w = np.array(cell_volume, dtype=float)  # a copy: the operator's weights are not the caller's array
        shapes = f"({N},)" if self._grid_shape == (N,) else f"({N},) or {self._grid_shape}"
        if w.ndim == 0:
            raise ValueError(
                f"cell_volume on a regular grid is per node, shape {shapes}; got the scalar "
                f"{float(w)!r}. One weight for every node is the rectangle rule, which gives each end node a full cell "
                "past the domain; leave cell_volume out for the trapezoid weights (#1642)."
            )
        if w.shape not in ((N,), self._grid_shape):
            raise ValueError(f"cell_volume on a regular grid is per node, shape {shapes}; got {w.shape}.")
        return w.ravel()

    def _build_grid_points(self) -> NDArray:
        """Construct the regular grid point set, shape ``(N, d)`` (origin at 0)."""
        mesh = np.meshgrid(*self._grid_axes(), indexing="ij")
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
            return self._fft_apply(m * self._w)
        return self._dense_matrix() @ (m * self._w)

    def _rmatvec(self, g: NDArray) -> NDArray:
        """Adjoint ``F^T[g]_j = w_j * sum_i K(x_i - x_j) g_i``.

        Radial kernels are symmetric (``K(x_i - x_j) = K(x_j - x_i)``), so the adjoint of
        ``W @ diag(w)`` is ``w * (W @ g)`` on both paths. The weights are not uniform on a
        regular grid either (the trapezoid ends), so the operator is not self-adjoint.
        """
        g = np.asarray(g, dtype=float).ravel()
        if self._use_fft:
            return self._w * self._fft_apply(g)
        return self._w * (self._dense_matrix() @ g)

    def _fft_apply(self, v: NDArray) -> NDArray:
        """``sum_j K(x_i - x_j) v_j`` on the regular grid, by FFT (Euclidean: no wrap)."""
        return fftconvolve(v.reshape(self._grid_shape), self._kernel_samples, mode="same").ravel()

    def _dense_matrix(self) -> NDArray:
        """Return the cached kernel matrix ``W_ij = K(|x_i - x_j|)`` (build if needed)."""
        if self._W is None:
            self._W = self._kernel.matrix(self._points)
        return self._W

    def as_dense(self) -> NDArray:
        """Return the full operator matrix ``M = W @ diag(w)`` (shape ``(N, N)``).

        ``F[m] = M @ m`` exactly. With per-node weights -- a regular grid's trapezoid
        weights included -- it is not symmetric.
        """
        return self._dense_matrix() * self._w[None, :]

    def kernel_matrix(self) -> NDArray:
        """Return the raw kernel matrix ``W_ij = K(|x_i - x_j|)`` (shape ``(N, N)``).

        Unweighted and symmetric for a radial kernel -- the quadrature factor
        lives in :attr:`weights`, so ``as_dense() == kernel_matrix() * weights``.
        This is the second variation ``delta^2 F / delta m^2`` of the quadratic
        interaction energy. Returns a copy; allocates ``(N, N)`` even on the FFT
        path, which otherwise never forms a dense matrix.
        """
        return self._dense_matrix().copy()

    @property
    def points(self) -> NDArray:
        """The point set backing the operator: shape ``(N, d)`` on a regular grid, whose first node is the origin; the
        ``points`` given, copied, in point mode. Only differences between points enter the kernel."""
        return self._points

    @property
    def weights(self) -> NDArray:
        """Quadrature weights of the point set, shape ``(N,)``.

        Always an array, whether the operator was built from ``spacings`` (the
        trapezoid weights, or a per-node ``cell_volume``) or from a scalar/array
        ``cell_volume`` (scattered cloud). The single spelling of this quantity: a
        scalar-vs-array polymorphic accessor invites two conventions for one number.
        """
        return self._w

    @property
    def uses_fft(self) -> bool:
        """Whether the operator evaluates ``matvec`` via FFT."""
        return self._use_fft

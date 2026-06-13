"""
Graphon kernel definitions for continuum-limit network MFG.

A graphon W: [0,1]^2 -> [0,1] is a symmetric measurable function that
encodes the connection probability between agents at positions x, y
in the continuum limit of dense graph sequences.

Issue #963: GraphonCouplingOperator.

References:
    Lovasz (2012), "Large Networks and Graph Limits"
    Caines & Huang (2021), "Graphon Mean Field Games"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class GraphonKernel(Protocol):
    """Protocol for graphon kernel functions W(x, y).

    A graphon is symmetric: W(x, y) = W(y, x) for all x, y in [0,1].
    Values are in [0, 1] (connection probability).
    """

    def evaluate(self, x: NDArray, y: NDArray) -> NDArray:
        """Evaluate W(x, y).

        Args:
            x: First argument, shape (N,).
            y: Second argument, shape (M,).

        Returns:
            W(x_i, y_j), shape (N, M).
        """
        ...

    @property
    def is_symmetric(self) -> bool:
        """Whether W(x,y) = W(y,x)."""
        ...


class ConstantGraphon:
    """Erdos-Renyi graphon: W(x, y) = p for all x, y.

    The continuum limit of G(n, p) random graphs.

    Parameters
    ----------
    p : float
        Connection probability, in [0, 1].
    """

    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0,1], got {p}")
        self._p = p

    def evaluate(self, x: NDArray, y: NDArray) -> NDArray:
        return np.full((len(x), len(y)), self._p)

    @property
    def is_symmetric(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"ConstantGraphon(p={self._p})"


class StochasticBlockModelGraphon:
    """Stochastic Block Model graphon: piecewise constant on blocks.

    W(x, y) = B[k(x), k(y)] where k(x) = community assignment
    based on position in [0, 1].

    Parameters
    ----------
    block_matrix : NDArray
        Connection probabilities between blocks, shape (K, K). Symmetric.
    block_sizes : NDArray | None
        Relative sizes of each block, shape (K,). Must sum to 1.
        Default: equal sizes.
    """

    def __init__(self, block_matrix: NDArray, block_sizes: NDArray | None = None):
        B = np.asarray(block_matrix, dtype=float)
        K = B.shape[0]
        if B.shape != (K, K):
            raise ValueError(f"block_matrix must be square, got {B.shape}")
        self._B = B
        self._K = K

        if block_sizes is None:
            self._sizes = np.ones(K) / K
        else:
            self._sizes = np.asarray(block_sizes, dtype=float)
            self._sizes = self._sizes / self._sizes.sum()

        # Cumulative boundaries for block assignment
        self._boundaries = np.cumsum(self._sizes)

    def _assign_block(self, x: NDArray) -> NDArray:
        """Assign block index to each position x in [0, 1]."""
        return np.searchsorted(self._boundaries, x, side="left").clip(0, self._K - 1)

    def evaluate(self, x: NDArray, y: NDArray) -> NDArray:
        kx = self._assign_block(np.asarray(x))
        ky = self._assign_block(np.asarray(y))
        return self._B[kx[:, None], ky[None, :]]

    @property
    def is_symmetric(self) -> bool:
        return np.allclose(self._B, self._B.T)

    def __repr__(self) -> str:
        return f"StochasticBlockModelGraphon(K={self._K})"


class GeometricGraphon:
    """Geometric graphon: W(x, y) = phi(|x - y|).

    Connection probability depends on distance. This is the
    continuum limit of geometric random graphs.

    Supports FFT-based fast convolution (translational invariance).

    Parameters
    ----------
    kernel_fn : callable
        phi(d) -> probability, where d = |x - y|. Must be in [0, 1].
    bandwidth : float
        Scale parameter for the kernel. Default 0.1.
    """

    def __init__(self, bandwidth: float = 0.1):
        self._h = bandwidth

    def evaluate(self, x: NDArray, y: NDArray) -> NDArray:
        dist = np.abs(np.asarray(x)[:, None] - np.asarray(y)[None, :])
        return np.exp(-((dist / self._h) ** 2))

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def is_translational(self) -> bool:
        """Whether W(x,y) = phi(|x-y|) — enables FFT convolution."""
        return True

    def __repr__(self) -> str:
        return f"GeometricGraphon(h={self._h})"

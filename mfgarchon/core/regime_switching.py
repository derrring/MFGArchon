"""
Regime switching configuration for Markov-switching MFG.

Defines the transition rate matrix Q for K regimes in a continuous-time
Markov chain. Used by RegimeSwitchingIterator to solve coupled HJB-FP
systems with inter-regime transition terms.

Issue #925: Part of Phase 2 (Generalized PDE & Institutional MFG Plan).

Mathematical background:
    The regime process xi_t in {1, ..., K} has generator matrix Q:
      Q[k,j] >= 0 for k != j  (transition rate from k to j)
      Q[k,k] = -sum_{j!=k} Q[k,j]  (rows sum to zero)

    In the MFG system, regime switching adds cross-terms:
      HJB: lambda_{kj}(v^k - v^j) coupling between value functions
      FP:  lambda_{jk} m^j - lambda_{kj} m^k mass transfer between populations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class RegimeSwitchingConfig:
    """Markov chain generator matrix for regime switching.

    Parameters
    ----------
    transition_matrix : NDArray
        K x K generator matrix Q where:
        - Q[k,j] >= 0 for k != j (transition rate from regime k to regime j)
        - Q[k,k] = -sum_{j!=k} Q[k,j] (each row sums to zero)
    regime_names : list[str] | None
        Optional descriptive names for regimes (e.g., ["high_protection", "low_protection"]).

    Example
    -------
    Two-state regime switching (high/low property rights protection):

    >>> Q = np.array([[-0.1, 0.1],   # high -> low at rate 0.1
    ...               [0.2, -0.2]])   # low -> high at rate 0.2
    >>> config = RegimeSwitchingConfig(
    ...     transition_matrix=Q,
    ...     regime_names=["high_protection", "low_protection"],
    ... )
    >>> config.validate()
    >>> pi = config.stationary_distribution()  # [2/3, 1/3]
    """

    transition_matrix: NDArray
    regime_names: list[str] | None = field(default=None)

    @property
    def n_regimes(self) -> int:
        """Number of regimes K."""
        return self.transition_matrix.shape[0]

    def validate(self) -> None:
        """Check that Q is a valid generator matrix.

        Raises
        ------
        ValueError
            If Q is not square, has negative off-diagonal entries,
            or rows don't sum to zero.
        """
        Q = self.transition_matrix
        K = Q.shape[0]

        if Q.shape != (K, K):
            msg = f"Transition matrix must be square, got shape {Q.shape}"
            raise ValueError(msg)

        # Off-diagonal entries must be non-negative
        off_diag = Q - np.diag(np.diag(Q))
        if np.any(off_diag < -1e-12):
            msg = f"Off-diagonal entries must be non-negative, min={off_diag.min():.2e}"
            raise ValueError(msg)

        # Rows must sum to zero
        row_sums = Q.sum(axis=1)
        if not np.allclose(row_sums, 0, atol=1e-10):
            msg = f"Rows must sum to zero, got row sums: {row_sums}"
            raise ValueError(msg)

        if self.regime_names is not None and len(self.regime_names) != K:
            msg = f"regime_names length ({len(self.regime_names)}) must match n_regimes ({K})"
            raise ValueError(msg)

    def stationary_distribution(self) -> NDArray:
        """Solve pi @ Q = 0, sum(pi) = 1 for the stationary distribution.

        Returns
        -------
        NDArray
            Stationary probability vector, shape (K,).
        """
        from scipy.linalg import null_space

        ns = null_space(self.transition_matrix.T)
        if ns.shape[1] == 0:
            msg = "Generator matrix has no null space — check validity"
            raise ValueError(msg)
        pi = ns[:, 0].real
        pi = pi / pi.sum()
        # Ensure non-negative (numerical artifacts)
        pi = np.maximum(pi, 0.0)
        pi = pi / pi.sum()
        return pi

    def transition_rate(self, from_regime: int, to_regime: int) -> float:
        """Get transition rate from regime k to regime j.

        Parameters
        ----------
        from_regime : int
            Source regime index.
        to_regime : int
            Target regime index.

        Returns
        -------
        float
            Transition rate Q[from_regime, to_regime].
        """
        return float(self.transition_matrix[from_regime, to_regime])

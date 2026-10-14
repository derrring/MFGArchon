"""
Array norms.

The checker classes that used to live here are removed (#2648, ruling 5):
- `ConvergenceConfig`, `MFGConvergenceChecker`, `HJBConvergenceChecker`, `FPConvergenceChecker`;
- the `ConvergenceChecker` protocol and `create_convergence_checker`.

No solver took a `ConvergenceConfig`, and eight of its fourteen fields had no reader, `_history` among them.
The convergence owner is `sweep_change` (`convergence_metrics.py`), which the coupling iterators call. #2657
records one convergence configuration across solvers, built on it.

`compute_norm` stays. It is #2566's row: a plain array norm that chooses its own measure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_norm(
    diff: NDArray[np.floating],
    method: Literal["l1", "l2", "linf"] = "l2",
) -> float:
    """
    Compute norm of difference array.

    Args:
        diff: Difference array (new - old)
        method: Norm type ('l1', 'l2', 'linf')

    Returns:
        Scalar norm value
    """
    if method == "l1":
        return float(np.sum(np.abs(diff)))
    elif method == "linf":
        return float(np.max(np.abs(diff)))
    else:  # l2 default
        return float(np.sqrt(np.sum(diff**2)))

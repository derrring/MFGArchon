"""Single-source batch Hamiltonian evaluation for HJB solvers (re-scope of Issue #1071).

The Hamiltonian *value* and its *gradient* are already single-source (``HamiltonianBase``
in ``core/hamiltonian.py``, reached via ``problem.hamiltonian_class``). What was duplicated
is the per-solver *evaluation glue* -- every HJB solver inlined the same
``np.asarray(H_class(x, m, p, t=t), dtype=float)`` batch call. This module is the one home
for that call, so a future change to the batch contract (dtype, shape handling, NaN policy)
happens in exactly one place.

These are byte-identical extractions of the inline expressions they replace: the callers
still own their own ``.ravel()`` / reshape / sign conventions (e.g. ``alpha* = -dp``) and the
discrete operators (gradient, Laplacian) they feed in. This is Layer A of #1071; the
residual/Jacobian *assembly* harness is Layer B.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.core.hamiltonian import HamiltonianBase


def eval_H_batch(H_class: HamiltonianBase, x: NDArray, m: NDArray, p: NDArray, t: float) -> NDArray:
    """Evaluate the Hamiltonian value ``H(x, m, p, t)`` over a batch of points.

    Returns a float ``ndarray`` shaped as ``H_class`` returns it (callers ``.ravel()`` or
    reshape as their assembly needs). Byte-identical to the inline
    ``np.asarray(H_class(x, m, p, t=t), dtype=float)`` it replaces.
    """
    return np.asarray(H_class(x, m, p, t=t), dtype=float)


def eval_dH_dp_batch(H_class: HamiltonianBase, x: NDArray, m: NDArray, p: NDArray, t: float) -> NDArray:
    """Evaluate the Hamiltonian gradient ``∂H/∂p(x, m, p, t)`` over a batch of points.

    Returns a float ``ndarray`` as ``H_class.dp`` returns it. Callers keep their own sign
    convention (the FP drift is ``alpha* = -∂H/∂p``, so several callers negate the result).
    Byte-identical to the inline ``np.asarray(H_class.dp(x, m, p, t=t), dtype=float)``.
    """
    return np.asarray(H_class.dp(x, m, p, t=t), dtype=float)

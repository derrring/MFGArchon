"""Finite difference operators for FP equation discretization.

This module provides common utilities for the FDM discretization
of the Fokker-Planck equation.

Module structure per issue #388:
    fp_fdm_operators.py - Common utilities

Advection Scheme Files:
    fp_fdm_alg_divergence_centered.py - divergence_centered (conservative, oscillates)
    fp_fdm_alg_divergence_upwind.py   - divergence_upwind (conservative via telescoping)

Scheme Comparison:
    | Scheme             | PDE Form   | Spatial | Conservative | Stable |
    |--------------------|------------|---------|--------------|--------|
    | divergence_centered| div(v*m)   | Central | YES (flux)   | Pe<2   |
    | divergence_upwind  | div(v*m)   | Upwind  | YES (flux)   | Always |

Functions:
    is_boundary_point: Utility to check if a point is on the boundary
    ito_diffusion: The FP diffusion of a spatially varying A, d_i d_j(A_ij m) (#2841)

Note:
    - Boundary condition enforcement is in fp_fdm_bc.py
    - Advection term computation is in fp_fdm_advection.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def ito_diffusion(unit: Mapping[Any, Any], coefficients: Mapping[Any, float | np.ndarray]) -> Any:
    """The Fokker-Planck diffusion of a spatially varying A, in Ito form: ``sum_ij d_i d_j (A_ij m)`` (#2841).

    The FP equation of ``dX = alpha dt + Sigma dW`` is ``d_t m + div(alpha m) = d_i d_j (A_ij m)`` with
    ``A = Sigma Sigma^T / 2``, the adjoint of the HJB's ``tr(A D^2 u)`` (CONVENTIONS § 5; user ruling 2026-10-30).
    For a field, ``A = D(x) I``, it is ``Delta(D m)``; neither ``D Delta m`` nor ``div(D grad m)`` is.

    ``unit`` maps each component key to the caller's own unit second-difference operator for it, with its own wall
    closure, and ``coefficients`` maps the same keys to that component of A at every node. A field passes one key
    (its Laplacian with D); a tensor passes ``(i, j)`` for ``i <= j``, the operator for ``i < j`` being
    ``d_i d_j + d_j d_i`` so that both off-diagonal entries are counted. The result is ``sum_k U_k diag(a_k)``: the
    operator acts on ``a_k m``, never on ``m`` scaled afterwards. A sparse ``U_k`` gives a sparse matrix; a callable
    one, applied to arrays of the grid's shape, gives a callable ``m -> sum_k U_k(a_k m)``.

    Each ``U_k`` that conserves mass (``w^T U_k = 0``) leaves the sum conserving it, whatever A is; with no drift,
    ``m = c / D`` is a discrete fixed point of a field, since every ``U_k`` takes a constant to 0.
    """
    if unit.keys() != coefficients.keys():
        raise ValueError(
            f"ito_diffusion: unit keys {sorted(map(str, unit))} != coefficient keys {sorted(map(str, coefficients))}"
        )
    sparse_ops = [sparse.issparse(op) for op in unit.values()]
    if all(sparse_ops):
        terms = [op @ sparse.diags(np.asarray(coefficients[k], dtype=float).ravel()) for k, op in unit.items()]
        return sparse.csr_matrix(sum(terms[1:], terms[0]))
    if any(sparse_ops):
        raise TypeError("ito_diffusion: the unit operators must all be sparse matrices or all be callables")
    ops: dict[Any, Callable[[np.ndarray], np.ndarray]] = dict(unit)

    def apply(m: np.ndarray) -> np.ndarray:
        total = np.zeros(np.shape(m))
        for k, op in ops.items():
            total = total + op(np.asarray(coefficients[k], dtype=float) * m)
        return total

    return apply


def is_boundary_point(multi_idx: tuple[int, ...], shape: tuple[int, ...], ndim: int) -> bool:
    """Check if a grid point is on the boundary."""
    return any(multi_idx[d] == 0 or multi_idx[d] == shape[d] - 1 for d in range(ndim))

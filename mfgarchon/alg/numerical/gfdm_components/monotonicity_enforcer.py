"""
Checks of the M-matrix property on an assembled GFDM HJB system (Issue #1074).

The derivative-coefficient QP this module was named for, ``MonotonicityEnforcer``, is deleted (#1689):
it constrained the solution's Taylor coefficients, not the stencil's weights, so it was not a monotone
scheme. What is left measures monotonicity rather than enforcing it.

Mathematical Background:
    For a monotone finite difference scheme, the discretization matrix must
    have the M-matrix property:
    - Diagonal elements: a_ii <= 0
    - Off-diagonal elements: a_ij >= 0 for i != j

    This ensures convergence to the viscosity solution (Barles-Souganidis 1991).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def verify_assembled_m_matrix(
    matrix: Any,
    interior_indices: np.ndarray | None = None,
    atol: float = 1e-9,
) -> dict[str, Any]:
    r"""Verify the M-matrix property of the *assembled* HJB iteration matrix (Issue #1074).

    This routine exists *precisely because* per-stencil SOCP feasibility is not the same as
    the assembled-matrix M-matrix property (Issue #1074). The per-stencil SOCP constraints
    hold on the raw Laplacian weights at each collocation point. They do **not** imply that the assembled Newton iteration matrix
    $J = I/\Delta t - D\,L + \alpha\cdot D_{\mathrm{grad}}$ is an M-matrix: the signed drift
    term $\alpha\cdot D_{\mathrm{grad}}$ can flip an off-diagonal positive once $|\alpha|$ is
    large relative to $D$ (a Péclet-like condition; see :func:`critical_drift_for_dmp`). This
    checks the assembled system directly.

    The ``joint_socp`` scheme therefore does not provide an a-priori discrete-maximum-principle
    (DMP) guarantee for the assembled system. The DMP claim for a *given* problem and
    discretization should be backed by running this check (the scoped claim): a passing
    ``is_m_matrix`` result certifies the assembled M-matrix / DMP property for that specific run, on the rows it
    checked. Restricted to ``interior_indices`` it says nothing of the boundary rows, which are the BC rows, and a
    no-flux or Neumann one is not an M-matrix row (#1689 comment 6246936251).

    Args:
        matrix: Assembled iteration matrix (any scipy sparse format).
        interior_indices: Restrict the check to these rows. Boundary rows carry BC structure
            that the solver applies separately and that is not expected to be M-matrix
            pre-BC, so pass the solver's ``interior_indices`` to test the scheme's claim.
        atol: Sign/zero tolerance.

    Returns:
        dict with ``is_m_matrix`` (bool) and the diagnostics ``n_bad_diag`` (non-positive
        diagonals), ``n_positive_offdiag`` (wrong-sign off-diagonals), ``worst_positive_offdiag``
        (max wrong-sign off-diagonal magnitude), ``worst_dominance_margin`` (max of
        $\sum_{j\ne i}|J_{ij}| - J_{ii}$ over checked rows; $\le 0$ means weakly diagonally
        dominant).
    """
    J = matrix.tocsr()
    n = J.shape[0]
    rows = np.arange(n) if interior_indices is None else np.asarray(interior_indices)

    n_bad_diag = 0
    n_positive_offdiag = 0
    worst_positive_offdiag = 0.0
    worst_dominance_margin = -np.inf
    for i in rows:
        row = J.getrow(i).toarray().ravel()
        d_i = row[i]
        off = row.copy()
        off[i] = 0.0
        if d_i <= atol:
            n_bad_diag += 1
        positive = off[off > atol]
        if positive.size:
            n_positive_offdiag += int(positive.size)
            worst_positive_offdiag = max(worst_positive_offdiag, float(positive.max()))
        worst_dominance_margin = max(worst_dominance_margin, float(np.abs(off).sum() - d_i))

    is_m_matrix = n_bad_diag == 0 and n_positive_offdiag == 0 and worst_dominance_margin <= atol
    return {
        "is_m_matrix": is_m_matrix,
        "n_bad_diag": n_bad_diag,
        "n_positive_offdiag": n_positive_offdiag,
        "worst_positive_offdiag": worst_positive_offdiag,
        "worst_dominance_margin": worst_dominance_margin,
    }


def critical_drift_for_dmp(
    d_lap: Any,
    d_grad: list,
    diffusion_coeff: float | np.ndarray,
    interior_indices: np.ndarray | None = None,
    atol: float = 1e-12,
) -> float:
    r"""Largest drift magnitude $|\alpha|$ for which the assembled interior matrix keeps its
    M-matrix off-diagonal sign (Issue #1074).

    For interior off-diagonal $(i, j)$ the assembled entry is
    $\sum_d \alpha_d (D_{\mathrm{grad}})_{d,ij} - D\,L_{ij}$. The worst-case unit-$|\alpha|$
    direction makes the drift part $|\alpha|\,\lVert (D_{\mathrm{grad}})_{:,ij}\rVert$
    (Cauchy-Schwarz), so the M-matrix sign holds iff
    $|\alpha| \le D_i\,L_{ij}/\lVert (D_{\mathrm{grad}})_{:,ij}\rVert$, where
    ``diffusion_coeff`` is either one scalar $D$ or a row-wise field $(D_i)$.

    When $L_{ij} \le 0$ (relaxed-SOCP slack row; Issue #1253) and $\lVert D_{:,j}\rVert > 0$,
    the right-hand side is $\le 0$: the off-diagonal entry is already non-negative at zero
    drift, so $\alpha_{\mathrm{crit}}$ collapses to 0 (or below) for that edge. The
    iteration therefore covers **all** edges with nonzero gradient weight, treating
    $L_{ij} \le atol$ edges as immediate-violation contributors.

    Returns ``inf`` when no edge constrains the sign (e.g. a zero gradient operator).
    Returns a value $\le 0$ when any assembled off-diagonal is already non-negative at zero
    drift (assembled matrix non-M regardless of advection).
    """
    L = d_lap.tocsr()
    grads = [g.tocsr() for g in d_grad]
    n = L.shape[0]
    rows = np.arange(n) if interior_indices is None else np.asarray(interior_indices)
    diffusion = np.asarray(diffusion_coeff, dtype=float)
    if diffusion.ndim > 0 and diffusion.shape != (n,):
        raise ValueError(
            f"diffusion_coeff must be a scalar or a row-wise field with shape ({n},), got {diffusion.shape}."
        )

    alpha_crit = np.inf
    for i in rows:
        diffusion_i = float(diffusion) if diffusion.ndim == 0 else float(diffusion[i])
        l_row = L.getrow(i).toarray().ravel()
        g_rows = [g.getrow(i).toarray().ravel() for g in grads]
        # Issue #1253 2026-06-10 audit: iterate over ALL off-diagonal edges with
        # nonzero gradient weight, not only those with L_ij > atol. A negative L_ij
        # (relaxed-SOCP slack) means D*L_ij < 0, so the assembled off-diagonal entry
        # alpha*D_grad_ij - D*L_ij > 0 at zero drift → alpha_crit → 0 or negative.
        # Iterating only l_row > atol silently drops these edges and over-estimates
        # alpha_crit (false-negative DMP warning).
        g_norms = np.sqrt(sum(gr**2 for gr in g_rows))  # vectorized over all n columns
        for j in np.nonzero(g_norms > atol)[0]:
            if j == i:
                continue
            alpha_crit = min(alpha_crit, diffusion_i * float(l_row[j]) / float(g_norms[j]))
    return alpha_crit

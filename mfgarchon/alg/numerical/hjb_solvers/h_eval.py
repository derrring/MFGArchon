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

from mfgarchon.core.hamiltonian import HEvalState
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.sparse import spmatrix

    from mfgarchon.core.hamiltonian import HamiltonianBase


def eval_H_batch(H_class: HamiltonianBase, x: NDArray, m: NDArray, p: NDArray, t: float) -> NDArray:
    """Evaluate the Hamiltonian value ``H(x, m, p, t)`` over a batch of points.

    Thin shim over the single-source primitive ``H_class.evaluate_H`` (Issue #1071):
    this is no longer a parallel implementation, it delegates to the method on the
    Hamiltonian so the batch contract has exactly one home. Byte-identical to the
    inline ``np.asarray(H_class(x, m, p, t=t), dtype=float)`` it replaced; callers
    ``.ravel()`` / reshape as their assembly needs.
    """
    return H_class.evaluate_H(HEvalState(x=x, p=p, m=m, t=t))


def eval_dH_dp_batch(H_class: HamiltonianBase, x: NDArray, m: NDArray, p: NDArray, t: float) -> NDArray:
    """Evaluate the Hamiltonian gradient ``∂H/∂p(x, m, p, t)`` over a batch of points.

    Thin shim over the single-source primitive ``H_class.evaluate_dp`` (Issue #1071);
    delegates to the method on the Hamiltonian rather than re-implementing the batch
    call. Callers keep their own sign convention (the FP drift is ``alpha* = -∂H/∂p``,
    so several callers negate the result). Byte-identical to the inline
    ``np.asarray(H_class.dp(x, m, p, t=t), dtype=float)``.
    """
    return H_class.evaluate_dp(HEvalState(x=x, p=p, m=m, t=t))


def assemble_hjb_residual(
    *,
    H_class: HamiltonianBase,
    x: NDArray,
    m: NDArray,
    p: NDArray,
    lap_u: NDArray,
    sigma: float | NDArray,
    t: float,
    u_t: NDArray,
    running_cost: NDArray | None = None,
) -> NDArray:
    r"""Assemble the implicit-backward-Euler HJB residual (Layer B of #1071).

    Returns ``-u_t + H(+running_cost) - D·lap_u`` with ``D = diffusion_from_volatility(sigma)``,
    so the diffusion-term convention (Issue #1073/#811: ``D = σ²/2``) lives in one place. The
    caller supplies its own discrete operators (gradient ``p``, Laplacian ``lap_u``) and the
    time-derivative ``u_t = (u^{n+1}-u^n)/dt``; it owns its own framing -- this is the implicit
    residual ``-u_t + H - D·lap``, NOT the WENO explicit-RHS framing ``-H + D·Δu``. Byte-identical
    to the inline gfdm expression it replaces.
    """
    H = eval_H_batch(H_class, x, m, p, t)
    if running_cost is not None:
        H = H + running_cost
    return -u_t + H - diffusion_from_volatility(sigma) * lap_u


def assemble_hjb_jacobian_diag(
    *,
    H_class: HamiltonianBase,
    x: NDArray,
    m: NDArray,
    p: NDArray,
    sigma: float | NDArray,
    t: float,
    dt: float,
    D_grad: list,
    D_lap: spmatrix,
) -> spmatrix:
    r"""Assemble the sparse HJB Newton Jacobian for the implicit residual (Layer B of #1071).

    Returns ``(1/dt)I + Σ_d diag(∂H/∂p_d) @ D_grad[d] - D·D_lap`` with
    ``D = diffusion_from_volatility(sigma)``. ``D_grad`` (per-dimension first-derivative
    matrices) and ``D_lap`` (Laplacian matrix) are the caller's discrete operators -- scattered
    GFDM stencils, structured FDM matrices, etc. Byte-identical to the inline gfdm assembly.
    """
    from scipy.sparse import diags, eye

    dH_dp = eval_dH_dp_batch(H_class, x, m, p, t)
    n = D_lap.shape[0]
    jacobian = (1.0 / dt) * eye(n, format="csr")
    for dim in range(dH_dp.shape[1]):
        jacobian = jacobian + diags(dH_dp[:, dim], format="csr") @ D_grad[dim]
    return jacobian - diffusion_from_volatility(sigma) * D_lap

"""Manufactured solutions: one owner for turning an exact pair into MFG source terms.

The method of manufactured solutions is the only EXTERNAL oracle a scheme has. Pick exact
``u*`` and ``m*``, substitute them into the MFG system, and whatever is left over is fed back as a
source term; the scheme must then reproduce ``u*`` and ``m*`` and its error must fall at its order.
Everything else this repository can check compares one implementation against another, and it has
repeatedly measured two implementations agreeing while both were wrong.

WHAT THIS OWNS, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
Nine fixtures currently manufacture a pair, and each **assembles its own source** from its own
derivatives. The differentiation is not the duplicated part -- three lines of ``cos`` derivatives
are easy and each fixture gets them right. What is restated nine times is the ASSEMBLY:

    S_hjb = -d_t u + H(x, grad u, m) - (1/2) tr(Sigma . Hess u)
    S_fp  =  d_t m - div(m * dH/dp)   - (1/2) tr(Sigma . Hess m)

and that is where a sign goes wrong quietly (this repository already has #1645, a DPP sign bug).
So a :class:`ManufacturedPair` carries its own analytic derivatives -- written once per family, not
once per fixture -- and the two functions below own the assembly.

``H`` is taken as the *problem's own* ``HamiltonianBase``, not re-implemented here. That is the
point: a source assembled from a different Hamiltonian than the solver integrates would verify a
different equation, and the test would pass while proving nothing about the solver.

WHY THE HESSIAN AND NOT THE LAPLACIAN
-------------------------------------
The diffusion term is written ``(1/2) tr(Sigma . Hess)``, not ``(sigma^2 / 2) Lap``. For isotropic
``Sigma = sigma^2 I`` the two agree exactly (verified to 1.1e-16), so nothing is lost. What is
gained is the off-diagonal: ``sum_{i != j} sigma_ij d2u/dx_i dx_j`` is precisely the cross-derivative
term of #2198, and a Laplacian-shaped source cannot express it at all. Writing this module against
``Lap`` would have made it structurally incapable of manufacturing the one problem the anisotropic
work needs -- discovered only after the fixtures were migrated onto it.

SCOPE
-----
- Separable Hamiltonians with a quadratic control cost, where ``dH/dp = grad u / lambda`` and hence
  ``div(dH/dp) = Lap u / lambda``. That covers every manufactured pair in this repository, and
  anything else raises rather than silently assembling a source from a different equation.
- **Constant** ``Sigma``, isotropic or not. For spatially varying diffusion the FP term is
  ``(1/2) sum_ij d2(sigma_ij m)/dx_i dx_j``, which is not ``tr(Sigma . Hess m)``; the two coincide
  only when ``Sigma`` does not depend on ``x``. A varying ``Sigma`` therefore needs a different
  assembly, and is refused rather than approximated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.core.hamiltonian import HEvalState, QuadraticControlCost, SeparableHamiltonian

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.core.hamiltonian import HamiltonianBase

# A field callable: (t, x) -> (N,) for scalars, (t, x) -> (N, d) for gradients, with x of shape
# (N, d). The same convention `base_hjb.py` already uses for a source term, so a pair's own
# derivatives and the assembled source speak one language.
Field = Callable[[float, "NDArray"], "NDArray"]


@dataclass(frozen=True)
class ManufacturedPair:
    """An exact ``(u*, m*)`` together with the analytic derivatives a source assembly needs.

    Every field is a callable of ``(t, x)`` with ``x`` of shape ``(N, d)``. Gradients return
    ``(N, d)``, Hessians ``(N, d, d)``, everything else ``(N,)``.

    The derivatives are supplied rather than computed. Differentiating numerically would put the
    differentiation's own truncation error into a source that is supposed to be exact, and the
    resulting convergence study would measure that error rather than the scheme's.
    """

    u: Field
    u_t: Field
    grad_u: Field
    hess_u: Field
    m: Field
    m_t: Field
    grad_m: Field
    hess_m: Field
    name: str = "unnamed"


def _quadratic_lambda(hamiltonian: HamiltonianBase) -> float:
    """The control cost of a separable quadratic Hamiltonian, or raise saying why not."""
    if not isinstance(hamiltonian, SeparableHamiltonian):
        raise NotImplementedError(
            f"manufactured sources need dH/dp and div(dH/dp); this module derives them for a "
            f"SeparableHamiltonian with a quadratic control cost, and got "
            f"{type(hamiltonian).__name__}. Supply div(dH/dp) explicitly, or extend this module -- "
            f"do not assemble a source from a Hamiltonian the solver does not integrate."
        )
    cost = hamiltonian.control_cost
    if not isinstance(cost, QuadraticControlCost):
        raise NotImplementedError(
            f"quadratic control cost required to derive div(dH/dp) = Lap u / lambda; got {type(cost).__name__}."
        )
    return float(cost.control_cost)


def _as_tensor(sigma: float | NDArray, dim: int) -> NDArray:
    """Normalise a scalar / per-axis / full-tensor sigma to a constant ``(d, d)`` covariance.

    One place that decides what ``sigma`` means, so the two assemblies below cannot disagree about
    it. A scalar is isotropic ``sigma^2 I``; a ``(d,)`` vector is the diagonal of per-axis
    variances; a ``(d, d)`` matrix is used as given and must be symmetric, since an antisymmetric
    part contributes nothing to ``tr(Sigma . Hess)`` for a symmetric Hessian and would therefore be
    silently discarded.
    """
    arr = np.asarray(sigma, dtype=float)
    if arr.ndim == 0:
        return float(arr) ** 2 * np.eye(dim)
    if arr.ndim == 1:
        return np.diag(arr.astype(float) ** 2)
    if arr.ndim == 2:
        if not np.allclose(arr, arr.T):
            raise ValueError(
                "a full-tensor sigma must be symmetric: an antisymmetric part contributes nothing "
                "to tr(Sigma . Hess) and would be dropped without trace."
            )
        return arr
    raise ValueError(f"sigma must be scalar, (d,) or (d, d); got shape {arr.shape}")


def hjb_source(pair: ManufacturedPair, hamiltonian: HamiltonianBase, sigma: float | NDArray) -> Field:
    """``S_hjb = -d_t u + H(x, grad u, m) - (1/2) tr(Sigma . Hess u)``.

    ``H`` is evaluated through the problem's own Hamiltonian, so the manufactured equation is the
    one the solver integrates rather than a re-implementation that could drift from it.
    """

    def source(t: float, x: NDArray) -> NDArray:
        x = np.asarray(x, dtype=float)
        cov = _as_tensor(sigma, x.shape[-1])
        p = pair.grad_u(t, x)
        m = pair.m(t, x)
        h = np.asarray(hamiltonian.evaluate_H(HEvalState(x=x, p=p, m=m, t=t))).ravel()
        diffusion = 0.5 * np.einsum("ij,nij->n", cov, pair.hess_u(t, x))
        return -pair.u_t(t, x) + h - diffusion

    return source


def fp_source(pair: ManufacturedPair, hamiltonian: HamiltonianBase, sigma: float | NDArray) -> Field:
    """``S_fp = d_t m - div(m * dH/dp) - (1/2) tr(Sigma . Hess m)``.

    The transport expands as ``div(m * alpha) = grad m . alpha + m * div(alpha)``, with
    ``alpha = dH/dp = grad u / lambda`` and ``div(alpha) = Lap u / lambda = tr(Hess u) / lambda``
    for the quadratic control cost this module is scoped to. Note that this Laplacian is the
    divergence of the DRIFT and not a diffusion term -- it is unaffected by ``Sigma``.
    """
    lam = _quadratic_lambda(hamiltonian)

    def source(t: float, x: NDArray) -> NDArray:
        x = np.asarray(x, dtype=float)
        cov = _as_tensor(sigma, x.shape[-1])
        div_alpha = np.einsum("nii->n", pair.hess_u(t, x)) / lam
        advection = (pair.grad_m(t, x) * pair.grad_u(t, x)).sum(axis=-1) / lam
        diffusion = 0.5 * np.einsum("ij,nij->n", cov, pair.hess_m(t, x))
        return pair.m_t(t, x) - (advection + pair.m(t, x) * div_alpha) - diffusion

    return source

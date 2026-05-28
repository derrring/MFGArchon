"""
Dimension-agnostic Moving Least Squares (MLS) shape functions with full
derivatives. Two interchangeable derivative backends:

- ``"numpy"`` (default): analytic derivatives, core dependencies only.
- ``"jax"`` (optional): autodiff through the moment matrix; requires jax.

Both compute the *full* MLS derivative (differentiating through M(x)), not the
diffuse derivative, and must agree to rounding. Dimension enters only through the
multi-index monomial basis; nothing branches on it.

Issue #1131 Phase 2.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def monomial_exponents(d: int, k: int) -> NDArray:
    """Exponent multi-indices for the total-degree-<= k space in d variables.

    Returns an integer array of shape ``(m, d)``, ``m = C(d+k, k)``, ordered by
    total degree then lexicographically. Row ``alpha`` encodes the monomial
    ``prod_l x_l**alpha_l``.
    """
    exps = [e for e in product(range(k + 1), repeat=d) if sum(e) <= k]
    exps.sort(key=lambda e: (sum(e), e))
    return np.array(exps, dtype=np.int64)


# --- Wendland C^2 weight (radial; dimension enters only via ||x - x_j||) -------
def _wendland_c2(r: NDArray) -> NDArray:
    return np.where(r < 1.0, (1.0 - r) ** 4 * (4.0 * r + 1.0), 0.0)


def _wendland_c2_deriv(r: NDArray) -> NDArray:
    return np.where(r < 1.0, -20.0 * r * (1.0 - r) ** 3, 0.0)


# --- monomial basis and its gradient (numpy) -----------------------------------
def _poly_batch(pts: NDArray, exponents: NDArray) -> NDArray:
    """Monomial basis at many points. pts (P,d), exponents (m,d) -> (P,m)."""
    return np.prod(pts[:, None, :] ** exponents[None, :, :], axis=2)


def _poly_grad_batch(pts: NDArray, exponents: NDArray) -> NDArray:
    """Gradient of the monomial basis. pts (P,d), exponents (m,d) -> (P,m,d).

    d(prod x_e^a_e)/dx_c = a_c x_c^{a_c-1} prod_{e!=c} x_e^{a_e}. The a_c factor
    zeroes terms with a_c=0, so the reduced exponent is clamped at 0 to avoid
    x_c^{-1} (the term is multiplied by 0 anyway).
    """
    P, d = pts.shape
    m = exponents.shape[0]
    out = np.zeros((P, m, d))
    for c in range(d):
        reduced = exponents.copy()
        reduced[:, c] = np.maximum(reduced[:, c] - 1, 0)
        term = np.prod(pts[:, None, :] ** reduced[None, :, :], axis=2)  # (P,m)
        out[:, :, c] = exponents[:, c][None, :] * term
    return out


def shape_functions_and_grads_numpy(
    x_eval: NDArray, nodes: NDArray, rho: float, exponents: NDArray
) -> tuple[NDArray, NDArray]:
    r"""MLS shape functions and full gradients via analytic differentiation.

    phi_j(x) = omega_j(x) p(x)^T M(x)^{-1} p(x_j),  M(x) = sum_l omega_l p_l p_l^T.

    Using gamma = M^{-1} p, s_j = (P gamma)_j, the full gradient is
        d_c phi_j = (d_c omega_j) s_j + omega_j (P (beta_c - w_c))_j,
    with beta_c = M^{-1} (d_c p), w_c = M^{-1} (d_c M) gamma, and
    d(M^{-1}) = -M^{-1} (d M) M^{-1}.

    x_eval (Q,d) -> phi (Q,N), grad (Q,N,d).
    """
    x_eval = np.asarray(x_eval, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    Q, d = x_eval.shape

    diffs = x_eval[:, None, :] - nodes[None, :, :]  # (Q,N,d), x - x_l
    dist = np.linalg.norm(diffs, axis=2)  # (Q,N)
    r = dist / rho
    omega = _wendland_c2(r)  # (Q,N)
    wprime = _wendland_c2_deriv(r)  # (Q,N)

    P = _poly_batch(nodes, exponents)  # (N,m)
    p = _poly_batch(x_eval, exponents)  # (Q,m)
    dp = _poly_grad_batch(x_eval, exponents)  # (Q,m,d)

    M = np.einsum("qn,ni,nj->qij", omega, P, P)  # (Q,m,m)
    # numpy>=2.0: a (Q,m,m) with b (Q,m) is read as a single matrix RHS; pass an
    # explicit (Q,m,1) column stack and squeeze to get batched vector solves.
    gamma = np.linalg.solve(M, p[..., None])[..., 0]  # (Q,m)
    s = np.einsum("qm,nm->qn", gamma, P)  # (Q,N)
    phi = omega * s

    # weight gradient: d omega_l/dx_c = w'(r_l) (x - x_l)_c / (rho * dist_l).
    # dist_l = 0 only when x hits a node, where w'(0) = 0, so the row is 0;
    # the safe denominator just avoids 0/0.
    safe = np.where(dist == 0.0, 1.0, dist)
    domega = (wprime / (rho * safe))[..., None] * diffs  # (Q,N,d)

    grad = np.empty((Q, P.shape[0], d))
    for c in range(d):
        dM = np.einsum("qn,ni,nj->qij", domega[:, :, c], P, P)  # (Q,m,m)
        beta = np.linalg.solve(M, dp[:, :, c][..., None])[..., 0]  # (Q,m)
        u = np.einsum("qij,qj->qi", dM, gamma)  # (Q,m)
        w_c = np.linalg.solve(M, u[..., None])[..., 0]  # (Q,m)
        ds = np.einsum("qm,nm->qn", beta - w_c, P)  # (Q,N)
        grad[:, :, c] = domega[:, :, c] * s + omega * ds
    return phi, grad


def shape_functions_and_grads_jax(
    x_eval: NDArray, nodes: NDArray, rho: float, exponents: NDArray
) -> tuple[NDArray, NDArray]:
    """MLS shape functions and full gradients via JAX autodiff (optional backend).

    Same signature/result as the numpy backend. Raises ImportError if jax is
    unavailable -- no silent fallback (Issue #1072).
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("backend='jax' requires jax. Install jax, or use backend='numpy'.") from None

    jax.config.update("jax_enable_x64", True)
    nodes_j = jnp.asarray(nodes, dtype=jnp.float64)
    exps_j = jnp.asarray(exponents)
    P = jnp.prod(nodes_j[:, None, :] ** exps_j[None, :, :], axis=2)  # (N,m)

    def phi_at(x):  # x (d,) -> (N,)
        sq = jnp.sum((nodes_j - x[None, :]) ** 2, axis=1)
        is_zero = sq == 0.0
        dist = jnp.where(is_zero, 0.0, jnp.sqrt(jnp.where(is_zero, 1.0, sq)))
        r = dist / rho
        w = jnp.where(r < 1.0, (1.0 - r) ** 4 * (4.0 * r + 1.0), 0.0)
        p = jnp.prod(x[None, :] ** exps_j, axis=1)  # (m,)
        M = jnp.einsum("n,ni,nj->ij", w, P, P)
        return (P @ jnp.linalg.solve(M, p)) * w

    x_eval_j = jnp.asarray(x_eval, dtype=jnp.float64)
    phi = jax.vmap(phi_at)(x_eval_j)
    grad = jax.vmap(jax.jacobian(phi_at))(x_eval_j)
    return np.asarray(phi), np.asarray(grad)


def shape_functions_and_grads(
    x_eval: NDArray,
    nodes: NDArray,
    rho: float,
    exponents: NDArray,
    backend: str = "numpy",
) -> tuple[NDArray, NDArray]:
    """Dispatch to the numpy (default) or jax MLS derivative backend."""
    if backend == "numpy":
        return shape_functions_and_grads_numpy(x_eval, nodes, rho, exponents)
    if backend == "jax":
        return shape_functions_and_grads_jax(x_eval, nodes, rho, exponents)
    raise ValueError(f"Unknown backend {backend!r}; expected 'numpy' or 'jax'.")

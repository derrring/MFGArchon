"""
Dimension-agnostic Moving Least Squares (MLS) shape functions with full
derivatives and evaluation-centered shifted/scaled moments. Two interchangeable
derivative backends:

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
    x_eval: NDArray, nodes: NDArray, rho: float, exponents: NDArray, check_conditioning: bool = False
) -> tuple[NDArray, NDArray]:
    r"""MLS shape functions and full gradients via analytic differentiation.

    At each evaluation point, the polynomial basis is frozen at that point and
    scaled by ``rho``. This is an invertible basis change, so it preserves the
    exact-arithmetic MLS shape functions and derivatives while making the moment
    condition number translation- and scale-independent.

    Using gamma = M^{-1} p, s_j = (P gamma)_j, the full gradient is
        d_c phi_j = (d_c omega_j) s_j + omega_j (P (beta_c - w_c))_j,
    with beta_c = M^{-1} (d_c p), w_c = M^{-1} (d_c M) gamma, and
    d(M^{-1}) = -M^{-1} (d M) M^{-1}.

    x_eval (Q,d) -> phi (Q,N), grad (Q,N,d).
    """
    x_eval = np.asarray(x_eval, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    Q, d = x_eval.shape
    N = len(nodes)
    m = len(exponents)
    if Q == 0:
        return np.empty((0, N)), np.empty((0, N, d))

    origin = np.zeros((1, d), dtype=np.float64)
    p = _poly_batch(origin, exponents)[0]
    dp = _poly_grad_batch(origin, exponents)[0] / rho
    phi = np.empty((Q, N), dtype=np.float64)
    grad = np.empty((Q, N, d), dtype=np.float64)
    batch_size = max(1, min(Q, 4_000_000 // max(N * m, 1)))

    for start in range(0, Q, batch_size):
        stop = min(start + batch_size, Q)
        evaluation_batch = x_eval[start:stop]
        diffs = evaluation_batch[:, None, :] - nodes[None, :, :]
        dist = np.linalg.norm(diffs, axis=2)
        r = dist / rho
        omega = _wendland_c2(r)
        wprime = _wendland_c2_deriv(r)

        local_nodes = -diffs / rho
        P = np.ones((stop - start, N, m), dtype=np.float64)
        for c in range(d):
            P *= local_nodes[:, :, c, None] ** exponents[None, None, :, c]
        M = np.einsum("qn,qni,qnj->qij", omega, P, P)
        if check_conditioning:
            conds = np.linalg.cond(M)
            worst = float(np.max(conds)) if conds.size else 0.0
            if not np.isfinite(worst) or worst > 1e12:
                local_bad = int(np.argmax(conds))
                q_bad = start + local_bad
                raise np.linalg.LinAlgError(
                    f"MLS moment matrix is ill-conditioned at quadrature point {q_bad} (cond={worst:.2e} "
                    f"> 1e12): too few nodes cover its support (rho={rho} too small, or a gap / degenerate "
                    f"cloud). The shape functions would be silently garbage. Increase the support radius "
                    f"rho, densify the cloud, or lower the polynomial degree (Issue #1485)."
                )

        right_hand_side = np.broadcast_to(p, (stop - start, m))
        gamma = np.linalg.solve(M, right_hand_side[..., None])[..., 0]
        s = np.einsum("qnm,qm->qn", P, gamma)
        phi[start:stop] = omega * s

        safe = np.where(dist == 0.0, 1.0, dist)
        domega = (wprime / (rho * safe))[..., None] * diffs
        for c in range(d):
            dM = np.einsum("qn,qni,qnj->qij", domega[:, :, c], P, P)
            derivative_rhs = np.broadcast_to(dp[:, c], (stop - start, m))
            beta = np.linalg.solve(M, derivative_rhs[..., None])[..., 0]
            u = np.einsum("qij,qj->qi", dM, gamma)
            w_c = np.linalg.solve(M, u[..., None])[..., 0]
            ds = np.einsum("qnm,qm->qn", P, beta - w_c)
            grad[start:stop, :, c] = domega[:, :, c] * s + omega * ds
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

    def phi_at(x):  # x (d,) -> (N,)
        center = jax.lax.stop_gradient(x)
        local_nodes = (nodes_j - center[None, :]) / rho
        local_evaluation = (x - center) / rho
        P = jnp.prod(local_nodes[:, None, :] ** exps_j[None, :, :], axis=2)
        sq = jnp.sum((nodes_j - x[None, :]) ** 2, axis=1)
        is_zero = sq == 0.0
        dist = jnp.where(is_zero, 0.0, jnp.sqrt(jnp.where(is_zero, 1.0, sq)))
        r = dist / rho
        w = jnp.where(r < 1.0, (1.0 - r) ** 4 * (4.0 * r + 1.0), 0.0)
        p = jnp.prod(local_evaluation[None, :] ** exps_j, axis=1)
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
    check_conditioning: bool = False,
) -> tuple[NDArray, NDArray]:
    """Dispatch to the numpy (default) or jax MLS derivative backend.

    ``check_conditioning`` (numpy backend only) fails loud on a near-singular MLS moment matrix; the
    Gauss-quadrature assembly path passes ``True``, the SCNI path leaves it ``False`` (Issue #1485).
    """
    if backend == "numpy":
        return shape_functions_and_grads_numpy(x_eval, nodes, rho, exponents, check_conditioning=check_conditioning)
    if backend == "jax":
        return shape_functions_and_grads_jax(x_eval, nodes, rho, exponents)
    raise ValueError(f"Unknown backend {backend!r}; expected 'numpy' or 'jax'.")

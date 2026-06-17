"""Contract tests for the single-source batch Hamiltonian evaluation (Issue #1071, Layer A).

``eval_H_batch`` / ``eval_dH_dp_batch`` are byte-identical extractions of the inline
``np.asarray(H_class(...), dtype=float)`` calls every HJB solver used to duplicate. These
pin that contract so a future change to the helper cannot silently diverge from the form the
solvers relied on.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.h_eval import eval_dH_dp_batch, eval_H_batch
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian


def _batch(n=7, d=1):
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n).reshape(-1, 1) if d == 1 else rng.uniform(size=(n, d))
    m = rng.uniform(0.1, 1.0, size=n)
    p = rng.uniform(-1.0, 1.0, size=(n, d))
    return x, m, p


def test_eval_H_batch_is_byte_identical_to_inline():
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=2.0))
    x, m, p, t = *_batch(), 0.3
    out = eval_H_batch(H, x, m, p, t)
    ref = np.asarray(H(x, m, p, t=t), dtype=float)
    assert out.dtype == np.float64
    assert np.array_equal(out, ref)


def test_eval_dH_dp_batch_is_byte_identical_to_inline():
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=2.0))
    x, m, p, t = *_batch(), 0.3
    out = eval_dH_dp_batch(H, x, m, p, t)
    ref = np.asarray(H.dp(x, m, p, t=t), dtype=float)
    assert out.dtype == np.float64
    assert np.array_equal(out, ref)


def test_eval_batch_passes_through_non_lq_coupling():
    """The helper must call through to whatever H_class computes (not assume the LQ form):
    a non-LQ congestion coupling f(m)=m**3 changes the value, and the helper reflects it."""
    H_lq = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=2.0))
    H_cong = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=2.0),
        coupling=lambda m: m**3,
        coupling_dm=lambda m: 3.0 * m**2,
    )
    x, m, p, t = *_batch(), 0.0
    assert np.array_equal(eval_H_batch(H_cong, x, m, p, t), np.asarray(H_cong(x, m, p, t=t), dtype=float))
    # the congestion term must actually change the value (helper is not LQ-hardcoded)
    assert not np.allclose(eval_H_batch(H_cong, x, m, p, t), eval_H_batch(H_lq, x, m, p, t))


def test_assemble_hjb_residual_byte_identical():
    """Layer B: assemble_hjb_residual == -u_t + H(+running_cost) - D*lap_u (D = sigma^2/2)."""
    from mfgarchon.alg.numerical.hjb_solvers.h_eval import assemble_hjb_residual
    from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=2.0))
    x, m, p = _batch()
    n = m.shape[0]
    lap_u = np.linspace(-1.0, 1.0, n)
    u_t = np.linspace(0.0, 0.5, n)
    rc = np.full(n, 0.05)
    sigma, t = 0.3, 0.1
    out = assemble_hjb_residual(H_class=H, x=x, m=m, p=p, lap_u=lap_u, sigma=sigma, t=t, u_t=u_t, running_cost=rc)
    ref = -u_t + (np.asarray(H(x, m, p, t=t), dtype=float) + rc) - diffusion_from_volatility(sigma) * lap_u
    assert np.array_equal(out, ref)


def test_assemble_hjb_jacobian_diag_byte_identical():
    """Layer B: assemble_hjb_jacobian_diag == (1/dt)I + sum_d diag(dH/dp_d)@D_grad[d] - D*D_lap."""
    from scipy.sparse import diags, eye

    from mfgarchon.alg.numerical.hjb_solvers.h_eval import assemble_hjb_jacobian_diag, eval_dH_dp_batch
    from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=2.0))
    x, m, p = _batch()
    n, d = m.shape[0], p.shape[1]
    sigma, t, dt = 0.3, 0.1, 0.05
    D_grad = [eye(n, format="csr") for _ in range(d)]
    D_lap = (diags(-2.0 * np.ones(n)) + diags(np.ones(n - 1), 1) + diags(np.ones(n - 1), -1)).tocsr()
    out = assemble_hjb_jacobian_diag(H_class=H, x=x, m=m, p=p, sigma=sigma, t=t, dt=dt, D_grad=D_grad, D_lap=D_lap)
    dH_dp = eval_dH_dp_batch(H, x, m, p, t)
    ref = (1.0 / dt) * eye(n, format="csr")
    for dim in range(d):
        ref = ref + diags(dH_dp[:, dim], format="csr") @ D_grad[dim]
    ref = ref - diffusion_from_volatility(sigma) * D_lap
    assert np.allclose(out.toarray(), ref.toarray())


# --- Issue #1071 phase 7: per-node (field) sigma — collapse the GFDM inline LLF branches ---


def test_assemble_hjb_residual_array_sigma_matches_inline_llf_1071():
    """A per-node sigma field (GFDM Local-Lax-Friedrichs sigma_eff) yields D = sigma_i^2/2
    elementwise; the helper must be byte-identical to the inline LLF residual it replaces
    (hjb_gfdm.py:2209-2213: -u_t + H - diffusion_from_volatility(sigma_eff, kind='field') * lap_u)."""
    from mfgarchon.alg.numerical.hjb_solvers.h_eval import assemble_hjb_residual, eval_H_batch
    from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=2.0))
    x, m, p = _batch()
    n = m.shape[0]
    lap_u = np.linspace(-1.0, 1.0, n)
    u_t = np.linspace(0.0, 0.5, n)
    sigma_eff = np.random.default_rng(11).uniform(0.2, 0.9, size=n)  # per-node field
    t = 0.1
    out = assemble_hjb_residual(H_class=H, x=x, m=m, p=p, lap_u=lap_u, sigma=sigma_eff, t=t, u_t=u_t)
    ref = -u_t + eval_H_batch(H, x, m, p, t) - diffusion_from_volatility(sigma_eff, kind="field") * lap_u
    assert np.array_equal(out, ref)
    # field-sigma must genuinely differ from collapsing it to a scalar (it row-varies the diffusion)
    scalar_out = assemble_hjb_residual(
        H_class=H, x=x, m=m, p=p, lap_u=lap_u, sigma=float(sigma_eff.mean()), t=t, u_t=u_t
    )
    assert not np.allclose(out, scalar_out)


def test_assemble_hjb_jacobian_diag_array_sigma_matches_inline_llf_1071():
    """A per-node sigma field must ROW-SCALE the Laplacian via diags(D) @ D_lap, byte-identical to
    the inline LLF Jacobian (hjb_gfdm.py:2262-2268). Exact sparse equality, and it must differ
    from the scalar form (proving the per-node diffusion is actually applied)."""
    from scipy.sparse import diags, eye

    from mfgarchon.alg.numerical.hjb_solvers.h_eval import assemble_hjb_jacobian_diag, eval_dH_dp_batch
    from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=2.0))
    x, m, p = _batch()
    n, d = m.shape[0], p.shape[1]
    sigma_eff = np.random.default_rng(12).uniform(0.2, 0.9, size=n)
    t, dt = 0.1, 0.05
    D_grad = [eye(n, format="csr") for _ in range(d)]
    D_lap = (diags(-2.0 * np.ones(n)) + diags(np.ones(n - 1), 1) + diags(np.ones(n - 1), -1)).tocsr()
    out = assemble_hjb_jacobian_diag(H_class=H, x=x, m=m, p=p, sigma=sigma_eff, t=t, dt=dt, D_grad=D_grad, D_lap=D_lap)
    dH_dp = eval_dH_dp_batch(H, x, m, p, t)
    ref = (1.0 / dt) * eye(n, format="csr")
    for dim in range(d):
        ref = ref + diags(dH_dp[:, dim], format="csr") @ D_grad[dim]
    ref = ref - diags(diffusion_from_volatility(sigma_eff, kind="field"), format="csr") @ D_lap
    assert (out != ref).nnz == 0  # exact sparse equality to the inline LLF form
    scalar_out = assemble_hjb_jacobian_diag(
        H_class=H, x=x, m=m, p=p, sigma=float(sigma_eff.mean()), t=t, dt=dt, D_grad=D_grad, D_lap=D_lap
    )
    assert (out != scalar_out).nnz != 0  # per-node diffusion genuinely row-varies the Jacobian

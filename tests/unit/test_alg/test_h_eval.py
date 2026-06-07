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

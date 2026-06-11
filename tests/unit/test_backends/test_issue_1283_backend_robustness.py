"""
Pinning tests for Issue #1283 — backend robustness fixes.

Three bugs:
  (1) NumbaBackend.diff() returned size-n (derivative) instead of size-(n-1) (np.diff contract).
  (2) HybridParticleStrategy.solve() passed constructor args to CPUParticleStrategy
      which accepts none → TypeError on small-N and backend=None fallback paths.
  (3) TorchBackend defaulted to float32; all other backends default to float64
      → silent precision drift; also called torch.set_default_dtype() (process-global).

Each test below FAILS on the pre-fix code and PASSES after.
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# (1) NumbaBackend.diff() output length
# ---------------------------------------------------------------------------


def test_numba_diff_returns_n_minus_1():
    """NumbaBackend.diff(a) must return length len(a)-1, matching np.diff contract."""
    pytest.importorskip("numba")
    from mfgarchon.backends.numba_backend import NumbaBackend

    backend = NumbaBackend()
    for n in (5, 10, 100):
        a = np.arange(n, dtype=np.float64)
        result = backend.diff(a)
        assert len(result) == n - 1, (
            f"NumbaBackend.diff() returned length {len(result)} for input length {n}; "
            f"expected {n - 1} (np.diff contract). Issue #1283 fix not applied."
        )

    # Values must also match np.diff exactly
    a = np.array([1.0, 4.0, 9.0, 16.0])
    np.testing.assert_array_equal(
        backend.diff(a),
        np.diff(a),
        err_msg="NumbaBackend.diff() values do not match np.diff(). Issue #1283.",
    )


# ---------------------------------------------------------------------------
# (2) HybridParticleStrategy small-N and backend=None paths do not raise TypeError
# ---------------------------------------------------------------------------


class _MockProblem:
    """Minimal MFGProblem stand-in for strategy routing tests."""

    T = 1.0
    Nt = 5
    Nx = 10
    xmin = 0.0
    xmax = 1.0
    sigma = 0.1


def test_hybrid_strategy_small_n_no_typeerror(monkeypatch):
    """
    HybridParticleStrategy.solve() with num_particles < 10000 must not raise TypeError.

    Before the fix, CPUParticleStrategy(self.backend) was called, but CPUParticleStrategy
    has no __init__ and raises TypeError: object.__init__() takes exactly one argument.
    """
    from mfgarchon.backends.strategies.particle_strategies import (
        CPUParticleStrategy,
        HybridParticleStrategy,
    )

    # Patch CPUParticleStrategy.solve so we only test the instantiation path
    captured = {}

    def _fake_solve(self, *args, **kwargs):
        captured["called"] = True
        return np.zeros((6, 11))  # shape (Nt+1, Nx+1)

    monkeypatch.setattr(CPUParticleStrategy, "solve", _fake_solve)

    strategy = HybridParticleStrategy(backend=object())  # non-None backend
    m_initial = np.ones(11) / 11
    U_drift = np.zeros((6, 11))

    # Must not raise TypeError (pre-fix: CPUParticleStrategy(self.backend) would crash)
    strategy.solve(
        m_initial,
        U_drift,
        _MockProblem(),
        num_particles=100,  # < 10000 → small-N path
        kde_bandwidth="scott",
        normalize_kde_output=True,
        boundary_conditions=None,
        backend=None,
    )
    assert captured.get("called"), "CPUParticleStrategy.solve was not called on small-N path"


def test_hybrid_strategy_backend_none_fallback_no_typeerror(monkeypatch):
    """
    HybridParticleStrategy.solve() with backend=None and large N must not raise TypeError.

    Before the fix, CPUParticleStrategy(None) was called on the fallback path.
    """
    from mfgarchon.backends.strategies.particle_strategies import (
        CPUParticleStrategy,
        HybridParticleStrategy,
    )

    captured = {}

    def _fake_solve(self, *args, **kwargs):
        captured["called"] = True
        return np.zeros((6, 11))

    monkeypatch.setattr(CPUParticleStrategy, "solve", _fake_solve)

    strategy = HybridParticleStrategy(backend=None)  # backend=None → CPU fallback
    m_initial = np.ones(11) / 11
    U_drift = np.zeros((6, 11))

    # Must not raise TypeError (pre-fix: CPUParticleStrategy(None) would crash)
    strategy.solve(
        m_initial,
        U_drift,
        _MockProblem(),
        num_particles=50_000,  # ≥ 10000 → large-N path, but backend=None → CPU fallback
        kde_bandwidth="scott",
        normalize_kde_output=True,
        boundary_conditions=None,
        backend=None,
    )
    assert captured.get("called"), "CPUParticleStrategy.solve was not called on backend=None fallback"


# ---------------------------------------------------------------------------
# (3) TorchBackend default dtype is float64, no process-global side effect
# ---------------------------------------------------------------------------


def test_torch_backend_default_precision_is_float64():
    """
    TorchBackend() with no precision argument must default to float64,
    matching numpy/jax/numba backends. Issue #1283 fix: change default from float32.
    """
    torch = pytest.importorskip("torch")
    from mfgarchon.backends.torch_backend import TorchBackend

    backend = TorchBackend(device="cpu")
    assert backend.precision == "float64", (
        f"TorchBackend default precision is '{backend.precision}', expected 'float64'. Issue #1283 fix not applied."
    )
    assert backend.torch_dtype == torch.float64, (
        f"TorchBackend.torch_dtype is {backend.torch_dtype}, expected torch.float64. Issue #1283 fix not applied."
    )


def test_torch_backend_does_not_mutate_global_default_dtype():
    """
    Constructing a TorchBackend must not call torch.set_default_dtype() and
    must not change the process-global dtype. Issue #1283 fix: remove global side effect.

    The pre-fix code called torch.set_default_dtype() in _setup_backend, so constructing
    TorchBackend(precision='float32') after the global was set to float64 would silently
    mutate it back to float32 — observable by any subsequent torch.tensor([1.0]) call.
    """
    torch = pytest.importorskip("torch")
    from mfgarchon.backends.torch_backend import TorchBackend

    original_dtype = torch.get_default_dtype()
    try:
        # Force global to float64 so any float32 mutation is detectable
        torch.set_default_dtype(torch.float64)
        pre_dtype = torch.get_default_dtype()
        assert pre_dtype == torch.float64, "setup failed: could not set global dtype to float64"

        # Construct with float32 — pre-fix code would call
        # torch.set_default_dtype(torch.float32) and mutate the global back to float32.
        _backend = TorchBackend(device="cpu", precision="float32")
        post_construct_dtype = torch.get_default_dtype()
        assert post_construct_dtype == torch.float64, (
            f"TorchBackend(precision='float32') mutated process-global dtype from float64 "
            f"to {post_construct_dtype}. Issue #1283 fix not applied: torch.set_default_dtype() "
            f"must not be called in _setup_backend."
        )
    finally:
        torch.set_default_dtype(original_dtype)

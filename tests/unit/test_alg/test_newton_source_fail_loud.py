"""Issue #1430 (closing the #1424 sibling): the Newton residual path
(``MFGResidual.compute_hjb_output`` / ``compute_fp_output``) must fail loud on a problem-level
source a solver cannot accept — exactly like the Picard path (Issue #1424,
``base_mfg._build_*_kwargs``, pinned in ``test_source_term_fail_loud.py``).

Before this fix the Newton path composed the source ONLY when the signature already had
``source_term``, silently dropping it otherwise — so a source-defining problem solved on a
non-accepting solver gave a silently-wrong Newton fixed point while Picard correctly raised.

Tests bind the two ``MFGResidual`` methods to a minimal carrier and monkeypatch the composers, so
they isolate the fail-loud dispatch from the composition internals.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import mfg_residual as mr
from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual


def _src(t, x):
    return np.zeros_like(x)


class _FakeSolver:
    def solve_hjb_system(self, M, U_terminal, U_prev, **kw):
        return M

    def solve_fp_system(self, M_initial, U, **kw):
        return U


class _Carrier:
    """Minimal carrier exposing the two Newton-residual methods under test."""

    compute_hjb_output = MFGResidual.compute_hjb_output
    compute_fp_output = MFGResidual.compute_fp_output

    def __init__(self, hjb_params, fp_params):
        self._hjb_sig_params = hjb_params
        self._fp_sig_params = fp_params
        self.hjb_solver = _FakeSolver()
        self.fp_solver = _FakeSolver()
        self.problem = object()  # composers are monkeypatched; problem is never inspected
        self.U_terminal = None
        self.M_initial = None
        self.volatility_field = None
        self.drift_field = None
        self._fp_drift_convention = None
        self.solution_shape = (3, 4)


def test_newton_hjb_source_incapable_raises(monkeypatch):
    monkeypatch.setattr(mr, "compose_hjb_source", lambda p, m, u: _src)
    c = _Carrier(hjb_params={"M_density", "U_terminal"}, fp_params=set())
    with pytest.raises(NotImplementedError, match="1430"):
        c.compute_hjb_output(np.zeros((3, 4)), np.zeros((3, 4)))


def test_newton_fp_source_incapable_raises(monkeypatch):
    monkeypatch.setattr(mr, "compose_fp_source", lambda p, m, u: _src)
    c = _Carrier(hjb_params=set(), fp_params={"M_initial", "U"})
    with pytest.raises(NotImplementedError, match="1424"):
        c.compute_fp_output(np.zeros((3, 4)), M=np.zeros((3, 4)))


def test_newton_hjb_source_capable_passes_through(monkeypatch):
    """Capable solver: no raise; the source is routed (solver runs), not dropped."""
    monkeypatch.setattr(mr, "compose_hjb_source", lambda p, m, u: _src)
    c = _Carrier(hjb_params={"source_term"}, fp_params=set())
    out = c.compute_hjb_output(np.ones((3, 4)), np.zeros((3, 4)))
    np.testing.assert_array_equal(out, np.ones((3, 4)))


def test_newton_hjb_no_source_is_noop_even_when_incapable(monkeypatch):
    """Baseline-safe: no problem source -> no raise even for an incapable solver (the common case)."""
    monkeypatch.setattr(mr, "compose_hjb_source", lambda p, m, u: None)
    c = _Carrier(hjb_params={"M_density"}, fp_params=set())
    out = c.compute_hjb_output(np.full((3, 4), 7.0), np.zeros((3, 4)))
    np.testing.assert_array_equal(out, np.full((3, 4), 7.0))

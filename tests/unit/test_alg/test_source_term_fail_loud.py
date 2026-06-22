#!/usr/bin/env python3
"""Issue #1424: a problem-level source (source_term_hjb / source_term_fp / nonlocal_operator /
obstacle, composed into a non-None source callable) must NOT be silently dropped for solvers whose
``solve_*_system`` signature lacks ``source_term`` — the iterator now fails loud instead.

Tests bind the kwargs builders to a minimal carrier (no full iterator construction). They pin:
incapable solver + active source -> NotImplementedError; capable solver -> source passed through;
no source -> no raise even for an incapable solver (the baseline-safe guard).
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import BaseCouplingIterator


class _Builder:
    _build_hjb_kwargs = BaseCouplingIterator._build_hjb_kwargs
    _build_fp_kwargs = BaseCouplingIterator._build_fp_kwargs

    def __init__(self, hjb_params, fp_params):
        self._hjb_sig_params = hjb_params
        self._fp_sig_params = fp_params
        self._hjb_solver_name = "FakeNonFDMSolver"
        self._fp_solver_name = "FakeNonFDMSolver"


def _src(t, x):
    return np.zeros_like(x)


class TestSourceTermFailLoud:
    def test_hjb_source_incapable_raises(self):
        b = _Builder(hjb_params={"M_density", "U_terminal"}, fp_params=set())
        with pytest.raises(NotImplementedError, match="source_term"):
            b._build_hjb_kwargs(source_term=_src)

    def test_fp_source_incapable_raises(self):
        b = _Builder(hjb_params=set(), fp_params={"M_initial", "U"})
        with pytest.raises(NotImplementedError, match="1424"):
            b._build_fp_kwargs(source_term=_src)

    def test_hjb_source_capable_passes_through(self):
        b = _Builder(hjb_params={"source_term", "volatility_field"}, fp_params=set())
        assert b._build_hjb_kwargs(source_term=_src)["source_term"] is _src

    def test_fp_source_capable_passes_through(self):
        b = _Builder(hjb_params=set(), fp_params={"source_term", "drift_field"})
        assert b._build_fp_kwargs(source_term=_src)["source_term"] is _src

    def test_no_source_is_noop_even_when_incapable(self):
        """Baseline-safe: source_term=None never raises (the common case is untouched)."""
        b = _Builder(hjb_params={"M_density"}, fp_params={"M_initial"})
        assert b._build_hjb_kwargs(source_term=None) == {}
        assert b._build_fp_kwargs(source_term=None) == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

#!/usr/bin/env python3
"""Issue #1424: a problem-level source (source_term_hjb / source_term_fp / nonlocal_operator,
composed into a non-None source callable) must NOT be silently dropped for solvers whose
``solve_*_system`` signature lacks ``source_term`` — the iterator fails loud instead.

Issue #2207 moved WHERE the source comes from, and these tests moved with it. The builders used to
take a ``source_term=`` callable the caller had composed; they now compose it themselves from the
iterates, which are required. So a case here supplies a PROBLEM carrying a source, not a callable:
the chain under test is composition -> capability check -> forward, and the old shape could only
ever pin its second half. That mattered — three of the seven coupling loops never composed at all,
so the guard these tests defend was unreachable from them and every case here still passed.

Tests bind the kwargs builders to a minimal carrier (no full iterator construction). They pin:
incapable solver + a problem with a source -> NotImplementedError; capable solver -> the composed
source is forwarded; a problem with no source -> no raise even for an incapable solver.
"""

from types import SimpleNamespace

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import BaseCouplingIterator

_M = np.zeros((3, 5))
_U = np.zeros((3, 5))


def _problem(*, hjb_source=None, fp_source=None):
    """The fields `compose_hjb_source` / `compose_fp_source` read to decide there is a source.

    Only those three, plus `sigma` for `resolve_volatility_kwarg` and `dt` for the closure's time
    slicing. The closure is never invoked here: the capability check fires on whether a source
    EXISTS, and evaluating it would be testing `source_composition`, which has its own tests.
    """
    return SimpleNamespace(
        source_term_hjb=hjb_source,
        source_term_fp=fp_source,
        nonlocal_operator=None,
        sigma=0.3,
        dt=0.1,
    )


def _src(x, m, v, t):
    return np.zeros_like(np.asarray(x, dtype=float))


class _Builder(BaseCouplingIterator):
    """Subclass, not a method-borrowing double.

    Borrowing `_build_*_kwargs` onto a bare class was green only because these tests never pass
    `volatility_field`; adding one such case raised `AttributeError` on the seam's other
    dependencies instead of testing anything (#1783 review). Inheriting keeps the double honest.
    """

    def solve(self, *args, **kwargs):  # pragma: no cover - abstract stub, never called
        raise NotImplementedError

    def get_results(self, *args, **kwargs):  # pragma: no cover - abstract stub, never called
        raise NotImplementedError

    def __init__(self, hjb_params, fp_params, problem=None):
        self.problem = problem if problem is not None else _problem()
        self._hjb_sig_params = hjb_params
        self._fp_sig_params = fp_params
        self._hjb_solver_name = "FakeNonFDMSolver"
        self._fp_solver_name = "FakeNonFDMSolver"


class TestSourceTermFailLoud:
    def test_hjb_source_incapable_raises(self):
        b = _Builder({"M_density", "U_terminal"}, set(), _problem(hjb_source=_src))
        with pytest.raises(NotImplementedError, match="source_term"):
            b._build_hjb_kwargs(M=_M, U=_U)

    def test_fp_source_incapable_raises(self):
        b = _Builder(set(), {"M_initial", "U"}, _problem(fp_source=_src))
        with pytest.raises(NotImplementedError, match="1424"):
            b._build_fp_kwargs(M=_M, U=_U)

    def test_hjb_source_capable_is_forwarded(self):
        b = _Builder({"source_term", "volatility_field"}, set(), _problem(hjb_source=_src))
        assert callable(b._build_hjb_kwargs(M=_M, U=_U)["source_term"])

    def test_fp_source_capable_is_forwarded(self):
        b = _Builder(set(), {"source_term", "drift_field"}, _problem(fp_source=_src))
        assert callable(b._build_fp_kwargs(M=_M, U=_U)["source_term"])

    def test_no_source_is_noop_even_when_incapable(self):
        """Baseline-safe: a problem with no source never raises (the common case is untouched)."""
        b = _Builder({"M_density"}, {"M_initial"}, _problem())
        assert b._build_hjb_kwargs(M=_M, U=_U) == {}
        assert b._build_fp_kwargs(M=_M, U=_U) == {}


def test_the_iterates_are_required_not_optional():
    """Issue #2207: the omission this change removes must not be re-expressible.

    A default of ``None`` on ``M`` / ``U`` would restore exactly the hole: a loop that forgets them
    would compose nothing, raise nothing, and solve without the source. The absence of a default is
    the mechanism, so it is asserted rather than left to review.
    """
    b = _Builder({"source_term"}, {"source_term"}, _problem(hjb_source=_src, fp_source=_src))
    with pytest.raises(TypeError, match="M"):
        b._build_hjb_kwargs()
    with pytest.raises(TypeError, match="M"):
        b._build_fp_kwargs()


def test_the_double_carries_the_whole_seam_not_one_borrowed_method():
    """Guard on the fixture itself: it must reach every dependency of `_build_hjb_kwargs`.

    A double that borrows one method passes this file's cases and fails the moment an unrelated
    test adds a `volatility_field`, reporting an `AttributeError` about the fixture rather than a
    fact about the code. Exercising the other branch here keeps that discoverable in this file.
    """
    b = _Builder({"M_density", "U_terminal", "volatility_field"}, set())
    assert b._build_hjb_kwargs(M=_M, U=_U, volatility_field=0.42) == {"volatility_field": 0.42}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

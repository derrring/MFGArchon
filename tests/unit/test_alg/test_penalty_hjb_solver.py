"""`PenaltyHJBSolver` is retired and must refuse construction (#2002).

WHAT THIS FILE USED TO CONTAIN, AND WHY IT WAS DELETED RATHER THAN ADAPTED
--------------------------------------------------------------------------
Eight tests asserting the wrapper worked: that the penalty changed the answer, that a larger
`penalty_parameter` changed it more, that a zero obstacle was a pass-through, that an existing
`source_term` was composed with it. Every one of them passed, and every one of them was true.

They measured the wrong quantity. The wrapper applied
``penalty_parameter * max(0, Psi(x))`` -- no ``v`` -- so "the penalty changes the answer" and
"a bigger penalty changes it more" are statements about a POSITION penalty, and hold exactly as
well when the constraint ``v >= Psi`` is satisfied everywhere as when it is violated everywhere.
A test suite can be fully green over a term that cannot express the thing its module is named
after. Adapting them would have kept that shape; they are gone.

WHY THE CLASS IS RETIRED
------------------------
Not an arithmetic bug. Its design is to add ``v >= Psi(x)`` to ANY inner solver by injecting a
penalty into that solver's ``source_term``. The penalty for that constraint is
``max(0, Psi - v)``, which needs the value function; ``source_term`` is ``(t, x) -> array``.
**There is nowhere for ``v`` to enter.** The channel it chose cannot carry the quantity its
purpose requires, so the wrapper was unimplementable in the shape it was written, and the
``(1/eps) * max(0, Psi - v)`` in its own docstring was a description of the intent.

The wrapper was also the only mechanism claiming to give obstacle support to solvers other than
``HJBFDMSolver``, so retiring it removes a capability that was never there. What replaces it is
``constraint=`` on that one solver today, and #2046 for the general case.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_penalty import PenaltyHJBSolver


class _StubInner:
    """Enough of a solver to reach the constructor. It must never be used."""

    problem = None
    config = None

    def solve_hjb_system(self, M, U_T, U_prev, volatility_field=None, source_term=None):  # pragma: no cover
        raise AssertionError("the retired wrapper must not reach its inner solver")


def test_construction_refuses():
    with pytest.raises(NotImplementedError) as excinfo:
        PenaltyHJBSolver(inner_solver=_StubInner(), obstacle=lambda x: np.zeros(np.asarray(x).shape[0]))

    message = str(excinfo.value)
    # The refusal has to say WHY, or the next reader reimplements it the same way.
    assert "RETIRED" in message
    assert "source_term" in message, "must name the channel that cannot carry v"
    assert "constraint=" in message, "must name the replacement"
    assert "#2036" in message, "must name the replacement's own limits"


def test_the_refusal_survives_a_penalty_parameter():
    """No value of the knob makes the term depend on `v`, so none of them re-enable the class."""
    for penalty in (1e-6, 1.0, 1e4, 1e12):
        with pytest.raises(NotImplementedError):
            PenaltyHJBSolver(
                inner_solver=_StubInner(),
                obstacle=lambda x: np.zeros(np.asarray(x).shape[0]),
                penalty_parameter=penalty,
            )


def test_it_is_still_exported():
    """Retired, not deleted: the public name must resolve and explain itself.

    Removing it from `hjb_solvers.__init__` would turn an explanation into an ImportError, which
    tells a caller nothing about why their obstacle was never enforced.
    """
    from mfgarchon.alg.numerical import hjb_solvers

    assert "PenaltyHJBSolver" in hjb_solvers.__all__
    assert hjb_solvers.PenaltyHJBSolver is PenaltyHJBSolver

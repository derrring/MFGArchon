#!/usr/bin/env python3
"""Issue #2207: a problem-level source must reach the solver through EVERY coupling loop.

WHAT THIS PINS, AND WHY IT IS NOT A HAPPY-PATH ASSERTION
--------------------------------------------------------
The mutation is ``source_term_hjb = None``: solve the same problem with and without the source and
compare. A loop that composes and forwards moves; a loop that drops it returns a bit-identical
``U``. That is the whole discrimination, and before #2207 it separated the three loops here into
two groups -- measured on this fixture, ``FixedPointIterator`` moved ``max|U|`` by 8.75 having
called the source 15 times, while ``FictitiousPlayIterator`` and ``BlockGaussSeidelIterator`` both
returned ``0.000000e+00`` having called it **0 times**. Both were solving a different equation than
the one posed, with no error and no warning.

The call counter is the second half and is not redundant with the norm. A norm of zero has two
causes -- the source never arrived, or it arrived and cancelled -- and only the counter separates
them. It is also what would catch a future regression that composes the source, forwards it, and
has the solver discard it downstream: the norm would go to zero and look exactly like the defect
this file was written for, while the counter would stay non-zero and say where to look.

WHY d = 1
---------
The property is source-term plumbing -- named in AGENTS.md as fully expressed in one dimension.
Nothing here is directional: there is no normal/tangential split, no axis pairing, no corner. A 2-D
fixture would buy runtime and no discrimination.

NOT COVERED, AND NAMED RATHER THAN IMPLIED
------------------------------------------
``RegimeSwitchingIterator`` and ``GraphMFGSolver`` also reach a solver's ``source_term``, and are
NOT rows here. Both compose a cross-regime / cross-node term of their own into the same channel, so
for them the question is how a problem-level source COMBINES with it -- addition or replacement,
and with which sign -- which #2207 deliberately left open (cf. #1681, #1803).
``MultiPopulationIterator`` is not a ``BaseCouplingIterator`` subclass at all and is folded into
``MFGProblem`` by #2173.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.block_iterators import BlockGaussSeidelIterator
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.model import Conditions, Model
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_N = 21
_ITERATIONS = 3

#: Looser than the inner Newton's own default. A tighter outer tolerance makes the solver warn
#: that the outer loop asks for more than the inner one delivers -- true, and not this file's
#: subject.
_TOLERANCE = 1e-6

#: Large against every field this fixture produces, so a solver that receives it cannot return a
#: value function within rounding of the sourceless one. A small source would make a real drop and
#: a real forward differ by an amount the assertion could not name a threshold for.
_SOURCE_MAGNITUDE = 50.0

_LOOPS = [FixedPointIterator, FictitiousPlayIterator, BlockGaussSeidelIterator]

#: The loop whose row must move for any other row to mean anything. Without it, a fixture that
#: never delivers a source at all reports every loop as dropping one, and the file passes while
#: measuring nothing.
_POSITIVE_CONTROL = FixedPointIterator


class _CountingSource:
    """The problem-level ``(x, m, v, t)`` source, counting its own invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x, m, v, t):
        self.calls += 1
        a = np.asarray(x, dtype=float)
        return np.full(a.shape[0] if a.ndim == 2 else a.size, _SOURCE_MAGNITUDE)


def _m_initial(x):
    """Mass exactly 1 on this node-centred grid: the cosine sums to 0 over a whole period, so the
    trapezoid (#2145) gives 1 with no normalising constant to keep in step."""
    return 1.0 + 0.5 * np.cos(2.0 * np.pi * np.asarray(x, dtype=float))


def _problem(source=None):
    """Built through the v1.0 API on purpose: the legacy keyword form warns, and #2203 is already
    open against the two fixtures that still use it. A third would be a third."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=no_flux_bc(dimension=1))
    extra = {"source_term_hjb": source} if source is not None else {}
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: np.asarray(m) * 0.0,
                coupling_dm=lambda m: np.asarray(m) * 0.0,
            ),
            sigma=0.3,
        ),
        domain=grid,
        conditions=Conditions(m_initial=_m_initial, u_terminal=lambda x: np.asarray(x, dtype=float) * 0.0, T=0.2),
        Nt=5,
        coupling_coefficient=0.0,
        **extra,
    )


def _solve(loop_cls, source):
    problem = _problem(source)
    loop = loop_cls(problem, HJBFDMSolver(problem), FPFDMSolver(problem))
    try:
        result = loop.solve(max_iterations=_ITERATIONS, tolerance=_TOLERANCE)
    except TypeError:  # BlockIterator.solve has no `tolerance=` keyword
        result = loop.solve(max_iterations=_ITERATIONS)
    U = result[0] if isinstance(result, tuple) else getattr(result, "U", loop.U)
    return np.asarray(U, dtype=float)


def _measure(loop_cls):
    """``(max|U_with - U_without|, number of source invocations)`` for one coupling loop."""
    without = _solve(loop_cls, None)
    source = _CountingSource()
    with_source = _solve(loop_cls, source)
    return float(np.nanmax(np.abs(with_source - without))), source.calls


@pytest.mark.parametrize("loop_cls", _LOOPS, ids=lambda c: c.__name__)
def test_the_problems_source_reaches_the_solver(loop_cls):
    delta, calls = _measure(loop_cls)

    assert calls > 0, (
        f"{loop_cls.__name__} never invoked the problem's source_term_hjb. The coupling loop did "
        f"not compose it (Issue #2207): pass the iterates to _build_hjb_kwargs / _build_fp_kwargs "
        f"rather than calling the solver directly."
    )
    assert delta > 0.0, (
        f"{loop_cls.__name__} invoked the source {calls} times and the value function did not "
        f"move (max|dU| = {delta:.6e}). The source was composed but did not reach the "
        f"discretisation -- look downstream of the coupling layer, not at it."
    )


def test_the_measurement_can_distinguish_a_drop_from_a_forward():
    """Control on the instrument, in the same run as the rows above.

    ``_measure`` compares two solves of a problem this file builds. If the fixture were mis-built
    -- a source too small to move the answer, a solver that ignores it, ``_ITERATIONS`` too low for
    the source to propagate -- every row above would report a drop, and the file would fail for a
    reason having nothing to do with the coupling loops. This asserts the positive direction on the
    one loop that has composed and forwarded a source since #1424.
    """
    delta, calls = _measure(_POSITIVE_CONTROL)
    broken = (
        f"The instrument is broken, not the library: {_POSITIVE_CONTROL.__name__} has forwarded a "
        f"composed source since #1424 and this fixture measured max|dU| = {delta:.6e} over "
        f"{calls} invocations. Every other row in this file is uninterpretable until this passes."
    )
    assert calls > 0, broken
    assert delta > 0.0, broken


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

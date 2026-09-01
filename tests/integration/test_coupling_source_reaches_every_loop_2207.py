#!/usr/bin/env python3
"""Issue #2207: a problem-level source must reach the solver through EVERY coupling loop.

WHAT THIS PINS, AND WHY IT IS NOT A HAPPY-PATH ASSERTION
--------------------------------------------------------
The mutation is "the problem field is absent": solve the same problem with and without it and
compare. A loop that composes and forwards moves; a loop that drops it returns a bit-identical
answer. Before #2207 that separated the loops here into two groups -- measured, ``FixedPointIterator``
moved ``max|U|`` by 8.75 having called the source 15 times, while ``FictitiousPlayIterator`` and
``BlockGaussSeidelIterator`` both returned ``0.000000e+00`` having called it **0 times**. Both were
solving a different equation than the one posed, with no error and no warning.

THREE FIELDS, NOT ONE -- AND WHY
--------------------------------
``MFGProblem`` carries three composable source fields and they travel different channels:
``source_term_hjb`` and ``nonlocal_operator`` through ``compose_hjb_source``, ``source_term_fp``
through ``compose_fp_source``. The first version of this file exercised only ``source_term_hjb``,
and an adversarial review measured what that cost: disabling the FP composition entirely
(``compose_fp_source(...)`` -> ``None`` inside the owner) left this file **4 passed**. Half of what
the change claims was unpinned by the test written to pin it. Each field now has a row, and each is
measured on the array it actually moves -- an HJB source on ``U``, an FP source on ``M``.

The call counter is the second half and is not redundant with the norm. A norm of zero has two
causes -- the source never arrived, or it arrived and cancelled -- and only the counter separates
them. It is also what would catch a regression that composes and forwards correctly while the
solver discards downstream: the norm would go to zero and look exactly like the defect this file
exists for, while the counter would stay non-zero and say where to look.

WHY d = 1
---------
The property is source-term plumbing -- named in AGENTS.md as fully expressed in one dimension.
Nothing here is directional: no normal/tangential split, no axis pairing, no corner. A 2-D fixture
would buy runtime and no discrimination.

NOT COVERED, AND NAMED RATHER THAN IMPLIED
------------------------------------------
``RegimeSwitchingIterator`` and ``GraphMFGSolver`` also reach a solver's ``source_term``, and are
NOT rows here. Both compose a cross-regime / cross-node term of their own into the same channel, so
for them the question is how a problem-level source COMBINES with it -- addition or replacement, and
with which sign -- which #2207 deliberately left open (cf. #1681, #1803). Neither routes through
``_build_*_kwargs``, so the required-iterate mechanism cannot reach them either.
``MultiPopulationIterator`` is not a ``BaseCouplingIterator`` subclass at all; #2173 proposes
folding ``MultiPopulationProblem`` into ``MFGProblem``, which is the related but distinct question.
"""

from __future__ import annotations

import inspect

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import BaseCouplingIterator
from mfgarchon.alg.numerical.coupling.block_iterators import BlockGaussSeidelIterator, BlockJacobiIterator
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

#: Large against every field this fixture produces, so a solver that receives it cannot return an
#: answer within rounding of the sourceless one. A small source would make a real drop and a real
#: forward differ by an amount no threshold could separate.
_MAGNITUDE = 50.0

#: Looser than the inner Newton's own default. A tighter outer tolerance makes the solver warn that
#: the outer loop asks for more than the inner one delivers -- true, and not this file's subject.
_TOLERANCE = 1e-6

_LOOPS = [FixedPointIterator, FictitiousPlayIterator, BlockGaussSeidelIterator, BlockJacobiIterator]

#: The loop whose row must move for any other row to mean anything. It has composed and forwarded a
#: source since #1424, so a zero here indicts the fixture, not the library.
_POSITIVE_CONTROL = FixedPointIterator


class _CountingSource:
    """The problem-level ``(x, m, v, t)`` callback, counting its own invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x, m, v, t):
        self.calls += 1
        a = np.asarray(x, dtype=float)
        return np.full(a.shape[0] if a.ndim == 2 else a.size, _MAGNITUDE)


def _nonlocal_operator():
    """A dense ASYMMETRIC (N, N) kernel, applied as ``K @ v_t`` (#1259).

    Asymmetric because a symmetric kernel could not express a transpose defect at all -- a necessary
    condition, and AGENTS.md's reason for breaking the symmetry of the 2-D MMS pair (#2016). It is
    NOT sufficient and this file does not exploit it: every assertion here is ``|delta| > 0`` against
    a no-kernel baseline, which a transposed kernel satisfies too. Measured -- mutating
    ``nonlocal_operator @ v_t`` to ``.T @ v_t`` leaves this file at 17 passed. The asymmetry is kept
    so a future provenance pin (#2211) has a fixture that can separate the two; claiming it buys
    discrimination here would be the "necessary read as sufficient" error.
    """
    i = np.arange(_N)[:, None]
    j = np.arange(_N)[None, :]
    return (_MAGNITUDE / _N) * np.exp(-0.5 * np.abs(i - 2 * j) / _N)


def _u_terminal(x):
    """NOT identically zero, and that is load-bearing rather than decoration.

    The nonlocal term is ``K @ v_t`` (#1259). With a zero terminal cost and zero coupling this
    problem has ``U == 0`` everywhere, so ``K @ v_t == 0`` for EVERY kernel and the nonlocal row
    cannot fail -- measured: ``max|U| = 0.0`` on the first draft of this fixture, which failed the
    nonlocal row on all four loops INCLUDING the positive control. That is the file's own control
    doing its job: when the control row fails, the fixture is wrong and not the library.
    """
    a = np.asarray(x, dtype=float)
    return (a - 0.5) ** 2


def _m_initial(x):
    """Mass 1 on this node-centred grid to within a ulp, with no normalising constant to keep in
    step with #2145: under TRAPEZOID weights the cosine contributes 0 over a whole period, so the
    constant carries the whole mass. Under the plain node sum it does not -- `np.sum(m) * dx` is
    1.075 here, a 7.5% error -- and which measure is meant is the whole subject of #2145, so the
    weighting is stated rather than left to "sums to 0". ``np.trapezoid`` returns exactly ``1.0``; the library's own ``_measure_initial_density()``
    returns ``0.9999999999999999``, which is what matters because that is the value its warning
    tests. An earlier draft of this line said "exactly 1", which is true of one of those two
    measures and was written without running the other."""
    return 1.0 + 0.5 * np.cos(2.0 * np.pi * np.asarray(x, dtype=float))


def _problem(**source_fields):
    """Built through the v1.0 API on purpose: the legacy keyword form warns, and #2203 is already
    open against the two fixtures that still use it. A third would be a third."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=no_flux_bc(dimension=1))
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
        conditions=Conditions(m_initial=_m_initial, u_terminal=_u_terminal, T=0.2),
        Nt=5,
        coupling_coefficient=0.0,
        **source_fields,
    )


def _solve(loop_cls, **source_fields):
    """``(U, M)`` from one coupling loop."""
    problem = _problem(**source_fields)
    loop = loop_cls(problem, HJBFDMSolver(problem), FPFDMSolver(problem))
    result = loop.solve(max_iterations=_ITERATIONS, tolerance=_TOLERANCE)
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=float), np.asarray(result[1], dtype=float)
    return np.asarray(loop.U, dtype=float), np.asarray(loop.M, dtype=float)


#: (field name, factory, which array the field moves). The pairing matters: an FP source leaves U
#: untouched at this coupling strength, so measuring it on U would report a correct forward as a
#: drop -- the same fixture mistake in the opposite direction.
_FIELDS = [
    ("source_term_hjb", _CountingSource, "U"),
    ("source_term_fp", _CountingSource, "M"),
    ("nonlocal_operator", _nonlocal_operator, "U"),
]


def _measure(loop_cls, field, factory, which):
    """``(max|Δ| on the array this field moves, invocation count or None)``."""
    u0, m0 = _solve(loop_cls)
    obj = factory()
    u1, m1 = _solve(loop_cls, **{field: obj})
    a, b = (u0, u1) if which == "U" else (m0, m1)
    calls = getattr(obj, "calls", None)  # a nonlocal_operator is a matrix and counts nothing
    return float(np.nanmax(np.abs(b - a))), calls


@pytest.mark.parametrize(("field", "factory", "which"), _FIELDS, ids=[f[0] for f in _FIELDS])
@pytest.mark.parametrize("loop_cls", _LOOPS, ids=lambda c: c.__name__)
def test_the_problems_source_reaches_the_solver(loop_cls, field, factory, which):
    delta, calls = _measure(loop_cls, field, factory, which)

    if calls is not None:
        assert calls > 0, (
            f"{loop_cls.__name__} never invoked the problem's {field}. The coupling loop did not "
            f"compose it (Issue #2207): pass the iterates to _build_hjb_kwargs / _build_fp_kwargs "
            f"rather than calling the solver directly."
        )
    assert delta > 0.0, (
        f"{loop_cls.__name__}: {field} left {which} bit-identical (max|d{which}| = {delta:.6e}"
        + (f", callback invoked {calls} times" if calls is not None else "")
        + "). The problem posed and the problem solved are different."
    )


def test_the_measurement_can_distinguish_a_drop_from_a_forward():
    """Control on the instrument, in the same run as the rows above.

    If the fixture were mis-built -- a source too small to move the answer, ``_ITERATIONS`` too low
    for it to propagate, the wrong array compared -- every row above would report a drop and the
    file would fail for a reason having nothing to do with the coupling loops.
    """
    for field, factory, which in _FIELDS:
        delta, calls = _measure(_POSITIVE_CONTROL, field, factory, which)
        broken = (
            f"The instrument is broken, not the library: {_POSITIVE_CONTROL.__name__} has forwarded "
            f"a composed source since #1424, and this fixture measured {field} -> "
            f"max|d{which}| = {delta:.6e}. Every other row in this file is uninterpretable."
        )
        assert delta > 0.0, broken
        if calls is not None:
            assert calls > 0, broken


@pytest.mark.parametrize("builder", ["_build_hjb_kwargs", "_build_fp_kwargs"])
@pytest.mark.parametrize("name", ["M", "U"])
def test_both_iterates_are_required_not_optional(builder, name):
    """Issue #2207: the omission this change removes must not be re-expressible.

    A default of ``None`` on EITHER iterate restores the hole -- a loop that forgets it composes
    nothing, raises nothing, and solves without the source. Asserted on the signature rather than
    through a ``pytest.raises(TypeError, match=...)``, because an adversarial review measured that
    ``match="M"`` is satisfied by the message for a missing ``M`` alone: giving ``U`` a default left
    the whole file green, so half the mechanism was unpinned by the test that named it.
    """
    param = inspect.signature(getattr(BaseCouplingIterator, builder)).parameters[name]
    assert param.default is inspect.Parameter.empty, (
        f"BaseCouplingIterator.{builder} gave '{name}' a default. The composition is then skippable "
        f"by omission, which is exactly the #2207 hole one level up."
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("loop_cls", [BlockGaussSeidelIterator, BlockJacobiIterator], ids=lambda c: c.__name__)
def test_the_strict_adjoint_path_refuses_an_fp_source(loop_cls):
    """#2207: the one path that has no route for a source must refuse it, not drop it.

    `BlockIterator._solve_fp` returns to `_solve_fp_strict_adjoint` BEFORE reaching the builders,
    so the owner is unreachable from `adjoint_mode != "off"` and the source has nowhere to go --
    that path assembles and steps its own FP operator instead of calling the FP solver. Dropping it
    silently is #1424; dropping it HERE while the HJB source still arrives through the builder is
    worse than either, because the two halves of one problem would then be solved from different
    equations.

    Three arms, and the last two are the controls without which the first proves nothing about
    over-firing: an FP source must raise; no source at all must still run; and an HJB source, which
    this path CAN carry (it enters through `build_linearized_operator`, not the FP transport
    operator), must also still run.

    This test exists because three review rounds passed over the branch's only user-visible
    behaviour change with no oracle named -- AGENTS.md makes naming one mandatory, and "verified by
    a probe I ran" corresponds to no committed artifact.
    """
    p_src = _problem(source_term_fp=_CountingSource())
    solver = loop_cls(p_src, HJBFDMSolver(p_src), FPFDMSolver(p_src), adjoint_mode="jacobian_transpose")
    with pytest.raises(NotImplementedError, match="cannot carry an FP source term"):
        solver.solve(max_iterations=2, tolerance=_TOLERANCE)

    for label, fields in (("no source", {}), ("hjb source only", {"source_term_hjb": _CountingSource()})):
        q = _problem(**fields)
        loop = loop_cls(q, HJBFDMSolver(q), FPFDMSolver(q), adjoint_mode="jacobian_transpose")
        u, _ = (loop.solve(max_iterations=2, tolerance=_TOLERANCE), None)
        assert u is not None, f"the guard over-fired on {label}: this configuration must still run"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

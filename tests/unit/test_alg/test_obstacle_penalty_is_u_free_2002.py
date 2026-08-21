"""#2002 -- the ``obstacle`` field is documented as a variational inequality and implemented
as a position penalty, in two places that differ by 1e10.

``mfg_problem.py`` declares ``obstacle`` as ``v >= Psi(x)``. Two code paths act on it and
**neither reads ``v``**:

===========================================  =============================  ==============
path                                         expression                     ``psi = 0.5``
===========================================  =============================  ==============
``coupling/source_composition.py``           ``(1 / eps) * max(0, psi)``    ``5.0e-07``
``hjb_solvers/hjb_penalty.py``               ``penalty * max(0, psi)``      ``5.0e+03``
===========================================  =============================  ==============

The headline number is that those are 1e10 apart, but the disqualifying property is cheaper to
state and independent of both constants: **the term is identical at a node that satisfies the
constraint and one that violates it**, because ``max(0, psi)`` contains no ``v``. It penalises
position, not violation.

**The two are not alternatives.** ``FixedPointIterator`` passes ``compose_hjb_source``'s closure
as ``source_term=`` to whatever HJB solver it holds, and ``PenaltyHJBSolver.penalized_source``
starts from ``base = source_term(t, x)`` and *adds* its own term. A problem with ``obstacle`` set
and wrapped in ``PenaltyHJBSolver`` applies both, summed. The 1e10 ratio is therefore a statement
about two spellings of one knob, not a discrepancy between rival implementations to reconcile.

Why the defect survived. It is NOT that no test exercises ``problem.obstacle`` --
``test_issue1285_newton_fail_loud.py`` builds a problem with ``obstacle=`` and asserts
``u_gap < 1e-8``. But that test compares **Picard against coupled-Newton**, and both consume the
one ``compose_hjb_source``, so no amount of ``v``-freedom in the shared term could make it fail.
The field has end-to-end coverage of *path agreement* and none of *what the term computes*.

A constraint-shaped alternative exists under a different entry point --
``ObstacleConstraint.project`` (#591), applied by ``HJBFDMSolver(constraint=...)``. It reads ``u``
and enforces ``u >= psi`` on what it returns, which is more than these terms can say. Calling it
*correct* overstates it: that path clips post-hoc in 1D and returns an infeasible terminal slice
in nD (#2036).

**Retirement condition.** When #2002 makes either term read ``v``:

- ``test_composed_hjb_source_is_byte_identical_for_violated_and_satisfied_u`` and
  ``test_penalty_source_is_byte_identical_for_violated_and_satisfied_v`` assert current-wrong
  behaviour on purpose and must be **DELETED, not adjusted**.
- ``test_the_two_obstacle_spellings_are_1e10_apart`` reddens under a ``v``-dependence fix too,
  because the fix moves the measured values -- **update its expected values, do not delete it**.
  It is the only guard on the two constructor defaults, and it fires on its own when the knobs
  are unified without any ``v`` change (measured).
- ``test_the_constraint_owner_exists_and_is_u_dependent`` survives; it describes the target.
- Expect a fourth failure outside this file:
  ``test_issue1361_source_composition.py::test_hjb_composition_matches_reference_and_delegate``
  reimplements the production obstacle formula as its oracle (lines 58-71) and reddens under any
  compose-side fix. That is the oracle needing the same edit, not a regression.
"""

import numpy as np

from mfgarchon.alg.numerical.coupling.source_composition import compose_hjb_source
from mfgarchon.alg.numerical.hjb_solvers.hjb_penalty import PenaltyHJBSolver
from mfgarchon.geometry.boundary import ObstacleConstraint

EPS_DEFAULT = 1e6  # source_composition.py: getattr(problem, "_penalty_eps", 1e6)
PENALTY_DEFAULT = 1e4  # hjb_penalty.py: penalty_parameter -- NOT passed below, so the real
# constructor default is what gets exercised.

PSI = 0.5
NX = 5
NT = 3
# Two value functions on opposite sides of the obstacle. A term enforcing `v >= Psi` vanishes on
# SATISFIED and is strictly positive on VIOLATED. Neither may be all-zero: `max(0, psi - 0)` is
# `max(0, psi)`, so a zero `v` cannot distinguish the fix from the defect.
U_VIOLATED = np.full((NT, NX), -9.0)
U_SATISFIED = np.full((NT, NX), +9.0)


def _flat_obstacle(x):
    return np.full(np.asarray(x).shape[0], PSI)


def _x():
    return np.linspace(0.0, 1.0, NX).reshape(-1, 1)


class _ProblemStub:
    """Minimal duck type carrying only the fields the obstacle branch reads.

    Verified byte-equal against a real ``MFGProblem(obstacle=...)``; it omits ``_penalty_eps`` on
    purpose so the ``getattr`` default is the thing under test.
    """

    obstacle = staticmethod(_flat_obstacle)
    nonlocal_operator = None
    source_term_hjb = None
    dt = 0.1


class _SpyInner:
    """Captures the ``source_term`` the wrapper hands down, per call."""

    problem = _ProblemStub()
    config = None

    def __init__(self):
        self.captured = []

    def solve_hjb_system(self, M, U_T, U_prev, volatility_field=None, source_term=None):
        self.captured.append(source_term)
        return np.zeros_like(U_T)


def _penalty_source_for(u_prev):
    """Drive the real ``PenaltyHJBSolver`` with a given ``U_coupling_prev`` and return the
    ``source_term`` closure it produced, evaluated on the grid."""
    inner = _SpyInner()
    solver = PenaltyHJBSolver(inner_solver=inner, obstacle=_flat_obstacle)
    solver.solve_hjb_system(np.zeros((NT, NX)), np.zeros(NX), u_prev)
    assert inner.captured, "sanity: the wrapper must pass a source_term down"
    return inner.captured[-1](0.0, _x())


def test_the_constraint_owner_exists_and_is_u_dependent():
    """``ObstacleConstraint.project`` enforces ``u >= psi`` and reads ``u``.

    SURVIVES the fix. Establishes that #2002 is about a term that cannot express the constraint,
    not about the constraint being hard to express. It says nothing about whether the solver path
    around ``project`` solves an obstacle problem -- it does not; see #2036.
    """
    psi = np.full(NX, PSI)
    u = np.array([1.0, 0.2, 0.5, -3.0, 0.9])  # entries 1 and 3 violate u >= psi

    constraint = ObstacleConstraint(psi, constraint_type="lower")
    projected = constraint.project(u)

    assert np.all(projected >= psi - 1e-14), f"project did not enforce u >= psi: {projected}"
    # Feasible entries untouched -- a projection, not a clamp-everything.
    assert projected[0] == u[0]
    assert projected[4] == u[4]
    # Violated entries land ON the obstacle.
    assert projected[1] == psi[1]
    assert projected[3] == psi[3]
    # And the output genuinely depends on u.
    assert not np.allclose(projected, constraint.project(u + 5.0))


def test_composed_hjb_source_is_byte_identical_for_violated_and_satisfied_u():
    """DEFECT (#2002). Delete on fix -- do not adjust.

    Drives the real ``compose_hjb_source``, which receives ``u_current`` and could therefore
    distinguish the two regimes. It does not.
    """
    x = _x()
    m = np.zeros((NT, NX))

    f_violated = compose_hjb_source(_ProblemStub(), m, U_VIOLATED)
    f_satisfied = compose_hjb_source(_ProblemStub(), m, U_SATISFIED)
    assert f_violated is not None, "sanity: an obstacle field must produce a closure"
    assert f_satisfied is not None

    out_violated = f_violated(0.0, x)
    out_satisfied = f_satisfied(0.0, x)

    assert out_satisfied.shape == (NX,), f"degenerate grid would vacuously pass: {out_satisfied.shape}"
    assert np.array_equal(out_violated, out_satisfied), (
        "the obstacle source distinguished the two regimes -- if #2002 is fixed, DELETE this test"
    )
    # Not merely small-but-different: positive where a constraint penalty must vanish.
    assert np.all(out_satisfied > 0.0), "u is 9 above the obstacle everywhere; a penalty must be 0"
    assert np.allclose(out_satisfied, (1.0 / EPS_DEFAULT) * PSI)


def test_penalty_source_is_byte_identical_for_violated_and_satisfied_v():
    """DEFECT (#2002). Delete on fix -- do not adjust.

    ``PenaltyHJBSolver.solve_hjb_system`` receives ``U_coupling_prev``, so the ``v`` its own
    withdrawn docstring promised to use is already in hand. Drive it twice with ``v`` on opposite
    sides of the obstacle and compare the closures it produced.

    The comparison must use a NON-ZERO ``v``: with ``v = 0`` the fix ``max(0, psi - v)`` collapses
    to ``max(0, psi)`` and this test would pass on a fixed implementation.
    """
    out_violated = _penalty_source_for(U_VIOLATED)
    out_satisfied = _penalty_source_for(U_SATISFIED)

    assert out_satisfied.shape == (NX,), f"degenerate grid would vacuously pass: {out_satisfied.shape}"
    assert np.array_equal(out_violated, out_satisfied), (
        "the penalty source distinguished v = -9 from v = +9 -- if #2002 is fixed, DELETE this test"
    )
    assert np.all(out_satisfied > 0.0), "v is 9 above the obstacle everywhere; a penalty must be 0"


def test_the_two_obstacle_spellings_are_1e10_apart():
    """Pins BOTH constructor defaults. Reddens under a ``v``-dependence fix as well -- update its
    numbers rather than deleting it -- and fires alone when the knobs are unified.

    Neither default is passed in by the caller here: ``_ProblemStub`` omits ``_penalty_eps`` so
    ``source_composition``'s ``getattr`` default fires, and ``_penalty_source_for`` omits
    ``penalty_parameter`` so the constructor default fires. Changing either silently moves the
    gap and reddens this test.
    """
    compose_out = compose_hjb_source(_ProblemStub(), np.zeros((NT, NX)), U_VIOLATED)(0.0, _x())
    penalty_out = _penalty_source_for(U_VIOLATED)

    assert np.allclose(compose_out, (1.0 / EPS_DEFAULT) * PSI), f"eps default moved: {compose_out}"
    assert np.allclose(penalty_out, PENALTY_DEFAULT * PSI), f"penalty default moved: {penalty_out}"
    assert np.allclose(penalty_out / compose_out, 1.0e10), f"the gap moved: {penalty_out / compose_out}"

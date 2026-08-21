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

**Two of the four tests here have been RETIRED, by their own condition (2026-08-21).**
``PenaltyHJBSolver`` was retired in #2002: its constructor now raises, because the penalty it
injected could never read ``v`` -- ``source_term`` is ``(t, x) -> array`` and there is nowhere for
the value function to enter. So ``test_penalty_source_is_byte_identical_for_violated_and_satisfied_v``
is gone, per its own "delete on fix, do not adjust", and
``test_the_two_obstacle_spellings_are_1e10_apart`` is gone as well -- it compared two spellings of
one knob and there is now only one spelling. That second deletion goes beyond what its condition
said ("update its numbers rather than deleting it"), because that instruction assumed the second
side would still exist.

**Remaining retirement condition.** When #2002 makes the compose-side term read ``v``:

- ``test_composed_hjb_source_is_byte_identical_for_violated_and_satisfied_u`` asserts
  current-wrong behaviour on purpose and must be **DELETED, not adjusted**.
- ``test_the_constraint_owner_exists_and_is_u_dependent`` survives; it describes the target.
- Expect a fourth failure outside this file:
  ``test_issue1361_source_composition.py::test_hjb_composition_matches_reference_and_delegate``
  reimplements the production obstacle formula as its oracle (lines 58-71) and reddens under any
  compose-side fix. That is the oracle needing the same edit, not a regression.
"""

import numpy as np

from mfgarchon.alg.numerical.coupling.source_composition import compose_hjb_source
from mfgarchon.geometry.boundary import ObstacleConstraint

EPS_DEFAULT = 1e6  # source_composition.py: getattr(problem, "_penalty_eps", 1e6)

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

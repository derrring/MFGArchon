"""#2002 -- ``problem.obstacle`` is a second, broken spelling of a concept the library
already implements correctly.

``mfg_problem.py`` declares ``obstacle`` as the variational inequality ``v >= Psi(x)``.
Two code paths act on it and **neither reads ``v``**:

===========================================  =============================  ==============
path                                         expression                     ``psi = 0.5``
===========================================  =============================  ==============
``source_composition.py``                    ``(1 / eps) * max(0, psi)``    ``5.0e-07``
``hjb_solvers/hjb_penalty.py``               ``penalty * max(0, psi)``      ``5.0e+03``
===========================================  =============================  ==============

The headline number is that those are 1e10 apart, but the disqualifying property is
cheaper to state: **the term is identical at a node that satisfies the constraint and a
node that violates it**, because ``max(0, psi)`` contains no ``v``. It penalises position,
not violation. A docstring in ``source_composition`` asserted the two paths "match rather
than silently diverge"; that sentence is withdrawn (see the module docstring there).

What makes this a single-source-of-truth defect rather than a missing feature: the
constraint IS implemented, correctly and reachably, under a different entry point --
``ObstacleConstraint.project`` (#591), which ``HJBFDMSolver`` calls at ``hjb_fdm.py:499``
and ``:762`` when constructed with ``constraint=``. So the repo holds two spellings of one
concept, and the reachable-from-``problem.obstacle`` one is the broken one.

These tests pin the defect so that reading the (now corrected) comments cannot restore the
belief that the paths agree. **Retirement condition:** when #2002 routes ``problem.obstacle``
to the constraint machinery or makes it fail loud, ``test_obstacle_term_cannot_tell_satisfied_from_violated``
and ``test_the_two_obstacle_scalings_are_1e10_apart`` must be DELETED, not adjusted -- they
assert current-wrong behaviour on purpose. ``test_the_correct_owner_exists_and_is_u_dependent``
survives; it describes the target.
"""

import numpy as np

from mfgarchon.alg.numerical.coupling.source_composition import compose_hjb_source
from mfgarchon.alg.numerical.hjb_solvers.hjb_penalty import PenaltyHJBSolver
from mfgarchon.geometry.boundary import ObstacleConstraint

# The two literal defaults, read off the call sites named in the module docstring.
EPS_DEFAULT = 1e6  # source_composition.py: getattr(problem, "_penalty_eps", 1e6)
PENALTY_DEFAULT = 1e4  # hjb_penalty.py: penalty_parameter

PSI = 0.5
NX = 5
# Two value functions on opposite sides of the obstacle. A term that enforced `v >= Psi`
# would vanish on SATISFIED and be strictly positive on VIOLATED.
U_VIOLATED = np.full((3, NX), -9.0)
U_SATISFIED = np.full((3, NX), +9.0)


def _flat_obstacle(x):
    return np.full(np.asarray(x).shape[0], PSI)


class _ProblemStub:
    """Minimal duck type carrying only the fields the obstacle branch reads."""

    obstacle = staticmethod(_flat_obstacle)
    nonlocal_operator = None
    source_term_hjb = None
    dt = 0.1


def test_the_correct_owner_exists_and_is_u_dependent():
    """``ObstacleConstraint.project`` enforces ``u >= psi`` and reads ``u``.

    This is the test that SURVIVES the fix: it establishes that #2002 is a routing
    defect, not a missing capability. Without it the other two read as "this is hard".
    """
    psi = np.full(NX, PSI)
    u = np.array([1.0, 0.2, 0.5, -3.0, 0.9])  # entries 1 and 3 violate u >= psi

    constraint = ObstacleConstraint(psi, constraint_type="lower")
    projected = constraint.project(u)

    assert np.all(projected >= psi - 1e-14), f"project did not enforce u >= psi: {projected}"
    # Feasible entries must be left alone -- a projection, not a clamp-everything.
    assert projected[0] == u[0]
    assert projected[4] == u[4]
    # Violated entries land ON the obstacle.
    assert projected[1] == psi[1]
    assert projected[3] == psi[3]
    # And the output genuinely depends on u.
    assert not np.allclose(projected, constraint.project(u + 5.0))


def test_composed_hjb_source_is_byte_identical_for_violated_and_satisfied_u():
    """DEFECT (#2002). Delete on fix -- do not adjust.

    Drives the REAL ``compose_hjb_source``, which receives ``u_current`` and therefore
    could distinguish the two regimes. It does not: the obstacle branch computes
    ``max(0, psi)``, so feeding it a ``u`` nine units BELOW the obstacle and one nine units
    ABOVE returns the same array. Any fix that makes the term read ``v`` breaks this.
    """
    x = np.linspace(0.0, 1.0, NX).reshape(-1, 1)
    m = np.zeros((3, NX))

    f_violated = compose_hjb_source(_ProblemStub(), m, U_VIOLATED)
    f_satisfied = compose_hjb_source(_ProblemStub(), m, U_SATISFIED)
    assert f_violated is not None, "sanity: an obstacle field must produce a closure"
    assert f_satisfied is not None

    out_violated = f_violated(0.0, x)
    out_satisfied = f_satisfied(0.0, x)

    # The defect, stated as the property that disqualifies the term.
    assert np.array_equal(out_violated, out_satisfied), (
        "the obstacle source distinguished the two regimes -- if #2002 is fixed, DELETE "
        "this test rather than loosening it"
    )
    # And it is not merely small-but-different: it is positive where it should vanish.
    assert np.all(out_satisfied > 0.0), (
        "u is 9 above the obstacle everywhere, so a constraint penalty must be exactly 0"
    )
    assert np.allclose(out_satisfied, (1.0 / EPS_DEFAULT) * PSI)


def test_penalty_solver_source_is_u_free_and_1e10_from_the_other_path():
    """DEFECT (#2002). Delete on fix -- do not adjust.

    Drives the REAL ``PenaltyHJBSolver``, capturing the ``source_term`` it hands to its
    inner solver, and evaluates it. Pins both halves: no ``v`` dependence, and a scaling
    1e10 from ``compose_hjb_source`` -- one divides by ``eps`` (defaulted LARGE, so the
    coefficient is 1e-6), the other multiplies by ``penalty_parameter`` (1e4).
    """
    x = np.linspace(0.0, 1.0, NX).reshape(-1, 1)
    captured = {}

    class _SpyInner:
        problem = _ProblemStub()
        config = None

        def solve_hjb_system(self, M, U_T, U_prev, volatility_field=None, source_term=None):
            captured["source_term"] = source_term
            return np.zeros_like(U_T)

    solver = PenaltyHJBSolver(
        inner_solver=_SpyInner(),
        obstacle=_flat_obstacle,
        penalty_parameter=PENALTY_DEFAULT,
    )
    solver.solve_hjb_system(np.zeros((3, NX)), np.zeros(NX), np.zeros((3, NX)))

    penalized = captured["source_term"]
    assert penalized is not None, "sanity: the wrapper must pass a source_term down"

    # The closure takes (t, x) only -- there is nowhere for v to enter.
    value = penalized(0.0, x)
    assert np.allclose(value, PENALTY_DEFAULT * PSI), f"hjb_penalty term moved: {value}"

    # The other path, measured through its own real code, on the same obstacle.
    other = compose_hjb_source(_ProblemStub(), np.zeros((3, NX)), U_VIOLATED)(0.0, x)
    assert np.allclose(other, (1.0 / EPS_DEFAULT) * PSI)
    assert np.allclose(value / other, 1.0e10), f"the gap between the two paths moved: {value / other}"

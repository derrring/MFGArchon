"""Pinning tests for the H<->L convention and the admissible control set.

Issue #1642, capabilities B1 (hl-convention-pin) and B3 (controlcost-effective-domain).

B1 pins the (V, f) sign convention documented on ``MFGOperatorBase``:

    L(x, alpha, m, t) = L_ctrl(alpha) - V(x, t) - f(m)

asserted in its conjugate form, ``sup_alpha { p.alpha - L } == H``.

Which tests carry that pin, precisely -- a conjugate round trip only
discriminates when the two sides have INDEPENDENT sources for V and f:

- ``TestSeparableRoundTrip.test_conjugate_of_lagrangian_recovers_hamiltonian``
  -- its 9 ``V + f != 0`` rows (3 control costs x 3 non-zero (V, f) cells). These
  carried strict xfail until Issue #1645 (B2) flipped the sign in
  ``SeparableLagrangian.__call__``; they are now the load-bearing pin on the
  Separable side. Its 3 ``V0_f0`` rows are non-discriminating by construction,
  the disputed term being identically zero there.
- ``TestCongestionRoundTrip`` (12 tests = 3 costs x 4 (V, f) cells). The L side
  is an analytic closed form written out in this module, NOT
  ``H.legendre_transform()``, so V and f are independently sourced and a sign
  error in ``CongestionHamiltonian`` breaks the identity by 2(V+f).

What these tests catch:

- Adding V and f to BOTH H and L (the pre-B2 ``SeparableLagrangian`` fork):
  breaks the identity by exactly 2(V+f), constant in p (Issue #1645).
- A (V, f) sign error in ``CongestionHamiltonian.__call__``.
- A conjugate/optimization box that ignores the admissible control set: for
  ``L1ControlCost(lambda_=0.5)`` at p=5 an unrestricted sup returns 225.0
  against a true H of 4.5 (50x), because ``lagrangian()`` omits the indicator.
- A second owner of the admissible set diverging from ``effective_domain()``.
- Losing the domain for a Moreau-Yosida-wrapped cost (regularizing H must not
  enlarge A).

What they do NOT catch: the alpha-sign question on ``LagrangianBase.optimal_control``
(Issue #1642, B5) -- every shipped L_ctrl is even in alpha, so both sign
conventions give the same conjugate value. ``test_conjugate_is_alpha_sign_blind``
pins that evenness rather than leaving it implicit.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

import numpy as np
from scipy.optimize import minimize_scalar

import mfgarchon
from mfgarchon.core.hamiltonian import (
    BoundedControlCost,
    CongestionHamiltonian,
    ControlCostBase,
    L1ControlCost,
    QuadraticControlCost,
    SeparableLagrangian,
    _MoreauYosidaControlCost,
)

# Sweep points chosen to straddle every kink the shipped costs have, so the pin
# is not evaluated only on smooth interiors:
#   0.0  L1 dead zone (|p| <= lambda) and the kink at the origin
#   0.5  exactly on the L1 activation threshold lambda=0.5
#   1.5  Bounded still quadratic (threshold lambda*max_control = 2.0)
#   3.0  Bounded saturated, L1 active
#   8.0  deep saturation, where an unrestricted conjugate diverges worst
P_SWEEP = [0.0, 0.5, 1.5, 3.0, 8.0]

X_POINT = np.array([0.25])
M_VALUE = 2.0

# Non-trivial V and f: constant-in-x V is enough because the fork is an additive
# offset, and a constant makes the expected gap 2(V+f) exactly computable.
V_NONZERO = 0.7
F_SLOPE = 0.3


def _potential(x, t):
    return V_NONZERO


def _coupling(m):
    return F_SLOPE * m


def _conjugate(L, p, *, bounds, alpha_sign=1.0):
    """sup_alpha { alpha_sign * p * alpha - L(x, alpha, m, t) } over ``bounds``.

    ``bounds`` must be the admissible control set: the shipped ``lagrangian()``
    implementations omit the indicator of their effective domain, so an
    unrestricted sup silently overshoots (see module docstring).
    """

    def neg_objective(a):
        return -(alpha_sign * p * a - float(L(X_POINT, np.array([a]), M_VALUE, 0.0)))

    res = minimize_scalar(neg_objective, bounds=bounds, method="bounded", options={"xatol": 1e-13})
    return -res.fun


def _search_box(cost: ControlCostBase) -> tuple[float, float]:
    """Admissible set from the single owner, widened when A = R."""
    return cost.effective_domain() or (-50.0, 50.0)


CONTROL_COSTS = {
    "quadratic": lambda: QuadraticControlCost(lambda_=2.0),
    "bounded": lambda: BoundedControlCost(lambda_=1.0, max_control=2.0),
    "l1": lambda: L1ControlCost(lambda_=0.5),
}

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# WHY THE DICT ABOVE IS NOT THE POPULATION (#2386).
#
# `test_conjugate_is_alpha_sign_blind` below used to promise: "If a non-even control cost is ever
# added, this test fails and B5 stops being optional." It could not. Its population was the dict
# above -- three hand-written entries -- so a fourth cost was covered only if whoever added it
# remembered. A hand-written list cannot notice a subclass it does not contain, and the dict was
# already missing `_MoreauYosidaControlCost`, which has been a `ControlCostBase` subclass all along.
#
# The dict stays, because `CONGESTION_L_CTRL` pairs each entry with a conjugate transcribed BY HAND
# and that independence is what gives the congestion round trip teeth. What it gains is a
# completeness gate: `test_the_hand_written_cost_dict_covers_every_subclass` fails and NAMES the
# missing class, so a new cost cannot arrive without its conjugate.
#
# The evenness claim needs no transcription, so it runs over the enumeration directly and covers a
# new cost the moment it is defined.
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# Required constructor arguments we know how to supply. A cost whose required argument is NOT here
# raises rather than being skipped: a silently skipped subclass is the defect this section exists to
# remove, and it would look exactly like coverage.
_REQUIRED_ARG_STRATEGY = {
    "base": lambda: QuadraticControlCost(lambda_=2.0),
    "epsilon": lambda: 0.1,
}


def _concrete_control_cost_classes():
    """Every concrete `ControlCostBase` subclass DEFINED IN THE LIBRARY.

    Recursive, because a future cost may subclass a concrete one rather than the base.

    THE `mfgarchon.` FILTER IS LOAD-BEARING AND WAS ADDED BY A RED GATE. `__subclasses__()` is
    interpreter-global, so under full collection it also returns subclasses defined inside OTHER TEST
    FILES -- `_NarrowL1` and `_NarrowBounded` in `test_sl_control_set_single_source_1642.py`. Without
    the filter this file passed standalone at 42 and failed in the gate, which is precisely the
    "coverage depends on collection order" failure that
    `test_s1_drift_convention_routing.py:236` warns about, arriving in the fix written from its
    lesson. The enumeration was right; the expectation was not.

    Test fixtures are also not the population: a test may legitimately define a non-even cost to
    probe something, and the library's guarantee says nothing about it.
    """

    def walk(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from walk(sub)

    return sorted(
        {
            c
            for c in walk(ControlCostBase)
            if not getattr(c, "__abstractmethods__", ())
            and (c.__module__ == "mfgarchon" or c.__module__.startswith("mfgarchon."))
        },
        key=lambda c: c.__name__,
    )


def _construct(cls):
    """Build `cls` with defaults, filling required arguments from the strategy table."""
    signature = inspect.signature(cls.__init__)
    kwargs = {}
    for name, param in signature.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        assert name in _REQUIRED_ARG_STRATEGY, (
            f"{cls.__name__}.__init__ requires {name!r}, which this file does not know how to "
            f"supply. Add it to _REQUIRED_ARG_STRATEGY -- do NOT skip the class, because a skipped "
            f"subclass is indistinguishable from a covered one and that is exactly the gap #2386 "
            f"was filed about."
        )
        kwargs[name] = _REQUIRED_ARG_STRATEGY[name]()
    return cls(**kwargs)


def _control_cost_classes_in_source():
    """The same population read from the SOURCE, independent of what has been imported.

    `__subclasses__()` only sees classes that some module has already imported, so a cost defined in
    a module nobody imports is invisible to it while looking like an empty result. This is the second
    instrument: an AST scan over the package. The two must agree, and
    `test_the_runtime_enumeration_sees_every_cost_in_the_source` is where they are compared.

    The hazard is not hypothetical -- `test_s1_drift_convention_routing.py:236` records the same
    enumeration reaching 7 classes standalone and 10 under full collection, and its comment is the
    rule this follows: a pin whose coverage depends on collection order is not a pin.
    """
    root = pathlib.Path(mfgarchon.__file__).parent
    found: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = {b.id for b in node.bases if isinstance(b, ast.Name)} | {
                b.attr for b in node.bases if isinstance(b, ast.Attribute)
            }
            if base_names & ({"ControlCostBase"} | found):
                found.add(node.name)
    return found


CONGESTION_SLOPE = 3.0


def _congestion_factor(m):
    return 1.0 + CONGESTION_SLOPE * m


# Analytic control part of L for CongestionHamiltonian, written out by hand.
#
# H_kin(p) = g(p) / c(m) with g = control_cost.evaluate, so the control part of L
# is the conjugate (g/c)*(alpha) = g*(c*alpha) / c. The lambda_ values below
# restate those of CONTROL_COSTS deliberately: an independent transcription is
# the point, so that these formulas share no source with the Hamiltonian.
#
#   quadratic  g* = lambda/2 |a|^2          -> lambda*c/2 |a|^2,   A = R
#   bounded    g* = lambda/2 |a|^2 + I_A    -> lambda*c/2 |a|^2,   A/c
#   l1         g* = lambda |a| + I_A        -> lambda |a|,         A/c
CONGESTION_L_CTRL = {
    "quadratic": lambda a, c_m: 0.5 * 2.0 * c_m * a**2,
    "bounded": lambda a, c_m: 0.5 * 1.0 * c_m * a**2,
    "l1": lambda a, c_m: 0.5 * abs(a),
}


def _congestion_lagrangian(cost_name, potential, coupling):
    """Analytic L(x, alpha, m, t) = L_ctrl^{c(m)}(alpha) - V(x, t) - f(m).

    Independently sourced from ``CongestionHamiltonian`` -- see the note on
    ``TestCongestionRoundTrip`` for why that independence is what makes the
    round trip discriminating.
    """
    l_ctrl = CONGESTION_L_CTRL[cost_name]

    def L(x, alpha, m, t=0.0):
        v = potential(x, t) if potential is not None else 0.0
        f = coupling(m) if coupling is not None else 0.0
        return l_ctrl(float(np.atleast_1d(alpha)[0]), _congestion_factor(m)) - v - f

    return L


def _congestion_box(cost) -> tuple[float, float]:
    """A / c(m). Congestion SHRINKS the admissible set: dom((g/c)*) = dom(g*)/c."""
    domain = cost.effective_domain()
    if domain is None:
        return (-50.0, 50.0)
    c_m = _congestion_factor(M_VALUE)
    return (domain[0] / c_m, domain[1] / c_m)


# (V, f) grid. The separable fork is invisible when V + f == 0, so the
# (zero, zero) cell must pass today and the other three must not.
VF_GRID = {
    "V0_f0": (None, None),
    "V0_fnz": (None, _coupling),
    "Vnz_f0": (_potential, None),
    "Vnz_fnz": (_potential, _coupling),
}


def _separable_params():
    """Cartesian product. Every cell passes since Issue #1645 (B2) closed the fork."""
    params = []
    for cost_name in CONTROL_COSTS:
        for vf_name, (pot, coup) in VF_GRID.items():
            params.append(pytest.param(cost_name, pot, coup, id=f"{cost_name}-{vf_name}"))
    return params


class TestSeparableRoundTrip:
    """sup_alpha { p.alpha - L } == H for SeparableLagrangian / SeparableHamiltonian."""

    @pytest.mark.parametrize(("cost_name", "potential", "coupling"), _separable_params())
    def test_conjugate_of_lagrangian_recovers_hamiltonian(self, cost_name, potential, coupling):
        cost = CONTROL_COSTS[cost_name]()
        L = SeparableLagrangian(control_cost=cost, potential=potential, coupling=coupling)
        H = L.as_hamiltonian()
        box = _search_box(cost)

        for p in P_SWEEP:
            expected = float(H(X_POINT, M_VALUE, np.array([p])))
            got = _conjugate(L, p, bounds=box)
            assert got == pytest.approx(expected, abs=1e-6), (
                f"{cost_name} p={p}: conjugate of L gave {got}, H gave {expected}"
            )

    @pytest.mark.parametrize("cost_cls", _concrete_control_cost_classes(), ids=lambda c: c.__name__)
    def test_conjugate_is_alpha_sign_blind(self, cost_cls):
        """Every shipped L_ctrl is even in alpha, so sup{+p.a - L} == sup{-p.a - L}.

        This is why the (V, f) pin above is independent of the Issue #1642 B5
        alpha-sign question, and it is what #2375 ruling 5's "nothing in this library changes
        value" rests on: the pairing flip is a no-op exactly while every L is even.

        PARAMETRISED OVER THE ENUMERATION, NOT `CONTROL_COSTS` (#2386). The old version promised
        "if a non-even control cost is ever added, this test fails and B5 stops being optional"
        while iterating a three-entry hand-written dict, so it could not keep that promise -- and
        it was already missing `_MoreauYosidaControlCost`. Adding a class to the library is now
        sufficient; nobody has to remember this file.
        """
        cost = _construct(cost_cls)
        L = SeparableLagrangian(control_cost=cost)
        box = _search_box(cost)
        for p in P_SWEEP:
            plus = _conjugate(L, p, bounds=box, alpha_sign=+1.0)
            minus = _conjugate(L, p, bounds=box, alpha_sign=-1.0)
            assert plus == pytest.approx(minus, abs=1e-6), (
                f"{cost_cls.__name__} p={p}: L_ctrl is NOT even in alpha. This is legitimate -- "
                f"#2375 ruling 5 exists so the library is correct for a non-even cost -- but it "
                f"means the four analytic closed forms on ControlCostBase subclasses "
                f"(evaluate / optimal_control / dp) must be RE-DERIVED: each is hand-written and "
                f"correct only while L(-a) == L(a). Do not make the new cost even to silence this."
            )


class TestCongestionRoundTrip:
    """sup_alpha { p.alpha - L } == H for the non-separable CongestionHamiltonian.

    The L side is the ANALYTIC Lagrangian built by ``_congestion_lagrangian``,
    deliberately not ``H.legendre_transform()``. That method returns a
    ``DualLagrangian``, which computes L = sup_p { p.alpha - H } from the SAME H
    object, so the round trip collapses to ``H** == H`` -- true for every convex
    H whatever the (V, f) signs are. Such a test asserts convexity, not a
    convention, and stays green under a sign flip in
    ``CongestionHamiltonian.__call__``.

    Sourcing V and f independently here is what gives the assertion teeth: a
    sign error on either term shifts H by 2V or 2f while the conjugate stays
    put. There is no shipped analytic CongestionLagrangian to import (Issue
    #1642, B6 owns that), hence the closed forms in this module.

    Tolerance is 1e-6, set by the single bounded scalar sup; the errors measured
    are ~4e-8. The fork this guards against is 2(V+f) = 2.6.
    """

    @pytest.mark.parametrize("cost_name", list(CONTROL_COSTS))
    @pytest.mark.parametrize(("vf_name", "vf"), list(VF_GRID.items()))
    def test_analytic_conjugate_recovers_hamiltonian(self, cost_name, vf_name, vf):
        potential, coupling = vf
        cost = CONTROL_COSTS[cost_name]()
        H = CongestionHamiltonian(
            control_cost=cost,
            congestion_factor=_congestion_factor,
            potential=potential,
            coupling=coupling,
        )
        L = _congestion_lagrangian(cost_name, potential, coupling)
        box = _congestion_box(cost)

        for p in P_SWEEP:
            expected = float(H(X_POINT, M_VALUE, np.array([p])))
            got = _conjugate(L, p, bounds=box)
            assert got == pytest.approx(expected, abs=1e-6), (
                f"{cost_name}/{vf_name} p={p}: conjugate of the analytic L gave {got}, H gave {expected}"
            )


class TestEffectiveDomain:
    """ControlCostBase.effective_domain() is the single owner of A (B3)."""

    def test_quadratic_is_unbounded(self):
        assert QuadraticControlCost(lambda_=2.0).effective_domain() is None

    def test_bounded_reports_max_control(self):
        assert BoundedControlCost(lambda_=1.0, max_control=2.0).effective_domain() == (-2.0, 2.0)

    def test_l1_reports_bang_bang_interval(self):
        assert L1ControlCost(lambda_=0.5).effective_domain() == (-1.0, 1.0)

    @pytest.mark.parametrize(
        ("cost_factory", "expected"),
        [
            (lambda: QuadraticControlCost(lambda_=2.0), None),
            (lambda: BoundedControlCost(lambda_=1.0, max_control=2.0), (-2.0, 2.0)),
            (lambda: L1ControlCost(lambda_=0.5), (-1.0, 1.0)),
        ],
    )
    def test_control_bounds_delegates_to_effective_domain(self, cost_factory, expected):
        """The isinstance ladder is gone; SeparableLagrangian must read the owner."""
        cost = cost_factory()
        assert SeparableLagrangian(control_cost=cost).control_bounds() == expected
        assert SeparableLagrangian(control_cost=cost).control_bounds() == cost.effective_domain()

    @pytest.mark.parametrize(
        ("cost_factory", "expected"),
        [
            (lambda: BoundedControlCost(lambda_=1.0, max_control=2.0), (-2.0, 2.0)),
            (lambda: L1ControlCost(lambda_=0.5), (-1.0, 1.0)),
            (lambda: QuadraticControlCost(lambda_=2.0), None),
        ],
    )
    def test_moreau_yosida_preserves_domain(self, cost_factory, expected):
        """dom(L + eps/2|.|^2) == dom(L). Smoothing H must not enlarge A.

        BEHAVIOR CHANGE vs the removed isinstance ladder, which returned None
        here because a wrapped cost matched neither branch -- a regularized
        bounded problem silently claimed an unbounded control set. Reachable via
        SeparableHamiltonian.regularize() -> MFGComponents deriving a
        SeparableLagrangian -> hjb_semi_lagrangian control_bounds().
        """
        wrapped = cost_factory().regularize(0.1)
        assert wrapped.effective_domain() == expected
        assert SeparableLagrangian(control_cost=wrapped).control_bounds() == expected

    def test_lagrangian_does_not_enforce_the_domain(self):
        """Forcing evidence for B3, pinned so the gap cannot be forgotten.

        ``lagrangian()`` returns a finite value for infeasible alpha. Making it
        fail loud is Issue #1644 (B4); when that lands, this test must be
        replaced by one asserting the raise -- not deleted.
        """
        cost = BoundedControlCost(lambda_=1.0, max_control=2.0)
        infeasible = np.array([5.0])
        assert cost.effective_domain() == (-2.0, 2.0)
        assert float(cost.lagrangian(infeasible)) == pytest.approx(12.5)

    def test_unrestricted_conjugate_overshoots_without_the_domain(self):
        """Why every conjugate consumer must read effective_domain().

        L1 at p=5: true H is 4.5; a sup taken over (-50, 50) instead of the
        effective domain returns ~225.
        """
        cost = L1ControlCost(lambda_=0.5)
        L = SeparableLagrangian(control_cost=cost)
        # Issue #1653: evaluate reduces over the component axis, so a (1,) momentum
        # yields one scalar rather than a length-1 array.
        true_h = float(cost.evaluate(np.array([5.0])))

        on_domain = _conjugate(L, 5.0, bounds=cost.effective_domain())
        off_domain = _conjugate(L, 5.0, bounds=(-50.0, 50.0))

        assert on_domain == pytest.approx(true_h, abs=1e-6)
        assert off_domain > 40.0 * true_h


class TestTheEnumerationIsThePopulation:
    """#2386. Two instruments over the same population, which must agree.

    `test_conjugate_is_alpha_sign_blind` is only as good as the set it iterates, and this class is
    what makes that set trustworthy. One check guards the hand-written dict the congestion round
    trip needs; the other guards the runtime enumeration itself, which can under-report silently.
    """

    def test_the_runtime_enumeration_sees_every_cost_in_the_source(self):
        """`__subclasses__()` sees only what has been IMPORTED, and reports the rest as absent.

        A cost defined in a module nobody imports is invisible to the runtime enumeration while
        the result looks like a complete population -- an empty difference and a missing class are
        the same observation. So the source is read independently, by AST, and the two compared.

        `test_s1_drift_convention_routing.py:236` records this exact failure on a sibling
        enumeration: 7 classes standalone against 10 under full collection, because one module was
        imported only by a neighbouring test. Its comment is the rule followed here.
        """
        runtime = {c.__name__ for c in _concrete_control_cost_classes()}
        in_source = _control_cost_classes_in_source()
        assert in_source, "the AST scan found no ControlCostBase subclass at all -- it is broken"
        missing = in_source - runtime
        assert not missing, (
            f"{sorted(missing)} subclass ControlCostBase in the source but are not reachable from "
            f"`__subclasses__()`, so the evenness pin never sees them. Import the defining module "
            f"at the top of this file -- that is what makes the enumeration independent of "
            f"collection order."
        )

    def test_the_hand_written_cost_dict_covers_every_subclass(self):
        """`CONTROL_COSTS` is deliberately hand-written; this is what stops it going stale.

        The congestion round trip pairs each entry with a conjugate transcribed BY HAND in
        `CONGESTION_L_CTRL`, and that independence is what gives it teeth -- so the dict cannot be
        derived. What it can have is a gate that fails and NAMES the gap, so a new cost cannot
        arrive without someone writing its conjugate.

        Note what this does NOT require: the evenness pin covers a new class immediately, with no
        conjugate. Only the congestion round trip needs the transcription.
        """
        covered = {type(factory()) for factory in CONTROL_COSTS.values()}
        enumerated = set(_concrete_control_cost_classes())
        uncovered = enumerated - covered
        assert uncovered == {_MoreauYosidaControlCost}, (
            f"{sorted(c.__name__ for c in uncovered)} are ControlCostBase subclasses with no "
            f"CONTROL_COSTS entry, so the congestion round trip does not exercise them. Add an "
            f"instance to CONTROL_COSTS and its hand-transcribed conjugate to CONGESTION_L_CTRL, "
            f"or extend the expected set here with the reason."
        )
        assert set(CONGESTION_L_CTRL) == set(CONTROL_COSTS), (
            "every CONTROL_COSTS entry needs a CONGESTION_L_CTRL conjugate and vice versa; "
            f"dict keys {sorted(CONTROL_COSTS)} against conjugates {sorted(CONGESTION_L_CTRL)}"
        )

    def test_the_enumeration_excludes_subclasses_defined_in_tests(self):
        """NEGATIVE CONTROL for the `mfgarchon.` filter, which a red gate added.

        Two test files define `ControlCostBase` subclasses (`_NarrowL1`, `_NarrowBounded` in
        `test_sl_control_set_single_source_1642.py`). `__subclasses__()` is interpreter-global, so
        they appear here under full collection and not standalone -- and without the filter this
        file's completeness gate failed in the gate while passing alone.

        Asserts both directions: a locally-defined subclass is excluded, and a real library class
        is still included, so the filter cannot pass by excluding everything.
        """

        class _LocalProbeCost(QuadraticControlCost):
            pass

        enumerated = _concrete_control_cost_classes()
        assert _LocalProbeCost not in enumerated, (
            "a subclass defined in a test file is in the library population; the mfgarchon. filter "
            "is not working and this file's coverage depends on collection order"
        )
        assert QuadraticControlCost in enumerated, "the filter excluded a real library cost"

    def test_the_construction_strategy_refuses_rather_than_skips(self):
        """CONTROL: a cost with an unknown required argument must RAISE, not be skipped.

        A skipped subclass is indistinguishable from a covered one, which is the whole defect
        #2386 records one layer up. Probed with a class the strategy table cannot build.
        """

        class _NeedsSomethingUnknown(QuadraticControlCost):
            def __init__(self, mystery_parameter):
                raise AssertionError("must not be reached")

        with pytest.raises(AssertionError, match=r"mystery_parameter.*does not know how to supply"):
            _construct(_NeedsSomethingUnknown)

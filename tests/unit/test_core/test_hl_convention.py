"""Pinning tests for the H<->L convention and the admissible control set.

Issue #1642, capabilities B1 (hl-convention-pin) and B3 (controlcost-effective-domain).

B1 pins the (V, f) sign convention documented on ``MFGOperatorBase``. Both are
cost-signed (#2375 ruling 3), so they enter the running cost with a plus sign:

    L(t, x, alpha, m) = L_ctrl(alpha) + V(x, t) + f(m)

asserted in its conjugate form under the library's pairing (#2375 ruling 5),
``sup_alpha { -p.alpha - L } == H``.

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

What they do NOT catch: the sign of the pairing itself -- every shipped L_ctrl is even
in alpha, so ``sup{-p.alpha - L}`` and ``sup{+p.alpha - L}`` give the same value. That
sign is pinned with a non-even L in ``test_legendre_sign_convention_2375.py``. The
evenness that makes it invisible here is pinned by ``test_conjugate_is_alpha_sign_blind``
and ``test_every_library_control_cost_is_even`` rather than left implicit.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

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


def _potential(t, x):
    return V_NONZERO


def _coupling(m):
    return F_SLOPE * m


def _conjugate(L, p, *, bounds, alpha_sign=-1.0):
    """sup_alpha { alpha_sign * p * alpha - L(t, x, alpha, m) } over ``bounds``.

    ``bounds`` must be the admissible control set: the shipped ``lagrangian()``
    implementations omit the indicator of their effective domain, so an
    unrestricted sup silently overshoots (see module docstring).
    """

    def neg_objective(a):
        return -(alpha_sign * p * a - float(L(x=X_POINT, alpha=np.array([a]), m=M_VALUE, t=0.0)))

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

# The dict above is NOT the population the evenness claim is about (#2386). It is hand-written
# because `CONGESTION_L_CTRL` pairs each entry with a conjugate transcribed by hand, and it was
# already missing `_MoreauYosidaControlCost`. The evenness pins below enumerate `ControlCostBase`
# subclasses instead, so adding a cost to the library is enough to put it under them.

# Probed across the l1 kink at 0, the interior and the boundary of a bounded admissible set. Only
# values with both a and -a admissible are used: `lagrangian()` omits the domain's indicator, and
# neither conjugate looks outside the domain, so L there says nothing about the pairing.
_EVENNESS_PROBES = (0.0, 1e-12, 0.25, 0.5, 0.999, 1.0, 1.5, 3.0, 8.0, 120.0)


def _evenness_probes(cost):
    """The probes a for which both a and -a lie in `cost.effective_domain()`."""
    domain = cost.effective_domain()
    if domain is None:
        return _EVENNESS_PROBES
    return tuple(a for a in _EVENNESS_PROBES if domain[0] <= -a and a <= domain[1])


# Required constructor arguments we know how to supply. A cost whose required argument is NOT here
# raises rather than being skipped: a silently skipped subclass is the defect this section exists to
# remove, and it would look exactly like coverage.
_REQUIRED_ARG_STRATEGY = {
    # `base` MUST be non-smooth. `regularize()` returns `self` for a smooth cost
    # (`QuadraticControlCost.is_smooth()` is True), so `_MoreauYosidaControlCost(base=Quadratic)`
    # is a specimen the library CANNOT produce -- and its `effective_domain()` is None, so the
    # search box widens to (-50, 50) and neither an l1 kink nor a saturated boundary is probed.
    # Measured: `L1ControlCost(0.5).regularize(0.1)` gives base=L1, domain=(-1.0, 1.0).
    "base": lambda: L1ControlCost(lambda_=0.5),
    "epsilon": lambda: 0.1,
}


def _concrete_control_cost_classes():
    """Every concrete `ControlCostBase` subclass defined in the library AND ALREADY IMPORTED.

    Recursive, because a future cost may subclass a concrete one rather than the base.

    `__subclasses__()` sees only imported classes, so in this process the result depends on what
    collection happened to import -- `test_every_library_control_cost_is_even` is what covers the
    rest. It is also interpreter-global: under full collection it returns subclasses defined in
    OTHER TEST FILES (`_NarrowL1`, `_NarrowBounded` in `test_sl_control_set_single_source_1642.py`),
    which the `mfgarchon.` module filter excludes, because the library's guarantee says nothing
    about a cost a test defines to probe something.
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


# Run in a CHILD interpreter by `test_every_library_control_cost_is_even`. Only a process that has
# imported every module sees every cost, and that import must not happen in the test process: it
# would pre-import the renamed shim modules for every later test on the same xdist worker, and the
# warning census would then lose their DeprecationWarnings depending on scheduling.
_CHILD_SWEEP = r"""
import importlib, importlib.util, json, pathlib, pkgutil, sys
import numpy as np
import mfgarchon

spec = importlib.util.spec_from_file_location("_hl_convention_child", sys.argv[1])
hl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hl)

failures, walked = [], {"mfgarchon"}
for info in pkgutil.walk_packages(mfgarchon.__path__, prefix="mfgarchon.", onerror=failures.append):
    walked.add(info.name)
    try:
        importlib.import_module(info.name)
    except Exception as exc:
        failures.append(f"{info.name}: {type(exc).__name__}: {exc}")

# walk_packages only descends into directories with an __init__.py, so its population is checked
# against the files on disk rather than trusted.
root = pathlib.Path(mfgarchon.__file__).parent
on_disk = set()
for path in root.rglob("*.py"):
    parts = ("mfgarchon",) + path.relative_to(root).with_suffix("").parts
    on_disk.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
unwalked = sorted(on_disk - walked)

classes = hl._concrete_control_cost_classes()
checked, uneven = [], []
for cls in classes:
    cost = hl._construct(cls)
    if type(cost) is not cls:
        continue
    for a in hl._evenness_probes(cost):
        plus = float(cost.lagrangian(np.array([a])))
        minus = float(cost.lagrangian(np.array([-a])))
        if plus != minus:
            uneven.append(f"{cls.__module__}.{cls.__qualname__}: L({a}) = {plus!r}, L({-a}) = {minus!r}")
            break
    domain = cost.effective_domain()
    if domain is not None and domain[0] != -domain[1]:
        uneven.append(f"{cls.__module__}.{cls.__qualname__}: effective_domain() = {domain!r} is not symmetric")
    checked.append(cls.__name__)

print(json.dumps({
    "tree": mfgarchon.__file__,
    "import_failures": failures,
    "unwalked": unwalked,
    "classes": [c.__name__ for c in classes],
    "checked": checked,
    "uneven": uneven,
}))
"""


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
    """Analytic L(t, x, alpha, m) = L_ctrl^{c(m)}(alpha) + V(x, t) + f(m).

    Independently sourced from ``CongestionHamiltonian`` -- see the note on
    ``TestCongestionRoundTrip`` for why that independence is what makes the
    round trip discriminating.
    """
    l_ctrl = CONGESTION_L_CTRL[cost_name]

    def L(x, alpha, m, t=0.0):
        v = potential(t=t, x=x) if potential is not None else 0.0
        f = coupling(m) if coupling is not None else 0.0
        return l_ctrl(float(np.atleast_1d(alpha)[0]), _congestion_factor(m)) + v + f

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
    """sup_alpha { -p.alpha - L } == H for SeparableLagrangian / SeparableHamiltonian."""

    @pytest.mark.parametrize(("cost_name", "potential", "coupling"), _separable_params())
    def test_conjugate_of_lagrangian_recovers_hamiltonian(self, cost_name, potential, coupling):
        cost = CONTROL_COSTS[cost_name]()
        L = SeparableLagrangian(control_cost=cost, potential=potential, coupling=coupling)
        H = L.as_hamiltonian()
        box = _search_box(cost)

        for p in P_SWEEP:
            expected = float(H(x=X_POINT, m=M_VALUE, p=np.array([p]), t=0.0))
            got = _conjugate(L, p, bounds=box)
            assert got == pytest.approx(expected, abs=1e-6), (
                f"{cost_name} p={p}: conjugate of L gave {got}, H gave {expected}"
            )

    @pytest.mark.parametrize("cost_cls", _concrete_control_cost_classes(), ids=lambda c: c.__name__)
    def test_conjugate_is_alpha_sign_blind(self, cost_cls):
        """Every imported library L_ctrl is even on a symmetric admissible set, so the pairing
        sign cannot change the conjugate: sup{+p.a - L} == sup{-p.a - L}.

        This pins the premise of #2375 ruling 5's "nothing in this library changes value", and
        retires with that paragraph, as `test_every_library_control_cost_is_even` does.
        Parametrised over the enumeration rather than `CONTROL_COSTS` (#2386); a cost in a module
        collection did not import is covered by that test instead.
        """
        cost = _construct(cost_cls)
        # Bit-exact first. The conjugate comparison below goes through `minimize_scalar` at
        # abs=1e-6, and an odd term eps*a on a quadratic cost moves it by 2*p*eps/lambda, so over
        # P_SWEEP it cannot see eps below lambda * 6.25e-8. The direct check has no tolerance.
        for probe_a in _evenness_probes(cost):
            assert float(cost.lagrangian(np.array([probe_a]))) == float(cost.lagrangian(np.array([-probe_a]))), (
                f"{cost_cls.__name__}: L({probe_a}) != L({-probe_a}), so the cost is not even in alpha. "
                f"See the conjugate assertion's message below for what that means."
            )
        # `_construct` is a factory: a mis-constructed class would advertise one class in the test
        # id while exercising another.
        assert type(cost) is cost_cls, f"_construct({cost_cls.__name__}) returned {type(cost).__name__}"
        L = SeparableLagrangian(control_cost=cost)
        box = _search_box(cost)
        for p in P_SWEEP:
            plus = _conjugate(L, p, bounds=box, alpha_sign=+1.0)
            minus = _conjugate(L, p, bounds=box, alpha_sign=-1.0)
            assert plus == pytest.approx(minus, abs=1e-6), (
                f"{cost_cls.__name__} p={p}: sup{{+p.a - L}} != sup{{-p.a - L}}. Two possible causes.\n"
                f"(1) L_ctrl is not even in alpha. That is legitimate -- #2375 ruling 5 exists so the "
                f"library is correct for a non-even cost -- but the class's hand-written closed forms "
                f"for H (`evaluate`, `optimal_control`, `dp`) must then be derived under "
                f"H = sup{{-p.a - L}}, and #2375's 'nothing changes value' no longer holds. Do not "
                f"make the cost even to silence this.\n"
                f"(2) L is even but its admissible set is asymmetric, so the two sups range over "
                f"different sets: check effective_domain(). The bit-exact probes above passing points "
                f"here first, though they sample ten values of alpha, not the whole line."
            )


class TestCongestionRoundTrip:
    """sup_alpha { -p.alpha - L } == H for the non-separable CongestionHamiltonian.

    The L side is the ANALYTIC Lagrangian built by ``_congestion_lagrangian``,
    deliberately not ``H.legendre_transform()``. That method returns a
    ``DualLagrangian``, which computes L = sup_p { -p.alpha - H } from the SAME H
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
            expected = float(H(x=X_POINT, m=M_VALUE, p=np.array([p]), t=0.0))
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


_PACKAGE_ROOT = Path(mfgarchon.__file__).resolve().parent.parent


class TestEveryLibraryCost:
    """#2386. The evenness claim over the WHOLE library, not only what collection imported."""

    def test_every_library_control_cost_is_even(self):
        """Every concrete `ControlCostBase` subclass in the package is sign-symmetric in alpha.

        That is the premise of #2375 ruling 5's "nothing in this library changes value":
        sup{-p.a - L} equals sup{+p.a - L} when L is even AND the admissible set is symmetric.
        `test_conjugate_is_alpha_sign_blind` cannot establish it for the whole library: it is
        parametrised at collection, and `__subclasses__()` misses a cost in a module nothing has
        imported yet. So a child interpreter imports every module `pkgutil.walk_packages` reaches,
        fails if any `.py` file under the package was not reached, builds each concrete class with
        `_construct`, and checks L(a) == L(-a) bit-exactly at `_evenness_probes(cost)` and that
        `effective_domain()` is symmetric. The child reads the tree this process reads (`-P` plus
        this package prepended to PYTHONPATH), and that is asserted rather than assumed, because
        mfgarchon is also editable-installed from the main checkout.

        What it does NOT check: an instance built with other arguments than `_construct` supplies
        (a class even only at its defaults passes); alpha with more than one component (the probes
        are one-component); a class defined inside a function or behind an import guard that did
        not fire.

        Retire this test, with `test_conjugate_is_alpha_sign_blind`, if #2375's "nothing changes
        value" paragraph is withdrawn: both pin that sentence's premise, which ruling 5 does not
        require to stay true. The module docstring's last paragraph and the comment above
        `_EVENNESS_PROBES` go with them.
        """
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [str(_PACKAGE_ROOT), os.environ.get("PYTHONPATH")])),
        }
        proc = subprocess.run(
            [sys.executable, "-P", "-c", _CHILD_SWEEP, str(Path(__file__).resolve())],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, f"the child sweep failed:\n{proc.stderr[-4000:]}"
        report = json.loads(proc.stdout.strip().splitlines()[-1])

        assert Path(report["tree"]).resolve() == Path(mfgarchon.__file__).resolve(), (
            f"the child imported {report['tree']}, not the tree under test {mfgarchon.__file__}"
        )
        assert report["import_failures"] == [], (
            f"{len(report['import_failures'])} module(s) failed to import, so a cost defined there is "
            f"invisible to this check: {report['import_failures'][:5]}"
        )
        assert report["unwalked"] == [], (
            f"{len(report['unwalked'])} module(s) under the package were never imported, because "
            f"pkgutil.walk_packages does not enter a directory without __init__.py, so a cost defined "
            f"there is invisible to this check: {report['unwalked'][:5]}"
        )
        in_process = {c.__name__ for c in _concrete_control_cost_classes()}
        assert in_process <= set(report["classes"]), (
            f"the child enumerated {report['classes']} but this process already sees {sorted(in_process)}; "
            f"the sweep saw less than collection did, so its population is broken"
        )
        assert report["checked"] == report["classes"], (
            f"enumerated {report['classes']} but probed only {report['checked']}: `_construct` returned "
            f"the wrong class for the rest"
        )
        assert report["uneven"] == [], (
            f"not sign-symmetric in alpha: {report['uneven']}. That is legitimate -- #2375 ruling 5 exists "
            f"so the library is correct for such a cost -- but the class's hand-written closed forms "
            f"for H (`evaluate`, `optimal_control`, `dp`) must be derived under H = sup{{-p.a - L}}, and "
            f"#2375's 'nothing changes value' no longer holds. Do not make the cost symmetric to silence "
            f"this; withdraw or amend that changelog paragraph."
        )

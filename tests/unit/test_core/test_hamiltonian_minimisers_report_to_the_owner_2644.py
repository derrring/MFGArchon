"""core/hamiltonian.py's nine scipy minimiser calls hand a failed search to the one owner, which raises (#2644 B2, rows
16-24, stage 1). Before, each read ``res.x`` or ``res.fun`` and returned it whatever scipy said.

Each member is driven so that its own call fails, through a cost, Lagrangian or Hamiltonian a user can write:
- the bounded scalar searches (rows 17 and 19) spend their evaluation budget on a V-shaped L over a box of +-1e150;
- Brent's search in the Moreau-Yosida prox (row 16) finds no bracket for the middle of three momenta, [2, 0, 2], under a
  cost that is NaN below -1; the searches before and after it succeed, so a check of only the first search, or of only
  the last, would not raise;
- each L-BFGS-B search (rows 18 and 20-24) starts where a barrier L or H is NaN, and stops ABNORMAL.

The message is the failure's own, compared field by field: the library's parts (the member, its input, the routine,
the counts) literally, and scipy's message as the member's own call returned it, captured through a pass-through
wrapper, so that no scipy wording is written here. For rows 17 and 19 the fun scipy reported is checked against the
member's objective at its x, recomputed here; for the others, x is the NaN start (or NaN) and fun NaN. Each failure is
raised in the owner's frame, `report_inner_failure`.

Exhaustion, the fact the plan names, has its own pins for the members whose routine is Brent or L-BFGS-B. No input
was found that makes either spend its default budget, so the test caps it: the member's own call goes to scipy with
its own objective, start and bounds, and with maxiter=1 added to its options. Without the cap the same call succeeds.
"""

from __future__ import annotations

import math
import re
import traceback

import pytest

import numpy as np
import scipy.optimize

from mfgarchon.core.hamiltonian import (
    ControlCostBase,
    DualHamiltonian,
    DualLagrangian,
    L1ControlCost,
    LagrangianBase,
    SeparableHamiltonian,
    SeparableLagrangian,
)
from mfgarchon.utils import exceptions
from mfgarchon.utils.exceptions import ConvergenceError

EPS = 0.5
BOX = 1e150
TAU = 1e300


def _barrier(v):
    """-log(1 - |v|^2): a barrier, NaN outside the unit ball."""
    return -np.log(1.0 - np.sum(np.asarray(v, dtype=float) ** 2, axis=-1))


class BarrierCost(ControlCostBase):
    """H(p) = -log(1 - |p|^2), and not smooth, so ``regularize`` builds its Moreau-Yosida envelope."""

    def evaluate(self, p):
        return _barrier(p)

    def dp(self, p):
        p = np.asarray(p, dtype=float)
        return 2 * p / (1.0 - np.sum(p**2, axis=-1, keepdims=True))

    def optimal_control(self, p):
        return -self.dp(p)

    def lagrangian(self, alpha):
        return np.zeros(np.shape(alpha)[:-1])

    def is_smooth(self):
        return False


class HalfLineCost(ControlCostBase):
    """H(p) = p^2 / 2 for p >= -1 and NaN below, declared not smooth, so ``regularize`` builds its Moreau-Yosida
    envelope. Its prox finds no bracket at p = 0, and succeeds at p = 2."""

    def evaluate(self, p):
        p = np.asarray(p, dtype=float)
        return np.sum(np.where(p >= -1.0, p**2 / 2, np.nan), axis=-1)

    def dp(self, p):
        return np.asarray(p, dtype=float)

    def optimal_control(self, p):
        return -self.dp(p)

    def lagrangian(self, alpha):
        return np.zeros(np.shape(alpha)[:-1])

    def is_smooth(self):
        return False


class VeeLagrangian(LagrangianBase):
    """L(alpha) = |alpha - 0.3| on a control set of +-1e150."""

    def __init__(self):
        super().__init__()

    def __call__(self, t, x, alpha, m):
        return float(np.sum(np.abs(np.asarray(alpha, dtype=float) - 0.3)))

    def control_bounds(self):
        return (-BOX, BOX)


class BarrierLagrangian(LagrangianBase):
    """L(alpha) = -log(1 - |alpha|^2), NaN outside the unit ball, on a declared box of [-2, 2]."""

    def __init__(self):
        super().__init__()

    def __call__(self, t, x, alpha, m):
        return float(_barrier(alpha))

    def control_bounds(self):
        return (-2.0, 2.0)


X2 = (0.5, 0.5)
OUTSIDE = np.array([1.5, 0.5])  # each L-BFGS-B search starts here, where the barrier is NaN
LBFGSB = ("minimize", "minimize (L-BFGS-B)")


def _abnormal(res):
    return res.status == 2  # L-BFGS-B's abnormal stop: its line search failed


def _budget(res):
    return res.status == 1 and res.nit == res.nfev  # the bounded search's evaluation budget ran out


def _dual_h():
    return DualHamiltonian(BarrierLagrangian(), alpha_bounds=(-2.0, 2.0))


def _dual_l():
    return DualLagrangian(SeparableHamiltonian(BarrierCost(lambda_=1.0)), p_bounds=(-2.0, 2.0))


def _vee_value(x):
    return abs(x[0] - 0.3)


#: member -> (its call, (the scipy.optimize name it calls, the routine its message names), the name and input its message
#: gives, how scipy stops, and a check of the x and fun scipy reported: against the member's objective for rows 17 and
#: 19, and for the NaN start, or NaN, for the others).
MEMBERS = {
    "row 16 _MoreauYosidaControlCost._prox_h": (
        lambda: HalfLineCost(lambda_=1.0).regularize(EPS).dp(np.array([2.0, 0.0, 2.0])),
        ("minimize_scalar", "minimize_scalar (Brent)"),
        "_MoreauYosidaControlCost._prox_h at p=0.0",
        lambda res: res.nit == 0,  # Brent found no bracket
        lambda x, value: math.isnan(x[0]) and math.isnan(value),
    ),
    "row 17 LagrangianBase.conjugate_argmax, 1-D": (
        lambda: VeeLagrangian().conjugate_argmax(t=0.0, x=(0.5,), p=np.array([0.0]), m=1.0),
        ("minimize_scalar", "minimize_scalar (bounded)"),
        "VeeLagrangian.conjugate_argmax at p=0.0",
        _budget,
        lambda x, value: value == _vee_value(x),  # at p = 0 the objective is L
    ),
    "row 18 LagrangianBase.conjugate_argmax, n-D": (
        lambda: BarrierLagrangian().conjugate_argmax(t=0.0, x=X2, p=-OUTSIDE, m=1.0),
        LBFGSB,
        "BarrierLagrangian.conjugate_argmax at p=[-1.5, -0.5]",
        _abnormal,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
    ),
    "row 19 LagrangianBase.proximal, 1-D": (
        lambda: VeeLagrangian().proximal(TAU, np.array([0.3])),
        ("minimize_scalar", "minimize_scalar (bounded)"),
        "VeeLagrangian.proximal at z=0.3, tau=1e+300",
        _budget,
        lambda x, value: value == _vee_value(x) + (x[0] - 0.3) ** 2 / (2 * TAU),
    ),
    "row 20 LagrangianBase.proximal, n-D": (
        lambda: BarrierLagrangian().proximal(1.0, OUTSIDE, x=X2),
        LBFGSB,
        "BarrierLagrangian.proximal at z=[1.5, 0.5], tau=1.0",
        _abnormal,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
    ),
    "row 21 DualHamiltonian.__call__": (
        lambda: _dual_h()(t=0.0, x=X2, p=-OUTSIDE, m=1.0),
        LBFGSB,
        "DualHamiltonian.__call__ at p=[-1.5, -0.5]",
        _abnormal,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
    ),
    "row 22 DualHamiltonian._find_optimal_alpha": (
        lambda: _dual_h().dp(t=0.0, x=X2, p=-OUTSIDE, m=1.0),
        LBFGSB,
        "DualHamiltonian._find_optimal_alpha at p=[-1.5, -0.5]",
        _abnormal,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
    ),
    "row 23 DualLagrangian.__call__": (
        lambda: _dual_l()(t=0.0, x=X2, alpha=-OUTSIDE, m=1.0),
        LBFGSB,
        "DualLagrangian.__call__ at alpha=[-1.5, -0.5]",
        _abnormal,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
    ),
    "row 24 DualLagrangian._find_optimal_p": (
        lambda: _dual_l().d_alpha(t=0.0, x=X2, alpha=-OUTSIDE, m=1.0),
        LBFGSB,
        "DualLagrangian._find_optimal_p at alpha=[-1.5, -0.5]",
        _abnormal,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
    ),
}

MESSAGE = re.compile(
    r"(?P<where>.+) stopped without success after (?P<nit>\d+) iterations and (?P<nfev>\d+) evaluations "
    r"\((?P<why>.*)\)\. scipy reported x=\[(?P<x>[^\]]*)\] and fun=(?P<fun>\S+); that x is not known to be the optimum, "
    r"and fun is scipy's, not known to be the objective at x \(#2644\)\.",
    re.S,
)


def _spy(monkeypatch, name, maxiter=None):
    """Every result of ``scipy.optimize.<name>``, which hamiltonian.py imports when it calls it, as the member's own call
    returned it; with ``maxiter`` added to the call's options when given."""
    real = getattr(scipy.optimize, name)
    results = []

    def spied(*args, **kwargs):
        if maxiter is not None:
            kwargs["options"] = {**(kwargs.get("options") or {}), "maxiter": maxiter}
        results.append(real(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(scipy.optimize, name, spied)
    return results


def _raised_by_the_owner(error):
    """The failure was raised in the one owner's frame, `report_inner_failure`, not by its caller."""
    frames = traceback.extract_tb(error.__traceback__)
    assert (frames[-1].name, frames[-1].filename) == ("report_inner_failure", exceptions.__file__), frames


def _carries(message, where, routine, res):
    """The library's message names ``where`` and ``routine`` literally, scipy's own message as ``res`` holds it, and
    ``res``'s counts."""
    fields = MESSAGE.fullmatch(message)
    assert fields is not None, message
    assert fields["where"] == f"{where}: scipy's {routine}"
    assert fields["why"] == " ".join(str(res.message).split())
    assert (int(fields["nit"]), int(fields["nfev"])) == (res.nit, res.nfev)
    return fields


@pytest.mark.parametrize("member", list(MEMBERS))
def test_a_failed_search_is_raised_by_the_owner_with_its_own_message(member, monkeypatch):
    call, (name, routine), where, stops, stopped_at = MEMBERS[member]
    results = _spy(monkeypatch, name)
    # The inputs leave the barrier's domain, and the 1e150 box overflows scipy's parabolic step, by construction.
    with pytest.raises(ConvergenceError) as raised, np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        call()
    error = raised.value
    _raised_by_the_owner(error)
    failed = results[-1]  # the search the owner refused; nothing runs after it
    assert not failed.success
    assert stops(failed), failed
    fields = _carries(str(error), where, routine, failed)
    x = [float(v) for v in fields["x"].split(", ")]
    assert stopped_at(x, float(fields["fun"])), str(error)
    assert error.solver_type == where.split(" at ")[0]
    assert error.iterations == failed.nit
    assert math.isnan(error.residual)


class QuarticCost(ControlCostBase):
    """H(p) = |p|^4 / 4, declared not smooth, so ``regularize`` builds its Moreau-Yosida envelope. Not quadratic, so
    L-BFGS-B over it takes more than one iteration."""

    def evaluate(self, p):
        return np.sum(np.asarray(p, dtype=float) ** 4, axis=-1) / 4

    def dp(self, p):
        return np.asarray(p, dtype=float) ** 3

    def optimal_control(self, p):
        return -self.dp(p)

    def lagrangian(self, alpha):
        return np.zeros(np.shape(alpha)[:-1])

    def is_smooth(self):
        return False


class SmoothLagrangian(LagrangianBase):
    """L(alpha) = |alpha|^2 / 2 + |alpha|^4 / 4 on a box of [-5, 5]."""

    def __init__(self):
        super().__init__()

    def __call__(self, t, x, alpha, m):
        a2 = float(np.sum(np.asarray(alpha, dtype=float) ** 2))
        return a2 / 2 + a2**2 / 4

    def control_bounds(self):
        return (-5.0, 5.0)


P = np.array([0.7, -0.4])
BRENT_SPENT = ("minimize_scalar", "minimize_scalar (Brent)")
LBFGSB_SPENT = LBFGSB

#: member -> (a call whose search succeeds at scipy's default budget, the name and input its message gives, and
#: (the scipy.optimize name it calls, the routine its message names)).
CAPPED = {
    "row 16 _MoreauYosidaControlCost._prox_h": (
        lambda: QuarticCost(lambda_=1.0).regularize(EPS).dp(np.array([1.3])),
        "_MoreauYosidaControlCost._prox_h at p=1.3",
        BRENT_SPENT,
    ),
    "row 18 LagrangianBase.conjugate_argmax, n-D": (
        lambda: SmoothLagrangian().conjugate_argmax(t=0.0, x=X2, p=P, m=1.0),
        "SmoothLagrangian.conjugate_argmax at p=[0.7, -0.4]",
        LBFGSB_SPENT,
    ),
    "row 20 LagrangianBase.proximal, n-D": (
        lambda: SmoothLagrangian().proximal(1.0, P, x=X2),
        "SmoothLagrangian.proximal at z=[0.7, -0.4], tau=1.0",
        LBFGSB_SPENT,
    ),
    "row 21 DualHamiltonian.__call__": (
        lambda: DualHamiltonian(SmoothLagrangian(), alpha_bounds=(-5.0, 5.0))(t=0.0, x=X2, p=P, m=1.0),
        "DualHamiltonian.__call__ at p=[0.7, -0.4]",
        LBFGSB_SPENT,
    ),
    "row 22 DualHamiltonian._find_optimal_alpha": (
        lambda: DualHamiltonian(SmoothLagrangian(), alpha_bounds=(-5.0, 5.0)).dp(t=0.0, x=X2, p=P, m=1.0),
        "DualHamiltonian._find_optimal_alpha at p=[0.7, -0.4]",
        LBFGSB_SPENT,
    ),
    "row 23 DualLagrangian.__call__": (
        lambda: DualLagrangian(SeparableHamiltonian(QuarticCost(lambda_=1.0)), p_bounds=(-5.0, 5.0))(
            t=0.0, x=X2, alpha=P, m=1.0
        ),
        "DualLagrangian.__call__ at alpha=[0.7, -0.4]",
        LBFGSB_SPENT,
    ),
    "row 24 DualLagrangian._find_optimal_p": (
        lambda: DualLagrangian(SeparableHamiltonian(QuarticCost(lambda_=1.0)), p_bounds=(-5.0, 5.0)).d_alpha(
            t=0.0, x=X2, alpha=P, m=1.0
        ),
        "DualLagrangian._find_optimal_p at alpha=[0.7, -0.4]",
        LBFGSB_SPENT,
    ),
}


@pytest.mark.parametrize("member", list(CAPPED))
def test_a_search_that_spends_its_budget_is_raised_by_the_owner(member, monkeypatch):
    call, where, (name, routine) = CAPPED[member]
    call()  # the control: at scipy's default budget the search succeeds, and the member returns
    results = _spy(monkeypatch, name, maxiter=1)
    with pytest.raises(ConvergenceError) as raised:
        call()
    assert len(results) == 1, results  # the member's own call, capped
    assert not results[0].success, results
    _raised_by_the_owner(raised.value)
    _carries(str(raised.value), where, routine, results[0])
    assert results[0].nit == raised.value.iterations == 1


def test_a_non_smooth_lagrangian_in_2d_is_refused_at_p_0_until_stage_2(monkeypatch):
    """What #2939 ships for a Lagrangian that is not differentiable, in 2-D: the library's L1 cost through
    DualHamiltonian, at p = 0. The maximiser is alpha = 0, where L-BFGS-B starts, and its line search stops ABNORMAL at
    the kink there, so the search is refused, although the point it stopped at is the maximiser. L is minimised on its
    kink at alpha = 0, where the search starts for p = 0, so every evaluation of this H at p = 0 is refused.

    This holds until #2644 stage 2 changes the method for rows 18 and 20-24 to one whose verdict means something for a
    non-smooth objective. If H at p = 0 returns, this test fails with RETIRE in its output, which says what to check."""
    H = DualHamiltonian(SeparableLagrangian(L1ControlCost(lambda_=1.0)))
    results = _spy(monkeypatch, "minimize")
    try:
        H(t=0.0, x=(0.5, 0.5), p=np.zeros(2), m=1.0)
    except ConvergenceError as error:
        assert len(results) == 1, results
        assert not results[0].success
        assert _abnormal(results[0])
        assert np.array_equal(np.abs(results[0].x), np.zeros(2))  # the point it stopped at is the maximiser
        _raised_by_the_owner(error)
        _carries(str(error), "DualHamiltonian.__call__ at p=[0.0, 0.0]", "minimize (L-BFGS-B)", results[0])
    else:
        pytest.fail(RETIRE)


#: What the non-smooth pin prints when H at p = 0 returns.
RETIRE = (
    "H at p = 0 returned with no ConvergenceError. If #2644 stage 2 has changed the method for rows 18 and 20-24, this "
    "pin is retired: delete test_a_non_smooth_lagrangian_in_2d_is_refused_at_p_0_until_stage_2, and replace it with "
    "stage 2's pin of the value at p = 0 against the exact maximum. If it has not, read row 21's own pins in this file, "
    "the failure pin and the exhaustion pin of DualHamiltonian.__call__: red there means row 21's check is gone."
)

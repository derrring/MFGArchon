"""core/hamiltonian.py's nine scipy minimiser calls hand a failed search to the one owner, which raises (#2644 B2, rows
16-24, stage 1). Before, each read ``res.x`` or ``res.fun`` and returned it whatever scipy said.

Each member is driven so that its own call fails, through a cost, Lagrangian or Hamiltonian a user can write:
- the bounded scalar searches (rows 17 and 19) spend their 500 evaluations on a V-shaped L over a box of +-1e150;
- Brent's search in the Moreau-Yosida prox (row 16) finds no bracket under a barrier cost that is NaN outside the unit
  ball;
- each L-BFGS-B search (rows 18 and 20-24) starts where a barrier L or H is NaN, and stops ABNORMAL.

The message is the failure's own, compared field by field: the member, its input, the routine and scipy's message;
the point and the value are checked against the member's objective, recomputed here.
"""

from __future__ import annotations

import math
import re

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import (
    ControlCostBase,
    DualHamiltonian,
    DualLagrangian,
    LagrangianBase,
    SeparableHamiltonian,
)
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
ABNORMAL = ("minimize (L-BFGS-B)", "ABNORMAL:")


def _dual_h():
    return DualHamiltonian(BarrierLagrangian(), alpha_bounds=(-2.0, 2.0))


def _dual_l():
    return DualLagrangian(SeparableHamiltonian(BarrierCost(lambda_=1.0)), p_bounds=(-2.0, 2.0))


def _vee_value(x):
    return abs(x[0] - 0.3)


#: member -> (its call, the name and input its message gives, (routine, scipy's message), the point and value it
#: stopped at, checked against its objective, and the iterations if the budget is what ran out).
MEMBERS = {
    "row 16 _MoreauYosidaControlCost._prox_h": (
        lambda: BarrierCost(lambda_=1.0).regularize(EPS).dp(np.array([0.0])),
        "_MoreauYosidaControlCost._prox_h at p=0.0",
        (
            "minimize_scalar (Brent)",
            "The algorithm terminated without finding a valid bracket. Consider trying different initial points.",
        ),
        lambda x, value: math.isnan(x[0]) and math.isnan(value),
        None,
    ),
    "row 17 LagrangianBase.conjugate_argmax, 1-D": (
        lambda: VeeLagrangian().conjugate_argmax(t=0.0, x=(0.5,), p=np.array([0.0]), m=1.0),
        "VeeLagrangian.conjugate_argmax at p=0.0",
        ("minimize_scalar (bounded)", "Maximum number of function calls reached."),
        lambda x, value: value == _vee_value(x),  # at p = 0 the objective is L
        500,
    ),
    "row 18 LagrangianBase.conjugate_argmax, n-D": (
        lambda: BarrierLagrangian().conjugate_argmax(t=0.0, x=X2, p=-OUTSIDE, m=1.0),
        "BarrierLagrangian.conjugate_argmax at p=[-1.5, -0.5]",
        ABNORMAL,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
        None,
    ),
    "row 19 LagrangianBase.proximal, 1-D": (
        lambda: VeeLagrangian().proximal(TAU, np.array([0.3])),
        "VeeLagrangian.proximal at z=0.3, tau=1e+300",
        ("minimize_scalar (bounded)", "Maximum number of function calls reached."),
        lambda x, value: value == _vee_value(x) + (x[0] - 0.3) ** 2 / (2 * TAU),
        500,
    ),
    "row 20 LagrangianBase.proximal, n-D": (
        lambda: BarrierLagrangian().proximal(1.0, OUTSIDE, x=X2),
        "BarrierLagrangian.proximal at z=[1.5, 0.5], tau=1.0",
        ABNORMAL,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
        None,
    ),
    "row 21 DualHamiltonian.__call__": (
        lambda: _dual_h()(t=0.0, x=X2, p=-OUTSIDE, m=1.0),
        "DualHamiltonian.__call__ at p=[-1.5, -0.5]",
        ABNORMAL,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
        None,
    ),
    "row 22 DualHamiltonian._find_optimal_alpha": (
        lambda: _dual_h().dp(t=0.0, x=X2, p=-OUTSIDE, m=1.0),
        "DualHamiltonian._find_optimal_alpha at p=[-1.5, -0.5]",
        ABNORMAL,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
        None,
    ),
    "row 23 DualLagrangian.__call__": (
        lambda: _dual_l()(t=0.0, x=X2, alpha=-OUTSIDE, m=1.0),
        "DualLagrangian.__call__ at alpha=[-1.5, -0.5]",
        ABNORMAL,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
        None,
    ),
    "row 24 DualLagrangian._find_optimal_p": (
        lambda: _dual_l().d_alpha(t=0.0, x=X2, alpha=-OUTSIDE, m=1.0),
        "DualLagrangian._find_optimal_p at alpha=[-1.5, -0.5]",
        ABNORMAL,
        lambda x, value: x == OUTSIDE.tolist() and math.isnan(value),
        None,
    ),
}

MESSAGE = re.compile(
    r"(?P<where>.+) stopped without success after (?P<nit>\d+) iterations and (?P<nfev>\d+) evaluations "
    r"\((?P<why>.*)\)\. The point it stopped at, x=\[(?P<x>[^\]]*)\] with value (?P<value>\S+), is not known to be "
    r"the optimum \(#2644\)\.",
    re.S,
)


@pytest.mark.parametrize("member", list(MEMBERS))
def test_a_failed_search_is_raised_by_the_owner_with_its_own_message(member):
    call, where, (routine, why), stopped_at, budget = MEMBERS[member]
    # The inputs leave the barrier's domain, and the 1e150 box overflows scipy's parabolic step, by construction.
    with pytest.raises(ConvergenceError) as raised, np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        call()
    error = raised.value
    fields = MESSAGE.fullmatch(str(error))
    assert fields is not None, str(error)
    assert fields["where"] == f"{where}: scipy's {routine}"
    assert fields["why"] == why
    x = [float(v) for v in fields["x"].split(", ")]
    assert stopped_at(x, float(fields["value"])), str(error)
    if budget is not None:
        assert int(fields["nit"]) == int(fields["nfev"]) == budget, str(error)
    assert error.solver_type == where.split(" at ")[0]
    assert error.iterations == int(fields["nit"])
    assert math.isnan(error.residual)

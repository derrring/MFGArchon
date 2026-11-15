"""#2901 stage 1c-ii: the Hamiltonian the solvers read, after the soft wall is composed, is the one its methods describe.

The solvers read a Hamiltonian's doors: HJB-SL hands ``evaluate_H`` and ``evaluate_dp`` one point, the HJB batch paths
hand them a batch of rows, and ``compute_fp_velocity_field`` hands ``evaluate_optimal_control`` a batch. A problem compares
each door with its method at the consistency check's probes, in the forms its readers hand it, at 0, T/2 and T, after the
wall is composed, and reads the wall back through ``__call__`` (`refuse_a_hamiltonian_whose_reads_disagree`,
`validate_composed_wall`).

Each case below names the defect it pins; #2968's body gives each one's behaviour at ad5a7726.
"""

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core import hamiltonian as hm
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.validation import ValidationError


def _a(v):
    return np.asarray(v, dtype=float)


def _coupling(m):
    return 0.5 * _a(m)


def _problem(hamiltonian, **wall):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.3),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.5 * (_a(x[0]) - 0.3) ** 2, T=0.5),
        Nt=4,
        **wall,
    )


def _interior_wall(x):
    """Nonzero at the node 0.7 alone, and zero at every probe node (0, 0.5, 1): only the largest-wall node reads it."""
    return ((_a(x[0]) > 0.62) & (_a(x[0]) < 0.78)).astype(float)


def _edge_wall(x):
    return (_a(x[0]) < 0.15).astype(float)


class _PointDoorDoubled(hm.SeparableHamiltonian):
    """The dp door twice dp at one point, right on a batch: HJB-SL's `_sl_value_update` reads it at one point, and only
    the point form sees it."""

    def evaluate_dp(self, state):
        p = _a(state.p)
        return (2.0 if p.ndim == 1 else 1.0) * super().evaluate_dp(state)


class _StaticControlDoorDoubled(hm.SeparableHamiltonian):
    """The control as a staticmethod naming p alone, read by its declared names, and its door twice it: a method read
    with all four slots raises, and the door goes uncompared (#2901's F3)."""

    @staticmethod
    def optimal_control(p):
        return -_a(p)

    def evaluate_optimal_control(self, state):
        return -2.0 * _a(state.p)


class _DpDoorDoubledAfterStart(hm.SeparableHamiltonian):
    """The dp door twice dp after t = 0: a check at t = 0 alone passes it."""

    def evaluate_dp(self, state):
        return (2.0 if state.t > 0 else 1.0) * super().evaluate_dp(state)


class _PointOnlyDpBesideALargePotential(hm.SeparableHamiltonian):
    """dp written for one point, reading row 0's momentum at every row of a batch, beside a potential of 1e6 at x = 0, a
    probe node: a margin taken from the largest H over a batch (2.2 there) hid the misread from every row."""

    def __call__(self, t, x, p, m):
        return super().__call__(t=t, x=x, p=p, m=m) - 1e6 * (_a(x[0]) < 0.05).astype(float)

    def dp(self, t, x, p, m):
        p = _a(p)
        return p[0] * np.ones_like(p)


class _ValueDoorOffByOneInAHundredMillion(hm.SeparableHamiltonian):
    """The value door 1 + 1e-8 times ``__call__``: above roundoff, below the differencing floor the dp and control doors
    are compared above. A door and its method of the same value are one function, so it is refused."""

    def evaluate_H(self, state):
        return (1.0 + 1e-8) * super().evaluate_H(state)


_X_INFINITE = 0.6  # a probe node on 11 nodes: one of the three `spread_samples` picks


class _DpDoorDoubledWhereHIsInfinite(hm.SeparableHamiltonian):
    """H infinite at one probe node, and the dp door twice dp there alone: an infinite margin at that row hid it."""

    def __call__(self, t, x, p, m):
        return super().__call__(t=t, x=x, p=p, m=m) + np.where(np.isclose(_a(x[0]), _X_INFINITE), np.inf, 0.0)

    def evaluate_dp(self, state):
        g = super().evaluate_dp(state)
        rows = g.shape[0] if g.ndim == 2 else 1
        factor = np.where(np.isclose(_a(state.x).reshape(rows, -1)[:, 0], _X_INFINITE), 2.0, 1.0)
        return (factor[:, None] if g.ndim == 2 else factor) * g


class _PointOnlyDefinedAboveTwoThirds(hm.SeparableHamiltonian):
    """H written for one point, reading row 0's momentum at every row, and defined only for x > 0.65. A batch skipped
    whole for a row left out compared nothing on a batch (review 1 of #2968). The probe nodes left are 1 and 0.9, whose
    probes, numbered by node, lay 3 apart and shared p; numbered by row, they do not (review 2 of #2968)."""

    def __call__(self, t, x, p, m):
        if np.any(_a(x[0]) <= 0.65):
            raise ValueError("H is defined for x > 0.65")
        return 0.5 * _a(p)[0] ** 2 - _coupling(m)


@pytest.mark.parametrize(
    "hamiltonian",
    [
        _PointDoorDoubled,
        _StaticControlDoorDoubled,
        _DpDoorDoubledAfterStart,
        _PointOnlyDpBesideALargePotential,
        _DpDoorDoubledWhereHIsInfinite,
        _PointOnlyDefinedAboveTwoThirds,
        _ValueDoorOffByOneInAHundredMillion,
    ],
    ids=[
        "point form",
        "method bound by its names",
        "after t = 0",
        "a large H at one row",
        "an infinite H at one row",
        "rows left out that would share p",
        "the value door off by 1e-8",
    ],
)
def test_a_door_that_reads_another_hamiltonian_than_its_method_is_refused(hamiltonian):
    # An infinite H differenced in the base class's methods is numpy's invalid value, not the subject here.
    with (
        np.errstate(invalid="ignore"),
        pytest.raises(ValueError, match=r"\.evaluate_(H|dp|optimal_control) .*reads differently"),
    ):
        _problem(hamiltonian(hm.QuadraticControlCost(), coupling=_coupling))


class _ControlDoorReadsABatchOnly(hm.SeparableHamiltonian):
    """The control door raises at one point and reads right on a batch, the one form its one reader,
    `compute_fp_velocity_field`, hands it."""

    def evaluate_optimal_control(self, state):
        if _a(state.p).ndim == 1:
            raise ValueError("this door reads a batch")
        return super().evaluate_optimal_control(state)


def test_the_control_door_is_compared_on_a_batch_only():
    """No solver hands the control door one point (#2901's plan revision), so a door that cannot read one is accepted."""
    assert (
        _problem(_ControlDoorReadsABatchOnly(hm.QuadraticControlCost(), coupling=_coupling)).hamiltonian_class
        is not None
    )


class _SkipsTheBaseInit(hm.HamiltonianBase):
    """Every method its own, and ``__init__`` skipping ``super().__init__()``, so it holds no ``finite_diff_eps``."""

    def __init__(self):
        self.cost = 1.0

    def __call__(self, t, x, p, m):
        return 0.5 * self.cost * np.sum(_a(p) ** 2, axis=-1) - _coupling(m)

    def dp(self, t, x, p, m):
        return self.cost * _a(p)

    def optimal_control(self, t, x, p, m):
        return -self.cost * _a(p)

    def dm(self, t, x, p, m):
        return -0.5 + 0.0 * _a(m)


def test_a_hamiltonian_that_skips_the_base_init_is_compared_with_the_checks_step():
    """It cannot be using the base class's finite differences, which read ``finite_diff_eps``; the doors are compared
    with the consistency check's step in its place, and the problem builds as it did at ad5a7726."""
    assert _problem(_SkipsTheBaseInit()).hamiltonian_class is not None


class _BatchDropsPotential(hm.SeparableHamiltonian):
    """``evaluate_H`` vectorised for batches of two or more, written without the potential; the library's door at a
    point. Under a wall that misses every probe node, its batch reads drop the wall (#2901's MM1)."""

    def evaluate_H(self, state):
        p = _a(state.p)
        if p.ndim == 2 and p.shape[0] > 1:
            return _a(self.control_cost.evaluate(p)) - _coupling(state.m)
        return super().evaluate_H(state)


class _CallIgnoresPotential(hm.SeparableHamiltonian):
    """``__call__`` written out, reading no potential: every solve drops the wall (#2002)."""

    def __call__(self, t, x, p, m):
        return _a(self.control_cost.evaluate(_a(p))) - _coupling(m)

    def evaluate_H(self, state):
        return np.asarray(self(t=state.t, x=state.x, p=state.p, m=state.m), dtype=float)


@pytest.mark.parametrize(
    ("hamiltonian", "wall", "refusal"),
    [
        (_BatchDropsPotential, _interior_wall, (ValueError, r"evaluate_H .*reads differently")),
        (_CallIgnoresPotential, _edge_wall, (ValidationError, r"does not reach its __call__")),
    ],
    ids=["the largest-wall node", "the wall read back"],
)
def test_a_hamiltonian_that_drops_the_composed_wall_is_refused(hamiltonian, wall, refusal):
    error, message = refusal
    with pytest.raises(error, match=message):
        _problem(
            hamiltonian(hm.QuadraticControlCost(), coupling=_coupling), state_penalty=wall, state_penalty_scale=5.0
        )


class _PointOnly(hm.SeparableHamiltonian):
    """Declares that it reads no batch; its doors raise on one, as a ``DualHamiltonian``'s do."""

    reads_a_batch = False

    def evaluate_H(self, state):
        if _a(state.p).ndim == 2:
            raise ValueError("one point at a time")
        return super().evaluate_H(state)


class _PointOnlyBatchWrong(hm.SeparableHamiltonian):
    """Declares that it reads no batch, and its value door returns a wrong row on one."""

    reads_a_batch = False

    def evaluate_H(self, state):
        value = super().evaluate_H(state)
        return value + 1.0 if _a(state.p).ndim == 2 else value


def test_reads_a_batch_false_excuses_a_raise_and_not_a_wrong_answer():
    """No solver reads the flag, so a batch reaches such a door on every batch path (#2901's 1c-ii, ruling Q2 as
    revised): a raise on the batch is the declaration holding, and is not compared, silently; a wrong row is refused."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _problem(_PointOnly(hm.QuadraticControlCost(), coupling=_coupling))
    assert not [str(w.message) for w in caught if "_PointOnly" in str(w.message)]
    with pytest.raises(ValueError, match=r"evaluate_H .*reads differently"):
        _problem(_PointOnlyBatchWrong(hm.QuadraticControlCost(), coupling=_coupling))

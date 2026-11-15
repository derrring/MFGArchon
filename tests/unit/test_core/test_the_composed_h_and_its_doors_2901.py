"""#2901 stage 1c-ii: the Hamiltonian the solvers read, after the soft wall is composed, is the one its methods describe.

The solvers read a Hamiltonian's doors (``evaluate_H``, ``evaluate_dp``, ``evaluate_optimal_control``): the
semi-Lagrangian path hands one point, every other path a batch of rows. A problem compares each door with its method at
the consistency check's probes, in both forms, at 0, T/2 and T, after the wall is composed, and reads the wall back
through ``__call__`` (`refuse_a_hamiltonian_whose_reads_disagree`, `validate_composed_wall`).

Every case below built at ad5a7726, and each names the defect it pins. On that sha's default solve, against the
shipped Hamiltonian under the same wall: three moved M by 0.18 to 0.88; ``_PointOnly`` raised in the solve; and three
read right by accident, since that solve does not read what they misread: ``_PointDoorDoubled``'s point door (M equal),
``_DpDoorDoubledAfterStart``'s dp door, which Newton's Jacobian alone reads (4.1e-07), and ``_PointOnlyBatchWrong``'s
constant shift of H on a batch (4.4e-16).
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
    """The control door twice the method at one point, right on a batch: only the point form sees it."""

    def evaluate_optimal_control(self, state):
        p = _a(state.p)
        return (2.0 if p.ndim == 1 else 1.0) * super().evaluate_optimal_control(state)


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


@pytest.mark.parametrize(
    "hamiltonian",
    [_PointDoorDoubled, _StaticControlDoorDoubled, _DpDoorDoubledAfterStart],
    ids=["point form", "method bound by its names", "after t = 0"],
)
def test_a_door_that_reads_another_hamiltonian_than_its_method_is_refused(hamiltonian):
    with pytest.raises(ValueError, match=r"\.evaluate_(optimal_control|dp) .*reads differently"):
        _problem(hamiltonian(hm.QuadraticControlCost(), coupling=_coupling))


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

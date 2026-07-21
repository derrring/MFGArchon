"""RL base-class hooks must not fabricate a metric nobody computed (Issue #1688).

`BaseRLSolver.evaluate_nash_gap()` returned `0.0` from a body that reads nothing. A Nash gap is
what an RL caller trusts to decide whether training reached equilibrium, and `0.0` is the
strongest possible claim about it. `scale_to_mean_field()` returned the finite-population solution
unchanged while its name and docstring promised a limit.

Neither had an override or a caller, so this pins a trap before it is sprung rather than a wrong
answer already shipped. The RL subtree is 39 files and growing; a default that reports exact
equilibrium is the kind of thing that gets inherited silently and then believed.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.base_solver import BaseRLSolver


class _MinimalRLSolver(BaseRLSolver):
    """Smallest concrete subclass.

    The abstract set is enumerated from ``BaseRLSolver.__abstractmethods__`` rather than guessed;
    a first version of this file implemented three of the five and every test failed on
    instantiation instead of on the thing under test.
    """

    def __init__(self, population_size=None):
        self.population_size = population_size
        # `solution` is a read-only property guarded by `_is_solved`; set the backing fields
        # rather than working around the guard, so these tests exercise the real accessor.
        self._solution = np.array([1.0, 2.0, 3.0])
        self._is_solved = True

    def create_agents(self):
        return []

    def create_environment(self):
        return None

    def solve(self, *args, **kwargs):
        return self.solution

    def train_agents(self):
        return {"reward": 0.0}

    def validate_solution(self):
        return {}


def test_nash_gap_raises_instead_of_reporting_exact_equilibrium():
    """`0.0` is not a safe default for a gap: it is the claim that there is none."""
    solver = _MinimalRLSolver()

    with pytest.raises(NotImplementedError, match="exact Nash equilibrium"):
        solver.evaluate_nash_gap()


def test_scale_to_mean_field_raises_for_a_finite_population():
    """The identity is not a conversion when the population is finite."""
    solver = _MinimalRLSolver(population_size=100)

    with pytest.raises(NotImplementedError, match="finite"):
        solver.scale_to_mean_field()


@pytest.mark.parametrize("size", [None, float("inf")])
def test_scale_to_mean_field_is_the_identity_for_an_infinite_population(size):
    """That branch is genuinely correct and must not be caught by the refusal.

    The mean field limit of an infinite population is itself, so returning the solution unchanged
    is the right answer rather than a missing implementation.
    """
    solver = _MinimalRLSolver(population_size=size)

    np.testing.assert_array_equal(solver.scale_to_mean_field(), solver.solution)


def test_an_override_is_honoured():
    """Pin the mechanism: a subclass that implements the hook is not blocked by the guard."""

    class _Implemented(_MinimalRLSolver):
        def evaluate_nash_gap(self) -> float:
            return 0.25

        def scale_to_mean_field(self):
            return self.solution / self.population_size

    solver = _Implemented(population_size=4)

    assert solver.evaluate_nash_gap() == pytest.approx(0.25)
    np.testing.assert_allclose(solver.scale_to_mean_field(), np.array([0.25, 0.5, 0.75]))

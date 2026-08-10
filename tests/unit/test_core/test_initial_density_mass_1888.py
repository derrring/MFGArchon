"""The initial density must carry mass 1, in every dimension.

Nothing pinned this. `MFGProblem` normalises `m_initial` so that its discrete integral is 1, but in
n-D it divided by a cell volume it computed itself as `prod(L/n)` -- the spacing of n *intervals*,
where a grid of n *nodes* spanning `[a, b]` inclusive has spacing `L/(n-1)`, which is what
`geometry.get_grid_spacing()` returns and what every mass measurement in the library uses. So every
n-D problem started at mass `(n/(n-1))^d`: 21% heavy on an 11-point 2-D grid, 42% on a 9-point 3-D
one, converging to 1 under refinement so that it reads as a first-order-convergent error rather than
a bug.

Why 5943 tests did not notice. Mass conservation is measured as *drift* from the initial mass, not
as deviation from 1 -- deliberately, and correctly, since drift is the physical property. A ratio is
invariant to the cell measure, so the entire mass-oracle family is structurally blind to the initial
value being wrong. See `test_mass_conservation_error_1672.py`, whose docstring recorded this exact
fork on 2026-07-21 and set it out of scope.

This file is the missing half: an external oracle on the absolute value, computed from the geometry's
own spacing rather than from the normaliser's.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem(dimension: int, n: int) -> MFGProblem:
    def m0(x):
        arr = np.asarray(x)
        r2 = np.sum((arr - 0.5) ** 2, axis=-1) if dimension > 1 else (arr - 0.5) ** 2
        return np.exp(-30 * r2)

    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)] * dimension,
            Nx_points=[n] * dimension,
            boundary_conditions=no_flux_bc(dimension=dimension),
        ),
        Nt=4,
        T=0.2,
        sigma=0.4,
        components=MFGComponents(
            m_initial=m0,
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


def _mass(problem: MFGProblem) -> float:
    """Integrate with the geometry's own volume element, not the normaliser's."""
    return float(np.sum(np.asarray(problem.m_initial, dtype=float)) * np.prod(problem.geometry.get_grid_spacing()))


@pytest.mark.parametrize(("dimension", "n"), [(1, 11), (1, 21), (2, 11), (2, 15), (2, 21), (3, 9)])
def test_the_initial_density_integrates_to_one(dimension: int, n: int):
    """The 1-D rows are the control: they take a different branch and were always correct.

    Without them, a regression that broke both branches equally would leave this file green in the
    only way that matters -- every row failing for the same reason reads as a bad fixture.
    """
    assert _mass(_problem(dimension, n)) == pytest.approx(1.0, rel=1e-12)


def test_the_error_this_replaces_had_a_closed_form_and_this_would_have_caught_it():
    """Pin the discrimination, not just the value.

    The pre-fix mass was exactly `(n/(n-1))^d`, verified against measurement on all six
    configurations above. Asserting that the current mass is NOT that number, on the configurations
    where the two differ most, states what this file would catch rather than leaving it to be
    inferred from a passing assertion.
    """
    for dimension, n in ((2, 11), (3, 9)):
        old_prediction = (n / (n - 1)) ** dimension
        assert old_prediction > 1.2, f"chose a configuration where the two verdicts barely differ: {old_prediction}"
        assert _mass(_problem(dimension, n)) != pytest.approx(old_prediction, rel=1e-6)

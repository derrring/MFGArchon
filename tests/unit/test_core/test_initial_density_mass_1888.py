"""`m(0, .)` must integrate to 1 under the geometry's own volume element.

`MFGProblem` normalises `m_initial` by dividing by its discrete integral. It used to compute that
integral's cell volume itself, as `prod((b - a) / n)` from `self.spatial_discretization` -- an
attribute that means the INTERVAL count when passed to the constructor and the NODE count when it
comes from a geometry. Under the first reading the expression is exactly the spacing; under the
second it is one interval too many. So the same 11x11 grid gave mass 1.0 through
`spatial_discretization=[10, 10]` and 1.21 through `geometry=TensorProductGrid(Nx_points=[11, 11])`.
The normaliser now asks the geometry, so both agree.

**What this file is, and is not.** It is a consistency pin between the normaliser and
`geometry.get_grid_spacing()`, which is the fork that produced the defect. It is **not** an external
oracle on the cell volume: the normaliser divides by `sum(m) * V` and this file multiplies by the
same `V` from the same accessor, so `V` cancels and the assertion holds for any spacing the accessor
returns. Verified -- monkeypatching `get_grid_spacing` to `L/n` leaves this file at 7 passed. That
the convention itself is `L/(n-1)` is pinned elsewhere, by tests carrying literal spacing values,
which do go red under that mutation.

Why 5943 tests did not catch the fork. Mass conservation is measured as *drift from the initial
mass*, deliberately and correctly, since drift is the physical property -- and a ratio is invariant
to the cell measure, so the whole mass-oracle family is blind to the initial value. See
`tests/unit/test_utils/test_mass_conservation_error_1672.py`, whose docstring recorded this exact
fork on 2026-07-21 and set it out of scope without filing an issue.
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


def test_both_construction_paths_agree():
    """The fork was between two ways of building the same grid, so the pin has to compare them.

    `== 1.0` on one path cannot see a disagreement between paths, and the previous version of this
    test asserted only that the mass is *not* the old value -- which is implied by the rows above
    and so could never be the sole failure. This builds the identical 11x11 and 9x9x9 grids through
    both public constructors and requires them to agree, which is the property that was violated:
    1.0 against 1.21 in 2-D and 1.0 against 1.4238 in 3-D, measured on the revision before the fix.
    """
    for dimension, intervals in ((2, 10), (3, 8)):
        via_geometry = _problem(dimension, intervals + 1)
        via_bounds = MFGProblem(
            spatial_bounds=[(0.0, 1.0)] * dimension,
            spatial_discretization=[intervals] * dimension,
            Nt=4,
            T=0.2,
            sigma=0.4,
            components=via_geometry.components,
        )
        assert np.allclose(via_geometry.geometry.get_grid_spacing(), via_bounds.geometry.get_grid_spacing()), (
            "the two constructors were meant to describe the same grid"
        )
        assert _mass(via_geometry) == pytest.approx(_mass(via_bounds), rel=1e-12)

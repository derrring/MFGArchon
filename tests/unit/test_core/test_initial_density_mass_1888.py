"""`MFGProblem` normalises `m(0, .)` on the GEOMETRY'S OWN MEASURE, and the two constructors agree.

**What this file pins, and why the measure is the whole point.** The constructor rescales the
initial density so that its discrete integral is 1. Which integral was the defect: it used
`sum(m) * dx`, the cell-centred one, while `TensorProductGrid` is endpoint-inclusive -- the wall
lies ON `x_0`, so the two end nodes own half a cell each and the measure is the trapezoid (#2145).
Normalising with one functional while the FP wall conserves the other is how a solve reports
perfect conservation of a quantity nobody asked about.

So the discriminating assertion is not `mass == 1` -- it is `grid.integrate(m) == 1` **while**
`sum(m) * dx != 1`. Only the second half can tell the two normalisers apart, and it is measured
below rather than asserted in prose.

**Why the oracle is written by hand.** The reported value comes from `geometry.integrate`, so
computing the expected value the same way would be a tautology one layer along -- exactly the defect
the original version of this file admitted to in its own docstring:

    It is **not** an external oracle on the cell volume: the normaliser divides by `sum(m) * V` and
    this file multiplies by the same `V` from the same accessor, so `V` cancels and the assertion
    holds for any spacing the accessor returns. Verified -- monkeypatching `get_grid_spacing` to
    `L/n` leaves this file at 7 passed.

The weights are therefore spelled out here from the coordinates. Independent review measured that
this works: mutating `quadrature_weights_1d` to the rectangle rule reddens all six rows of
`test_it_matches_an_independently_derived_integral`, and 70 of the 346 tests that touch the owner.

**Two things the original defect record keeps.** The fork it was written for -- an attribute meaning
the INTERVAL count from the constructor and the NODE count from a geometry, so the same 11x11 grid
measured 1.0 one way and 1.21 the other -- is still pinned by `test_both_construction_paths_agree`,
now including d = 4, where `spatial_shape` was a COUNT rather than a shape until this branch. And
the reason 5943 tests missed the original is unchanged: mass conservation is measured as drift from
the initial mass, correctly, and a ratio is invariant to the cell measure, so the whole mass-oracle
family is blind to the initial value. This file is where that blindness is covered.

**Not here:** whether the library should normalise AT ALL. `fix/1887-validate-do-not-normalise`
replaces the rescale with validate-and-report; its tiers and its "handed back unchanged" assertions
live on that branch, because that change moves what `f(m)` is evaluated at and has to carry the
`examples/` tree with it.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _gaussian(dimension: int):
    def m0(x):
        arr = np.asarray(x)
        r2 = np.sum((arr - 0.5) ** 2, axis=-1) if dimension > 1 else (arr - 0.5) ** 2
        return np.exp(-30 * r2)

    return m0


def _problem(dimension: int, n: int, m0=None) -> MFGProblem:
    with warnings.catch_warnings():
        # Tier 3 fires on this fixture by construction -- a Gaussian's integral is not 1 -- and the
        # warning itself is asserted in `TestTheThreeTiers`, not incidentally here.
        warnings.filterwarnings("ignore", message="initial density mass")
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
                m_initial=m0 or _gaussian(dimension),
                u_terminal=lambda x: 0.0,
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0),
                    coupling=lambda m: m,
                    coupling_dm=lambda m: 1.0,
                ),
            ),
        )


def _mass_by_hand(problem: MFGProblem) -> float:
    """The trapezoid weights, written out, deliberately not taken from the owner.

    See the module docstring: the reported value comes from `geometry.integrate`, so an oracle that
    called it too would cancel exactly as the old `V` did and hold for any weights the grid returns.
    """
    m = np.asarray(problem.m_initial, dtype=float)
    weights = None
    for coords in problem.geometry.coordinates:
        x = np.asarray(coords, dtype=float)
        w = np.empty_like(x)
        w[0] = (x[1] - x[0]) / 2.0
        w[-1] = (x[-1] - x[-2]) / 2.0
        w[1:-1] = (x[2:] - x[:-2]) / 2.0
        weights = w if weights is None else np.multiply.outer(weights, w)
    return float((m * weights).sum())


class TestTheReportedMass:
    @pytest.mark.parametrize(("dimension", "n"), [(1, 11), (1, 21), (2, 11), (2, 15), (2, 21), (3, 9)])
    def test_it_matches_an_independently_derived_integral(self, dimension: int, n: int):
        """The 1-D rows are the control: they take a different measurement branch.

        Without them a regression that broke every branch equally would leave this file failing in
        the only way that reads as a bad fixture rather than as a defect.
        """
        problem = _problem(dimension, n)
        assert problem.initial_mass == pytest.approx(_mass_by_hand(problem), rel=1e-12)
        assert problem.initial_mass_measure == "grid"

    @pytest.mark.parametrize("dimension", [1, 2])
    def test_the_normaliser_used_the_trapezoid_and_not_the_rectangle(self, dimension: int):
        """The assertion that separates the two normalisers, which `mass == 1` cannot.

        `sum(m) * prod(dx)` and `grid.integrate(m)` both equal 1 after normalising by themselves, so
        `mass == 1` holds under either and pins nothing. What tells them apart is the OTHER
        functional: normalise on the trapezoid and the rectangle sum comes out at 1 + s, where s is
        the endpoint share. Measured on this fixture: 1.007518 in 1-D and 1.015093 in 2-D.

        Mutation: point the constructor's normaliser back at `sum(m) * dx` and both halves swap --
        `integrate` reads 1 - s and this fails on the first assertion.
        """
        n = 11
        problem = _problem(dimension, n)
        m = np.asarray(problem.m_initial, dtype=float)
        assert float(problem.geometry.integrate(m)) == pytest.approx(1.0, rel=1e-12), (
            "the constructor must normalise on the geometry's own measure"
        )
        spacing = np.prod(np.asarray(problem.geometry.get_grid_spacing(), dtype=float))
        rectangle = float(m.sum() * spacing)
        assert rectangle != pytest.approx(1.0, rel=1e-6), (
            f"the rectangle sum is {rectangle!r}. If it is 1, the normaliser is the cell-centred "
            "integral again and #2145 has been reverted -- the two measures cannot BOTH be 1 on a "
            "density whose endpoints are non-zero."
        )
        assert rectangle > 1.0, "the rectangle over-counts the two half-cells, so it must exceed 1"

    def test_both_construction_paths_agree(self):
        """The #1888 fork was between two ways of building the same grid, so the pin compares them.

        It was 1.0 against 1.21 in 2-D and 1.0 against 1.4238 in 3-D on the revision before the fix.
        Reported mass rather than normalised mass now, but the same property.

        **d = 4 is here because the fork was still open there**, and this migration is what found it.
        `MFGProblem.__init__` set `spatial_shape` from `spatial_discretization` at d <= 3 and from
        `geometry.num_spatial_points` above it -- a count, not a shape -- so `spatial_bounds=` built
        a flat (14641,) density where `geometry=` built (11, 11, 11, 11). Nothing in the repository
        compared a field against the grid shape, so it went unseen; `geometry.integrate` was the
        first caller that did, and it refused. Both constructors now read `get_grid_shape()`.

        What this still discriminates after that consolidation: the interval-to-point conversion,
        `Nx_points = [n + 1 for n in spatial_discretization]`. Both paths reaching the same owner
        makes the SHAPE agreement structural, so the assertion that carries weight is that the two
        argument conventions describe the same grid. Mutation -- drop the `+ 1` -- and every
        dimension below fails on the spacing assertion.
        """
        for dimension, intervals in ((2, 10), (3, 8), (4, 6)):
            via_geometry = _problem(dimension, intervals + 1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="initial density mass")
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
            expected_shape = (intervals + 1,) * dimension
            assert via_bounds.spatial_shape == expected_shape, (
                f"{dimension}-D: spatial_shape is a SHAPE, not a point count -- "
                f"a flat (prod,) here is the d >= 4 fork this row was added for"
            )
            assert via_geometry.spatial_shape == expected_shape
            assert np.asarray(via_bounds.m_initial).shape == expected_shape
            assert via_geometry.initial_mass == pytest.approx(via_bounds.initial_mass, rel=1e-12)


# `TestTheThreeTiers` lived here and has moved to `fix/1887-validate-do-not-normalise` with the
# behaviour it tested. On this branch the constructor still normalises, so there is no reported mass
# to warn about and no sub-probability density to admit -- those are that change's subject, not this
# one's. What stays here is the MEASURE, which is #2145's.

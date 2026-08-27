"""`MFGProblem` reports `m(0, .)`'s mass on the geometry's measure, and does not change it.

**The subject of this file changed with #1887.** It used to assert that the initial density
*integrates to 1*, because `MFGProblem` divided `m_initial` by its discrete integral. That rescale
is gone: normalising is the caller's job, and the library's job is to say what it received. So the
property under test is now *the mass is reported and the density is untouched*, and the old
assertion -- `== 1.0` -- would be wrong, since a caller may legitimately hand in a sub-probability
density or one population's share.

**What this file was, in its author's own words, and why that changes too.** It said:

    It is **not** an external oracle on the cell volume: the normaliser divides by `sum(m) * V` and
    this file multiplies by the same `V` from the same accessor, so `V` cancels and the assertion
    holds for any spacing the accessor returns. Verified -- monkeypatching `get_grid_spacing` to
    `L/n` leaves this file at 7 passed.

That was true and it is the reason the oracle below is written by hand. The reported mass now comes
from `geometry.integrate`, so computing the expected value the same way would be the identical
tautology one layer along. The weights are therefore spelled out here, from the coordinates. This is
the one place in the migration where a second derivation of the formula is the point rather than the
defect -- everywhere else, #2145's owner is called.

**Two things the original defect record should keep.** The fork it was written for -- an attribute
meaning the INTERVAL count from the constructor and the NODE count from a geometry, so the same
11x11 grid measured 1.0 one way and 1.21 the other -- is still pinned, by
`test_both_construction_paths_agree` below. And the reason 5943 tests missed it stands unchanged:
mass conservation is measured as *drift from the initial mass*, correctly, and a ratio is invariant
to the cell measure, so the whole mass-oracle family is blind to the initial value. This file is
where that blindness is covered.
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
    def test_the_density_is_handed_back_unchanged(self, dimension: int):
        """#1887's subject in one assertion: the library must not substitute a nearby object.

        Mutation: restore `self.m_initial /= integral` and this goes red while every mass-drift
        oracle in the repository stays green, which is exactly how the rescale survived.
        """
        n = 11
        problem = _problem(dimension, n)
        axis = np.linspace(0.0, 1.0, n)
        grid = np.stack(np.meshgrid(*([axis] * dimension), indexing="ij"), axis=-1) if dimension > 1 else axis
        expected = _gaussian(dimension)(grid)
        assert np.allclose(np.asarray(problem.m_initial), expected, rtol=0, atol=0)
        assert problem.initial_mass != pytest.approx(1.0, rel=1e-6), (
            "this fixture's Gaussian does not integrate to 1; if it now does, the rescale is back"
        )

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


class TestTheThreeTiers:
    """#1887's decision, one test per tier. Only tier 3 presumes a target of 1."""

    def test_tier_1_refuses_a_negative_density(self):
        with pytest.raises(ValueError, match="negative"):
            _problem(1, 11, m0=lambda x: np.asarray(x) - 0.5)

    def test_tier_1_refuses_a_zero_density(self):
        with pytest.raises(ValueError, match="mass|positive"):
            _problem(1, 11, m0=lambda x: np.zeros_like(np.asarray(x, dtype=float)))

    def test_tier_3_warns_off_one_and_says_how_to_silence_it(self):
        """The message must carry the number, the measure, and the remedy. A bare "mass is not 1"
        recreates the invisible convention the report exists to remove (#2145)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            MFGProblem(
                geometry=TensorProductGrid(
                    bounds=[(0.0, 1.0)],
                    Nx_points=[11],
                    boundary_conditions=no_flux_bc(dimension=1),
                ),
                Nt=4,
                T=0.2,
                sigma=0.4,
                components=MFGComponents(
                    m_initial=_gaussian(1),
                    u_terminal=lambda x: 0.0,
                    hamiltonian=SeparableHamiltonian(
                        control_cost=QuadraticControlCost(control_cost=1.0),
                        coupling=lambda m: m,
                        coupling_dm=lambda m: 1.0,
                    ),
                ),
            )
        messages = [str(w.message) for w in caught if "initial density mass" in str(w.message)]
        assert messages, "tier 3 did not fire on a density whose integral is not 1"
        assert "grid" in messages[0], "the warning must name the measure, not just the number"
        assert "Silence" in messages[0] or "silence" in messages[0]

    def test_tier_3_is_silent_on_a_density_that_already_integrates_to_one(self):
        """The tolerance is not a licence: a correctly normalised density must warn zero times, or
        the warning becomes noise and gets filtered along with the tiers that matter."""
        n = 11
        axis = np.linspace(0.0, 1.0, n)
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
        raw = _gaussian(1)(axis)
        normalised = raw / grid.integrate(raw)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            problem = MFGProblem(
                geometry=grid,
                Nt=4,
                T=0.2,
                sigma=0.4,
                components=MFGComponents(
                    m_initial=lambda x: np.interp(np.asarray(x), axis, normalised),
                    u_terminal=lambda x: 0.0,
                    hamiltonian=SeparableHamiltonian(
                        control_cost=QuadraticControlCost(control_cost=1.0),
                        coupling=lambda m: m,
                        coupling_dm=lambda m: 1.0,
                    ),
                ),
            )
        assert problem.initial_mass == pytest.approx(1.0, rel=1e-12)
        assert not [w for w in caught if "initial density mass" in str(w.message)]

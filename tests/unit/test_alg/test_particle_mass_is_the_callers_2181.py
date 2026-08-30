"""The particle solver returns the mass the caller handed in, not 1 (#2181).

**Why this needs its own file, and why the obvious test would have passed before the fix.** A
particle method does not carry mass. `sample_from_density` draws N particles from the density's
SHAPE -- a set of positions has no scale -- and the KDE reconstruction returns a density integrating
to about 1 whatever went in. So every assertion of the form "the solved mass is 1" held, and held for
a reason that was not the solver working.

That was invisible while `MFGProblem` normalised every initial density to 1. Since #1887 it does not:
the caller's mass is a modelling decision the library publishes as `problem.initial_mass`, and a
solver returning a different one contradicts its own library. Measured before the fix: an initial
density of mass 0.300000 came back as 1.000000, a factor of 3.33.

**`kde_normalization` was not the cause**, which is why the first proposed fix -- change the default
-- would have done nothing. Instrumented: under `NONE`, `_should_normalize_density()` returns False
on every call and `_normalize_density` is never invoked, and the mass is still 1.0. All three modes
are covered below for that reason.

**One measure, since #2181.** `_measure` answers "what is this slice's mass" on `geometry.integrate`'s
measure and nothing else does, so the rectangle rule no longer appears anywhere in this solver's
accounting. An earlier version of this paragraph said the opposite -- that `sum(M) * dx` was pinned
while the grid measure drifted -- which described a design this branch replaced, and contradicted two
assertions the same change wrote.

**The default is `INITIAL_ONLY`, not `ALL`.** Pinning every slice makes
`SolverResult.mass_conservation_error` identically 4.4e-16: mass conservation by fiat, which is what
#1683 removed from the FDM, GFDM and network FP paths and which this solver was the last instance of.
So t=0 carries the caller's mass exactly and later slices carry one factor, leaving the
reconstruction's own drift measurable.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver, KDENormalization
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem(grid: TensorProductGrid, centre: float, share: float) -> MFGProblem:
    """A 1-D Gaussian scaled so that its integral on THIS grid is exactly `share`.

    Built on the v1.0 API, and the reason is narrower than it first looks. It keeps this file from
    adding a DeprecationWarning identity to the warning ratchet -- but "the legacy constructor warns"
    is NOT on its own a reason to prefer v1.0, because that warning says legacy will be removed at
    v1.0.0 while `Model`/`Conditions` cannot yet express everything `MFGComponents` can:
    `potential_func` has no v1.0 home at all, and 20 sites pass it. So the conversion here rests on
    having CHECKED that the two paths build the same object for this fixture -- Nt, T, sigma,
    dimension, spatial_shape, initial_mass, initial_mass_measure and `m_initial` all identical --
    not on the deprecation being authoritative.

    The tier-3 mass warning is filtered rather than recorded: it fires by construction on every
    fixture below, since a share of 0.3 is the whole point, and it is pinned where it belongs, in
    `test_initial_density_mass_1888.py`.
    """
    x = np.asarray(grid.coordinates[0], dtype=float)
    scale = float(grid.integrate(np.exp(-50.0 * (x - centre) ** 2))) / share
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="initial density mass")
        return MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                sigma=0.4,
            ),
            domain=grid,
            conditions=Conditions(
                m_initial=lambda z, c=centre, k=scale: np.exp(-50.0 * (np.asarray(z) - c) ** 2) / k,
                u_terminal=lambda z: 0.0,
                T=0.2,
            ),
            Nt=4,
        )


def _grid(n: int = 21) -> TensorProductGrid:
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))


@pytest.mark.parametrize("mode", [KDENormalization.ALL, KDENormalization.INITIAL_ONLY, KDENormalization.NONE])
def test_a_sub_probability_density_keeps_its_mass(mode: KDENormalization):
    """All three modes, because `NONE` is the one that surprises.

    Mutation: replace `_restore_caller_scale` with the identity and every row reads 1.000000 against
    a target of 0.300000 -- including this parametrisation's `NONE`, where nothing normalises
    anything. (The scaling used to live in `solve_fp_system`; it moved to the reconstruction sites
    when review showed the exit was the wrong place.)
    """
    grid = _grid()
    problem = _problem(grid, centre=0.5, share=0.3)
    assert problem.initial_mass == pytest.approx(0.3, rel=1e-12), "the fixture itself must carry the mass"

    solver = FPParticleSolver(problem, num_particles=2000, kde_normalization=mode)
    M = solver.solve_fp_system(np.asarray(problem.m_initial, dtype=float))

    assert float(grid.integrate(M[0])) == pytest.approx(problem.initial_mass, rel=1e-12), (
        f"the solve returned a different mass from the one the library reports ({mode})"
    )


def test_a_unit_density_is_snapped_to_exactly_one():
    """Mass 1 is where the old behaviour and the new one nearly agree -- and the gap is the point.

    This started life as a positive control, on the claim that mass 1 is the case both behaviours get
    right so the row would stay green under the mutation. **That claim was false, measured**: remove
    the scaling and this fails at `0.9999916087845434 != 1.0`. The unscaled KDE lands NEAR 1, never
    exactly 1, and the tolerance here is tight enough to see the difference.

    So it is renamed to what it actually pins: the scaling is doing arithmetic even in the case that
    looks like a no-op, and `problem.initial_mass` is reproduced to twelve digits rather than to
    KDE accuracy. Kept at `rel=1e-12` deliberately -- loosening it to make the original "control"
    story true would discard the sharper fact.
    """
    grid = _grid()
    problem = _problem(grid, centre=0.5, share=1.0)
    M = FPParticleSolver(problem, num_particles=2000).solve_fp_system(np.asarray(problem.m_initial, dtype=float))
    assert float(grid.integrate(M[0])) == pytest.approx(1.0, rel=1e-12)


def test_a_later_slice_carries_the_mass_too():
    """The load-bearing property, and the one every other test here misses.

    The factor is calibrated ONCE per solve and reused for every later slice. Every other assertion
    in this file is on `M[0]`, which proves nothing about that: on the callable-drift path `M[0]` is
    the caller's own array and the factor is calibrated on `M[1]`, so `M[0]` would be right even if
    the reuse were broken. Review of PR #2185 named this gap.

    The bound is drift, not equality: the reconstruction has its own error and pinning `M[t]` to the
    target exactly would be the per-slice renormalisation this design rejects. Measured across five
    seeds on this fixture: 0.0098 to 0.0146, so 5e-2 is ~3.4x above the observed spread while a lost
    factor lands at 1/0.3 - 1 = 2.33, two orders up.
    """
    grid = _grid()
    problem = _problem(grid, centre=0.5, share=0.3)
    M = FPParticleSolver(problem, num_particles=2000, seed=7).solve_fp_system(
        np.asarray(problem.m_initial, dtype=float)
    )
    masses = np.array([float(grid.integrate(M[t])) for t in range(M.shape[0])])
    assert masses.shape[0] > 1, "a one-slice history cannot test reuse"
    assert np.max(np.abs(masses[1:] / 0.3 - 1.0)) < 5e-2, f"later slices drifted from the caller's mass: {masses}"


def test_the_callable_drift_path_calibrates_on_a_KDE_slice_not_the_input():
    """The path where `M[0]` is the caller's array, so the factor comes from `M[1]`.

    This is the configuration that broke the first design: `M_density_on_grid[0] = M_initial.copy()`
    made a single exit-calibrated factor read exactly 1.0, leaving `M[1:]` at mass 1 -- 0.300000
    followed by 0.999790 inside one returned history. Both ends are asserted, because asserting only
    `M[0]` is what let that pass.
    """
    grid = _grid()
    problem = _problem(grid, centre=0.5, share=0.3)
    M = FPParticleSolver(problem, num_particles=2000, seed=7).solve_fp_system(
        np.asarray(problem.m_initial, dtype=float),
        drift_field=lambda t, x, m: np.zeros_like(np.atleast_1d(x)),
    )
    masses = np.array([float(grid.integrate(M[t])) for t in range(M.shape[0])])
    assert masses[0] == pytest.approx(0.3, rel=1e-12)
    assert np.max(np.abs(masses[1:] / 0.3 - 1.0)) < 5e-2, f"the callable-drift path lost the mass after t=0: {masses}"


def test_two_populations_keep_their_shares():
    """The reason #1887 wanted this: a share is carried by the density, and had no other home.

    `MultiPopulationProblem` holds `list[MFGProblem]` and population k's share is the integral of
    m_k -- there is no `theta` field. Before this fix both populations came back at 1.0 and the
    decomposition summed to 2.0, so shares existed on the FDM/FVM paths and silently not on this one.
    """
    grid = _grid()
    solved = []
    for centre, share in ((0.35, 0.3), (0.65, 0.7)):
        problem = _problem(grid, centre, share)
        M = FPParticleSolver(problem, num_particles=2000).solve_fp_system(np.asarray(problem.m_initial, dtype=float))
        solved.append(float(grid.integrate(M[0])))

    assert solved[0] == pytest.approx(0.3, rel=1e-12)
    assert solved[1] == pytest.approx(0.7, rel=1e-12)
    assert sum(solved) == pytest.approx(1.0, rel=1e-12), (
        "the shares must still decompose; before #2181 this summed to 2.0"
    )


def test_ALL_pins_and_NONE_carries_so_the_modes_are_not_the_same_solve():
    """The design decision the whole change rests on, and which nothing pinned until now.

    The scale is put back two different ways: ALL divides each slice by its own mass and multiplies
    by the target, so every slice carries it exactly; NONE multiplies by ONE factor calibrated on the
    first slice, so the reconstruction's own drift stays visible. Recalibrating per slice would look
    like a harmless simplification and is not: it makes NONE behave like ALL and erases the drift,
    which is the information `kde_normalization=NONE` exists to expose.

    Round 3 of the PR #2185 review measured that mutation against all 32 test files that reach this
    solver and it killed **nothing** -- every assertion was an upper bound on drift, and driving the
    drift to zero sails through an upper bound. This test asserts the gap in the other direction.

    Measured on this fixture, seed 7, default `reflection`: ALL is flat to 2.2e-16 across the
    history while NONE drifts 8.0e-03. The bound below sits between those by two orders.
    """
    grid = _grid()
    problem = _problem(grid, centre=0.5, share=0.3)

    def masses(mode):
        M = FPParticleSolver(problem, num_particles=2000, kde_normalization=mode, seed=7).solve_fp_system(
            np.asarray(problem.m_initial, dtype=float)
        )
        return np.array([float(grid.integrate(M[t])) for t in range(M.shape[0])])

    pinned = masses(KDENormalization.ALL)
    carried = masses(KDENormalization.NONE)

    assert np.max(np.abs(pinned / 0.3 - 1.0)) < 1e-9, f"ALL must pin every slice to the caller's mass, got {pinned}"
    assert np.max(np.abs(carried / 0.3 - 1.0)) > 1e-4, (
        "NONE must leave the reconstruction's drift visible. A flat history here means the factor is "
        f"being recalibrated per slice, which is normalisation by another name. Got {carried}"
    )


def test_particles_without_a_density_are_left_alone():
    """No density in means no target, and the solver must not invent one.

    `solve_fp_system(initial_particles=...)` supplies positions rather than a density, so there is
    nothing to measure and the scaling is skipped. Pinned because the natural implementation --
    scaling by `problem.initial_mass` unconditionally -- would silently impose the problem's mass on
    a caller who deliberately did not supply one.
    """
    grid = _grid()
    problem = _problem(grid, centre=0.5, share=0.3)
    solver = FPParticleSolver(problem, num_particles=2000)
    assert solver._caller_mass(None) is None

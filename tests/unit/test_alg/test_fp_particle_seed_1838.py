"""`FPParticleSolver` must be able to repeat itself on demand. Issue #1838.

Without a seed no invariant can return a verdict on this solver. Measured on the #1822 periodic
capability matrix, three trials of one identical configuration gave `monotone = False, False, True`
-- so marking it `xfail` asserts a failure it does not reliably have, and marking it `pass` asserts
the opposite. It is skipped there and named in `STOCHASTIC_UNSEEDED`, whose comment already states
the conclusion: "the seeding is the fix."

Two contracts, and the second is the one that constrains the design:

- `seed=<int>` -- a private `Generator`, so two runs agree bit for bit and nothing else touching
  `np.random` can perturb them;
- `seed=None` -- the **global** stream, unchanged. Twelve files in this repository call
  `np.random.seed(...)` and then build this solver to compare two runs. A private Generator by
  default would stop honouring those seeds and make those comparisons non-deterministic rather
  than failing, which is the worse outcome: the tests would still pass, just meaninglessly.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

NX, NT = 25, 8


def _problem():
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1)),
        T=0.3,
        Nt=NT,
        sigma=0.25,
        components=MFGComponents(
            m_initial=lambda z: 1.0 + 0.5 * np.cos(2 * np.pi * np.asarray(z) + 1.1),
            u_terminal=lambda z: np.zeros_like(np.asarray(z)),
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


def _solve(seed, num_particles=2000):
    x = np.linspace(0.0, 1.0, NX)
    solver = FPParticleSolver(_problem(), num_particles=num_particles, seed=seed)
    m0 = 1.0 + 0.5 * np.cos(2 * np.pi * x + 1.1)
    return np.asarray(solver.solve_fp_system(m0, np.tile(np.sin(2 * np.pi * x), (NT + 1, 1))))


def test_the_same_seed_gives_the_same_density():
    """Bit-identical, not close: a seeded stochastic solver is a deterministic function."""
    first, second = _solve(20260805), _solve(20260805)
    assert first.shape == second.shape
    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_densities():
    """The discriminating half. Without it, a solver that ignored `seed` would pass the test above.

    A stochastic solver that returned the same field for every seed would satisfy reproducibility
    perfectly and be broken -- so equality alone certifies nothing.
    """
    first, second = _solve(20260805), _solve(11111)
    assert first.shape == second.shape
    assert not np.array_equal(first, second), (
        "two different seeds produced bit-identical densities; the seed is not reaching the draws"
    )


def test_an_unrelated_draw_cannot_perturb_a_seeded_solve():
    """The private stream must be private -- this is what the global state could never give.

    Interleaving an unrelated `np.random` call between two constructions is exactly what made the
    old behaviour irreproducible in practice, and it is not hypothetical: any library, fixture or
    plugin drawing from the global stream shifts every subsequent particle solve.
    """
    first = _solve(4242)
    np.random.random(12345)  # deliberately perturbing the GLOBAL stream
    second = _solve(4242)
    np.testing.assert_array_equal(first, second)


def test_seed_none_still_follows_the_global_seed():
    """The compatibility contract. Twelve files in this repo depend on exactly this.

    `np.random.seed(42)` before construction must still make an unseeded solver repeat itself, or
    every existing two-run comparison silently stops comparing anything.
    """
    np.random.seed(42)  # the legacy contract under test
    first = _solve(None)
    np.random.seed(42)
    second = _solve(None)
    np.testing.assert_array_equal(first, second)


def test_seed_none_is_not_secretly_deterministic():
    """Control for the test above: without the global seed being reset, the runs must differ.

    Otherwise `test_seed_none_still_follows_the_global_seed` would pass on a solver that ignored
    randomness altogether, and would be measuring nothing.
    """
    np.random.seed(7)
    first = _solve(None)
    second = _solve(None)  # no reset: the global stream has moved on
    assert not np.array_equal(first, second), (
        "two consecutive unseeded solves agreed bit for bit; the draws are not reaching the stream"
    )

"""`FPParticleSolver` must read the value function at the time level it is stepping. Issue #1792.

The FP sweep runs FORWARD while the HJB sweep runs backward, and the pairing of the two is what
makes the MFG fixed point mean anything. Reading `U[T - t_n]` instead of `U[t_n]` keeps every index
in range for any `Nt` -- negative indexing wraps -- so nothing raises and nothing warns, while the
density tracks the control's journey backwards.

That mutation survived a 25-axis sweep against the entire marker-filtered suite: 5770 passed, 0
failed. The reason is structural rather than a missing assertion. Every `U_solution` the particle
tests build is `np.zeros((Nt, Nx))` or `np.tile(f(x), (Nt, 1))` -- time-CONSTANT, so `U[n]` and
`U[-1-n]` are literally the same array and the mutation is a no-op by construction. A fixture where
time does not vary cannot test a coupling to time, and no assertion added to one would help.

So the datum is a well whose centre MOVES, 0.2 -> 0.8 over [0, T], and the observable is the
correlation between where the mass is and where the well is, over the whole trajectory rather than
at the endpoint alone. Measured here:

    k = 2.0    correct +0.9917    mirrored -0.0143
    k = 5.0    correct +0.9981    mirrored -0.6194

Both indexing sites are covered, because they are separate code paths with the same defect
available to each: `fp_particle.py` reads `U_solution_for_drift[n_time_idx]` on the 1D path and
`U_solution_for_drift[t_idx]` on the nD one.

The seed comes from #1838. Without it this is a coin flip on the Monte Carlo draw rather than a
statement about the solver.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

NT, T = 40, 2.0
SIGMA = 0.02
SEED = 1792
WELL_START, WELL_END = 0.2, 0.8
MASS_START = 0.2

# The correlation the correct solver achieves is +0.99; the mirrored read gives -0.01 at k=2.0 and
# -0.62 at k=5.0. Anywhere in between is not a threshold anyone has to tune.
TRACKS = 0.9
# ...and the control: correlation alone is satisfied by a mass that follows the well loosely from
# far away, so the run also has to get NEAR it. Correct is 0.114 (k=2) and 0.054 (k=5).
MAX_LAG = 0.2


def _centres() -> np.ndarray:
    return np.linspace(WELL_START, WELL_END, NT + 1)


def _travelling_well(k: float, nx: int, dim: int) -> np.ndarray:
    """U(t, x) = k/2 * |x - c(t)|^2, with the well centre c travelling WELL_START -> WELL_END.

    Quadratic because the drift it induces, -grad U, is then linear in the distance to the centre:
    the mass tracks c(t) rather than merely being pushed against a wall, so "where is the mass"
    answers "where does the solver think the well is" and nothing else.
    """
    x = np.linspace(0.0, 1.0, nx)
    if dim == 1:
        return 0.5 * k * (x[None, :] - _centres()[:, None]) ** 2
    grids = np.meshgrid(*([x] * dim), indexing="ij")
    # Only the first axis moves; the others are a stationary well, so the centroid along axis 0 is
    # the whole signal and the other axes cannot contribute to it.
    return np.stack([0.5 * k * ((grids[0] - c) ** 2 + sum(g**2 for g in grids[1:])) for c in _centres()])


def _initial_density(nx: int, dim: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, nx)
    if dim == 1:
        m0 = np.exp(-200.0 * (x - MASS_START) ** 2)
    else:
        grids = np.meshgrid(*([x] * dim), indexing="ij")
        m0 = np.exp(-200.0 * ((grids[0] - MASS_START) ** 2 + sum(g**2 for g in grids[1:])))
    return m0 / m0.sum()


def _problem(nx: int, dim: int) -> MFGProblem:
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)] * dim, Nx_points=[nx] * dim, boundary_conditions=no_flux_bc(dimension=dim)
        ),
        T=T,
        Nt=NT,
        sigma=SIGMA,
        components=MFGComponents(
            # *coords: the nD path calls these with one argument per axis.
            m_initial=lambda *c: np.exp(
                -200.0 * ((np.asarray(c[0]) - MASS_START) ** 2 + sum(np.asarray(a) ** 2 for a in c[1:]))
            ),
            u_terminal=lambda *c: np.zeros_like(np.asarray(c[0])),
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


def _centroid_trajectory(k: float, nx: int, dim: int, particles: int) -> np.ndarray:
    """Where the mass sits along axis 0, at every time level."""
    x = np.linspace(0.0, 1.0, nx)
    solver = FPParticleSolver(_problem(nx, dim), num_particles=particles, seed=SEED)
    density = np.asarray(
        solver.solve_fp_system(_initial_density(nx, dim), potential_field=_travelling_well(k, nx, dim))
    )
    along_axis0 = density.reshape(density.shape[0], nx, -1).sum(axis=2)
    weight = along_axis0.sum(axis=1)
    return (along_axis0 * x).sum(axis=1) / weight


@pytest.mark.parametrize(
    ("dim", "nx", "particles", "k"), [(1, 41, 20000, 2.0), (1, 41, 20000, 5.0), (2, 21, 20000, 5.0)]
)
def test_the_density_follows_the_well_forward_in_time(dim, nx, particles, k):
    """The density must track the well's journey in the direction the journey actually runs.

    The correlation is the load-bearing assertion and it comes first, because it is the one whose
    failure has a single likely cause. Under `U[-1 - n]` it collapses from +0.99 to -0.01 (k=2) or
    -0.62 (k=5) -- the mass is following the mirrored journey.

    The lag assertion below is the CONTROL, and the issue this test closes is explicit about why it
    is needed: a first attempt measured a 1% gap because the fixture was under-driven and neither
    run tracked anything, which is a null result that reads as a small effect. Correlation alone
    does not rule that out -- a mass drifting loosely in the same general direction correlates well
    while never going near the well.
    """
    centroids = _centroid_trajectory(k, nx, dim, particles)
    centres = _centres()
    correlation = float(np.corrcoef(centroids, centres)[0, 1])
    lag = float(np.mean(np.abs(centroids - centres)))

    assert correlation > TRACKS, (
        f"the density does not follow the moving well through time (corr {correlation:+.4f}, "
        f"mean|centroid - well| {lag:.4f}). The FP sweep steps FORWARD, so it must read the value "
        f"function at t_n, not at T - t_n; a mirrored read gives corr -0.01 to -0.62 here. "
        f"(If the correlation is near zero rather than negative, check the fixture drives at all "
        f"before concluding the time level is wrong.)"
    )
    assert lag < MAX_LAG, (
        f"the density correlates with the well but never gets near it (mean|centroid - well| "
        f"{lag:.4f}, corr {correlation:+.4f}): the fixture is under-driven, so the assertion above "
        f"is passing on a drift that follows the well only in the loosest sense"
    )

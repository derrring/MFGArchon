"""An absorbing particle wall returns the density of the survivors (Issue #2519).

DIRICHLET on `FPParticleSolver` absorbs particles, but every slice's reconstruction integrates to
about 1 whatever particle count built it, so the returned density kept the mass of no absorption:
1-D, 72% of particles absorbed, mass 1.03; 2-D, 92% absorbed, mass 1.07 (at `93c490ba`). That
slice is what a coupled solve hands the HJB. `_to_caller_mass` now multiplies each slice by N_t/N_0.

Oracle: the absorbing heat mode on [0, 1]^d with zero drift,

    m(t, x) = m0(x) exp(-d D pi^2 t),   m0 = prod sin(pi x_i), normalised to mass 1,

so the exact mass at time t is exp(-d D pi^2 t).

WHAT IS NOT ASSERTED, AND WHY. Euler-Maruyama tests for absorption only at the end of each step,
so a path that leaves and returns within a step survives. The returned mass is therefore biased
HIGH by an amount that falls with dt -- measured in 1-D at t = T, 4000 particles: +0.079 / +0.048 /
+0.011 at Nt = 20 / 80 / 320. The mass assertion carries a tolerance for that bias and a separate
test checks that it falls with dt. The density's wall values are not asserted at all: the KDE's
boundary bias dominates there, so the convergence test reads the interior half of the grid only.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc

SIGMA = 0.4
T = 1.0
D = SIGMA**2 / 2


def _problem(dim: int, n: int, nt: int):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=dirichlet_bc(dimension=dim, value=0.0)
    )
    X = np.meshgrid(*grid.coordinates, indexing="ij")
    m0 = np.prod([np.sin(np.pi * x) for x in X], axis=0)
    m0 = m0 / grid.integrate(m0)
    comps = MFGComponents(
        m_initial=lambda *z: np.ones_like(np.asarray(z[0], dtype=float)),
        u_terminal=lambda *z: 0.0,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )
    return grid, m0, MFGProblem(geometry=grid, T=T, Nt=nt, volatility=SIGMA, components=comps)


def _solve(dim: int, n: int, nt: int, particles: int, route: str, seed: int = 0):
    grid, m0, problem = _problem(dim, n, nt)
    solver = FPParticleSolver(problem, num_particles=particles, seed=seed)
    if route == "grid":
        M = solver.solve_fp_system(m0, np.zeros((nt + 1, *m0.shape)))
    else:
        M = solver.solve_fp_system(m0, drift_field=lambda t, x, m: np.zeros_like(np.asarray(x, dtype=float)))
    return grid, m0, np.asarray(M)


def _exact_mass(dim: int, t: float) -> float:
    return float(np.exp(-dim * D * np.pi**2 * t))


@pytest.mark.parametrize("route", ["grid", "callable"])
@pytest.mark.parametrize(("dim", "n"), [(1, 21), (2, 15)], ids=["1d", "2d"])
def test_the_returned_mass_is_the_surviving_mass(dim: int, n: int, route: str):
    """Before #2519 every case below returned mass ~1.03-1.08 against an exact 0.21-0.67."""
    nt = 80
    grid, _, M = _solve(dim, n, nt, 16000, route)
    for k in (nt // 2, nt):
        mass, exact = float(grid.integrate(M[k])), _exact_mass(dim, k * T / nt)
        # Measured ratios at this resolution: 1.05-1.10 (1-D), 1.11-1.20 (2-D), all from the step-end bias.
        assert 1.0 <= mass / exact < 1.25, f"{dim}-D {route}, t = {k * T / nt}: mass {mass:.4f}, exact {exact:.4f}"


def test_the_mass_excess_falls_with_dt():
    errors = []
    for nt in (20, 320):
        grid, _, M = _solve(1, 21, nt, 4000, "grid")
        errors.append(float(grid.integrate(M[nt])) - _exact_mass(1, T))
    assert errors[1] < 0.5 * errors[0], f"mass excess at t = T: Nt = 20 -> {errors[0]:.4f}, Nt = 320 -> {errors[1]:.4f}"


@pytest.mark.slow
def test_the_interior_density_converges_in_particles_and_grid():
    """Interior L1 at t = T/2, refining N, h and dt together. Measured 1.98e-2 -> 9.56e-3."""
    errors = []
    for n, nt, particles in ((21, 80, 16000), (41, 320, 64000)):
        _, m0, M = _solve(1, n, nt, particles, "grid")
        k = nt // 2
        exact = m0 * _exact_mass(1, k * T / nt)
        inner = slice(n // 4, n - n // 4)
        errors.append(float(np.abs(M[k] - exact)[inner].mean()))
    assert errors[1] < 0.75 * errors[0], f"interior L1 {errors}"


def test_a_nonzero_dirichlet_value_is_refused():
    """A particle wall absorbs: m = 0. `dirichlet_bc(value=0.7)` solved bit-identically to 0.0."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=dirichlet_bc(dimension=1, value=0.7)
    )
    comps = MFGComponents(
        m_initial=lambda z: np.ones_like(np.asarray(z, dtype=float)),
        u_terminal=lambda z: 0.0,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )
    problem = MFGProblem(geometry=grid, T=T, Nt=4, volatility=SIGMA, components=comps)
    with pytest.raises(NotImplementedError, match="#2519"):
        FPParticleSolver(problem, num_particles=100)

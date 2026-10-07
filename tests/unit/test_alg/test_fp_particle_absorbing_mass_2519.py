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
HIGH by an amount that falls with dt -- measured in 1-D at t = T, 4000 particles, seed 0: +0.079 / +0.048 /
+0.011 at Nt = 20 / 80 / 320. Over seeds 0-7 the Nt = 320 / Nt = 20 ratio is 0.13-0.35, median 0.26,
which is the sqrt(dt) scaling 1/sqrt(16) = 0.25. The mass assertion carries a tolerance for that bias and a separate
test checks that it falls with dt. The density's wall values are not asserted at all: the KDE's
boundary bias dominates there, so the convergence test reads the interior half of the grid only.
"""

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
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
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=SIGMA
        ),
        domain=grid,
        conditions=Conditions(u_terminal=lambda *z: 0.0, m_initial=lambda *z: 1.0, T=T),
        Nt=nt,
    )
    return grid, m0, problem


def _solve(dim: int, n: int, nt: int, particles: int, route: str, seed: int = 0):
    grid, m0, problem = _problem(dim, n, nt)
    solver = FPParticleSolver(problem, num_particles=particles, seed=seed)
    if route == "grid":
        M = solver.solve_fp_system(m0, np.zeros((nt + 1, *m0.shape)))
    else:
        M = solver.solve_fp_system(m0, drift_field=lambda t, x, m: np.zeros_like(np.asarray(x, dtype=float)))
    return grid, m0, np.asarray(M), solver


def _exact_mass(dim: int, t: float) -> float:
    return float(np.exp(-dim * D * np.pi**2 * t))


@pytest.mark.parametrize("route", ["grid", "callable"])
@pytest.mark.parametrize(("dim", "n"), [(1, 21), (2, 15)], ids=["1d", "2d"])
def test_the_returned_mass_is_the_surviving_mass(dim: int, n: int, route: str):
    """At `e506865e`, before #2519, these cases returned mass 0.975-0.996 at t = T/2 against an exact
    0.454-0.674: the reconstruction kept the mass of no absorption."""
    nt = 80
    grid, _, M, _ = _solve(dim, n, nt, 16000, route)
    for k in (nt // 2, nt):
        mass, exact = float(grid.integrate(M[k])), _exact_mass(dim, k * T / nt)
        # Measured over seeds 0-7 at this resolution: 1.048-1.115 (1-D), 1.094-1.218 (2-D), all from the
        # step-end bias. The 2-D margin to 1.25 is 0.032 at the worst seed.
        assert 1.0 <= mass / exact < 1.25, f"{dim}-D {route}, t = {k * T / nt}: mass {mass:.4f}, exact {exact:.4f}"


def test_the_mass_excess_falls_with_dt():
    errors = []
    for nt in (20, 320):
        grid, _, M, _ = _solve(1, 21, nt, 4000, "grid")
        errors.append(float(grid.integrate(M[nt])) - _exact_mass(1, T))
    assert errors[1] < 0.5 * errors[0], f"mass excess at t = T: Nt = 20 -> {errors[0]:.4f}, Nt = 320 -> {errors[1]:.4f}"


@pytest.mark.slow
def test_the_interior_density_converges_in_particles_and_grid():
    """Interior L1 at t = T/2, refining N, h and dt together. Measured 1.98e-2 -> 9.56e-3."""
    errors = []
    for n, nt, particles in ((21, 80, 16000), (41, 320, 64000)):
        _, m0, M, _ = _solve(1, n, nt, particles, "grid")
        k = nt // 2
        exact = m0 * _exact_mass(1, k * T / nt)
        inner = slice(n // 4, n - n // 4)
        errors.append(float(np.abs(M[k] - exact)[inner].mean()))
    assert errors[1] < 0.75 * errors[0], f"interior L1 {errors}"


def _refusal_problem():
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=dirichlet_bc(dimension=1, value=0.7)
    )
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=SIGMA
        ),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=T),
        Nt=4,
    )


def test_an_explicit_nonzero_dirichlet_value_is_refused():
    """A BC passed to the FP solver is its own: DIRICHLET(g) is a prescribed density m = g, which an
    absorbing particle wall cannot impose. `dirichlet_bc(value=0.7)` solved bit-identically to 0.0."""
    with pytest.raises(NotImplementedError, match="#2519"):
        FPParticleSolver(
            _refusal_problem(), num_particles=100, boundary_conditions=dirichlet_bc(dimension=1, value=0.7)
        )


def test_a_shared_nonzero_dirichlet_value_is_an_exit():
    """On the shared problem/geometry BC, DIRICHLET(g) is an exit: u = g for the HJB, m = 0 here
    (`fp_view_of_shared_bc`, #2512 convention row 5). So it constructs, and absorbs."""
    solver = FPParticleSolver(_refusal_problem(), num_particles=2000, seed=0)
    assert all(seg.value == 0.0 for seg in solver.boundary_conditions.segments)
    solver.solve_fp_system(np.ones(11), np.zeros((5, 11)))
    assert solver._count_alive(solver.M_particles_trajectory[-1]) < 2000


@pytest.mark.parametrize("route", ["grid", "callable"])
@pytest.mark.parametrize(("dim", "n"), [(1, 21), (2, 15)], ids=["1d", "2d"])
def test_every_slice_carries_the_survivors_of_this_solves_first_slice(dim: int, n: int, route: str):
    """N_0 is the count behind the solve's FIRST slice; take it a step late and every slice is off.

    The mass oracle above runs at a fine dt, where one step absorbs too little for a late N_0 to show.
    At Nt = 4 it shows in 2-D: with N_0 taken one step late, slice 1 read mass 0.960 (grid) and 1.000
    (callable) against a survivors' share of 0.834. Each calibration line is caught by a 2-D case. The
    1-D grid route never reaches the nD t = 0 line (it runs `_solve_fp_system_cpu`), and the 1-D callable
    deviation under that mutation, 0.084-0.097 over seeds 0-7, sits just inside the 0.1 tolerance.
    """
    nt = 4
    grid, m0, M, solver = _solve(dim, n, nt, 8000, route)
    mass0 = float(grid.integrate(m0))
    alive = [solver._count_alive(p) for p in solver.M_particles_trajectory]
    for k in range(1, nt + 1):
        expected = mass0 * alive[k] / alive[0]
        assert abs(float(grid.integrate(M[k])) - expected) < 0.1 * mass0, (
            f"{dim}-D {route}, slice {k}: mass {float(grid.integrate(M[k])):.3f}, survivors' share {expected:.3f}"
        )


def test_a_coupled_exit_cost_reaches_the_hjb_and_the_fp_absorbs():
    """One shared dirichlet_bc(value=0.5) on a coupled HJB-FDM + particle-FP solve: the exit cost reaches
    the HJB (u = g at both walls on every slice) and the FP absorbs (mass non-increasing, and well below
    its start). The survivors' share itself is pinned on the uncoupled solves above: the coupling damps
    M across Picard iterations, so the returned M is not the last FP solve's slice.

    The fixture takes u_T = g, so every slice holds g at the wall exactly; with u_T != g the iterator's
    damping of the initial guess leaves the wall short of g after three iterations (pre-existing).
    M is not asserted to vanish at the wall: the kernel reconstruction's boundary bias puts density there
    even when no particle is (wall/centre about 0.7 at T on this fixture), so the absorption is read off
    the mass instead.
    """
    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
    from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

    g = 0.5
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=dirichlet_bc(dimension=1, value=g)
    )
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m),
            volatility=SIGMA,
        ),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: g, m_initial=lambda x: 1.0, T=T),
        Nt=10,
    )
    fp = FPParticleSolver(problem, num_particles=4000, seed=0)
    result = FixedPointIterator(problem, hjb_solver=HJBFDMSolver(problem), fp_solver=fp).solve(
        max_iterations=3, tolerance=1e-3
    )
    U, M = np.asarray(result.U), np.asarray(result.M)
    np.testing.assert_array_equal(U[:, 0], g)
    np.testing.assert_array_equal(U[:, -1], g)
    mass = np.array([float(grid.integrate(M[k])) for k in range(M.shape[0])])
    assert np.all(np.diff(mass) <= 1e-12), f"mass rose: {mass}"
    assert mass[-1] < 0.5 * mass[0], f"the exit did not absorb: mass {mass[0]:.3f} -> {mass[-1]:.3f}"


@pytest.mark.parametrize("value", [np.zeros(2), np.array([0.5, 0.5])], ids=["zero_array", "nonzero_array"])
def test_an_array_valued_shared_dirichlet_is_translated_without_crashing(value):
    """An all-zero array is a legitimate g = 0, and a nonzero one is an exit cost; the translator decides
    "carries a value" with the same owner as the capability gate (`_describe_bc_value`)."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=dirichlet_bc(dimension=1, value=value)
    )
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=SIGMA
        ),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=T),
        Nt=4,
    )
    solver = FPParticleSolver(problem, num_particles=100)
    if not value.any():
        assert solver.boundary_conditions is grid.get_boundary_conditions(), "nothing to translate"
    else:
        assert all(np.all(np.asarray(seg.value) == 0) for seg in solver.boundary_conditions.segments)

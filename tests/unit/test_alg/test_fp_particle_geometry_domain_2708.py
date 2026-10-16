"""The particle solver keeps its particles in the problem's implicit geometry, not only in an `implicit_domain=` (#2708).

`FPParticleSolver._obstacle_domain` is the one owner of the domain the particles must end in. Two channels declare it:
the problem's geometry, when it is an `ImplicitDomain`, and the `implicit_domain=` keyword. Before, only the keyword
was read: on `DifferenceDomain(box, disc)` as the geometry a solve recorded up to 335 of 500 particles inside the disc
(i2688/geomprobe/probe.py, at 0b5481fd), and on a disc geometry up to 499 of 500 outside it (#2708, the audit
session's probe). Both channels naming one object is accepted; two objects are refused, and the refusal's advice runs.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.implicit import DifferenceDomain, Hyperrectangle, Hypersphere

_N = 500


def _box():
    return Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))


def _obstacle():
    return DifferenceDomain(_box(), Hypersphere(center=[0.5, 0.5], radius=0.2))


def _problem(geometry):
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.05
    )
    conditions = Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.5)
    return MFGProblem(model=model, domain=geometry, conditions=conditions, Nt=20)


def _solve(solver, start, target):
    """Callable drift toward ``target``; the recorded particles at every time, stacked."""
    solver.solve_fp_system(
        initial_particles=start,
        drift_field=lambda t, x, m: 3.0 * (np.asarray(target) - np.asarray(x)),
        drift_needs_density=False,
        show_progress=False,
    )
    return np.concatenate([np.asarray(P, dtype=float).reshape(-1, 2) for P in solver.M_particles_trajectory])


def _solver(geometry, **kwargs):
    return FPParticleSolver(
        _problem(geometry),
        num_particles=_N,
        seed=1,
        density_mode="query_only",
        boundary_conditions=no_flux_bc(2),
        **kwargs,
    )


def _corner_start():
    return np.random.default_rng(1).uniform(0.0, 0.25, (_N, 2))


@pytest.mark.parametrize(
    "case",
    ["the geometry's obstacle", "the keyword on a grid", "both channels, one object", "neither channel"],
)
def test_each_channel_keeps_the_particles_in_its_domain(case):
    """The product of the two channels, each run through a solve toward the disc's centre: the particles stay in the
    domain the channels declare, and the placement acts where there is an obstacle to reach."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[21, 21], boundary_conditions=no_flux_bc(2))
    geometry_obstacle = _obstacle()
    if case == "the geometry's obstacle":
        solver, domain = _solver(geometry_obstacle), geometry_obstacle
    elif case == "the keyword on a grid":
        solver, domain = _solver(grid, implicit_domain=geometry_obstacle), geometry_obstacle
    elif case == "both channels, one object":
        solver, domain = _solver(geometry_obstacle, implicit_domain=geometry_obstacle), geometry_obstacle
    else:
        solver, domain = _solver(grid), None
    recorded = _solve(solver, _corner_start(), [0.5, 0.5])
    placed = sum(sum(step.values()) for step in solver.obstacle_placement_history)
    if domain is None:
        assert placed == 0
        assert (np.linalg.norm(recorded - 0.5, axis=1) < 0.2).any(), "no obstacle: the drift carries particles in"
        return
    assert domain.contains(recorded).all()
    assert placed > 0


def test_two_domain_objects_are_refused_and_the_geometry_alone_is_the_advice():
    """Two equal domains built twice are two objects, and which one bounds the particles is not this solver's choice.
    The refusal says to drop implicit_domain=, and so the solve runs on the geometry's domain."""
    geometry = _obstacle()
    with pytest.raises(ValueError, match=r"implicit_domain= names a different object.*drop implicit_domain="):
        _solver(geometry, implicit_domain=_obstacle())
    recorded = _solve(_solver(geometry), _corner_start(), [0.5, 0.5])
    assert geometry.contains(recorded).all()


def test_a_curved_outer_boundary_is_a_boundary_too():
    """A disc geometry with no obstacle: the drift heads for (1, 1), outside the disc. Only the keyword was read, so
    up to 499 of 500 particles were recorded outside the disc; the geometry's own boundary now keeps them in."""
    disc = Hypersphere(center=[0.5, 0.5], radius=0.5)
    radius = 0.1 * np.sqrt(np.random.default_rng(1).uniform(size=_N))
    angle = np.random.default_rng(2).uniform(0.0, 2.0 * np.pi, _N)
    start = np.c_[0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)]
    recorded = _solve(_solver(disc), start, [1.0, 1.0])
    assert disc.contains(recorded).all()


@pytest.mark.parametrize(
    "case", ["a box geometry", "a box geometry beside an obstacle keyword", "a keyword box short of the grid"]
)
def test_a_box_on_the_grid_reaches_no_placement(case):
    """A domain that is exactly a `Hyperrectangle` on the grid's box carries nothing the boundary condition does not
    place, so no domain reaches the placement (the periodic box geometry is the torus of
    `test_fp_particle_gradient_bc_1255`, which solves). Beside it, an obstacle keyword declares the domain. A box
    that is not the grid's still reaches the placement, and the box refusal names it (#2706)."""
    if case == "a box geometry":
        assert _solver(_box())._implicit_domain is None
        return
    if case == "a box geometry beside an obstacle keyword":
        obstacle = _obstacle()
        solver = _solver(_box(), implicit_domain=obstacle)
        assert solver._implicit_domain is obstacle
        recorded = _solve(solver, _corner_start(), [0.5, 0.5])
        assert obstacle.contains(recorded).all()
        return
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[21, 21], boundary_conditions=no_flux_bc(2))
    short = Hyperrectangle(np.array([[0.0, 0.8], [0.0, 1.0]]))
    with pytest.raises(ValueError, match=r"is not the grid's box"):
        _solve(_solver(grid, implicit_domain=short), _corner_start() * 0.5, [0.5, 0.5])

"""#2490 B1d: what a particle does at the boundary has one owner, `particle_action_at`, and it reads the owner's corner
rule and ``default_bc``."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import needs_segment_aware_bc
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc
from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator


@pytest.mark.parametrize(
    ("x_max", "y_max", "fate"),
    [
        ((BCType.REFLECTING, 0), (BCType.DIRICHLET, 0), "absorbed"),
        ((BCType.DIRICHLET, 0), (BCType.REFLECTING, 1), "reflected"),
        ((BCType.DIRICHLET, 0), (BCType.REFLECTING, 0), "absorbed"),
    ],
    ids=[
        "equal priority: the exit on y, essential before natural",
        "the wall's higher priority",
        "equal priority: the exit on x, essential before natural",
    ],
)
def test_a_particle_beyond_a_corner_takes_the_corners_segment(x_max, y_max, fate):
    """A particle beyond the corner (1, 1) reads the corner, where ``corner_strategy="priority"`` orders the two faces'
    segments by priority, then DIRICHLET before a natural condition (`BoundaryConditions._corner_key`). The applicator
    named the first face it found the particle on, x before y, and asked that face alone, so a reflecting x_max
    swallowed an absorbing y_max (#1561 item 1), and an absorbing x_max a reflecting y_max of higher priority. The
    first and third rows put the exit on either face, so neither "the first face wins" nor "the last" passes both."""
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="x_max", bc_type=x_max[0], value=0.0, boundary="x_max", priority=x_max[1]),
            BCSegment(name="y_max", bc_type=y_max[0], value=0.0, boundary="y_max", priority=y_max[1]),
        ],
        dimension=2,
        default_bc=BCType.NO_FLUX,
        corner_strategy="priority",
    )
    remaining, absorbed, _ = ParticleApplicator().apply(np.array([[1.05, 1.05]]), bc, [(0.0, 1.0), (0.0, 1.0)])
    if fate == "absorbed":
        assert absorbed.tolist() == [True], remaining
        return
    assert absorbed.tolist() == [False]
    np.testing.assert_allclose(remaining, [[0.95, 0.95]], rtol=0, atol=1e-12)


def test_walls_that_default_bc_makes_absorbing_absorb():
    """A reflecting x_min and ``default_bc=DIRICHLET``: x_max absorbs. The solver's routing read the segments alone, so
    one reflecting segment sent the solve down the reflecting fast path, and x_max reflected every particle. It reads
    `BoundaryConditions.uniform_segment`, which counts ``default_bc``. 200 particles start in [0.8, 0.95] and drift
    right at speed 3 for t = 0.2, so their mean position ends past x = 1."""
    bc = BoundaryConditions(
        segments=[BCSegment(name="wall", bc_type=BCType.REFLECTING, value=0.0, boundary="x_min")],
        dimension=1,
        default_bc=BCType.DIRICHLET,
    )
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.05
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=5,
    )
    solver = FPParticleSolver(problem, num_particles=200, seed=0, boundary_conditions=bc)
    start = np.random.default_rng(0).uniform(0.8, 0.95, (200, 1))
    solver.solve_fp_system(
        initial_particles=start,
        drift_field=lambda t, x, m: np.full(np.shape(x), 3.0),
        drift_needs_density=False,
        show_progress=False,
    )
    final = np.asarray(solver.M_particles_trajectory[-1], dtype=float).reshape(-1)
    assert np.count_nonzero(np.isfinite(final)) < 100, f"{np.count_nonzero(np.isfinite(final))} of 200 remain"


def _segment(name, bc_type, **placement):
    return BCSegment(name=name, bc_type=bc_type, value=0.0, **placement)


def _drift_into_the_corner(bc, n=200):
    """``n`` particles from [0.8, 0.95]^2 drifting at (3, 3) for t = 0.2 on the unit square: how many remain."""
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.05
        ),
        domain=TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
        ),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=5,
    )
    solver = FPParticleSolver(problem, num_particles=n, seed=0, boundary_conditions=bc)
    solver.solve_fp_system(
        initial_particles=np.random.default_rng(0).uniform(0.8, 0.95, (n, 2)),
        drift_field=lambda t, x, m: np.full(np.shape(x), 3.0),
        drift_needs_density=False,
        show_progress=False,
    )
    final = np.asarray(solver.M_particles_trajectory[-1], dtype=float).reshape(-1, 2)
    return int(np.count_nonzero(np.all(np.isfinite(final), axis=1)))


def test_a_placed_segment_is_read_on_the_wall_on_the_solvers_grid():
    """- ``region={"x": (1, 1), "y": (0.4, 0.6)}`` is a piece of the wall x = 1. A particle beyond it, at (1.05, 0.5),
      is read at its clamp (1, 0.5), on the wall where the region is placed, and absorbed. The applicator read the
      region at the particle, off the wall, and reflected it.
    - A BC handed to the solver is bound to its grid, as the grid binds its own, so an ``sdf_region`` exit is read on
      it. Read at a corner without a grid, it raised. 200 particles drift into the corner (1, 1), away from the exit
      on x = 1 around y = 0.5, and all 200 remain."""
    bc = BoundaryConditions(
        segments=[_segment("exit", BCType.DIRICHLET, region={"x": (1.0, 1.0), "y": (0.4, 0.6)})],
        dimension=2,
        default_bc=BCType.NO_FLUX,
    )
    _, absorbed, _ = ParticleApplicator().apply(np.array([[1.05, 0.5]]), bc, [(0.0, 1.0), (0.0, 1.0)])
    assert absorbed.tolist() == [True]

    def disk(q):
        return float(np.hypot(q[0] - 1.0, q[1] - 0.5) - 0.1)

    exit_on_x = BoundaryConditions(
        segments=[
            _segment("exit", BCType.DIRICHLET, sdf_region=disk, priority=1),
            _segment("walls", BCType.NO_FLUX, boundary="all"),
        ],
        dimension=2,
    )
    assert _drift_into_the_corner(exit_on_x) == 200


@pytest.mark.parametrize(
    "segments",
    [
        [("wall", BCType.REFLECTING, "x_min")],
        [(face, BCType.NO_FLUX, face) for face in ("x_min", "x_max", "y_min", "y_max")],
    ],
    ids=["a reflecting segment beside a no-flux default", "four no-flux faces beside a no-flux default"],
)
def test_the_fast_path_is_one_particle_action_on_every_wall(segments):
    """Every wall reflects, so the reflecting fast path serves: the routing reads `BoundaryConditions.uniform_segment_under`
    the particle's sameness. Read by condition, the first row left it -- REFLECTING is not NO_FLUX -- and read by
    segment, the second did, four segments being "mixed"."""
    bc = BoundaryConditions(
        segments=[_segment(name, bc_type, boundary=face) for name, bc_type, face in segments],
        dimension=2,
        default_bc=BCType.NO_FLUX,
    )
    assert needs_segment_aware_bc(bc) is False


@pytest.mark.parametrize("x_first", [True, False], ids=["x declared first", "y declared first"])
def test_a_particle_beyond_a_periodic_face_and_a_wall_wraps_on_one_and_reflects_on_the_other(x_first):
    """Periodic in x and no-flux in y. A periodic axis is identified, not walled: the applicator wraps it first and
    reads the walls alone, so (1.05, 1.05) goes to (0.05, 0.95), in either declaration order. It applied one action to
    the whole particle: wrapped in both, or, read at the corner, whichever face was declared first."""
    xs = [_segment("xl", BCType.PERIODIC, boundary="x_min"), _segment("xr", BCType.PERIODIC, boundary="x_max")]
    ys = [_segment("yl", BCType.NO_FLUX, boundary="y_min"), _segment("yr", BCType.NO_FLUX, boundary="y_max")]
    bc = BoundaryConditions(segments=xs + ys if x_first else ys + xs, dimension=2)
    remaining, absorbed, _ = ParticleApplicator().apply(np.array([[1.05, 1.05]]), bc, [(0.0, 1.0), (0.0, 1.0)])
    assert absorbed.tolist() == [False]
    np.testing.assert_allclose(remaining, [[0.05, 0.95]], rtol=0, atol=1e-12)


def test_what_no_particle_action_describes_is_refused_at_construction():
    """- ``corner_strategy="average"`` with an exit on x_max and no-flux walls: no average of absorbing and reflecting
      says what a particle at the corner does. The solver refused it at the first step a particle reached the corner;
      it reads every vertex at construction. The remedy as the message gives it -- ``"priority"``, the exit's priority
      above the walls' -- absorbs all 200 particles drifting into the corner (1, 1).
    - A box with an axis of no width: a particle there lies on both faces of that axis at once."""
    exit_on_x = [_segment("exit", BCType.DIRICHLET, boundary="x_max")]
    averaged = BoundaryConditions(segments=exit_on_x, dimension=2, default_bc=BCType.NO_FLUX, corner_strategy="average")
    with pytest.raises(
        ValueError,
        match=r"the corner shared by x_max, y_min, at \(1\.0, 0\.0\).*'exit' \(absorbing\), 'default' \(reflecting\)"
        r".*corner_strategy='priority' and give the segment that should govern the corner the higher priority",
    ):
        _drift_into_the_corner(averaged)
    remedied = BoundaryConditions(
        segments=[_segment("exit", BCType.DIRICHLET, boundary="x_max", priority=1)],
        dimension=2,
        default_bc=BCType.NO_FLUX,
        corner_strategy="priority",
    )
    assert _drift_into_the_corner(remedied) == 0

    flat = BoundaryConditions(segments=exit_on_x, dimension=2, default_bc=BCType.NO_FLUX)
    with pytest.raises(ValueError, match=r"has no interior along axis \[1\]"):
        ParticleApplicator().apply(np.array([[0.3, 0.5]]), flat, [(0.0, 1.0), (0.5, 0.5)])

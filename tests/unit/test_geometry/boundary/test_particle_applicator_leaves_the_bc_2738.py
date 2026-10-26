"""The particle applicator reads the BC it is handed and writes nothing into it (Issue #2738).

`ParticleApplicator.apply` set ``bc.domain_bounds`` at its first step when the BC had none, and so did
``apply_with_flux_limits``, which no solver calls. On the shared route, when the solver's view translates nothing, that
BC is the problem's own; so on the legacy route, where the grid holds a bound copy of the components' BC, every later
``problem.get_boundary_conditions()`` refused the two as declaring the BC twice. Only the per-point path calls
``apply``; the fast path never wrote.

The applicator now writes nothing. A BC with no ``domain_bounds`` is queried through a copy that carries the step's box
in that slot, so the face a particle leaves through and the point a marked ``region_name`` reads past the wall come
from one box, as they did when the box was written, and the copy keeps the BC's grid binding (the last two tests).

The rows of the end-to-end test are the BCs the per-point path reaches. #2738's exit is per point at any routing. B1,
a REFLECTING segment placed by ``region_name``, and the absorbing ``default_bc`` with no segment are per point once
#2490's B1d routing counts placement and ``default_bc``. They raise under that routing without this fix. Pinned
here, the routing cannot bring the write back to them.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator

_N, _NT = 21, 10


def _exit_bc() -> BoundaryConditions:
    return BoundaryConditions(
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max")],
        dimension=1,
        default_bc=BCType.NO_FLUX,
    )


@pytest.mark.parametrize("method", ["apply", "apply_with_flux_limits"])
def test_the_applicator_writes_nothing_into_the_bc_it_reads(method):
    """The BC keeps ``domain_bounds=None``, and the step is the one a BC carrying the same box gives: a particle past
    x_max is absorbed, one past x_min is reflected."""
    bc = _exit_bc()
    boxed = _exit_bc()
    boxed.domain_bounds = np.array([[0.0, 1.0]])
    particles = np.array([[1.02], [-0.03], [0.5]])

    def step(b):
        applicator = ParticleApplicator()
        if method == "apply":
            return applicator.apply(particles.copy(), b, [(0.0, 1.0)])
        with pytest.warns(DeprecationWarning, match="apply_with_flux_limits is deprecated"):
            return applicator.apply_with_flux_limits(particles.copy(), b, [(0.0, 1.0)], {"exit": 10})

    got, expected = step(bc), step(boxed)
    assert bc.domain_bounds is None, f"{method} wrote domain_bounds into the BC it was handed: {bc.domain_bounds!r}"
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], [True, False, False])
    np.testing.assert_allclose(got[0][:, 0], [0.03, 0.5], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "bc",
    [
        _exit_bc(),
        BoundaryConditions(
            segments=[BCSegment(name="w", bc_type=BCType.REFLECTING, value=0.0, region_name="x_min")],
            dimension=1,
            default_bc=BCType.NO_FLUX,
        ),
        BoundaryConditions(segments=[], dimension=1, default_bc=BCType.DIRICHLET),
    ],
    ids=["#2738's exit", "REFLECTING placed by region_name, NO_FLUX default", "no segment, DIRICHLET default"],
)
# #2738 is a defect of the legacy route, which warns that it is deprecated; that warning is not this pin's subject.
@pytest.mark.filterwarnings("ignore:Specifying boundary conditions via MFGComponents:DeprecationWarning")
def test_a_particle_solve_leaves_the_problem_s_bc_readable(bc):
    """#2738's reproducer, on the legacy route: a particle solve, then the problem's BC read again."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The components= form", category=PendingDeprecationWarning)
        problem = MFGProblem(
            spatial_bounds=[(0.0, 1.0)],
            spatial_discretization=[_N - 1],
            components=MFGComponents(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
                boundary_conditions=bc,
            ),
            T=0.5,
            Nt=_NT,
            volatility=0.4,
        )
    solver = FPParticleSolver(problem, num_particles=200, seed=0)
    problem.get_boundary_conditions()
    solver.solve_fp_system(1.0 + np.linspace(0.0, 1.0, _N), drift_field=np.zeros((_NT + 1, _N)), show_progress=False)
    problem.get_boundary_conditions()


def _box_sdf(x):
    return np.max(np.abs(np.asarray(x, dtype=float) - 0.5)) - 0.5


@pytest.mark.parametrize("route", ["the problem's BC", "a BC handed to the solver"])
@pytest.mark.parametrize("domain_sdf", [None, _box_sdf], ids=["a box BC", "a BC whose domain is its domain_sdf"])
def test_a_grid_marked_exit_is_read_on_the_grid_the_bc_is_bound_to(domain_sdf, route):
    """An exit placed by a ``region_name`` the grid marks is decided on the grid the BC is bound to: 2000 particles
    drifting onto x = 1 leave through the door, and the problem's BC is still unwritten after the solve. A point query
    that lost the grid refuses the door at the first particle reaching it (#2767 review 1).

    A particle past the wall is read at its nearest point of the step's box, which the copy carries. A BC whose domain
    is its ``domain_sdf`` has no box of its own (``_query_bounds``): a query that read none refused the door as outside
    the grid (#2767 review 2)."""
    from mfgarchon import Conditions, Model
    from mfgarchon.geometry import TensorProductGrid

    n = 21

    def door():
        return BoundaryConditions(
            segments=[BCSegment(name="door", bc_type=BCType.DIRICHLET, value=0.0, region_name="door", priority=1)],
            dimension=2,
            default_bc=BCType.NO_FLUX,
            domain_sdf=domain_sdf,
        )

    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[n] * 2, boundary_conditions=door())
    grid.mark_region("door", predicate=lambda x: x[:, 0] > 0.99)
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.2
        ),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.5),
        Nt=20,
    )
    x = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * 2, indexing="ij")[0]
    handed = {} if route == "the problem's BC" else {"boundary_conditions": door()}
    solver = FPParticleSolver(problem, num_particles=2000, seed=3, **handed)
    solver.solve_fp_system(np.ones((n, n)), np.broadcast_to(-2.0 * x, (21, n, n)).copy(), show_progress=False)
    assert solver.total_absorbed > 1000, f"{solver.total_absorbed} of 2000 left through the door"
    assert grid.get_boundary_conditions().domain_bounds is None, "the solve wrote domain_bounds into the grid's BC"


# The components route on a geometry that declares no BC is deprecated; that warning is not this pin's subject.
@pytest.mark.filterwarnings("ignore:Specifying boundary conditions via MFGComponents:DeprecationWarning")
@pytest.mark.parametrize("on_the_domain", [False, True], ids=["the door alone", "with a corner of it on the domain"])
def test_a_door_marked_off_the_domain_lets_no_particle_out(on_the_domain):
    """A BC with no ``domain_bounds`` of its own, bound to a grid that is not the problem's geometry: on the components
    route over a ``Hyperrectangle``, which declares no BC, the problem's BC is the one the caller bound to a grid on x in
    [0, 1.2], whose door, x > 1.05, lies outside the domain [0, 1]^2. 500 particles driven onto x = 1 are read at their
    nearest point of the step's box, which stands in for where they crossed it (#2733 will pass that point), and none
    leaves. A query that read the bound grid's box instead read a particle past x = 1.05 as in the door: 492 left (#2767
    review 3). A BC that carries its own ``domain_bounds`` is read on them, and is not this row's case (#2733).

    The door alone governs no part of the domain's faces, so the solver refuses it by name (#2490 B1e-2). With a corner
    of it, y > 0.9 at x > 0.95, on the face x = 1, far from where the particles cross, the door governs that corner and
    is admitted, and the part past x = 1.05 still lets no particle out."""
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.implicit import Hyperrectangle

    marking = TensorProductGrid(
        bounds=[(0.0, 1.2), (0.0, 1.0)],
        Nx_points=[13, 11],
        boundary_conditions=BoundaryConditions(
            segments=[BCSegment(name="door", bc_type=BCType.DIRICHLET, value=0.0, region_name="door", priority=1)],
            dimension=2,
            default_bc=BCType.NO_FLUX,
        ),
    )
    marking.mark_region(
        "door",
        predicate=lambda x: (
            ((x[:, 0] > 1.05) & (x[:, 1] > 0.3) & (x[:, 1] < 0.7))
            | (on_the_domain & (x[:, 0] > 0.95) & (x[:, 1] > 0.9))
        ),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The components= form", category=PendingDeprecationWarning)
        problem = MFGProblem(
            geometry=Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])),
            components=MFGComponents(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
                boundary_conditions=marking.get_boundary_conditions(),
            ),
            T=0.5,
            Nt=20,
            volatility=0.05,
        )
    if not on_the_domain:
        with pytest.raises(ValueError, match=r"FPParticleSolver: segment\(s\) 'door' hold none of"):
            FPParticleSolver(problem, num_particles=500, seed=1, density_mode="query_only")
        return
    solver = FPParticleSolver(problem, num_particles=500, seed=1, density_mode="query_only")
    start = np.column_stack(
        [np.random.default_rng(1).uniform(0.6, 0.9, 500), np.random.default_rng(2).uniform(0.35, 0.65, 500)]
    )
    solver.solve_fp_system(
        initial_particles=start,
        drift_field=lambda t, x, m: np.tile([3.0, 0.0], (len(x), 1)),
        drift_needs_density=False,
        show_progress=False,
    )
    assert solver.total_absorbed == 0, f"{solver.total_absorbed} of 500 left through a door outside the domain"
    assert marking.get_boundary_conditions().domain_bounds is None, "the solve wrote domain_bounds into the BC"

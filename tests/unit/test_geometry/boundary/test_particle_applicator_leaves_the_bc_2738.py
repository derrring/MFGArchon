"""The particle applicator reads the BC it is handed and writes nothing into it (Issue #2738).

`ParticleApplicator.apply` and `apply_with_flux_limits` set ``bc.domain_bounds`` at their first step when the BC had
none. On the shared route that BC is the problem's own, so on the legacy route, where the grid holds a bound copy of
the components' BC, every later ``problem.get_boundary_conditions()`` refused the two as declaring the BC twice.
Only the per-point path calls the applicator. The fast path never wrote.

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
        warnings.simplefilter("ignore", PendingDeprecationWarning)  # the components= form's own notice
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

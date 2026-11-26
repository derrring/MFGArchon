"""FP-FDM reads a Dirichlet value at the level it solves for, and at the wall node (#2535, #2512 row B2).

External oracle. With zero drift, m(t, x) = a + b (|x|^2 + 2 d D t), D = sigma^2 / 2, solves m_t = D lap m exactly. The
centred second difference is exact for a quadratic, and implicit Euler is exact for a solution linear in t. So
with the walls held at g(t, x) = m(t, x), FP-FDM reproduces m to roundoff. A wall read at t = 0 misses by 2 d D b t,
and the error spreads inward.

Measured at `76d8b1dc` (main):
- in 1-D with a per-face g(t): 9.000e-2 at T, which is exactly 2 d D b T;
- in 2-D with a uniform g(t, x): 2.18, since a callable value was read as 0.
At the head, both are below 2e-15.

The BC goes to the FP's own channel, `FPFDMSolver(..., boundary_conditions=bc)`. On the shared route,
`fp_view_of_shared_bc` makes a Dirichlet wall absorbing by design.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, Model
from mfgarchon.alg.numerical.coupling import NewtonMFGSolver
from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fp_solvers import fp_fdm_time_stepping
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.config.damping import DampingPolicy
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc

A, B, SIGMA, T, NT = 1.0, 0.5, 0.6, 0.5, 10
D = SIGMA**2 / 2


def _problem(d: int, n: int) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[n] * d, boundary_conditions=no_flux_bc(dimension=d))
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    return MFGProblem(geometry=grid, components=comps, T=T, Nt=NT, volatility=SIGMA)


@pytest.mark.parametrize("d", [1, 2])
def test_a_time_dependent_dirichlet_wall_is_read_at_each_level(d, monkeypatch):
    n = 21 if d == 1 else 9
    exact = lambda t, x: A + B * (np.sum(np.atleast_2d(x) ** 2, axis=-1) + 2 * d * D * t)  # noqa: E731
    if d == 1:  # per face, g(t): the mixed path through the accessor
        bc = BoundaryConditions(
            dimension=1,
            segments=[
                BCSegment(name="left", bc_type=BCType.DIRICHLET, value=lambda t: A + B * 2 * D * t, boundary="x_min"),
                BCSegment(
                    name="right", bc_type=BCType.DIRICHLET, value=lambda t: A + B * (1 + 2 * D * t), boundary="x_max"
                ),
            ],
        )
    else:  # uniform, g(t, x), read at the node
        bc = dirichlet_bc(dimension=2, value=lambda t, x: A + B * (x[0] ** 2 + x[1] ** 2 + 2 * d * D * t))
    # FP-FDM reads a Dirichlet node's value from the segment governing the node, through the value owner
    # `segment_value` (#2512 row B1); `get_bc_value_at_boundary`, which this recorded until then, is a wrapper of it.
    times = []
    owner = fp_fdm_time_stepping.segment_value

    def recording(segment, *, t=None, **kwargs):
        times.append(t)
        return owner(segment, t=t, **kwargs)

    monkeypatch.setattr(fp_fdm_time_stepping, "segment_value", recording)
    pts = np.stack(np.meshgrid(*[np.linspace(0.0, 1.0, n)] * d, indexing="ij"), axis=-1).reshape(-1, d)
    shape = (n,) * d
    M = np.asarray(
        FPFDMSolver(_problem(d, n), boundary_conditions=bc).solve_fp_system(
            exact(0.0, pts).reshape(shape), drift_field=np.zeros((NT + 1, *shape)), volatility=SIGMA
        )
    )
    levels = np.linspace(0.0, T, NT + 1)
    exact_levels = np.stack([exact(t, pts).reshape(shape) for t in levels])
    assert np.abs(M - exact_levels).max() < 1e-12
    # The wall nodes, asserted on their own (#2512 comment 6042778010, correction 4: the bar for a wall that
    # carries data).
    idx = np.indices(shape)
    on_wall = np.any((idx == 0) | (idx == n - 1), axis=0)
    assert np.abs(M[:, on_wall] - exact_levels[:, on_wall]).max() < 1e-12
    # Every level after the first is read at its own time, and only there; level 0 is the caller's density and reads
    # no value (#2984).
    step_times = {round(k * T / NT, 12) for k in range(1, NT + 1)}
    read = {round(t, 12) for t in times}
    assert None not in times
    assert read == step_times, sorted(read ^ step_times)


def test_a_no_flux_wall_beside_a_dirichlet_one_keeps_its_density():
    """A BC whose only segment is Dirichlet, on x_min, leaves x_max to its no-flux default. The 1-D overwrite, at
    t = 0 and after every step, wrote both walls, since every segment was Dirichlet, and x_max read the BC's default
    value: the no-flux wall was held at 0.0 from level 0 on, at `76d8b1dc` (#2535). The post-step overwrite is gone,
    as the implicit step's rows hold g and the explicit steps refuse a Dirichlet wall. Level 0 is the caller's density
    at both walls (#2984)."""
    bc = BoundaryConditions(
        dimension=1,
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=2.0, boundary="x_min")],
        default_bc=BCType.NO_FLUX,
    )
    n = 21
    M = np.asarray(
        FPFDMSolver(_problem(1, n), boundary_conditions=bc).solve_fp_system(
            np.ones(n), drift_field=np.zeros((NT + 1, n)), volatility=SIGMA
        )
    )
    assert M[0, 0] == 1.0
    assert np.all(M[1:, 0] == 2.0)
    # Mass flows in from the wall held at 2: the no-flux wall never drops below its initial density.
    assert np.all(M[:, -1] >= 1.0), M[:, -1]


@pytest.mark.filterwarnings("ignore:initial density mass:UserWarning")  # the planted walls give mass 1.825, reported
def test_level_0_is_the_callers_density_at_a_dirichlet_wall_on_both_routes_and_the_coupled_solve_converges():
    """Level 0 is the caller's m_initial at a Dirichlet wall, on a direct FP-FDM call and in a coupled solve, which then
    agree there (#2984). At `3f362150` the direct call held g = 0 at the walls of a 1-D BC whose segments are all
    Dirichlet, while the coupled solve's iterator wrote the caller's density over the FP's row. The problem's own BC,
    which FP-FDM reads as an absorbing exit (#2512). The oracle is the planted wall value, 7.0.

    The coupled solve's Picard residual reads the FP's returned row, level 0 included, before the iterator writes
    m_initial over it. At `3f362150` the wall gap held the M residual at 0.700 and the solve did not converge in 50
    iterations; with one level 0 it converges (in 22). This pins that the two level-0 rows agree; a residual that
    skipped level 0 would converge as well.

    The routes that keep FP-FDM's row and write nothing back hold the planted walls at level 0 as well:
    `MultiPopulationIterator` and `GraphMFGSolver`, which at `3f362150` held 0.875 after three sweeps, decaying toward the
    prescribed 0; `RegimeSwitchingIterator`, which held 0; and Newton's phase (no Picard warm-up, which writes m_initial
    back), which held about 2e-08, on a 5-node problem so that it runs in under a second.

    In 2-D a direct call on the FP's own all-Dirichlet BC holds the planted walls at level 0 too. Main held them
    there already; this pins the 2-D own all-Dirichlet layout."""

    def m_initial(x):
        a = np.asarray(x[0], dtype=float)
        return np.where(np.isclose(a, 0.0) | np.isclose(a, 1.0), 7.0, 1.0 + 0.5 * a)

    def planted_problem(n, nt):
        return MFGProblem(
            model=Model(hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)), volatility=0.2),
            domain=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=dirichlet_bc(value=0.0, dimension=1)
            ),
            conditions=Conditions(
                m_initial=m_initial, u_terminal=lambda x: np.sin(np.pi * np.asarray(x[0], dtype=float)), T=0.5
            ),
            Nt=nt,
        )

    def planted_row(n):
        row = 1.0 + 0.5 * np.linspace(0.0, 1.0, n)
        row[[0, -1]] = 7.0
        return row

    problem = planted_problem(11, 5)
    planted = planted_row(11)
    direct = np.asarray(
        FPFDMSolver(problem).solve_fp_system(planted, potential_field=np.zeros((6, 11)), show_progress=False)
    )
    result = problem.solve(max_iterations=50, verbose=False)
    assert np.array_equal(direct[0], planted)
    assert np.array_equal(np.asarray(result.M)[0], planted)
    assert result.converged
    # The exit lets the wall's content out in the first step: the later levels hold the prescribed 0 there.
    assert np.all(direct[1:, [0, -1]] == 0.0)
    multi = MultiPopulationIterator(
        MultiPopulationProblem(populations=[problem], population_names=["the one"]),
        [HJBFDMSolver(problem)],
        [FPFDMSolver(problem)],
        damping=DampingPolicy.constant(0.5),
    ).solve(max_iterations=3, tolerance=1e-6)
    assert np.array_equal(np.asarray(multi.M[0])[0], planted)
    graph = GraphMFGSolver(
        [problem],
        AdjacencyCoupling(np.zeros((1, 1)), alpha=0.05, beta=0.02),
        [HJBFDMSolver(problem)],
        [FPFDMSolver(problem)],
        max_iterations=3,
        tolerance=1e-6,
        damping=DampingPolicy.constant(0.5),
    ).solve()
    assert np.array_equal(np.asarray(graph.M[0])[0], planted)
    regimes = [planted_problem(11, 5), planted_problem(11, 5)]
    regime = RegimeSwitchingIterator(
        regimes,
        RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]])),
        [HJBFDMSolver(q) for q in regimes],
        [FPFDMSolver(q) for q in regimes],
        max_iterations=3,
        tolerance=1e-6,
        damping=DampingPolicy.constant(0.5),
    ).solve()
    assert all(np.array_equal(np.asarray(density)[0], planted) for density in regime.M)
    small = planted_problem(5, 2)
    newton_result = NewtonMFGSolver(
        small, HJBFDMSolver(small), FPFDMSolver(small), picard_warmup=0, newton_max_iterations=1
    ).solve(max_iterations=1, verbose=False)
    _, newton_m, _ = newton_result.U, newton_result.M, newton_result.metadata
    assert np.array_equal(np.asarray(newton_m)[0], planted_row(5))
    n = 9
    planted_2d = 1.0 + 0.5 * np.linspace(0.0, 1.0, n)[:, None] + 0.25 * np.linspace(0.0, 1.0, n)[None, :]
    on_wall = np.zeros((n, n), dtype=bool)
    on_wall[[0, -1], :] = on_wall[:, [0, -1]] = True
    planted_2d[on_wall] = 7.0
    direct_2d = FPFDMSolver(_problem(2, n), boundary_conditions=dirichlet_bc(value=0.0, dimension=2)).solve_fp_system(
        planted_2d, drift_field=np.zeros((NT + 1, n, n)), volatility=SIGMA
    )
    assert np.array_equal(np.asarray(direct_2d)[0], planted_2d)

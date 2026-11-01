"""A capacity-limited exit on FP-FDM (#2581, PR 2a): on the exit, m >= 0, J.n <= c and m (c - J.n) = 0, as an active set
within the step (user rulings 2026-10-09 and 2026-10-10; the approver's on the plan, 2026-10-31).

Each exit node starts the step absorbing; a node whose outflow over the step would exceed c |face| dt takes the FLUX(c)
row, the no-flux row with -c/(dx/2) on its right-hand side, and the step is solved again. A node's outflow is read off
that no-flux row, which also gives the solver's internal record of the mass each Dirichlet segment let out. It is offered for a standalone FP solve
only, and a coupled solve refuses it: paired with an HJB reading u = g at the exit it is not an equilibrium (user
rulings 2026-10-10 and 2026-10-31).

The oracles, without symmetry that would let a rival branch agree:
- the saturated branch, 1-D and 2-D (the manifest's cells ``fp_fdm/capped_exit_{1,2}d``): a manufactured solution with
  J.n = c and m > 0 at the exit and a nonzero drift into it, at the upwind step's first order; the record's rate is
  c |exit| at every step; the same solve with the exit absorbing misses by O(1);
- the release: a queue that drains frees the exit, and the switch time converges under refinement;
- the mass identity: the trapezoid mass plus the recorded outflow is the mass at level 0 plus what came in;
- a cap past every outflow (1e300; an infinite one is refused) is the absorbing exit, bit for bit, except in a 1-D BC
  whose segments are all DIRICHLET: there the capped wall keeps the caller's density at level 0, and the cumulative
  record is offset at every level >= 1 by that node's level-0 mass, m0 dx/2 (the approver's ruling, RULE 3, amended);
- the cap's measure is the half cell normal to the exit face, on whichever axis that face lies.
"""

from __future__ import annotations

import pytest

import numpy as np
from scipy import sparse

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.block_iterators import BlockGaussSeidelIterator, BlockJacobiIterator
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_fp_nd_full_system, solve_timestep_full_nd
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid, no_flux_bc
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

ALPHA, SIGMA, CAP, GAMMA, T, LY = 0.5, 0.8, 0.5, 0.3, 0.5, 0.5
D = SIGMA**2 / 2
K = ALPHA / D
BETA = -CAP * np.exp(-K) / (2 * D)
LEVEL_FACTOR = 1.25
#: (levels, errors at T recorded at this file's first commit, ratio band): the upwind step's first order
CELLS = {
    1: ((21, 41, 81), (4.1591e-02, 2.0145e-02, 9.9017e-03), (1.6, 2.4)),
    2: ((21, 41, 81), (4.1591e-02, 2.0145e-02, 9.9017e-03), (1.6, 2.4)),
}


def _exact(t: float, x: np.ndarray) -> np.ndarray:
    """``m = e^{kx} (1 + beta x^2 + gamma e^{-t} cos(pi x))``, k = alpha/D: J = alpha m - D m_x = -D e^{kx} f', zero at
    x = 0 and c at x = 1 by beta's choice, with m > 0 there."""
    return np.exp(K * x) * (1 + BETA * x**2 + GAMMA * np.exp(-t) * np.cos(np.pi * x))


def _source(t: float, points: np.ndarray) -> np.ndarray:
    """``S = m_t + J_x``, with J = -D e^{kx} f'."""
    x = np.asarray(points)[:, 0]
    g = GAMMA * np.exp(-t)
    f1 = 2 * BETA * x - g * np.pi * np.sin(np.pi * x)
    f2 = 2 * BETA - g * np.pi**2 * np.cos(np.pi * x)
    return np.exp(K * x) * (-g * np.cos(np.pi * x) - D * (K * f1 + f2))


def _bc(
    dim: int,
    cap: float | None,
    x_max: BCType = BCType.DIRICHLET,
    y_max: BCType = BCType.NO_FLUX,
    value=0.0,
    y_max_priority: int = 0,
):
    faces = ["x_min", "x_max"] + (["y_min", "y_max"] if dim == 2 else [])
    segments = []
    for face in faces:
        if face == "x_max":
            segments.append(BCSegment(name="exit", bc_type=x_max, value=value, boundary=face, flux_capacity=cap))
        elif face == "y_max" and y_max == BCType.DIRICHLET:
            segments.append(
                BCSegment(
                    name="second_exit",
                    bc_type=BCType.DIRICHLET,
                    value=0.0,
                    boundary=face,
                    priority=y_max_priority,
                )
            )
        else:
            segments.append(BCSegment(name=face, bc_type=BCType.NO_FLUX, boundary=face))
    return BoundaryConditions(dimension=dim, segments=segments)


def _problem(dim: int, n: int, nt: int, t_end: float, sigma: float = SIGMA) -> tuple[MFGProblem, TensorProductGrid]:
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, LY)][:dim], Nx_points=[n] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=sigma)
    # the problem's own density is a constant of unit mass on the grid's measure; each solve is handed its own
    conditions = Conditions(m_initial=lambda x, c=1.0 / (LY if dim == 2 else 1.0): c, u_terminal=lambda x: 0.0, T=t_end)
    return MFGProblem(model=model, domain=grid, conditions=conditions, Nt=nt), grid


def _saturated(dim: int, n: int, cap: float | None) -> tuple[float, np.ndarray]:
    nt = 2 * (n - 1)
    problem, grid = _problem(dim, n, nt, T)
    x = np.meshgrid(*[np.asarray(a) for a in grid.coordinates], indexing="ij")[0]
    solver = FPFDMSolver(problem, boundary_conditions=_bc(dim, cap))
    m = solver.solve_fp_system(
        _exact(0.0, x),
        potential_field=np.broadcast_to(-ALPHA * x, (nt + 1, *x.shape)).copy(),
        source_term=_source,
        show_progress=False,
    )
    rate = np.diff(solver._boundary_outflow["exit"]) / (T / nt)
    return float(np.abs(np.asarray(m)[-1] - _exact(T, x)).max()), rate


@pytest.mark.parametrize("dim", [1, 2])
def test_the_saturated_branch_converges_at_the_steps_order(dim: int):
    """J.n = c with m > 0 at the exit, so every exit node is capped at every step: the record's rate is c |exit| there
    (the exit's measure is LY in 2-D), and the absorbing rival misses by 3.1."""
    levels, recorded, (low, high) = CELLS[dim]
    errors = []
    for n in levels:
        error, rate = _saturated(dim, n, CAP)
        errors.append(error)
        assert np.allclose(rate, CAP * (LY if dim == 2 else 1.0), rtol=1e-12, atol=0.0), rate
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(low < r < high for r in ratios), (errors, ratios)
    for got, want in zip(errors, recorded, strict=True):
        assert want / LEVEL_FACTOR < got < want * LEVEL_FACTOR, errors
    absorbing, _ = _saturated(dim, levels[1], None)
    assert absorbing > 100 * errors[1], absorbing


def _release(n: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    nt, t_end, cap = 4 * (n - 1), 4.0, 0.3
    problem, grid = _problem(1, n, nt, t_end, sigma=0.4)
    x = np.asarray(grid.coordinates[0])
    weights = grid.measure_weights()
    m0 = np.exp(-((x - 0.5) ** 2) / 0.02)
    m0 = m0 / (m0 @ weights)
    solver = FPFDMSolver(problem, boundary_conditions=_bc(1, cap))
    m = np.asarray(
        solver.solve_fp_system(m0, potential_field=np.broadcast_to(-x, (nt + 1, n)).copy(), show_progress=False)
    )
    out = solver._boundary_outflow["exit"]
    capped = np.diff(out) / (t_end / nt) >= cap * (1 - 1e-12)
    first = int(np.argmax(capped))
    released = first + int(np.argmax(~capped[first:]))
    assert capped[first], capped
    assert not capped[released:].any(), capped
    return (released + 1) * t_end / nt, m, out, weights


def test_the_cap_releases_as_its_queue_drains_and_the_switch_time_converges():
    """A drift of 1 carries a bump into an exit capped at 0.3: the outflow sits at the cap while the queue lasts, then the
    exit frees for good. Measured: the release at t* = 3.275, 3.3, 3.3125 for n = 41, 81, 161 (and 3.3156 at 321)."""
    times = []
    for n in (41, 81, 161):
        t_star, m, out, weights = _release(n)
        times.append(t_star)
        # The record closes the mass to round-off: the n node values are summed and recorded at each of the steps, so
        # the bound is steps * n * eps * mass (mass 1). A fixed 1e-12 failed on Linux at n = 161, 1.07e-12 (#2858).
        bound = (len(out) - 1) * len(weights) * np.finfo(float).eps
        closure = float(np.max(np.abs(m @ weights + out - 1.0)))
        assert closure < bound, f"n = {n}: closure {closure:.3e}, over steps * n * eps * mass = {bound:.3e}"
        assert m.min() >= 0.0
    assert abs(times[2] - times[1]) < 0.75 * abs(times[1] - times[0]), times


def test_the_record_closes_the_mass_beside_a_free_exit_and_a_flux_wall():
    """The trapezoid mass plus the record is the mass at level 0 plus what came in. In 1-D a capped exit beside a free
    one, both in the record, under a field volatility, whose Itô diffusion the wall rows carry: a BC of DIRICHLET
    segments alone holds its walls at g(0) at level 0 (#2535), and the capped wall keeps the caller's density there,
    which queues. In 2-D a capped exit across from a FLUX inflow into a nearly empty domain, whose q gives what came in:
    the inflow alone takes the exit to its cap, so the switch must read the flux's part of the step, and the exit never
    lets out more than c |exit| dt in a step."""
    problem, grid = _problem(1, 21, 20, 1.0)
    x = np.asarray(grid.coordinates[0])
    m0 = np.exp(-((x - 0.7) ** 2) / 0.05) + 0.5
    bc = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="free", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min"),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", flux_capacity=0.3),
        ],
    )
    solver = FPFDMSolver(problem, boundary_conditions=bc)
    m = np.asarray(
        solver.solve_fp_system(
            m0,
            potential_field=np.broadcast_to(-x, (21, 21)).copy(),
            volatility=0.2 + 0.6 * x,
            volatility_kind="field",
            show_progress=False,
        )
    )
    weights = grid.measure_weights()
    assert m[0][0] == 0.0
    assert m[0][-1] == m0[-1]
    out = solver._boundary_outflow
    assert np.max(np.abs(m @ weights + out["free"] + out["exit"] - m[0] @ weights)) < 1e-12

    problem, grid = _problem(2, 13, 5, 1.0)
    x = np.meshgrid(*[np.asarray(a) for a in grid.coordinates], indexing="ij")[0]
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="inflow", bc_type=BCType.FLUX, value=-1.0, boundary="x_min"),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", flux_capacity=0.3),
            BCSegment(name="y_min", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="y_max", bc_type=BCType.NO_FLUX, boundary="y_max"),
        ],
    )
    solver = FPFDMSolver(problem, boundary_conditions=bc)
    m = np.asarray(
        solver.solve_fp_system(
            0.01 * np.ones_like(x), potential_field=np.broadcast_to(-2.0 * x, (6, *x.shape)).copy(), show_progress=False
        )
    )
    # the FLUX face lets in 1.0 per unit length per unit time over its length LY, which the record does not hold
    let_in = 1.0 * LY * 0.2 * np.arange(6)
    weights = grid.measure_weights()
    total = np.einsum("tij,ij->t", m, weights)
    out = solver._boundary_outflow["exit"]
    assert np.max(np.abs(total + out - let_in - total[0])) < 1e-12
    assert np.max(np.diff(out)) > 0.5 * 0.3 * LY * 0.2  # the inflow reaches the exit
    assert np.max(np.diff(out)) <= 0.3 * LY * 0.2 * (1 + 1e-12)


def test_a_capped_face_off_axis_0_lets_out_c_times_its_measure():
    """A capped y_max under unequal spacing (dx 0.1, dy 0.0625), saturated at every step: each step lets out c |y_max| dt.
    The half cell is read on the exit face's own axis: read on axis 0, or off a corner node's first face, the step lets
    out 0.625 or 0.9625 of it (review 2 of #2849)."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 0.5)], Nx_points=[11, 9], boundary_conditions=no_flux_bc(dimension=2)
    )
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.3)
    conditions = Conditions(m_initial=lambda x: 2.0, u_terminal=lambda x: 0.0, T=0.3)  # unit mass on the box
    problem = MFGProblem(model=model, domain=grid, conditions=conditions, Nt=6)
    walls = [BCSegment(name=f, bc_type=BCType.NO_FLUX, boundary=f) for f in ("x_min", "x_max", "y_min")]
    exit_ = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="y_max", flux_capacity=0.05)
    solver = FPFDMSolver(problem, boundary_conditions=BoundaryConditions(dimension=2, segments=[*walls, exit_]))
    y = np.meshgrid(*[np.asarray(a) for a in grid.coordinates], indexing="ij")[1]
    solver.solve_fp_system(4.0 * np.ones_like(y), potential_field=np.broadcast_to(-2.0 * y, (7, *y.shape)).copy())
    assert np.allclose(np.diff(solver._boundary_outflow["exit"]), 0.05 * 1.0 * 0.05, rtol=1e-12, atol=0)


def test_a_cap_past_every_outflow_is_the_absorbing_exit_bit_for_bit():
    problem, grid = _problem(2, 21, 10, 0.2)
    x = np.meshgrid(*[np.asarray(a) for a in grid.coordinates], indexing="ij")[0]
    runs = []
    for cap in (None, 1e300):
        solver = FPFDMSolver(problem, boundary_conditions=_bc(2, cap))
        m = solver.solve_fp_system(_exact(0.0, x), potential_field=np.broadcast_to(-ALPHA * x, (11, *x.shape)).copy())
        runs.append((np.asarray(m), solver._boundary_outflow["exit"]))
    assert np.array_equal(runs[0][0], runs[1][0])
    assert np.array_equal(runs[0][1], runs[1][1])


def _solve(bc: BoundaryConditions, dim: int = 1, scheme: str = "divergence_upwind", drift=None) -> None:
    problem, grid = _problem(dim, 11, 4, 0.1)
    x = np.meshgrid(*[np.asarray(a) for a in grid.coordinates], indexing="ij")[0]
    solver = FPFDMSolver(problem, boundary_conditions=bc, advection_scheme=scheme)
    if drift is None:
        solver.solve_fp_system(
            np.ones_like(x), potential_field=np.broadcast_to(-x, (5, *x.shape)).copy(), show_progress=False
        )
    else:
        solver.solve_fp_system(np.ones_like(x), drift_field=drift, show_progress=False)


def _solve_set_after(bc: BoundaryConditions) -> None:
    """The solve with ``bc`` set on the solver after construction, which the constructor never checked."""
    problem, grid = _problem(1, 11, 4, 0.1)
    x = np.asarray(grid.coordinates[0])
    solver = FPFDMSolver(problem, boundary_conditions=_bc(1, CAP))
    solver.boundary_conditions = bc.bind_geometry(grid)
    solver.solve_fp_system(
        np.ones_like(x), potential_field=np.broadcast_to(-x, (5, *x.shape)).copy(), show_progress=False
    )


def _coupled(bc: BoundaryConditions) -> None:
    """A coupled solve, ``problem.solve``, whose FP solver carries ``bc`` as its own."""
    problem, _ = _problem(1, 11, 4, 0.1)
    fp = FPFDMSolver(problem, boundary_conditions=bc)
    problem.solve(hjb_solver=HJBFDMSolver(problem), fp_solver=fp, max_iterations=1, verbose=False)


def _capped() -> BCSegment:
    return BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", flux_capacity=CAP)


def _faces(x_max: BCSegment | None = None, y_max: BCSegment | None = None, capped_everywhere_else: bool = False):
    """A 2-D BC: NO_FLUX x_min and y_min; ``x_max`` and ``y_max`` as given, or one capped segment with no placement
    governing both."""
    segments = [BCSegment(name=f, bc_type=BCType.NO_FLUX, boundary=f) for f in ("x_min", "y_min")]
    if capped_everywhere_else:
        segments.append(BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, flux_capacity=CAP))
    else:
        segments += [x_max, y_max]
    return BoundaryConditions(dimension=2, segments=segments)


def _step_only(bc: BoundaryConditions) -> None:
    """``solve_timestep_full_nd`` called directly, below the solver and the full-system solve."""
    problem, grid = _problem(1, 11, 4, 0.1)
    x = np.asarray(grid.coordinates[0])
    solve_timestep_full_nd(
        M_current=np.ones(11),
        U_current=-x,
        problem=problem,
        dt=0.025,
        volatility=SIGMA,
        coupling_coefficient=1.0,
        spacing=(0.1,),
        grid=grid,
        ndim=1,
        shape=(11,),
        boundary_conditions=bc.bind_geometry(grid),
        advection_scheme="divergence_upwind",
    )


def _no_step(bc: BoundaryConditions) -> None:
    """``solve_fp_nd_full_system`` handed one time level, which writes level 0 and takes no step."""
    problem, grid = _problem(1, 11, 4, 0.1)
    x = np.asarray(grid.coordinates[0])
    solve_fp_nd_full_system(np.ones(11), -x[None, :].copy(), problem, boundary_conditions=bc.bind_geometry(grid))


def _coupled_set_after() -> None:
    """The cap set on the FP solver after the coupled solver holding it was built: the refusal reads it when it solves."""
    problem, grid = _problem(1, 11, 4, 0.1)
    fp = FPFDMSolver(problem, boundary_conditions=_bc(1, None))
    iterator = FixedPointIterator(problem, hjb_solver=HJBFDMSolver(problem), fp_solver=fp)
    fp.boundary_conditions = _bc(1, CAP).bind_geometry(grid)
    iterator.solve(**ONE_ITERATION)


def _adjoint_step(bc: BoundaryConditions) -> None:
    problem, _ = _problem(1, 11, 4, 0.1)
    FPFDMSolver(problem, boundary_conditions=bc).solve_fp_step_adjoint_mode(np.ones(11), sparse.csr_matrix((11, 11)))


REFUSALS = {
    "centered": (
        lambda: _solve(_bc(1, CAP), scheme="divergence_centered"),
        NotImplementedError,
        r"switch could take the wrong one",
    ),
    "a prescribed density": (
        lambda: _solve(_bc(1, CAP, value=0.5)),
        NotImplementedError,
        r"prescribed density m = g, not an exit",
    ),
    "a wall": (
        lambda: _solve(_bc(1, CAP, x_max=BCType.NO_FLUX)),
        NotImplementedError,
        r"is no_flux and carries a flux_capacity",
    ),
    "a negative cap": (lambda: _solve(_bc(1, -1.0)), ValueError, r"a cap is a finite mass per unit time"),
    # refused at construction, before any step: the constructor reads the owner as well as the step
    "a bool cap": (
        lambda: FPFDMSolver(_problem(1, 11, 4, 0.1)[0], boundary_conditions=_bc(1, False)),
        ValueError,
        r"not a bool",
    ),
    "a negative cap set after construction": (lambda: _solve_set_after(_bc(1, -1.0)), ValueError, r"finite mass"),
    "two exits at a corner": (
        lambda: _solve(_bc(2, CAP, y_max=BCType.DIRICHLET), dim=2),
        NotImplementedError,
        r"Make every face that meets the capped x_max face a NO_FLUX wall",
    ),
    "a corner the other face governs": (
        lambda: _solve(_bc(2, CAP, y_max=BCType.DIRICHLET, y_max_priority=1), dim=2),
        NotImplementedError,
        r"Make every face that meets the capped x_max face a NO_FLUX wall",
    ),
    "a cap past float's range": (lambda: _solve(_bc(1, 10**400)), ValueError, r"finite mass"),
    "a negative cap through the step directly": (lambda: _step_only(_bc(1, -1.0)), ValueError, r"finite mass"),
    "a negative cap on a run with no step": (lambda: _no_step(_bc(1, -1.0)), ValueError, r"finite mass"),
    "a FLUX face meeting the cap": (
        lambda: _solve(
            _faces(x_max=_capped(), y_max=BCSegment(name="in", bc_type=BCType.FLUX, value=-0.1, boundary="y_max")),
            dim=2,
        ),
        NotImplementedError,
        r"Make every face that meets the capped x_max face a NO_FLUX wall",
    ),
    "a capped segment on two meeting faces": (
        lambda: _solve(_faces(capped_everywhere_else=True), dim=2),
        NotImplementedError,
        r"Make every face that meets the capped",
    ),
    "a coupled solve": (lambda: _coupled(_bc(1, CAP)), NotImplementedError, r"standalone FP solve only"),
    # c = 0 is a cap FP-FDM honours, the limit the 2026-10-10 ruling turns on, not an absent one (review 3 of #2849)
    "a zero cap in a coupled solve": (lambda: _coupled(_bc(1, 0.0)), NotImplementedError, r"standalone FP solve only"),
    "a cap set after the coupled solver is built": (
        lambda: _coupled_set_after(),
        NotImplementedError,
        r"standalone FP solve only",
    ),
    "the explicit step": (
        lambda: _solve(_bc(1, CAP), drift=lambda t, x, m: np.ones_like(np.asarray(x)[..., 0])),
        NotImplementedError,
        r"runs no capped-exit switch",
    ),
    "the adjoint step": (
        lambda: _adjoint_step(_bc(1, CAP)),
        NotImplementedError,
        r"solve_fp_step_adjoint_mode runs no capped-exit switch",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_cap_is_refused_where_the_switch_does_not_hold(case: str):
    """Each refusal names what it refuses; the corner's remedy, making the other face a wall, and the coupled solve's,
    the FP solved alone, are run and solve."""
    run, error, message = REFUSALS[case]
    with pytest.raises(error, match=message):
        run()
    if case == "two exits at a corner":
        _solve(_bc(2, CAP), dim=2)
    if case == "a coupled solve":  # the remedy the refusal names: the FP solved alone
        _solve(_bc(1, CAP))


def _coupled_pairs(k: int) -> tuple[list[MFGProblem], list[HJBFDMSolver], list[FPFDMSolver]]:
    """k problems, each with an HJB solver and an FP solver whose own BC caps the exit at x_max."""
    problems = [_problem(1, 11, 4, 0.1)[0] for _ in range(k)]
    return (
        problems,
        [HJBFDMSolver(p) for p in problems],
        [FPFDMSolver(p, boundary_conditions=_bc(1, CAP)) for p in problems],
    )


def _single(build):
    problems, hjbs, fps = _coupled_pairs(1)
    return build(problems[0], hjbs[0], fps[0])


ONE_ITERATION = {"max_iterations": 1, "tolerance": 1e-12, "verbose": False}
#: every coupled solver that holds an FP solver, besides problem.solve (a REFUSALS row): each solves once, capped
COUPLED = {
    "fixed point": lambda: _single(
        lambda p, h, f: FixedPointIterator(p, hjb_solver=h, fp_solver=f).solve(**ONE_ITERATION)
    ),
    "block gauss-seidel": lambda: _single(
        lambda p, h, f: BlockGaussSeidelIterator(p, hjb_solver=h, fp_solver=f).solve(**ONE_ITERATION)
    ),
    "block jacobi": lambda: _single(lambda p, h, f: BlockJacobiIterator(p, h, f).solve(**ONE_ITERATION)),
    "fictitious play": lambda: _single(
        lambda p, h, f: FictitiousPlayIterator(p, hjb_solver=h, fp_solver=f).solve(**ONE_ITERATION)
    ),
    "newton": lambda: _single(lambda p, h, f: NewtonMFGSolver(p, h, f).solve(**ONE_ITERATION)),
    "mfg residual": lambda: _single(lambda p, h, f: MFGResidual(p, h, f).compute_fp_output(np.zeros((5, 11)))),
    "multi-population": lambda: (
        lambda ps, hs, fs: MultiPopulationIterator(
            MultiPopulationProblem(populations=ps, population_names=["a", "b"]), hs, fs
        ).solve(max_iterations=1, tolerance=1e-12)
    )(*_coupled_pairs(2)),
    "regime switching": lambda: (
        lambda ps, hs, fs: RegimeSwitchingIterator(
            ps, RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]])), hs, fs, max_iterations=1
        ).solve()
    )(*_coupled_pairs(2)),
    "graph": lambda: (
        lambda ps, hs, fs: GraphMFGSolver(
            ps, AdjacencyCoupling(np.ones((2, 2)) - np.eye(2), alpha=0.05, beta=0.02), hs, fs, max_iterations=1
        ).solve()
    )(*_coupled_pairs(2)),
}


@pytest.mark.parametrize("route", sorted(COUPLED))
def test_every_coupled_solver_refuses_a_capped_fp(route: str):
    """A capped exit is offered for a standalone FP solve only (user rulings 2026-10-10 and 2026-10-31): every coupled
    solver that holds an FP solver refuses one whose BC carries a cap, by name. Each row caps every FP solver before
    the coupled solver is built, so the rows pin that each site refuses, not that it reads at solve: that is pinned for
    the fixed point alone, by the REFUSALS row that caps after it is built. Its remedy, the FP solved alone, is the
    coupled REFUSALS row's."""
    with pytest.raises(NotImplementedError, match=r"standalone FP solve only"):
        COUPLED[route]()

"""#2997 stage 1b: GraphMFGSolver, RegimeSwitchingIterator and MultiPopulationIterator through the iterate's owner
(`IterationState`) and the damping run (`DampingRun`), one of each per component; and Newton's warm-up judged as every
sweep is. Each pin compares a route with FixedPointIterator, the route both owners were built from, or with the 1-D
solve a 2-D one must reproduce."""

from __future__ import annotations

import functools
import logging
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling import FixedPointIterator, NewtonMFGSolver
from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling, LaplacianCoupling
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.config.damping import DampingPolicy, FieldDamping
from mfgarchon.core.coupling_costs import InteractionCost, PotentialCost
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.operators.interaction.kernels import RadialKernel

ROUTES = ["GraphMFGSolver", "RegimeSwitchingIterator", "MultiPopulationIterator"]


class _Quadratic(RadialKernel):
    """``K(r) = c r^2``, so G's derivative moves with m(T)."""

    def __init__(self, c):
        self.c = c

    def profile(self, r):
        return self.c * np.asarray(r, dtype=float) ** 2

    @property
    def is_repulsive(self):
        return False


def _problem(centre, *, terminal_coupled, coupling=1.0):
    """A bump at ``centre`` on 11 nodes, coupling ``coupling * m``. With ``terminal_coupled``, G = (1/2) int int
    |x - y|^2 m m + int 3 x m moves u(T) with m(T), so U's terminal row moves from sweep to sweep; without it (the
    multi-population iterator refuses G, the user's ruling 5) u(T) is G's potential part, 3x, fixed."""
    xs = np.linspace(0.0, 1.0, 11)
    mass = np.trapezoid(np.exp(-40 * (xs - centre) ** 2), xs)
    G = {"terminal_coupling": InteractionCost(_Quadratic(1.0)) + PotentialCost(lambda x: 3.0 * x[0])}
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=lambda m: coupling * m),
            volatility=0.3,
            **(G if terminal_coupled else {}),
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            m_initial=lambda x: np.exp(-40 * (np.asarray(x[0], dtype=float) - centre) ** 2) / mass,
            u_terminal=(lambda x: 0.0) if terminal_coupled else (lambda x: 3.0 * np.asarray(x[0], dtype=float)),
            T=0.3,
        ),
        Nt=5,
    )


def _fp(problem, lifted):
    """FP-FDM, whose level 0 is m(0); or, ``lifted``, the same solver with its output's level 0 scaled by 1 + 1e-3, its
    signature kept, so a route that does not write level 0 back hands the scaled one to the next sweep."""
    fp = FPFDMSolver(problem)
    if lifted:
        solve = fp.solve_fp_system

        @functools.wraps(solve)
        def scaled(*args, **kwargs):
            M = np.array(solve(*args, **kwargs), dtype=float)
            M[0] *= 1 + 1e-3
            return M

        fp.solve_fp_system = scaled
    return fp


def _uncoupled(route, problems, damping, sweeps, tolerance, lifted=False):
    """``route`` on ``problems``, one component each, with nothing coupling the components: (values, densities, sweeps
    run, converged)."""
    hjbs, fps = [HJBFDMSolver(p) for p in problems], [_fp(p, lifted) for p in problems]
    K = len(problems)
    if route == "GraphMFGSolver":
        r = GraphMFGSolver(
            problems,
            AdjacencyCoupling(np.zeros((K, K)), alpha=0.1, beta=0.1),
            hjbs,
            fps,
            max_iterations=sweeps,
            tolerance=tolerance,
            damping=damping,
        ).solve()
        return r.values, r.densities, r.iterations, r.converged
    if route == "RegimeSwitchingIterator":
        r = RegimeSwitchingIterator(
            problems,
            RegimeSwitchingConfig(transition_matrix=np.zeros((K, K))),
            hjbs,
            fps,
            max_iterations=sweeps,
            tolerance=tolerance,
            damping=damping,
        ).solve()
        return r.values, r.densities, r.iterations, r.converged
    multi = MultiPopulationProblem(populations=problems, population_names=[f"p{k}" for k in range(K)])
    r = MultiPopulationIterator(multi, hjbs, fps, damping=damping).solve(max_iterations=sweeps, tolerance=tolerance)
    return r.U, r.M, r.iterations, r.converged


_ADAPTIVE = DampingPolicy(U=FieldDamping(theta=1.0, adaptive=True), M=FieldDamping(theta=1.0, adaptive=True))


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    ("case", "damping", "coupling", "sweeps", "tolerance", "lifted"),
    [
        ("default, 12 sweeps", None, 1.0, 12, 1e-12, False),
        ("default, a first sweep that passes", None, 1.0, 1, 1e9, False),
        ("adaptive", _ADAPTIVE, 5.0, 8, 1e-12, False),
        ("default, an FP whose level 0 is not m(0)", None, 1.0, 4, 1e-12, True),
    ],
)
def test_each_component_takes_fixed_points_path(route, case, damping, coupling, sweeps, tolerance, lifted):
    """Two components with nothing coupling them: each one's iterate is FixedPoint's on its own problem, bit for bit, at
    every sweep count run here, under the route's default policy and under an adaptive one.

    That is every concern the owners decide, read on the component axis: U's cold start, u(T) in every row (ruling 3);
    both fields damped from the first sweep (ruling 2); U's terminal row written back to the u(T) its sweep's HJB solve
    was given, which moves here with m(T) on Graph and RegimeSwitching; level 0 written back to m(0), seen only through
    an FP whose level 0 is not m(0), since FP-FDM's is; the first sweep judged (the approver's ruling of 2026-11-25); and
    one damping run per component, each field adapting on that component's own history (rulings 5 and 9). At coupling 5
    and theta 1 adaptation moves FixedPoint's path within eight sweeps; the control checks that it does, on the first
    component's problem. At 4cd1803f the default case differs on all three routes (U by 8.5e-4 on Graph after 12
    sweeps), and RegimeSwitching's first sweep does not stop.
    """
    terminal_coupled = route != "MultiPopulationIterator" and case != "adaptive"
    centres = (0.3, 0.6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            policy = damping if damping is not None else DampingPolicy.constant(0.5)
            references = []
            for centre in centres:
                p = _problem(centre, terminal_coupled=terminal_coupled, coupling=coupling)
                references.append(
                    FixedPointIterator(p, HJBFDMSolver(p), _fp(p, lifted), damping=policy).solve(
                        max_iterations=sweeps, tolerance=tolerance, verbose=False
                    )
                )
            if damping is not None:
                p = _problem(centres[0], terminal_coupled=terminal_coupled, coupling=coupling)
                plain = FixedPointIterator(p, HJBFDMSolver(p), FPFDMSolver(p), damping=DampingPolicy.constant(1.0))
                plain = plain.solve(max_iterations=sweeps, tolerance=tolerance, verbose=False)
                assert not np.array_equal(plain.U, references[0].U), (
                    "the control: adaptation must move FixedPoint's path"
                )
            problems = [_problem(c, terminal_coupled=terminal_coupled, coupling=coupling) for c in centres]
            values, densities, run, converged = _uncoupled(route, problems, damping, sweeps, tolerance, lifted)
        finally:
            logging.disable(logging.NOTSET)
    assert run == references[0].iterations == references[1].iterations
    assert converged == references[0].converged
    for k, reference in enumerate(references):
        assert np.array_equal(np.asarray(values[k]), reference.U), f"component {k}'s U is not FixedPoint's"
        assert np.array_equal(np.asarray(densities[k]), reference.M), f"component {k}'s M is not FixedPoint's"


@pytest.mark.parametrize("inner", ["converging", "capped"])
def test_a_warm_up_sweep_that_meets_the_criteria_ends_newtons_solve(inner):
    """Each phase judges the change of the map it iterates, through the one measure (the approver's ruling of
    2026-11-25): Newton's warm-up sweeps are FixedPoint's, so the first whose change meets the criteria ends the solve
    there with FixedPoint's iterate, count, error histories and verdict. A met criterion stops it either way, as it
    stops FixedPoint's loop (#1878): with each HJB step capped at one Newton iteration, the verdict is refused over the
    failed steps on both. At 4cd1803f the warm-up ran all twelve sweeps and then judged the residual at their output:
    a count of 13 where the HJB converges, and 16 against FixedPoint's 7 where it is capped, U 0.0148 apart."""
    cap = {"max_newton_iterations": 1, "newton_tolerance": 1e-14} if inner == "capped" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = _problem(0.3, terminal_coupled=True)
        fixed = FixedPointIterator(p, HJBFDMSolver(p, **cap), FPFDMSolver(p)).solve(
            max_iterations=12, tolerance=1e-2, verbose=False
        )
        q = _problem(0.3, terminal_coupled=True)
        newton = NewtonMFGSolver(q, HJBFDMSolver(q, **cap), FPFDMSolver(q), picard_warmup=12).solve(
            max_iterations=30, tolerance=1e-2, verbose=False
        )
    stopped = fixed.metadata["convergence_reason"]
    assert fixed.iterations < 12, f"the fixture: FixedPoint must stop inside the warm-up ({stopped})"
    if inner == "capped":
        assert stopped.startswith("inner_hjb_not_converged"), f"the fixture: the capped HJB must fail ({stopped})"
    assert newton.converged == fixed.converged == (inner == "converging")
    if inner == "capped":
        assert newton.metadata["convergence_reason"] == stopped
    assert newton.iterations == fixed.iterations
    assert newton.metadata["picard_iterations"] == fixed.iterations
    assert np.array_equal(newton.U, fixed.U)
    assert np.array_equal(newton.M, fixed.M)
    assert np.array_equal(newton.error_history_U, fixed.error_history_U)
    assert np.array_equal(newton.error_history_M, fixed.error_history_M)


_SHAPE = (5, 4)


def _flat_problem(k, axis, n=None):
    """Component k's problem on the 5 x 4 grid, its data varying along ``axis`` only; or, with ``n``, on ``n`` nodes of
    that axis alone. The components differ, so the coupling between them has something to read."""
    if n is None:
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=list(_SHAPE), boundary_conditions=no_flux_bc(dimension=2)
        )
    else:
        domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
        axis = 0
    centre, slope = (0.3, 0.7)[k], (1.0, -0.5)[k]
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=lambda m: 0.5 * m),
            volatility=0.3,
        ),
        domain=domain,
        conditions=Conditions(
            m_initial=lambda x: 1.0 + 0.8 * np.cos(np.pi * (np.asarray(x[axis], dtype=float) - centre)),
            u_terminal=lambda x: slope * (np.asarray(x[axis], dtype=float) - 0.5) ** 2,
            T=0.2,
        ),
        Nt=4,
    )


def _coupled(route, problems):
    hjbs = [HJBFDMSolver(p, newton_tolerance=1e-13, max_newton_iterations=60) for p in problems]
    fps = [FPFDMSolver(p) for p in problems]
    if route.startswith("GraphMFGSolver"):
        adjacency = np.ones((2, 2)) - np.eye(2)
        coupling = (
            LaplacianCoupling(adjacency, kappa=0.3)
            if route.endswith("Laplacian")
            else AdjacencyCoupling(adjacency, alpha=0.3, beta=0.2)
        )
        r = GraphMFGSolver(problems, coupling, hjbs, fps, max_iterations=4, tolerance=1e-300).solve()
        return r.values, r.densities
    if route == "RegimeSwitchingIterator":
        config = RegimeSwitchingConfig(transition_matrix=np.array([[-0.6, 0.6], [0.4, -0.4]]))
        r = RegimeSwitchingIterator(problems, config, hjbs, fps, max_iterations=4, tolerance=1e-300).solve()
        return r.values, r.densities
    multi = MultiPopulationProblem(populations=problems[:1], population_names=["a"])
    r = MultiPopulationIterator(multi, hjbs[:1], fps[:1]).solve(max_iterations=4, tolerance=1e-300)
    return r.U, r.M


@pytest.mark.parametrize("axis", [0, 1], ids=["along x", "along y"])
@pytest.mark.parametrize("route", ["GraphMFGSolver", "GraphMFGSolver, Laplacian", *ROUTES[1:]])
def test_a_2d_solve_whose_data_vary_along_one_axis_is_the_1d_solve(route, axis):
    """External oracle: under no-flux walls, data that vary along one axis only give a 2-D solution constant along the
    other, equal to the 1-D solve along the varying axis. Four sweeps of each route on a 5 x 4 grid (one population for
    the multi-population iterator, which refuses two off a 1-D grid by name, #2335) equal the 1-D route's, replicated,
    to 1e-12, with the HJB's Newton solved to 1e-13. At 4cd1803f all three raised on the 5 x 4 grid, each writing a
    level of that shape into a row sized from the grid's first axis (#2573's ledger).

    The grid is not square and the data run along each axis in turn, so the order in which a coupling closure reads a
    level at the geometry's points is seen: Graph with `AdjacencyCoupling` and with `LaplacianCoupling`, whose
    default sources are the library's four graph closures, and RegimeSwitching's two. Read in F order instead of C,
    whether in `IterationState.level_at_points` or in one closure, the routes it reaches fail both cases. The
    multi-population iterator has no such closure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        two = _coupled(route, [_flat_problem(k, axis) for k in range(2)])
        one = _coupled(route, [_flat_problem(k, axis, n=_SHAPE[axis]) for k in range(2)])
    for name, fields_2d, fields_1d in (("U", two[0], one[0]), ("M", two[1], one[1])):
        for k, (f2, f1) in enumerate(zip(fields_2d, fields_1d, strict=True)):
            f1 = np.asarray(f1)
            replicated = (
                np.repeat(f1[:, :, None], _SHAPE[1], axis=2)
                if axis == 0
                else np.repeat(f1[:, None, :], _SHAPE[0], axis=1)
            )
            np.testing.assert_allclose(np.asarray(f2), replicated, rtol=0, atol=1e-12, err_msg=f"{name}{k}")

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
from mfgarchon.alg.numerical.coupling.block_iterators import BlockGaussSeidelIterator
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
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
from mfgarchon.utils.solver_result import SolverResult
from mfgarchon.visualization.extract import extract_graph_trajectories

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
    """``route`` on ``problems``, one component each, with nothing coupling the components: its result."""
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
    elif route == "RegimeSwitchingIterator":
        r = RegimeSwitchingIterator(
            problems,
            RegimeSwitchingConfig(transition_matrix=np.zeros((K, K))),
            hjbs,
            fps,
            max_iterations=sweeps,
            tolerance=tolerance,
            damping=damping,
        ).solve()
    else:
        multi = MultiPopulationProblem(populations=problems, population_names=[f"p{k}" for k in range(K)])
        r = MultiPopulationIterator(multi, hjbs, fps, damping=damping).solve(max_iterations=sweeps, tolerance=tolerance)
    return r


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

    The result is FixedPoint's too, read on the component axis (stage 1c's K = 1 contract): each component's histories
    and mass drift in ``metadata["per_component"]`` are FixedPoint's on its problem, and the top level is the worst
    component's histories, the reason the same criteria give, and the drift of the mass the route conserves, the worst
    population's on MultiPopulation and the total's on Graph and RegimeSwitching. At ba36fa73 the three routes returned
    their own types: Graph's and RegimeSwitching's history the worst relative change per sweep, MultiPopulation's only
    the last sweep's relative changes, and none a mass.
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
            result = _uncoupled(route, problems, damping, sweeps, tolerance, lifted)
        finally:
            logging.disable(logging.NOTSET)
    assert result.components == 2
    assert result.U.shape == result.M.shape == (2, *references[0].U.shape)
    assert result.iterations == references[0].iterations == references[1].iterations
    assert result.converged == references[0].converged
    assert result.solver_name == route
    for k, reference in enumerate(references):
        assert np.array_equal(result.U[k], reference.U), f"component {k}'s U is not FixedPoint's"
        assert np.array_equal(result.M[k], reference.M), f"component {k}'s M is not FixedPoint's"
        own = result.metadata["per_component"][k]
        assert np.array_equal(own["l2distu_abs"], reference.error_history_U), f"component {k}'s history"
        assert np.array_equal(own["l2distm_abs"], reference.error_history_M), f"component {k}'s history"
        assert np.array_equal(own["l2distu_rel"], reference.metadata["l2distu_rel"]), f"component {k}'s history"
        assert np.array_equal(own["l2distm_rel"], reference.metadata["l2distm_rel"]), f"component {k}'s history"
        assert own["mass_conservation_error"] == reference.mass_conservation_error, f"component {k}'s mass drift"
    for field, key in (("error_history_U", "l2distu_abs"), ("error_history_M", "l2distm_abs")):
        worst = np.maximum(getattr(references[0], field), getattr(references[1], field))
        assert np.array_equal(getattr(result, field), worst), f"{field} is not the worst component's"
        assert np.array_equal(result.metadata["per_component"][0][key], getattr(references[0], field))
    for key in ("l2distu_rel", "l2distm_rel"):
        worst = np.maximum(references[0].metadata[key], references[1].metadata[key])
        assert np.array_equal(result.metadata[key], worst), f"{key} is not the worst component's"
    # The criteria's own reason for a met criterion, and the spent budget's where the budget ran out, as on FixedPoint (#3002)
    reason = result.metadata["convergence_reason"]
    assert (
        reason.startswith("Converged: ")
        if result.converged
        else reason == references[0].metadata["convergence_reason"] == "Maximum iterations reached"
    )
    assert result.metadata["inner_hjb_failures"] == []
    assert result.execution_time > 0.0
    assert result.metadata["damping"] == (damping if damping is not None else DampingPolicy.constant(0.5))
    if route == "MultiPopulationIterator":
        assert result.mass_conservation_error == max(r.mass_conservation_error for r in references)
    else:
        masses = [np.asarray(p.geometry.integrate(r.M), dtype=float) for p, r in zip(problems, references, strict=True)]
        total = masses[0] + masses[1]
        assert result.mass_conservation_error == float(np.max(np.abs(total / float(total[0]) - 1.0)))


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
        # The same refusal, of the criteria's reason with the phase that met it (stage 1c, ruling 3)
        assert stopped.count("were met (") == 1
        assert newton.metadata["convergence_reason"] == stopped.replace("were met (", "were met (Picard warm-up: ")
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
        return r.U, r.M
    if route == "RegimeSwitchingIterator":
        config = RegimeSwitchingConfig(transition_matrix=np.array([[-0.6, 0.6], [0.4, -0.4]]))
        r = RegimeSwitchingIterator(problems, config, hjbs, fps, max_iterations=4, tolerance=1e-300).solve()
        return r.U, r.M
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


def _drifting_fp(problem, rate):
    """FP-FDM whose output's level n is scaled by ``1 + rate * n / Nt``: a density whose mass drifts by ``rate`` by T,
    level 0 kept."""
    fp = FPFDMSolver(problem)
    solve = fp.solve_fp_system

    @functools.wraps(solve)
    def drifting(*args, **kwargs):
        M = np.array(solve(*args, **kwargs), dtype=float)
        steps = np.arange(M.shape[0]) / (M.shape[0] - 1)
        return M * (1.0 + rate * steps).reshape((-1,) + (1,) * (M.ndim - 1))

    fp.solve_fp_system = drifting
    return fp


def test_the_mass_drift_is_of_what_the_route_conserves():
    """``mass_conservation_error`` is the drift of the mass the route's model conserves (stage 1c, ruling 1).

    - **RegimeSwitching conserves the total.** External oracle: with Q = [[-1, 1], [0, 0]] regime 0 empties into regime
      1 at rate 1, so its mass is m_0(0) e^{-t} and its own drift 1 - e^{-T}, 0.259 at T = 0.3, while the total is
      conserved; the top level is the total's drift, within the scheme's error of 1e-2.
    - **MultiPopulation conserves each population's.** The control that separates the two: FP outputs whose masses
      drift by +2e-2 and -2e-2, equal initial masses, so the total holds while each moves. The top level is the worst
      population's, 2e-2, where the total's would read about 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problems = [_problem(c, terminal_coupled=False) for c in (0.3, 0.6)]
        config = RegimeSwitchingConfig(transition_matrix=np.array([[-1.0, 1.0], [0.0, 0.0]]))
        regime = RegimeSwitchingIterator(
            problems,
            config,
            [HJBFDMSolver(p) for p in problems],
            [FPFDMSolver(p) for p in problems],
            max_iterations=20,
            tolerance=1e-8,
        ).solve()
        populations = [_problem(c, terminal_coupled=False) for c in (0.3, 0.6)]
        multi = MultiPopulationIterator(
            MultiPopulationProblem(populations=populations, population_names=["up", "down"]),
            [HJBFDMSolver(p) for p in populations],
            [_drifting_fp(p, rate) for p, rate in zip(populations, (2e-2, -2e-2), strict=True)],
            damping=DampingPolicy.constant(1.0),
        ).solve(max_iterations=3, tolerance=1e-12)
    own = [c["mass_conservation_error"] for c in regime.metadata["per_component"]]
    assert abs(own[0] - (1.0 - np.exp(-0.3))) < 1e-2, f"regime 0's drift {own[0]:.4e} against 1 - e^-T"
    assert regime.mass_conservation_error < 1e-2 < min(own), (regime.mass_conservation_error, own)
    masses = [np.asarray(p.geometry.integrate(M), dtype=float) for p, M in zip(populations, multi.M, strict=True)]
    total = masses[0] + masses[1]
    assert float(np.max(np.abs(total / total[0] - 1.0))) < 1e-12, "the control: the populations' total must hold"
    own = [c["mass_conservation_error"] for c in multi.metadata["per_component"]]
    assert own == pytest.approx([2e-2, 2e-2], rel=1e-9)
    assert multi.mass_conservation_error == max(own)


def _grid_problem(n, nt, horizon):
    xs = np.linspace(0.0, 1.0, n)
    mass = np.trapezoid(np.exp(-40 * (xs - 0.4) ** 2), xs)
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=lambda m: m),
            volatility=0.3,
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            m_initial=lambda x: np.exp(-40 * (np.asarray(x[0], dtype=float) - 0.4) ** 2) / mass,
            u_terminal=lambda x: 3.0 * np.asarray(x[0], dtype=float),
            T=horizon,
        ),
        Nt=nt,
    )


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    ("second", "refusal"),
    [
        ((15, 5, 0.3), r"have an iterate of shape \[\(6, 15\)\] against {component} 0's \(6, 11\)"),
        ((11, 7, 0.3), r"step through time other than {component} 0's: \(T, Nt\) \[\(0\.3, 7\)\] against \(0\.3, 5\)"),
        ((11, 5, 0.5), r"step through time other than {component} 0's: \(T, Nt\) \[\(0\.5, 5\)\] against \(0\.3, 5\)"),
    ],
    ids=["other nodes", "other Nt", "other T"],
)
def test_components_off_one_grid_are_refused(route, second, refusal):
    """A route with a component axis reads every other component's level at the step it solves, by index, and returns
    every component's U and M in one array, so its components share one time grid and one shape (the user's rulings of
    2026-11-25). Refused by name before the first sweep. At ba36fa73 RegimeSwitching solved two regimes with other Nt,
    or other T at one Nt, each reading the other's level at its own time; Graph solved nodes of 11 and 15 points, its
    default couplings truncating to the shorter (#2765). MultiPopulation refuses other nodes by its own grid check."""
    component = {"GraphMFGSolver": "node", "RegimeSwitchingIterator": "regime"}.get(route, "population")
    refusal = refusal.format(component=component)
    if route == "MultiPopulationIterator" and second[0] != 11:
        refusal = r"population\(s\) \[1\] sit on a grid other than population 0's"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problems = [_grid_problem(11, 5, 0.3), _grid_problem(*second)]
        with pytest.raises(ValueError, match=refusal):
            _uncoupled(route, problems, None, 2, 1e-12)


ALL_ROUTES = ["FixedPointIterator", "FictitiousPlayIterator", "BlockGaussSeidelIterator", "NewtonMFGSolver", *ROUTES]


class _DivergingHJB(HJBFDMSolver):
    """HJB-FDM whose solve from its ``at``-th on returns its output with one NaN, as a diverged inner Newton does."""

    def __init__(self, problem, at):
        super().__init__(problem)
        self._calls, self._at = 0, at

    def solve_hjb_system(self, *args, source_term=None, volatility=None, **kwargs):
        U = np.array(
            super().solve_hjb_system(*args, source_term=source_term, volatility=volatility, **kwargs), dtype=float
        )
        self._calls += 1
        if self._at is not None and self._calls >= self._at:
            U[2, 5] = np.nan
        return U


def _any_route(route, problems, hjbs, sweeps, tolerance, **newton):
    """One of the seven coupled routes, its HJB solvers given: one problem for the four single-pair routes, all of them
    for the three component routes, with nothing coupling the components."""
    fps = [FPFDMSolver(p) for p in problems]
    p, h, f = problems[0], hjbs[0], fps[0]
    if route == "FixedPointIterator":
        return FixedPointIterator(p, h, f).solve(max_iterations=sweeps, tolerance=tolerance, verbose=False)
    if route == "FictitiousPlayIterator":
        return FictitiousPlayIterator(p, h, f).solve(max_iterations=sweeps, tolerance=tolerance, verbose=False)
    if route == "BlockGaussSeidelIterator":
        return BlockGaussSeidelIterator(p, h, f).solve(max_iterations=sweeps, tolerance=tolerance, verbose=False)
    if route == "NewtonMFGSolver":
        warmup = newton.get("picard_warmup", sweeps)
        return NewtonMFGSolver(p, h, f, picard_warmup=warmup).solve(
            max_iterations=sweeps, tolerance=tolerance, verbose=False
        )
    K = len(problems)
    if route == "GraphMFGSolver":
        return GraphMFGSolver(
            problems, AdjacencyCoupling(np.zeros((K, K))), hjbs, fps, max_iterations=sweeps, tolerance=tolerance
        ).solve()
    if route == "RegimeSwitchingIterator":
        config = RegimeSwitchingConfig(transition_matrix=np.zeros((K, K)))
        return RegimeSwitchingIterator(problems, config, hjbs, fps, max_iterations=sweeps, tolerance=tolerance).solve()
    multi = MultiPopulationProblem(populations=problems, population_names=[f"p{k}" for k in range(K)])
    return MultiPopulationIterator(multi, hjbs, fps).solve(max_iterations=sweeps, tolerance=tolerance)


def _quiet(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            return call()
        finally:
            logging.disable(logging.NOTSET)


@pytest.mark.parametrize("route", ALL_ROUTES)
def test_a_diverged_stop_counts_its_sweep_as_nan_on_every_route(route):
    """Ruling 10 (stage 1c): the sweep that diverged counts, its history entries are NaN, and the reason names the
    divergence, so a diverged solve's final change is not the last finite sweep's (#1672's shape). The first sweep is
    finite; at the second, the first component's HJB returns its output with one NaN. At ba36fa73 FixedPoint and
    FictitiousPlay published a 1.0 there, the Block iterators and Newton dropped the sweep, Graph and RegimeSwitching
    counted it with no entry, and MultiPopulation counted it with its last-sweep errors emptied."""
    problems = [_problem(c, terminal_coupled=False) for c in ((0.3,) if route not in ROUTES else (0.3, 0.6))]
    hjbs = [_DivergingHJB(p, 2 if k == 0 else None) for k, p in enumerate(problems)]
    result = _quiet(lambda: _any_route(route, problems, hjbs, 4, 1e-12))
    assert result.iterations == 2 == len(result.error_history_U) == len(result.error_history_M)
    assert np.isfinite(result.error_history_U[0])
    assert np.isfinite(result.error_history_M[0])
    assert np.isnan(result.error_history_U[-1])
    assert np.isnan(result.error_history_M[-1])
    assert np.isnan(result.metadata["l2distu_rel"][-1])
    assert np.isnan(result.metadata["l2distm_rel"][-1])
    assert result.metadata["convergence_reason"] == "diverged_nan"
    assert not result.converged


def _fp_turning(problem, at, value):
    """FP-FDM whose output from its ``at``-th solve on is ``value`` times its own, level 0 kept: NaN for a non-finite
    density, 1e6 for one whose mass blows up. The solver's signature is kept."""
    fp = FPFDMSolver(problem)
    solve = fp.solve_fp_system
    calls = [0]

    @functools.wraps(solve)
    def turned(*args, **kwargs):
        M = np.array(solve(*args, **kwargs), dtype=float)
        calls[0] += 1
        if calls[0] >= at:
            M[1:] = M[1:] * value
        return M

    fp.solve_fp_system = turned
    return fp


@pytest.mark.parametrize(("value", "reason"), [(np.nan, "diverged_nan"), (1e6, "diverged_mass_blowup")])
def test_fixed_points_density_stops_count_their_sweep_as_nan(value, reason):
    """FixedPoint's two stops on the density it was handed, a non-finite one (#688) and one whose mass blew up (#1489),
    end the sweep before its change is measured: the sweep counts, its entries NaN, as at its diverged HJB (ruling 10)."""
    problem = _problem(0.3, terminal_coupled=False)
    result = _quiet(
        lambda: FixedPointIterator(problem, HJBFDMSolver(problem), _fp_turning(problem, 2, value)).solve(
            max_iterations=4, tolerance=1e-12, verbose=False
        )
    )
    assert result.metadata["convergence_reason"] == reason
    assert result.iterations == 2 == len(result.error_history_U)
    assert np.isfinite(result.error_history_U[0])
    assert np.isnan(result.error_history_U[-1])
    assert np.isnan(result.error_history_M[-1])


def test_newtons_first_verdict_diverging_measures_no_mass():
    """With no warm-up, Newton's first verdict reads the cold start's map. Diverging there, the solve ends with M the cold
    start, which no FP of this solve produced, so its mass is not measured (#1717), and the sweep counts as NaN."""
    problem = _problem(0.3, terminal_coupled=False)
    result = _quiet(
        lambda: _any_route("NewtonMFGSolver", [problem], [_DivergingHJB(problem, 1)], 4, 1e-12, picard_warmup=0)
    )
    assert result.metadata["convergence_reason"] == "diverged_nan"
    assert result.iterations == 1
    assert np.isnan(result.error_history_U[-1])
    assert result.mass_conservation_error is None


@pytest.mark.parametrize(
    ("route", "picard_warmup", "tolerance"),
    [(route, None, 1e-6) for route in ALL_ROUTES] + [("NewtonMFGSolver", 0, 1e-6), ("NewtonMFGSolver", None, 1e12)],
    ids=[*ALL_ROUTES, "NewtonMFGSolver, its Newton phase", "NewtonMFGSolver, its warm-up's stop"],
)
def test_every_route_measures_the_mass_its_model_conserves(route, picard_warmup, tolerance):
    """`mass_conservation_error` on all seven routes (stage 1c, rulings 1 and 2): the drift of the geometry's integral of
    M from level 0, per component, and over the components the route conserves. Newton is run to each of its stops
    that holds this solve's density: the Newton phase, with no warm-up, and a warm-up sweep whose change meets the
    criteria. At ba36fa73 it was None on FictitiousPlay, Block and Newton."""
    problems = [_problem(c, terminal_coupled=False) for c in ((0.3,) if route not in ROUTES else (0.3, 0.6))]
    newton = {} if picard_warmup is None else {"picard_warmup": picard_warmup}
    result = _quiet(lambda: _any_route(route, problems, [HJBFDMSolver(p) for p in problems], 4, tolerance, **newton))
    densities = [result.M] if result.components is None else list(result.M)
    masses = [np.asarray(p.geometry.integrate(M), dtype=float) for p, M in zip(problems, densities, strict=True)]
    drifts = [float(np.max(np.abs(m / float(m[0]) - 1.0))) for m in masses]
    total = sum(masses)
    expected = (
        float(np.max(np.abs(total / float(total[0]) - 1.0)))
        if route in ("GraphMFGSolver", "RegimeSwitchingIterator")
        else max(drifts)
    )
    assert result.mass_conservation_error == expected, (result.mass_conservation_error, expected)
    assert result.execution_time > 0.0
    assert isinstance(result.metadata["damping"], DampingPolicy)


@pytest.mark.parametrize("route", ["BlockGaussSeidelIterator", *ROUTES])
def test_a_solve_that_runs_no_sweep_reports_so(route):
    """With no sweep run, a route's result counts none, gives FixedPoint's reason for that stop, and measures no mass,
    M being the cold start (#1717). FixedPoint, FictitiousPlay and Newton read 0 as their default budget. At ba36fa73
    MultiPopulation raised `UnboundLocalError` here."""
    problems = [_problem(c, terminal_coupled=False) for c in ((0.3,) if route not in ROUTES else (0.3, 0.6))]
    result = _quiet(lambda: _any_route(route, problems, [HJBFDMSolver(p) for p in problems], 0, 1e-12))
    assert result.iterations == 0 == len(result.error_history_U)
    assert result.metadata["convergence_reason"] == "Maximum iterations reached"
    assert result.mass_conservation_error is None


def test_a_result_with_no_component_axis_is_refused_by_the_graph_extractor(tmp_path):
    """`extract_graph_trajectories` iterates U and M along the component axis, so a single-pair result, whose first axis
    is time, would read its time levels as nodes; it is refused by name. A component result's `components` survives the
    HDF5 round trip, so a reloaded result keeps its layout."""
    problem = _problem(0.3, terminal_coupled=False)
    single = _quiet(lambda: _any_route("FixedPointIterator", [problem], [HJBFDMSolver(problem)], 2, 1e-12))
    with pytest.raises(ValueError, match="has no component axis"):
        extract_graph_trajectories(single)
    problems = [_problem(c, terminal_coupled=False) for c in (0.3, 0.6)]
    graph = _quiet(lambda: _any_route("GraphMFGSolver", problems, [HJBFDMSolver(p) for p in problems], 2, 1e-12))
    assert len(extract_graph_trajectories(graph)) == 2 * graph.U.shape[1] * graph.U.shape[2]
    graph.save_hdf5(tmp_path / "graph.h5")
    reloaded = SolverResult.load_hdf5(tmp_path / "graph.h5")
    assert reloaded.components == 2
    assert reloaded.U.shape == graph.U.shape

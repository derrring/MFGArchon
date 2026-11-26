"""One driver loop for the coupled routes (#3002, stage 2a): the pins its class lines owe.

Each test is a mutation-verified convention pin of a fact several routes decided on their own before the driver (AGENTS.md
§ Admission, class 1): each goes red under the mutant its docstring and the PR body name.

- **Class line 1, an FP output a solve cannot use** (#2997's folded item): a non-finite one stops with
  ``'diverged_nan'`` and one whose level's mass passes the magnitude check with ``'diverged_mass_blowup'``, on all seven
  routes and at Newton's post-warm-up check and Newton phase; the stop publishes the stopped evaluation's HJB and FP
  outputs. At 8598206f only FixedPoint stopped, and not on a NaN at level 0. On a non-finite output the others raised
  #2578's refusal; on ×1e5, four ran their budget out, Graph's FP raised its negative-density refusal, and Newton
  reported a line-search failure.
- **Class line 2, the budget and its report** (ruling 2): ``max_iterations=0`` runs no sweep, and a spent budget reads
  ``"Maximum iterations reached"``, on all seven; ``newton_max_iterations`` bounds Newton's phase within the budget.
- **Class line 3, the callback's convention** (#2886, ruling 4): False stops, with the verdict read at that iterate;
  True and None continue. FictitiousPlay read the opposite.
- **Class line 4, the owner's refusals during one solve** (#3005): a BC changed during a solve, so that the problem
  declares two that differ, is refused at the next step on every route, whichever solver carries its own BC, Newton's
  post-warm-up check and phase included.
- **Class line 5, a solver whose signature cannot be read** (#2209): refused when the route is built, on every route.
- **Class line 6, whether a published density is measured** (#1717): Newton's cold start is not reported as measured.
- **Newton's warm-up sweep** makes one HJB solve and one FP solve (ruling 6: ``picard_residuals`` is removed).
- **The order between the fields** is the sweep map's option: Block Jacobi's map is Newton's residual's, and differs
  from the Gauss-Seidel map in M only.

d = 1 expresses every one of these: they are stops, counts, a return convention and a source of BC, none directional.
"""

from __future__ import annotations

import functools
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling import (
    BlockGaussSeidelIterator,
    BlockJacobiIterator,
    FictitiousPlayIterator,
    FixedPointIterator,
)
from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBGFDMSolver
from mfgarchon.config.damping import DampingPolicy
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc

ROUTES = [
    "FixedPoint",
    "FictitiousPlay",
    "Block GS",
    "Block Jacobi",
    "Newton",
    "Graph",
    "RegimeSwitching",
    "MultiPopulation",
]
ONE_COMPONENT = ["FixedPoint", "FictitiousPlay", "Block GS", "Block Jacobi", "Newton"]


def _problem(centre: float = 0.3) -> MFGProblem:
    xs = np.linspace(0.0, 1.0, 11)
    mass = np.trapezoid(np.exp(-40 * (xs - centre) ** 2), xs)
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=lambda m: m),
            volatility=0.3,
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            m_initial=lambda x: np.exp(-40 * (np.asarray(x[0], dtype=float) - centre) ** 2) / mass,
            u_terminal=lambda x: 3.0 * np.asarray(x[0], dtype=float),
            T=0.3,
        ),
        Nt=5,
    )


def _solve(route, problems, hjbs, fps, *, sweeps, tol=1e-12, **kw):
    """One solve of ``route`` on its problems and solvers, ``sweeps`` its budget."""
    if route == "FixedPoint":
        # The pins run below HJB-FDM's default Newton tolerance; FixedPoint's warning about it (#1081) is the fixture's
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"HJB newton_tolerance", category=UserWarning)
            return FixedPointIterator(problems[0], hjbs[0], fps[0]).solve(
                max_iterations=sweeps, tolerance=tol, verbose=False, **kw
            )
    if route == "FictitiousPlay":
        return FictitiousPlayIterator(problems[0], hjbs[0], fps[0]).solve(
            max_iterations=sweeps, tolerance=tol, verbose=False, **kw
        )
    if route in ("Block GS", "Block Jacobi"):
        cls = BlockGaussSeidelIterator if route == "Block GS" else BlockJacobiIterator
        return cls(problems[0], hjbs[0], fps[0]).solve(max_iterations=sweeps, tolerance=tol, verbose=False, **kw)
    if route == "Newton":
        warmup = kw.pop("picard_warmup", 3)
        newton_max = kw.pop("newton_max_iterations", 20)
        solver = NewtonMFGSolver(problems[0], hjbs[0], fps[0], picard_warmup=warmup, newton_max_iterations=newton_max)
        return solver.solve(max_iterations=sweeps, tolerance=tol, verbose=False, **kw)
    if route == "Graph":
        coupling = AdjacencyCoupling(np.array([[0.0, 1.0], [1.0, 0.0]]))
        return GraphMFGSolver(problems, coupling, hjbs, fps, max_iterations=sweeps, tolerance=tol).solve()
    if route == "RegimeSwitching":
        config = RegimeSwitchingConfig(transition_matrix=np.array([[-0.5, 0.5], [0.5, -0.5]]))
        return RegimeSwitchingIterator(problems, config, hjbs, fps, max_iterations=sweeps, tolerance=tol).solve()
    multi = MultiPopulationProblem(populations=problems, population_names=["a", "b"][: len(problems)])
    return MultiPopulationIterator(multi, hjbs, fps).solve(max_iterations=sweeps, tolerance=tol)


def _problems(route):
    return (
        [_problem(0.3), _problem(0.7)] if route in ("Graph", "RegimeSwitching", "MultiPopulation") else [_problem(0.3)]
    )


def _reason(result) -> str:
    return result.metadata["convergence_reason"]


# --- Class line 1 ---------------------------------------------------------------------------------------------------


def _bad_fp(problem, at: int, how: str) -> FPFDMSolver:
    """An FP-FDM solver whose output goes bad from its ``at``-th solve: one entry NaN or +inf, one entry of level 0
    NaN, which the write-back re-imposes, or the whole output times 1e5, finite (#1489's blow-up)."""
    fp = FPFDMSolver(problem)
    solve, calls = fp.solve_fp_system, [0]

    @functools.wraps(solve)
    def bad(*args, **kwargs):
        M = np.array(solve(*args, **kwargs), dtype=float)
        calls[0] += 1
        if calls[0] >= at:
            if how == "nan":
                M[3, 4] = np.nan
            elif how == "nan at level 0":
                M[0, 4] = np.nan
            elif how == "inf":
                M[3, 4] = np.inf
            elif how == "x1e3":
                M = M * 1e3
            else:
                M = M * 1e5
        fp.last = M
        return M

    fp.solve_fp_system = bad
    return fp


def _recording_hjb(problem) -> HJBFDMSolver:
    """An HJB-FDM solver that keeps its last output, so a test can read what the pipeline published."""
    hjb = HJBFDMSolver(problem)
    solve = hjb.solve_hjb_system

    @functools.wraps(solve)
    def recording(*args, **kwargs):
        hjb.last = solve(*args, **kwargs)
        return hjb.last

    hjb.solve_hjb_system = recording
    return hjb


def _published(result, k: int, field: str):
    array = getattr(result, field)
    return array if result.components is None else array[k]


@pytest.mark.parametrize(
    ("case", "at", "how", "reason"),
    [
        ("NaN at sweep 1", 1, "nan", "diverged_nan"),
        ("NaN at sweep 2", 2, "nan", "diverged_nan"),
        ("+inf at sweep 2", 2, "inf", "diverged_nan"),
        ("NaN at level 0 only, at sweep 2", 2, "nan at level 0", "diverged_nan"),
        ("x1e5 from sweep 2", 2, "x1e5", "diverged_mass_blowup"),
    ],
)
@pytest.mark.parametrize("route", ROUTES)
def test_an_fp_output_the_solve_cannot_use_stops_it_on_every_route(route, case, at, how, reason):
    """Mutant: the driver's or a component loop's `fp_output_stop` call removed, or its magnitude half; the NaN rows
    then raise #2578's refusal of a non-finite change, and the x1e5 rows run their budget out. The level-0 row reds
    under the rule reading the iterate only, as FixedPoint's guard did."""
    problems = _problems(route)
    hjbs = [_recording_hjb(p) for p in problems]
    fps = [_bad_fp(p, at, how) for p in problems]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _solve(route, problems, hjbs, fps, sweeps=4)
    assert not result.converged
    assert _reason(result) == reason, f"{route}, {case}: {_reason(result)!r}"
    assert result.iterations == at
    # The publication rule (the approver's ruling E): the stopped evaluation's HJB output, and its FP output with its
    # bad values kept. Mutant: the damped, written-back iterate published, as before the rule.
    for k, (hjb, fp) in enumerate(zip(hjbs, fps, strict=True)):
        M = _published(result, k, "M")
        if route == "RegimeSwitching":
            # Its FP solver returns n = e^{q t} m; T's FP output is m, the integrating factor undone, bad values kept
            assert np.array_equal(np.isfinite(M), np.isfinite(fp.last)), f"{route}: M is not the FP output"
        else:
            assert np.array_equal(M, fp.last, equal_nan=True), f"{route}: M is not the FP output"
        assert np.array_equal(_published(result, k, "U"), hjb.last, equal_nan=True), f"{route}: U is not the HJB output"


@pytest.mark.parametrize(
    ("how", "reason"),
    [
        ("nan", "diverged_nan"),
        ("nan at level 0", "diverged_nan"),
        ("inf", "diverged_nan"),
        ("x1e5", "diverged_mass_blowup"),
    ],
)
@pytest.mark.parametrize(("site", "at"), [("the post-warm-up check", 1), ("the Newton phase", 2)])
def test_an_fp_output_at_newtons_residual_sites_stops_it(site, at, how, reason):
    """With no warm-up, FP solve 1 is the post-warm-up check's and FP solve 2 the Newton phase's first iterate. Mutant:
    the pipeline's FP-output rule bypassed at Newton's residual map; the NaN rows then raise #2578's refusal, and the
    x1e5 rows stop at the line search."""
    p = _problem()
    hjb, fp = _recording_hjb(p), _bad_fp(p, at, how)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _solve("Newton", [p], [hjb], [fp], sweeps=4, picard_warmup=0)
    assert not result.converged
    assert _reason(result) == reason, f"{site}: {_reason(result)!r}"
    assert np.array_equal(result.M, fp.last, equal_nan=True), f"{site}: M is not the FP output"
    assert np.array_equal(result.U, hjb.last, equal_nan=True), f"{site}: U is not the HJB output"


def test_the_magnitude_reads_each_levels_mass_not_their_sum():
    """A solve that keeps every level's mass at m(0)'s, over 20001 levels, against one level grown 2e4-fold. Mutant: the
    magnitude as the space-time sum against level 0, which stops the first, healthy one (the false stop review 1 measured
    at Nt = 9500)."""
    from mfgarchon.alg.numerical.coupling.fixed_point_utils import fp_output_stop

    integrate = _problem().spatial_measure().integrate
    xs = np.linspace(0.0, 1.0, 11)
    level = np.exp(-40 * (xs - 0.3) ** 2)
    level = level / float(np.asarray(integrate(level[None, :])).reshape(-1)[0])
    healthy = np.tile(level, (20001, 1))
    grown = healthy.copy()
    grown[7] *= 2e4
    stop = functools.partial(fp_output_stop, each_component=True, site="test", iteration=0)
    assert stop([healthy], [healthy], [integrate]) is None
    assert stop([grown], [healthy], [integrate]) == "diverged_mass_blowup"


# --- Class line 2 ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
def test_a_budget_of_zero_runs_no_sweep_and_a_spent_budget_says_so(route):
    """Mutant: `max_iterations or 100` restored (FixedPoint and FictitiousPlay then run 41 and 100 sweeps at 0 on this
    problem and tolerance, as at 8598206f), or a sweep's
    empty criteria reason left as the spent budget's (`""` on the six fixed-point routes, "Newton max iterations" on
    Newton's)."""
    for sweeps in (0, 4):
        problems = _problems(route)
        result = _solve(
            route, problems, [HJBFDMSolver(p) for p in problems], [FPFDMSolver(p) for p in problems], sweeps=sweeps
        )
        assert result.iterations == sweeps, f"{route} at {sweeps}: {result.iterations}"
        assert not result.converged
        assert _reason(result) == "Maximum iterations reached", f"{route} at {sweeps}: {_reason(result)!r}"


@pytest.mark.parametrize("route", ["FixedPoint", "FictitiousPlay", "Block GS", "Block Jacobi", "Newton"])
def test_a_negative_budget_is_refused(route):
    """The driver's budget is a count of sweeps (ruling 2). Mutant: its refusal of a negative one removed."""
    p = _problem()
    with pytest.raises(ValueError, match=r"max_iterations is a count of sweeps, an int >= 0, not -1"):
        _solve(route, [p], [HJBFDMSolver(p)], [FPFDMSolver(p)], sweeps=-1)


@pytest.mark.parametrize(("sweeps", "counted"), [(1, 1), (2, 2), (3, 3)])
def test_newtons_count_stays_within_its_budget_below_its_warm_up(sweeps, counted):
    """At e14eecd6 Newton at picard_warmup=3 ran 8 sweeps at budgets 0 to 3: its warm-up, then newton_max_iterations.
    Mutant: the Newton phase given newton_max_iterations when the warm-up spends the budget."""
    p = _problem()
    result = _solve("Newton", [p], [HJBFDMSolver(p)], [FPFDMSolver(p)], sweeps=sweeps)
    assert result.iterations == counted
    assert result.metadata["picard_iterations"] == counted
    assert result.metadata["newton_iterations"] == 0


def test_newton_max_iterations_bounds_the_phase_within_the_budget():
    """With budget to spare, the Newton phase's own bound is what runs out, and the reason says so."""
    p = _problem()
    result = _solve(
        "Newton", [p], [HJBFDMSolver(p)], [FPFDMSolver(p)], sweeps=30, picard_warmup=1, newton_max_iterations=2
    )
    assert result.iterations == 3
    assert _reason(result).startswith("Newton max iterations"), _reason(result)


# --- Class line 3 ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["FixedPoint", "FictitiousPlay", "Block GS", "Newton"])
@pytest.mark.parametrize(("returned", "stops"), [(False, True), (True, False), (None, False)])
def test_a_callback_returning_false_stops_each_route_that_takes_one(route, returned, stops):
    """Mutant: the driver's reader inverted, as FictitiousPlay read before (True stopped, False continued)."""
    seen = []

    def callback(iteration, U, M, error_U, error_M):
        seen.append(iteration)
        return returned

    p = _problem()
    result = _solve(route, [p], [HJBFDMSolver(p)], [FPFDMSolver(p)], sweeps=3, iteration_callback=callback)
    if stops:
        assert seen == [0]
        assert result.iterations == 1
        assert _reason(result).startswith("callback_stopped")
    else:
        assert seen == [0, 1, 2]
        assert result.iterations == 3


# --- Class line 4 ---------------------------------------------------------------------------------------------------


def _gfdm_pair(problem, carriage):
    """HJB-GFDM at the grid's points and FP-FDM, each given its own BC or not: with "both" or "HJB only" no solver
    re-reads the problem's owner during a solve (FP-FDM reads it once, when it is built)."""
    points = np.asarray(problem.geometry.get_spatial_grid()).reshape(-1, 1)
    own = no_flux_bc(dimension=1)
    hjb_kw = {"boundary_conditions": own} if carriage in ("both", "HJB only") else {}
    fp_kw = {"boundary_conditions": own} if carriage == "both" else {}
    # monotonicity_scheme="none" is the solver's default, named so it does not warn
    return HJBGFDMSolver(problem, collocation_points=points, monotonicity_scheme="none", **hjb_kw), FPFDMSolver(
        problem, **fp_kw
    )


def _changes_the_bc_after(hjb, problems, solves: int):
    """Set every problem's BC to a second declaration that differs, after the HJB solver's ``solves``-th solve."""
    solve, calls = hjb.solve_hjb_system, [0]

    @functools.wraps(solve)
    def changing(*args, **kwargs):
        out = solve(*args, **kwargs)
        calls[0] += 1
        if calls[0] == solves:
            for problem in problems:
                problem.components.boundary_conditions = dirichlet_bc(dimension=1, value=0.7)
        return out

    hjb.solve_hjb_system = changing


@pytest.mark.parametrize("carriage", ["both", "HJB only"])
@pytest.mark.parametrize("route", ROUTES)
def test_a_bc_changed_during_one_solve_is_refused_at_the_next_step(route, carriage):
    """Mutant: the per-step owner read removed, so the state reads only at `start`; every row then completes its budget.
    MultiPopulation runs one population, since its cross-coupling over two takes an HJB-FDM solver only (#1071)."""
    problems = [_problem()] if route == "MultiPopulation" else _problems(route)
    pairs = [_gfdm_pair(p, carriage) for p in problems]
    _changes_the_bc_after(pairs[0][0], problems, solves=1)
    with pytest.raises(ValueError, match=r"declares its"):
        _solve(route, problems, [h for h, _ in pairs], [f for _, f in pairs], sweeps=4, tol=1e-30)


def test_a_bc_changed_in_newtons_phase_is_refused_at_its_next_step():
    """The change is made after the post-warm-up residual, so only the Newton phase's own read can refuse it. Mutant:
    the read kept on warm-up sweeps only."""
    p = _problem()
    hjb, fp = _gfdm_pair(p, "both")
    # HJB solve 1 is the warm-up's sweep, solve 2 the residual judged after it
    _changes_the_bc_after(hjb, [p], solves=2)
    with pytest.raises(ValueError, match=r"declares its"):
        _solve("Newton", [p], [hjb], [fp], sweeps=6, tol=1e-30, picard_warmup=1)


# --- Newton's warm-up, and the order between the fields ---------------------------------------------------------------


def test_newtons_warm_up_sweep_makes_one_hjb_and_one_fp_solve():
    """Mutant: the warm-up's residual norm per sweep restored, which made 2 HJB and 2 FP solves per sweep."""
    p = _problem()
    hjb, fp = HJBFDMSolver(p), FPFDMSolver(p)
    counts = {"hjb": 0, "fp": 0}
    for solver, name, method in ((hjb, "hjb", "solve_hjb_system"), (fp, "fp", "solve_fp_system")):
        original = getattr(solver, method)

        def counted(*args, _original=original, _name=name, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        setattr(solver, method, functools.wraps(original)(counted))
    _solve("Newton", [p], [hjb], [fp], sweeps=1, picard_warmup=1)
    assert counts == {"hjb": 1, "fp": 1}


def test_block_jacobis_map_is_newtons_residual_map_and_differs_from_gauss_seidel_in_m():
    """One undamped sweep from the cold start, stopped by a met criterion. Mutant: Block Jacobi given the Gauss-Seidel
    order, which makes its M Gauss-Seidel's."""
    undamped = DampingPolicy.constant(1.0)
    results = {}
    for cls in (BlockJacobiIterator, BlockGaussSeidelIterator):
        p = _problem()
        results[cls] = cls(p, HJBFDMSolver(p), FPFDMSolver(p), damping=undamped).solve(
            max_iterations=1, tolerance=1e12, verbose=False
        )
    jacobi, gauss_seidel = results[BlockJacobiIterator], results[BlockGaussSeidelIterator]
    assert np.array_equal(jacobi.U, gauss_seidel.U)
    assert np.max(np.abs(jacobi.M - gauss_seidel.M)) > 1e-3
    p = _problem()
    newton = NewtonMFGSolver(p, HJBFDMSolver(p), FPFDMSolver(p), picard_warmup=1, damping=undamped)
    U0, M0 = newton.mfg_residual.state.start()
    _, F_HJB, F_FP = newton.mfg_residual.compute_residual(U0, M0, return_components=True)
    assert np.allclose(jacobi.U, U0 + F_HJB, rtol=0, atol=1e-12)
    assert np.allclose(jacobi.M, M0 + F_FP, rtol=0, atol=1e-12)


# --- Plan v1.1: the cases review 1 found ------------------------------------------------------------------------------


def test_newton_max_iterations_zero_with_budget_to_spare_names_its_bound():
    """Mutant: the Newton phase's early exit reading the solve's spent-budget reason whichever bound ran out."""
    p = _problem()
    result = _solve(
        "Newton", [p], [HJBFDMSolver(p)], [FPFDMSolver(p)], sweeps=30, picard_warmup=1, newton_max_iterations=0
    )
    assert result.iterations == 1
    assert _reason(result) == "Newton max iterations"


def test_a_callback_stop_in_newtons_phase_reads_that_iterates_inner_failures():
    """HJB-FDM capped at one Newton step, so its inner solves fail, and a tolerance every change meets: the criteria
    are met and the verdict is refused, so the callback runs, and stops. Mutant: the Newton phase not refreshing the
    inner-failure record, which reported converged=True over the failed steps (review 1, B1)."""
    p = _problem()
    hjb = HJBFDMSolver(p, max_newton_iterations=1, newton_tolerance=1e-14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = NewtonMFGSolver(p, hjb, FPFDMSolver(p), picard_warmup=0).solve(
            max_iterations=3, tolerance=1e12, verbose=False, iteration_callback=lambda *args: False
        )
    assert not result.converged
    assert _reason(result).startswith("callback_stopped (inner_hjb_not_converged"), _reason(result)


def test_a_callback_returning_false_stops_newtons_phase():
    """With no warm-up the callback's first call is at the Newton phase's first iterate. Mutant: the callback's reader
    removed from the Newton phase."""
    seen = []
    p = _problem()

    def callback(iteration, U, M, error_U, error_M):
        seen.append(iteration)
        return False

    result = _solve(
        "Newton", [p], [HJBFDMSolver(p)], [FPFDMSolver(p)], sweeps=6, picard_warmup=0, iteration_callback=callback
    )
    assert seen == [1]
    assert _reason(result).startswith("callback_stopped")


def test_a_bc_changed_before_newtons_post_warm_up_check_is_refused_there():
    """The BC changes during the warm-up's one sweep, and a tolerance between that sweep's change and the post-warm-up
    check's makes the check the evaluation that would end the solve. Mutant: the pipeline's owner read skipped at the
    check, which returned 'Converged during Picard warm-up' on a problem its owner refuses (review 1, M1)."""
    p = _problem()
    hjb, fp = _gfdm_pair(p, "both")
    probe = NewtonMFGSolver(p, hjb, fp, picard_warmup=1).solve(max_iterations=2, tolerance=1e-30, verbose=False)
    worst = [max(u, m) for u, m in zip(probe.metadata["l2distu_rel"], probe.metadata["l2distm_rel"], strict=True)]
    assert worst[1] < worst[0], worst
    tolerance = float(np.sqrt(worst[0] * worst[1]))
    p = _problem()
    hjb, fp = _gfdm_pair(p, "both")
    _changes_the_bc_after(hjb, [p], solves=1)
    with pytest.raises(ValueError, match=r"declares its"):
        NewtonMFGSolver(p, hjb, fp, picard_warmup=1).solve(max_iterations=6, tolerance=tolerance, verbose=False)


class _Unreadable:
    """A callable whose signature `inspect` refuses with ValueError, as a builtin's or an extension function's is."""

    def __init__(self, fn):
        self.fn = fn

    @property
    def __signature__(self):
        raise ValueError("no signature found")

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


# Graph's HJB hand-off reads no signature, so it has no unreadable arm to refuse (#2209 keeps that site)
HALVES = [(route, half) for route in ROUTES for half in ("HJB", "FP") if (route, half) != ("Graph", "HJB")]


@pytest.mark.parametrize(("route", "half"), HALVES)
def test_a_solver_whose_signature_cannot_be_read_is_refused_when_the_route_is_built(route, half):
    """Class line 5 (#2209; the user's ruling of 2026-11-26): a route reads its solvers' signatures through one reader,
    which refuses one `inspect` cannot read. Before it, T handed such an FP solver U positionally, into the drift, and
    neither the problem's source nor a volatility override. Mutant: the reader returning None on a failed read, as the
    four readers did at the base."""
    problems = _problems(route)
    hjbs, fps = [HJBFDMSolver(p) for p in problems], [FPFDMSolver(p) for p in problems]
    for solver in hjbs if half == "HJB" else fps:
        method = "solve_hjb_system" if half == "HJB" else "solve_fp_system"
        setattr(solver, method, _Unreadable(getattr(solver, method)))
    name = "HJBFDMSolver.solve_hjb_system" if half == "HJB" else "FPFDMSolver.solve_fp_system"
    with pytest.raises(
        ValueError, match=rf"{name}: the {half} solver's signature cannot be read \(ValueError: no signature found\)"
    ):
        _solve(route, problems, hjbs, fps, sweeps=1)


def _nan_hjb_at(problem, at: int) -> HJBFDMSolver:
    hjb = HJBFDMSolver(problem)
    solve, calls = hjb.solve_hjb_system, [0]

    @functools.wraps(solve)
    def bad(*args, **kwargs):
        U = np.array(solve(*args, **kwargs), dtype=float)
        calls[0] += 1
        if calls[0] == at:
            U[2, 3] = np.nan
        return U

    hjb.solve_hjb_system = bad
    return hjb


@pytest.mark.parametrize(
    ("stop", "measured"),
    [
        ("a callback stop at the first iterate", False),
        ("the HJB guard at the first iterate", False),
        ("a named stop at the first iterate", False),
        ("a callback stop after one step", True),
    ],
)
def test_newtons_cold_start_is_not_reported_as_measured(stop, measured):
    """Class line 6 (#1717; review 1's B6): with no warm-up, Newton's first iterate is the cold start, and a stop there
    publishes it; its mass is not a measurement of this solve. After one step the iterate is Newton's, and measured.
    Mutant: the flag decided by evaluations run rather than steps taken, which reported the cold start's drift as 0.0
    (the named stop and the HJB guard at 8598206f)."""
    p = _problem()
    hjb, fp, callback = HJBFDMSolver(p), FPFDMSolver(p), None
    if stop == "the HJB guard at the first iterate":
        hjb = _nan_hjb_at(p, 2)  # HJB solve 1 is the post-warm-up check's, at the same iterate
    elif stop == "a named stop at the first iterate":
        fp = _bad_fp(p, 1, "x1e3")  # under the magnitude: the line search fails
    else:
        calls = []

        def callback(*args):
            calls.append(args[0])
            return len(calls) < (1 if stop == "a callback stop at the first iterate" else 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _solve("Newton", [p], [hjb], [fp], sweeps=4, picard_warmup=0, iteration_callback=callback)
    assert not result.converged
    assert (result.mass_conservation_error is not None) is measured, f"{stop}: {result.mass_conservation_error!r}"


def _writes_into_its_input(solver, method: str):
    """After its solve, zero every array the call was handed, as a caller-supplied solver may."""
    solve = getattr(solver, method)

    @functools.wraps(solve)
    def writing(*args, **kwargs):
        out = np.array(solve(*args, **kwargs), dtype=float)
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, np.ndarray) and value.dtype.kind == "f":
                value[...] = 0.0
        return out

    setattr(solver, method, writing)
    return solver


ALIASING_ROWS = [(route, None) for route in ROUTES if route != "Newton"] + [("Newton", 3), ("Newton", 0), ("Newton", 1)]


@pytest.mark.parametrize("half", ["HJB", "FP"])
@pytest.mark.parametrize(("route", "warmup"), ALIASING_ROWS)
def test_a_solver_that_writes_into_its_input_leaves_the_solve_unchanged(route, warmup, half):
    """Class line 7 (the approver's ruling of 2026-11-27): a route reads the arrays it handed a solver after the call,
    so it hands copies (`base_mfg.handed`). The oracle is the same solve with the plain solver. Newton runs its warm-up
    only (picard_warmup 3), its post-warm-up check and phase (0), and a warm-up sweep then its phase (1), which reaches
    the finite-difference probes and the line search. Mutant: a hand-off passing its own arrays, which at 8598206f
    changed every route's result or raised."""
    problems = _problems(route)
    kw = {} if warmup is None else {"picard_warmup": warmup}

    def run(writing):
        hjbs, fps = [HJBFDMSolver(p) for p in problems], [FPFDMSolver(p) for p in problems]
        if writing:
            for solver in hjbs if half == "HJB" else fps:
                _writes_into_its_input(solver, "solve_hjb_system" if half == "HJB" else "solve_fp_system")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _solve(route, problems, hjbs, fps, sweeps=3, **kw)

    plain, writing = run(False), run(True)
    for k in range(len(problems)):
        assert np.array_equal(_published(writing, k, "U"), _published(plain, k, "U")), f"{route}, {half}: U moved"
        assert np.array_equal(_published(writing, k, "M"), _published(plain, k, "M")), f"{route}, {half}: M moved"


def test_newtons_result_carries_the_records_it_takes():
    """Mutant: Newton's result built without the driver's measure field and rate history (review 1, B3)."""
    p = _problem()
    result = _solve(
        "Newton", [p], [HJBFDMSolver(p)], [FPFDMSolver(p)], sweeps=3, track_measure_field=True, track_rate_history=True
    )
    assert result.metadata.get("measure_field") is not None
    assert result.metadata["learning_rate_history"] == [0.5, 0.5, 0.5]

"""One driver loop for the coupled routes (#3002, stage 2a): the pins its class lines owe.

Each test is a mutation-verified convention pin of a fact several routes decided on their own before the driver (AGENTS.md
§ Admission, class 1): each goes red under the mutant its docstring and the PR body name.

- **Class line 1, an iterate a solve cannot use** (#2997's folded item): a non-finite FP output stops with
  ``'diverged_nan'``, and one whose largest level mass passes 1e4 times its solve's first FP output's with
  ``'diverged_mass_blowup'``, on all seven routes and at Newton's post-warm-up check and Newton phase; the stop
  publishes the stopped evaluation's HJB and FP outputs. A source that fills the domain is not read as a blow-up. An
  update that leaves a non-finite iterate, a damping's or a Newton step's, stops with ``'diverged_nan'`` and publishes
  it; a non-finite warm start is refused where it is set.
- **Class line 2, the budget and its report** (ruling 2): ``max_iterations=0`` runs no sweep, a negative one is refused
  naming the route's class, and a spent budget reads ``"Maximum iterations reached"``, on all seven;
  ``newton_max_iterations`` bounds Newton's phase within the budget.
- **Class line 3, the callback's convention** (#2886, ruling 4): False stops, with the verdict read at that iterate;
  True and None continue. FictitiousPlay read the opposite.
- **Class line 4, the owner's refusals during one solve** (#3005): a BC changed during a solve, so that the problem
  declares two that differ, is refused at the next step on every route, whichever solver carries its own BC, Newton's
  post-warm-up check and phase included, and before each Newton step and the F of an exhausted budget.
- **Class line 5, what a route hands a solver** (#2209): a solver whose signature cannot be read, and an FP solver whose
  signature names neither ``potential_field`` nor ``drift_field``, are refused when the route is built, on every route.
- **Class line 6, whether a published density is measured** (#1717): Newton's cold start is not reported as measured,
  a stop after a Newton step is, and MultiPopulation reports each finite population's drift.
- **Class line 7, a solver writing into what it is handed** (#3009): the solve is unchanged, on every route and phase,
  for every method a route calls.
- **Newton's warm-up sweep** makes one HJB solve and one FP solve (ruling 6: ``picard_residuals`` is removed).
- **The order between the fields** is the sweep map's option: Block Jacobi's map is Newton's residual's, and differs
  from the Gauss-Seidel map in M only.

d = 1 expresses every one of these: they are stops, counts, a return convention and a source of BC, none directional.
"""

from __future__ import annotations

import functools
import logging
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
from mfgarchon.alg.numerical.coupling.fixed_point_utils import FPOutputRule
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
from mfgarchon.utils.numerical.nonlinear_solvers import NewtonRun

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


def _problem(centre: float = 0.3, *, initial_mass: float = 1.0, fp_source: float | None = None) -> MFGProblem:
    """The fixture: m(0) a Gaussian of mass ``initial_mass`` at ``centre``; ``fp_source``, when given, an FP source
    adding that much mass per unit time at every point."""
    xs = np.linspace(0.0, 1.0, 11)
    mass = np.trapezoid(np.exp(-40 * (xs - centre) ** 2), xs) / initial_mass
    source = {} if fp_source is None else {"source_term_fp": lambda x, m, v, t: fp_source * np.ones_like(m)}
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
        **source,
    )


def _solve(route, problems, hjbs, fps, *, sweeps, tol=1e-12, **kw):
    """One solve of ``route`` on its problems and solvers, ``sweeps`` its budget; ``damping`` and ``adjoint_mode``, when
    given, go to the route's constructor."""
    ctor = {key: kw.pop(key) for key in ("damping", "adjoint_mode") if key in kw}
    if route == "FixedPoint":
        # The pins run below HJB-FDM's default Newton tolerance; FixedPoint's warning about it (#1081) is the fixture's
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"HJB newton_tolerance", category=UserWarning)
            return FixedPointIterator(problems[0], hjbs[0], fps[0], **ctor).solve(
                max_iterations=sweeps, tolerance=tol, verbose=False, **kw
            )
    if route == "FictitiousPlay":
        return FictitiousPlayIterator(problems[0], hjbs[0], fps[0], **ctor).solve(
            max_iterations=sweeps, tolerance=tol, verbose=False, **kw
        )
    if route in ("Block GS", "Block Jacobi"):
        cls = BlockGaussSeidelIterator if route == "Block GS" else BlockJacobiIterator
        return cls(problems[0], hjbs[0], fps[0], **ctor).solve(
            max_iterations=sweeps, tolerance=tol, verbose=False, **kw
        )
    if route == "Newton":
        warmup = kw.pop("picard_warmup", 3)
        newton_max = kw.pop("newton_max_iterations", 20)
        solver = NewtonMFGSolver(
            problems[0], hjbs[0], fps[0], picard_warmup=warmup, newton_max_iterations=newton_max, **ctor
        )
        return solver.solve(max_iterations=sweeps, tolerance=tol, verbose=False, **kw)
    if route == "Graph":
        coupling = AdjacencyCoupling(np.array([[0.0, 1.0], [1.0, 0.0]]))
        return GraphMFGSolver(problems, coupling, hjbs, fps, max_iterations=sweeps, tolerance=tol, **ctor).solve()
    if route == "RegimeSwitching":
        config = RegimeSwitchingConfig(transition_matrix=np.array([[-0.5, 0.5], [0.5, -0.5]]))
        return RegimeSwitchingIterator(
            problems, config, hjbs, fps, max_iterations=sweeps, tolerance=tol, **ctor
        ).solve()
    multi = MultiPopulationProblem(populations=problems, population_names=["a", "b"][: len(problems)])
    return MultiPopulationIterator(multi, hjbs, fps, **ctor).solve(max_iterations=sweeps, tolerance=tol)


def _problems(route, **kw):
    if route in ("Graph", "RegimeSwitching", "MultiPopulation"):
        return [_problem(0.3, **kw), _problem(0.7, **kw)]
    return [_problem(0.3, **kw)]


#: Each route's class, which its refusals name
CLASSES = {
    "FixedPoint": "FixedPointIterator",
    "FictitiousPlay": "FictitiousPlayIterator",
    "Block GS": "BlockGaussSeidelIterator",
    "Block Jacobi": "BlockJacobiIterator",
    "Newton": "NewtonMFGSolver",
    "Graph": "GraphMFGSolver",
    "RegimeSwitching": "RegimeSwitchingIterator",
    "MultiPopulation": "MultiPopulationIterator",
}


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
    """Mutants: the driver's or a component loop's FP-output rule removed, and every NaN row, the level-0 one included,
    raises #2578's refusal of a non-finite change, since the measure runs before the update; its magnitude half
    removed, and no x1e5 row stops on ``'diverged_mass_blowup'``. The x1e5 rows grow from sweep 2, against sweep 1's
    output, the solve's first."""
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
    ("site", "at", "how", "reason"),
    [
        (site, at, how, reason)
        for site, at in (("the post-warm-up check", 1), ("the Newton phase", 2))
        for how, reason in (("nan", "diverged_nan"), ("nan at level 0", "diverged_nan"), ("inf", "diverged_nan"))
    ]
    + [("the Newton phase", 2, "x1e5", "diverged_mass_blowup")],
)
def test_an_fp_output_at_newtons_residual_sites_stops_it(site, at, how, reason):
    """With no warm-up, FP solve 1 is the post-warm-up check's and FP solve 2 the Newton phase's first iterate. FP
    solve 1 is the solve's first FP output, the magnitude's reference, so the x1e5 row grows from solve 2. Mutant: the
    pipeline's FP-output rule bypassed at Newton's residual map; the NaN rows then raise #2578's refusal, and the x1e5
    row stops at the line search."""
    p = _problem()
    hjb, fp = _recording_hjb(p), _bad_fp(p, at, how)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _solve("Newton", [p], [hjb], [fp], sweeps=4, picard_warmup=0)
    assert not result.converged
    assert _reason(result) == reason, f"{site}: {_reason(result)!r}"
    assert np.array_equal(result.M, fp.last, equal_nan=True), f"{site}: M is not the FP output"
    assert np.array_equal(result.U, hjb.last, equal_nan=True), f"{site}: U is not the HJB output"


def _level_of_mass_one(integrate):
    xs = np.linspace(0.0, 1.0, 11)
    level = np.exp(-40 * (xs - 0.3) ** 2)
    return level / float(np.asarray(integrate(level[None, :])).reshape(-1)[0])


def test_the_magnitude_reads_each_levels_mass_not_their_sum():
    """A solve whose every level keeps m(0)'s mass, over 20001 levels, against one level grown 2e4-fold. Mutant: the
    figure as the space-time sum, which a 2e4-fold level moves by a factor of 2 and does not stop (the sum against level
    0 was review 1's false stop at Nt = 9500)."""
    integrate = _problem().spatial_measure().integrate
    healthy = np.tile(_level_of_mass_one(integrate), (20001, 1))
    grown = healthy.copy()
    grown[7] *= 2e4
    rule = FPOutputRule([integrate])
    assert rule.stop([healthy], [healthy], site="test", iteration=0) is None
    assert rule.stop([healthy], [healthy], site="test", iteration=1) is None
    assert rule.stop([grown], [healthy], site="test", iteration=2) == "diverged_mass_blowup"


def test_the_magnitude_reference_is_each_components_first_output_and_a_zero_one_reads_nothing(mfg_caplog):
    """The user's ruling of 2026-11-27: each component's reference is its own first FP output's figure, and a reference
    of 0 leaves that component's magnitude unread, with a warning. Mutants: the components' figures summed, against
    which component 1's 1e5-fold growth is a 100-fold one; a reference of 0 read as one, which stops the second
    output."""
    integrate = _problem().spatial_measure().integrate
    level = np.tile(_level_of_mass_one(integrate), (6, 1))
    rule = FPOutputRule([integrate, integrate])
    assert rule.stop([level, 1e-3 * level], [level, level], site="test", iteration=0) is None
    assert rule.stop([level, 1e2 * level], [level, level], site="test", iteration=1) == "diverged_mass_blowup"

    logger = "mfgarchon.alg.numerical.coupling.fixed_point_utils"
    zero = FPOutputRule([integrate])
    with mfg_caplog.at_level(logging.WARNING, logger=logger):
        assert zero.stop([0.0 * level], [level], site="test", iteration=0) is None
    assert any("reads no magnitude for component(s) [0]" in m for m in mfg_caplog.messages), mfg_caplog.messages
    assert zero.stop([1e9 * level], [level], site="test", iteration=1) is None


@pytest.mark.parametrize("route", ROUTES)
def test_a_source_that_adds_mass_is_not_read_as_a_blow_up(route):
    """m(0) of mass 1e-5 and an FP source adding mass 1 per unit time: a healthy solve whose level mass grows 3e4-fold
    from m(0)'s (the text lane's N1 of review 2, where every route stopped at its second sweep). The reference is the
    solve's first FP output, which the source has already filled. Mutant: the reference taken from m(0)'s mass."""
    problems = _problems(route, initial_mass=1e-5, fp_source=1.0)
    hjbs, fps = [HJBFDMSolver(p) for p in problems], [FPFDMSolver(p) for p in problems]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _solve(route, problems, hjbs, fps, sweeps=4)
    assert "diverged" not in _reason(result), f"{route}: {_reason(result)!r}"
    largest = max(
        float(np.max(_problem().spatial_measure().integrate(_published(result, k, "M")))) for k in range(len(problems))
    )
    assert largest > 1e4 * 1e-5, f"{route}: the fixture's mass grew only to {largest:.3e}"


def _inf_at_the_first_update() -> DampingPolicy:
    """A damping whose U factor is 1 / log(k + 1), inf at sweep 0: T's outputs stay finite, the update does not."""
    return DampingPolicy.from_mapping(
        {"U": {"theta": 0.5, "schedule": lambda k: 1.0 / np.log(k + 1.0)}, "M": {"theta": 0.5}}
    )


@pytest.mark.parametrize("route", ROUTES)
def test_an_update_that_leaves_a_non_finite_iterate_stops_the_solve(route):
    """The post-update rule (review 2's code N1): an update can make a non-finite iterate of finite outputs, and the
    stop publishes that iterate as it is. Newton's row is its warm-up sweep. Mutant: the rule removed; FixedPoint and
    Block Gauss-Seidel then reported converged=True over a non-finite U at a tolerance of 1e12 (q6 at 0247587c)."""
    problems = _problems(route)
    hjbs, fps = [HJBFDMSolver(p) for p in problems], [FPFDMSolver(p) for p in problems]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _solve(route, problems, hjbs, fps, sweeps=3, tol=1e12, damping=_inf_at_the_first_update())
    assert not result.converged
    assert _reason(result) == "diverged_nan", f"{route}: {_reason(result)!r}"
    assert result.iterations == 1
    assert not all(np.all(np.isfinite(_published(result, k, "U"))) for k in range(len(problems)))


def test_a_newton_step_that_leaves_a_non_finite_iterate_stops_the_solve(monkeypatch):
    """The post-update rule after Newton's step: the stop publishes the iterate the step left. Mutant: the rule removed
    after the step; the next evaluation's HJB guard then publishes its own output instead."""
    left = {}
    step = NewtonRun.step

    def poisoning(self, *args, **kwargs):
        out = step(self, *args, **kwargs)
        if out is None:
            self.x = self.x.copy()
            self.x[-3] = np.nan  # an entry of M's last level
            left["x"] = self.x.copy()
        return out

    monkeypatch.setattr(NewtonRun, "step", poisoning)
    p = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _solve("Newton", [p], [HJBFDMSolver(p)], [FPFDMSolver(p)], sweeps=4, picard_warmup=0)
    assert _reason(result) == "diverged_nan", _reason(result)
    assert result.metadata["newton_iterations"] == 1
    size = result.U.size
    assert np.array_equal(result.U.ravel(), left["x"][:size]), "U is not the iterate the step left"
    assert np.array_equal(result.M.ravel(), left["x"][size:], equal_nan=True), "M is not the iterate the step left"


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


@pytest.mark.parametrize("route", ROUTES)
def test_a_negative_budget_is_refused(route):
    """The budget is a count of sweeps on every route (ruling 2; the text lane's N3 of review 2), and the refusal names
    the route's class (S1). Mutant: `validate_budget`'s refusal removed; Graph, RegimeSwitching and MultiPopulation
    accepted -1 at 0247587c."""
    problems = _problems(route)
    with pytest.raises(
        ValueError, match=rf"{CLASSES[route]}: max_iterations is a count of sweeps, an int >= 0, not -1"
    ):
        _solve(route, problems, [HJBFDMSolver(p) for p in problems], [FPFDMSolver(p) for p in problems], sweeps=-1)


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


def _counting(hjb):
    """Count the HJB solver's solves in ``hjb.solves``."""
    solve = hjb.solve_hjb_system
    hjb.solves = 0

    @functools.wraps(solve)
    def counted(*args, **kwargs):
        hjb.solves += 1
        return solve(*args, **kwargs)

    hjb.solve_hjb_system = counted
    return hjb


def test_a_bc_changed_by_the_callback_is_refused_before_newtons_step():
    """The owner is read before each Newton step, so the step's finite-difference probes and line-search trials run
    under it (review 2's text N4). The callback changes the BC at the Newton phase's first iterate. Mutant: the read
    before the step removed; the step then evaluates the map on the refused BC, 266 HJB solves at 0247587c, before the
    next iterate's read refuses it."""
    p = _problem()
    hjb, fp = _gfdm_pair(p, "both")
    _counting(hjb)
    at_change = []

    def callback(*args):
        p.components.boundary_conditions = dirichlet_bc(dimension=1, value=0.7)
        at_change.append(hjb.solves)

    with pytest.raises(ValueError, match=r"declares its"):
        _solve("Newton", [p], [hjb], [fp], sweeps=6, tol=1e-30, picard_warmup=0, iteration_callback=callback)
    assert at_change, "the callback never ran"
    assert hjb.solves == at_change[0], f"{hjb.solves - at_change[0]} HJB solves on the refused BC"


def test_a_bc_changed_inside_newtons_last_step_is_refused_before_the_last_f():
    """With one Newton step in the budget, the BC changes at the step's first HJB solve, a probe; the owner is read
    before the F of the exhausted budget. Mutant: that read removed, and the solve returns an iterate evaluated on the
    refused BC."""
    p = _problem()
    hjb, fp = _gfdm_pair(p, "both")
    solve, armed = hjb.solve_hjb_system, []

    @functools.wraps(solve)
    def changing(*args, **kwargs):
        if armed:
            p.components.boundary_conditions = dirichlet_bc(dimension=1, value=0.7)
        return solve(*args, **kwargs)

    hjb.solve_hjb_system = changing
    with pytest.raises(ValueError, match=r"declares its"):
        # Budget 2: the post-warm-up check, then one Newton iterate, its step and the exhausted budget's F
        _solve(
            "Newton",
            [p],
            [hjb],
            [fp],
            sweeps=2,
            tol=1e-30,
            picard_warmup=0,
            iteration_callback=lambda *a: armed.append(1),
        )


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


@pytest.mark.parametrize("route", ROUTES)
def test_an_fp_solver_whose_signature_names_neither_drift_keyword_is_refused(route):
    """The user's ruling of 2026-11-27 (q3): a route hands the value function by ``potential_field`` or ``drift_field``,
    so an FP solver taking neither is refused when the route is built, a ``(*args, **kwargs)`` wrapper included; the
    advice, naming the parameter ``potential_field``, solves as the plain solver does (RegimeSwitching hands every
    regime a source, so the wrapper names ``source_term`` too). Mutant: the refusal removed;
    the route then handed U positionally, into FP-FDM's drift (q3 at 0247587c)."""
    problems = _problems(route)

    def plain():
        return [HJBFDMSolver(p) for p in problems], [FPFDMSolver(p) for p in problems]

    def opaque(original):
        def wrapper(*args, **kwargs):
            return original(*args, **kwargs)

        return wrapper

    def named(original):
        def wrapper(M_initial, potential_field=None, source_term=None, **kwargs):
            return original(M_initial, potential_field=potential_field, source_term=source_term, **kwargs)

        return wrapper

    hjbs, fps = plain()
    for fp in fps:
        fp.solve_fp_system = opaque(fp.solve_fp_system)
    with pytest.raises(
        ValueError, match=r"names neither potential_field nor drift_field \(it reads \['args', 'kwargs'\]\)"
    ):
        _solve(route, problems, hjbs, fps, sweeps=2)

    hjbs, fps = plain()
    for fp in fps:
        fp.solve_fp_system = named(fp.solve_fp_system)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        advised = _solve(route, problems, hjbs, fps, sweeps=2)
        reference = _solve(route, problems, *plain(), sweeps=2)
    for k in range(len(problems)):
        assert np.array_equal(_published(advised, k, "M"), _published(reference, k, "M")), f"{route}: M differs"


@pytest.mark.parametrize("route", ONE_COMPONENT)
@pytest.mark.parametrize("field", ["U", "M"])
def test_a_non_finite_warm_start_is_refused_where_it_is_set(route, field):
    """The start is an iterate the solve carries forward, published as it is at a budget of 0 (amendment 1 to plan
    v1.2). Mutant: the refusal removed from `IterationState.set_warm_start`."""
    p = _problem()
    hjb, fp = HJBFDMSolver(p), FPFDMSolver(p)
    if route == "FixedPoint":
        solver = FixedPointIterator(p, hjb, fp)
    elif route == "FictitiousPlay":
        solver = FictitiousPlayIterator(p, hjb, fp)
    elif route == "Newton":
        solver = NewtonMFGSolver(p, hjb, fp)
    else:
        solver = (BlockGaussSeidelIterator if route == "Block GS" else BlockJacobiIterator)(p, hjb, fp)
    U, M = np.zeros((6, 11)), np.ones((6, 11))
    (U if field == "U" else M)[2, 3] = np.nan
    with pytest.raises(ValueError, match=rf"the warm start's {field} holds 1 non-finite value"):
        solver.set_warm_start_data((U, M))


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
        ("the HJB guard after one step", True),
    ],
)
def test_newtons_cold_start_is_not_reported_as_measured(stop, measured, monkeypatch):
    """Class line 6 (#1717; review 1's B6): with no warm-up, Newton's first iterate is the cold start, and a stop there
    publishes it; its mass is not a measurement of this solve. After one step the iterate is Newton's, and measured,
    whichever stop publishes it: the driver's `DensityRecord` decides for every stop (review 2's code N5). Mutants: the
    flag decided by evaluations run rather than steps taken, which reported the cold start's drift as 0.0 (the named
    stop and the HJB guard at 8598206f); the record never counting a Newton step, which reds the last two rows."""
    p = _problem()
    hjb, fp, callback = HJBFDMSolver(p), FPFDMSolver(p), None
    if stop == "the HJB guard at the first iterate":
        hjb = _nan_hjb_at(p, 2)  # HJB solve 1 is the post-warm-up check's, at the same iterate
    elif stop == "the HJB guard after one step":
        # The HJB solve after the first step that moved the iterate is the second iterate's evaluation
        armed, step = [], NewtonRun.step

        def arming(self, *args, **kwargs):
            out = step(self, *args, **kwargs)
            if out is None:
                armed.append(1)
            return out

        monkeypatch.setattr(NewtonRun, "step", arming)
        hjb = HJBFDMSolver(p)
        solve = hjb.solve_hjb_system

        @functools.wraps(solve)
        def poisoned(*args, **kwargs):
            U = np.array(solve(*args, **kwargs), dtype=float)
            if armed:
                U[2, 3] = np.nan
            return U

        hjb.solve_hjb_system = poisoned
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
    if stop == "the HJB guard after one step":
        assert _reason(result) == "diverged_nan", _reason(result)
    assert (result.mass_conservation_error is not None) is measured, f"{stop}: {result.mass_conservation_error!r}"


def test_multipopulation_reports_the_drift_of_each_finite_population():
    """A stop that publishes one population's non-finite density leaves that population's drift unmeasured, and the
    worst over populations unknown; the other population's drift stands (review 2's code N6). Mutant: every
    population's drift None whenever one density is not finite."""
    problems = _problems("MultiPopulation")
    fps = [FPFDMSolver(problems[0]), _bad_fp(problems[1], 2, "nan")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _solve("MultiPopulation", problems, [HJBFDMSolver(p) for p in problems], fps, sweeps=4)
    assert _reason(result) == "diverged_nan"
    drifts = [entry["mass_conservation_error"] for entry in result.metadata["per_component"]]
    assert isinstance(drifts[0], float), drifts
    assert drifts[1] is None, drifts
    assert result.mass_conservation_error is None


class _Writer:
    """A solver as a route reaches it: every method the route calls through this proxy zeroes, after the call, every
    array it was handed, as a caller-supplied solver may. It covers each method by construction, whatever its name;
    the solver's calls on itself do not pass through it. ``writes`` False forwards unchanged, the oracle's twin."""

    def __init__(self, inner, writes: bool) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_writes", writes)

    @property
    def __class__(self):  # isinstance reads the solver's class, as the routes' capability checks do
        return type(self._inner)

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not (callable(attr) and self._writes):
            return attr

        @functools.wraps(attr)
        def writing(*args, **kwargs):
            out = attr(*args, **kwargs)
            for value in list(args) + list(kwargs.values()):
                if isinstance(value, np.ndarray) and value.dtype.kind == "f" and value.flags.writeable:
                    value[...] = 0.0
            return out

        return writing

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


ALIASING_ROWS = (
    [(route, None, "off") for route in ROUTES if route != "Newton"]
    + [("Newton", 3, "off"), ("Newton", 0, "off"), ("Newton", 1, "off")]
    + [("Block GS", None, "jacobian_transpose"), ("Block Jacobi", None, "jacobian_transpose")]
)


@pytest.mark.parametrize("half", ["HJB", "FP"])
@pytest.mark.parametrize(("route", "warmup", "adjoint"), ALIASING_ROWS)
def test_a_solver_that_writes_into_its_input_leaves_the_solve_unchanged(route, warmup, adjoint, half):
    """Class line 7 (the approver's ruling of 2026-11-27): a route reads the arrays it handed a solver after the call,
    so it hands copies (`base_mfg.call_handed`), whatever the method. The oracle is the same solve through a proxy that
    does not write. Newton runs its warm-up only (picard_warmup 3), its post-warm-up check and phase (0), and a warm-up
    sweep then its phase (1), which reaches the finite-difference probes and the line search; Block runs with
    ``adjoint_mode`` off and on, which calls `build_linearized_operator` and `solve_fp_step_adjoint_mode`. Mutant: a
    hand-off passing its own arrays, which at 8598206f changed every route's result or raised, and at 0247587c
    changed Block's adjoint results (review 2's code N3)."""
    problems = _problems(route)
    kw = {} if warmup is None else {"picard_warmup": warmup}
    if adjoint != "off":
        kw["adjoint_mode"] = adjoint

    def run(writes):
        hjbs, fps = [HJBFDMSolver(p) for p in problems], [FPFDMSolver(p) for p in problems]
        if half == "HJB":
            hjbs = [_Writer(solver, writes) for solver in hjbs]
        else:
            fps = [_Writer(solver, writes) for solver in fps]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _solve(route, problems, hjbs, fps, sweeps=3, **kw)

    plain, writing = run(False), run(True)
    for k in range(len(problems)):
        assert np.array_equal(_published(writing, k, "U"), _published(plain, k, "U")), f"{route}, {half}: U moved"
        assert np.array_equal(_published(writing, k, "M"), _published(plain, k, "M")), f"{route}, {half}: M moved"


class _ReportsNegative(FPFDMSolver):
    """An FP-FDM solver that reports on the density a solve returns, as the weak-form family does (#2845): the coupled
    route hands it the result's M after the solve."""

    reports_negative_density = True

    def negative_density_of(self, M):
        return {"minimum": float(np.min(M))}

    def warn_negative_density(self, report):
        pass


@pytest.mark.parametrize("route", ROUTES)
def test_a_solver_that_writes_into_the_density_it_reports_on_leaves_the_result_unchanged(route):
    """Class line 7: `_report_negative_density` hands each FP solver the density the result returns, after the solve,
    and a census that could not follow its loop missed the call. Mutant: that call handed the result's own M, which a
    writing solver then zeroes in the result."""
    problems = _problems(route)

    def run(writes):
        fps = [_Writer(_ReportsNegative(p), writes) for p in problems]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _solve(route, problems, [HJBFDMSolver(p) for p in problems], fps, sweeps=2)

    plain, writing = run(False), run(True)
    for k in range(len(problems)):
        assert np.array_equal(_published(writing, k, "M"), _published(plain, k, "M")), f"{route}: M moved"


def test_newtons_result_carries_the_records_it_takes():
    """Mutant: Newton's result built without the driver's measure field and rate history (review 1, B3)."""
    p = _problem()
    result = _solve(
        "Newton", [p], [HJBFDMSolver(p)], [FPFDMSolver(p)], sweeps=3, track_measure_field=True, track_rate_history=True
    )
    assert result.metadata.get("measure_field") is not None
    assert result.metadata["learning_rate_history"] == [0.5, 0.5, 0.5]

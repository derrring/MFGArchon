"""One driver loop for the coupled routes (#3002, stage 2a): the pins its class lines owe.

Each test is a labelled pin of a fact several routes decided on their own before the driver (AGENTS.md § Admission,
class 3), and each goes red under the mutant the PR body names beside it.

- **Class line 1, an FP output a solve cannot use** (#2997's folded item): a non-finite one stops with
  ``'diverged_nan'`` and one whose mass passes the magnitude check with ``'diverged_mass_blowup'``, on all seven routes.
  Before the driver only FixedPoint stopped; the six others raised #2578's refusal or ran their budget out.
- **Class line 2, the budget and its report** (ruling 2): ``max_iterations=0`` runs no sweep, and a spent budget reads
  ``"Maximum iterations reached"``, on all seven; ``newton_max_iterations`` bounds Newton's phase within the budget.
- **Class line 3, the callback's convention** (#2886, ruling 4): False stops, with the verdict read at that iterate;
  True and None continue. FictitiousPlay read the opposite.
- **Class line 4, the owner's refusals during one solve** (#3005): a BC changed during a solve, so that the problem
  declares two that differ, is refused at the next step on every route, whichever solver carries its own BC, Newton's
  phase included.
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
    """An FP-FDM solver whose output goes bad from its ``at``-th solve: one entry NaN or +inf, or the whole output
    times 1e5, finite (#1489's blow-up)."""
    fp = FPFDMSolver(problem)
    solve, calls = fp.solve_fp_system, [0]

    @functools.wraps(solve)
    def bad(*args, **kwargs):
        M = np.array(solve(*args, **kwargs), dtype=float)
        calls[0] += 1
        if calls[0] >= at:
            if how == "nan":
                M[3, 4] = np.nan
            elif how == "inf":
                M[3, 4] = np.inf
            else:
                M = M * 1e5
        return M

    fp.solve_fp_system = bad
    return fp


@pytest.mark.parametrize(
    ("case", "at", "how", "reason"),
    [
        ("NaN at sweep 1", 1, "nan", "diverged_nan"),
        ("NaN at sweep 2", 2, "nan", "diverged_nan"),
        ("+inf at sweep 2", 2, "inf", "diverged_nan"),
        ("x1e5 from sweep 2", 2, "x1e5", "diverged_mass_blowup"),
    ],
)
@pytest.mark.parametrize("route", ROUTES)
def test_an_fp_output_the_solve_cannot_use_stops_it_on_every_route(route, case, at, how, reason):
    """Mutant: the driver's or a component loop's `fp_output_stop` call removed, or its magnitude half; the NaN rows
    then raise #2578's refusal of a non-finite change, and the x1e5 rows run their budget out."""
    problems = _problems(route)
    hjbs = [HJBFDMSolver(p) for p in problems]
    fps = [_bad_fp(p, at, how) for p in problems]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _solve(route, problems, hjbs, fps, sweeps=4)
    assert not result.converged
    assert _reason(result) == reason, f"{route}, {case}: {_reason(result)!r}"
    assert result.iterations == at


# --- Class line 2 ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
def test_a_budget_of_zero_runs_no_sweep_and_a_spent_budget_says_so(route):
    """Mutant: `max_iterations or 100` restored (FixedPoint, FictitiousPlay run 48 and 100 sweeps at 0), or a sweep's
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
def test_a_callback_returning_false_stops_on_every_route(route, returned, stops):
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

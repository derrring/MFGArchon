"""The GFDM FP path stops instead of renormalising over a clip (#1683).

It clipped, renormalised to the initial mass, and warned only above 1% drift. Every
configuration therefore returned a final mass of exactly 1.0000 -- including one measured
to clip **61%** of the present mass at a single step. Reporting perfect conservation over
that is the defect, not the diagnostic that was missing.

Migrating this path broke **no existing test**, which is the other half of the finding: a
public solver whose plausible configurations fabricate most of their mass had no coverage
of that behaviour at all. These are that coverage.

Two mechanisms drive it, and they call for opposite changes -- measured on a 21-point
grid, `sigma=0.5` clips 61% at `dt*D/dx^2 = 2.5` (five times the explicit-diffusion
limit), while `sigma=0.1` with a steep drift clips 9.6% at `dt*D/dx^2 = 0.1`, where the
driver is advection. The remedy text names both rather than guessing which one bound.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

N = 21
NT = 10
T = 0.5


def _build(sigma, Nt=NT, **solver_kw):
    """Same construction as `_solver`, with Nt exposed for the refinement-sensitive tests."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=grid,
        Nt=Nt,
        T=T,
        volatility=sigma,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-30 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )
    return FPGFDMSolver(problem, collocation_points=np.linspace(0, 1, N).reshape(-1, 1), **solver_kw)


#: The clip-gate tests below pin the clip gate, so they keep the solve going past the mass-drift gate
#: (#2512 S5), which stops these drifting configurations first by default.
_WARN = {"mass_drift": "warn"}


def _solver(sigma, **solver_kw):
    return _build(sigma, **{**_WARN, **solver_kw})


def _inputs(drift_scale):
    x = np.linspace(0, 1, N)
    m0 = np.exp(-30 * (x - 0.5) ** 2)
    m0 /= m0.sum()
    return m0, np.tile(drift_scale * (x - 0.5) ** 2, (NT + 1, 1))


def test_a_diffusion_limited_configuration_stops():
    """sigma=0.5 gives dt*D/dx^2 = 2.5, five times the explicit limit.

    It used to return final mass 1.0000 over a 61% clip -- a configuration a user would
    reasonably pick, reporting perfect conservation.
    """
    m0, drift = _inputs(1.0)
    with pytest.raises(ValueError, match="would fabricate"):
        _solver(0.5).solve_fp_system(m0, drift)


def test_an_advection_driven_configuration_stops():
    """sigma=0.1 with a steep drift: dt*D/dx^2 = 0.1, so diffusion is not the binding limit."""
    m0, drift = _inputs(25.0)
    with pytest.raises(ValueError, match="would fabricate"):
        _solver(0.1).solve_fp_system(m0, drift)


def test_the_remedy_names_the_lever_that_works_and_disowns_the_one_that_does_not():
    """Review measured both of the first version's suggestions and neither helped.

    "Reduce dt" is worse than useless here -- refining dt at fixed h makes the mass drift
    grow monotonically (2.79 -> 8.73 over Nt = 10..1280). "Add diffusion" never reaches the
    threshold on the advection-driven configuration and turns back up. The lever that does
    move it, `upwind_scheme`, went unmentioned. A remedy that names only non-levers sends
    the reader to spend an afternoon refining a grid.
    """
    m0, drift = _inputs(1.0)
    with pytest.raises(ValueError) as exc:
        _solver(0.5).solve_fp_system(m0, drift)
    message = str(exc.value)
    assert "GFDM FP solve: at t_idx=" in message
    assert "upwind_scheme" in message
    assert "1752" in message
    assert "Do NOT reduce dt" in message, "refining dt silences this gate; the message must say so"
    assert "2.5e+09" in message, "name the number the refinement leads to, not just the direction"


def test_a_configuration_with_no_negatives_at_all_still_runs():
    """The gate must not stop a solve that never goes negative.

    Named for what it does. The first version called this "a converging configuration" and
    "the régime this path is usable in", which review measurement contradicted: the very
    next test asserts this same run fabricates 179% of its mass, and refining either dt or h
    makes it worse. It converges to nothing; its density merely stays positive
    (min +8.5e-05).

    That also bounds what this test can guard. With no negatives anywhere it exercises only
    `mass_fabricated_by_clip`'s `if not negatives.any(): return 0.0` early return, so it
    would stay green under any threshold down to zero. It is a smoke check, not the
    threshold guard -- `test_the_threshold_is_not_satisfiable_by_a_marginal_clip` is that.
    """
    m0, drift = _inputs(5.0)
    result = _solver(0.3).solve_fp_system(m0, drift)
    assert np.all(np.isfinite(result))
    assert result.min() >= 0.0


def test_the_scheme_does_not_conserve_mass_and_now_says_so(record_property):
    """Records #1752: removing the renormalisation exposed a defect larger than the clip.

    This configuration clips **nothing** across all ten steps, so no positivity repair is
    involved -- and its mass still goes 1.000000 -> 2.794967, a 179% gain. The per-step
    `M *= mass_initial / mass_current` was not masking the clip; it was masking the
    scheme. Every configuration returned exactly the initial mass because it was forced
    to.

    The assertion is the measurement, not the desired behaviour. It is written to fail if
    the drift **improves**, so fixing #1752 cannot land silently: a conservative
    discretisation would bring this near 1.0 and turn this test red, which is when it
    should be deleted.
    """
    m0, drift = _inputs(5.0)
    result = _solver(0.3).solve_fp_system(m0, drift)
    final = float(result[-1].sum())
    record_property("gfdm_final_mass", final)
    assert final > 2.0, (
        f"final mass {final:.6f} -- if this dropped toward {float(m0.sum()):.1f} the scheme "
        f"became conservative and #1752 is fixed; delete this test rather than relax it"
    )


def test_the_drift_is_reported_at_warning_level(mfg_caplog):
    """The renormalisation's removal left this line as the only signal for the drift.

    It was `logger.debug`, which is off by default -- a 179% mass error that nothing
    printed. Pinned because a diagnostic nobody reads is the same failure as no
    diagnostic, and log levels are the kind of thing a later edit lowers without noticing.
    """
    import logging

    m0, drift = _inputs(5.0)
    with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.fp_solvers.fp_gfdm"):
        _solver(0.3).solve_fp_system(m0, drift)
    assert mfg_caplog.records, "the drift was not reported at WARNING or above"
    assert "1752" in mfg_caplog.messages[0], "the message must name the issue tracking the defect"


def test_the_remedy_does_not_tell_a_stabilised_solve_to_stabilise():
    """The first fix for this interpolated `upwind_scheme` and then ignored it.

    On a solve already using `'linear'` it printed "upwind_scheme is 'linear'. 'none' leaves
    the flux divergence unstabilised ... 'linear' or 'exponential' measurably reduce it" --
    naming a mechanism the caller is not in and prescribing what they already did. Caught in
    re-review, and unpinned by my own fix until this test: mutating the branch to always take
    the 'none' text left all eight other tests green.
    """
    n_x, n_t = 21, 10
    x = np.linspace(0, 1, n_x)
    m0 = np.exp(-30 * (x - 0.5) ** 2)
    m0 /= m0.sum()
    drift = np.tile(25.0 * (x - 0.5) ** 2, (n_t + 1, 1))

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n_x], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=grid,
        Nt=n_t,
        T=T,
        volatility=0.1,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-30 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )
    solver = FPGFDMSolver(problem, collocation_points=np.linspace(0, 1, n_x).reshape(-1, 1), upwind_scheme="linear")
    with pytest.raises(ValueError) as exc:
        solver.solve_fp_system(m0, drift)
    message = str(exc.value)
    assert "already 'linear'" in message, "the message must acknowledge the scheme the caller is on"
    assert "is not enough here" in message
    assert "'linear' or 'exponential' measurably reduce it" not in message, (
        "do not prescribe the stabilisation the caller already enabled"
    )


def test_a_clip_far_below_one_percent_still_stops_the_solve():
    """Pins the THRESHOLD, which the large-clip configurations above do not.

    My first attempt here copied the FDM sibling's guard -- assert the reported percentage
    is above 1% -- and claimed it pinned the threshold. Measured, it does not: at
    `MAX_CLIP_MASS_FABRICATION = 0.02`, six orders looser than shipped, that assertion and
    every other one in this file stays green, because a 61% clip is above 1% either way.

    This configuration is the discriminating one. It fabricates so little that the message
    rounds it to 0.000%, and it must still stop -- because the campaign's premise is that
    there is no interesting régime between round-off (~1e-15) and a failed scheme, so
    anything measurably above round-off is the scheme. Loosening the threshold to any
    percent-scale value turns this red.
    """
    n_t = 640
    x = np.linspace(0, 1, N)
    m0 = np.exp(-30 * (x - 0.5) ** 2)
    m0 /= m0.sum()
    drift = np.tile(25.0 * (x - 0.5) ** 2, (n_t + 1, 1))

    with pytest.raises(ValueError) as exc:
        _build(sigma=0.1, Nt=n_t, **_WARN).solve_fp_system(m0, drift)
    percent = float(str(exc.value).split("would fabricate")[1].split("%")[0])
    assert percent < 0.01, (
        f"message reported {percent}% -- this test is only a threshold pin while the clip "
        f"here is far below the percent scale; pick a finer Nt if the scheme changed"
    )


def test_refining_the_timestep_silences_the_gate_while_the_answer_gets_worse():
    """Records the campaign invariant's structural limit. Recorded, not fixed.

    `fabricated = |sum(negatives)| / sum(positives)` is scale-invariant AND evaluated per
    step, so refining dt shrinks what any one step can fabricate whether or not the answer
    improves. On this configuration the observable falls monotonically to exactly zero --
    nothing goes negative at all -- while the final mass climbs seven orders:

        Nt=10    max fabricated 9.591e-02   final mass 8.40e+02   raises
        Nt=640   max fabricated 3.396e-05   final mass 1.70e+09   raises
        Nt=2560  max fabricated 0.000e+00   final mass 2.55e+09   PASSES

    The configuration pinned below is a stronger instance found in re-review: N=41, Nt=640,
    sigma=0.5, drift=50 fabricates **nothing** and returns a final mass of 1.06e+23. It is
    pinned instead of the Nt=2560 case because it is 13 orders further past the assertion and
    does not depend on sitting past a refinement boundary -- the Nt=2560 case was measured to
    flip to raising somewhere between Nt=900 and Nt=1100, which is a 2.6x margin rather than a
    structural one.

    No threshold closes it **here**: any fixed value can be driven below on this
    configuration. Whether that generalises to the gate's other callers is open -- the
    attempt to reproduce it at the FDM time-stepping site during review produced a null with
    no working positive control, so the claim is scoped to this site rather than to the
    invariant. The failure is adversarial in shape wherever it occurs, because the natural
    response to the gate firing is to refine the timestep, and here that silences the gate and
    makes the answer worse -- which is why the remedy string tells the reader not to.

    Asserted so nobody later reads a passing gate as "the solve is healthy", and so the drift
    WARNING is not mistaken for redundant with the gate: only the pair covers this.
    """
    n_t = 640
    n_x = 41
    x = np.linspace(0, 1, n_x)
    m0 = np.exp(-30 * (x - 0.5) ** 2)
    m0 /= m0.sum()
    drift = np.tile(50.0 * (x - 0.5) ** 2, (n_t + 1, 1))

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n_x], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=grid,
        Nt=n_t,
        T=T,
        volatility=0.5,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-30 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )
    points = np.linspace(0, 1, n_x).reshape(-1, 1)
    result = FPGFDMSolver(problem, collocation_points=points, **_WARN).solve_fp_system(m0, drift)

    # No assertion on result.min(): `clip_nonnegative_or_raise` returns `np.maximum(density, 0)`
    # into every row, so non-negativity is imposed by the gate and cannot fail. Asserting it
    # would look like a check and be a tautology.
    assert float(result[-1].sum()) > 1e18, (
        f"final mass {float(result[-1].sum()):.3e}: this configuration is meant to diverge past "
        f"1e+20 while fabricating nothing, which is the blind spot being recorded"
    )
    # The mass-drift gate (#2512 S5) is the other half of the pair, and by default it stops this.
    with pytest.raises(ValueError, match="mass ratio"):
        FPGFDMSolver(problem, collocation_points=points).solve_fp_system(m0, drift)


# =============================================================================
# The mass-drift gate (#2512 S5)
# =============================================================================
#
# The conserved quantity is the integral of m with the geometry's measure, checked only when the
# collocation points are the grid's nodes. An unweighted sum is a different functional: on the exact mode
# 1 + 0.5 cos(2 pi x) e^{-D (2 pi)^2 t} it drifts 1.37% at 21 uniform points (trapezoid: 3e-16), and the
# first version of this gate measured it. On other points only a gross blow-up check runs.


def _diffusion_solver(n_x, n_t, *, points=None, **solver_kw):
    """Pure diffusion (no drift) on [0, 1], no-flux walls, sigma = 0.3, T = 0.5."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n_x], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=grid,
        Nt=n_t,
        T=T,
        volatility=0.3,
        components=MFGComponents(
            m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        ),
    )
    x = np.linspace(0, 1, n_x) if points is None else points
    return grid, x, FPGFDMSolver(problem, collocation_points=x.reshape(-1, 1), **solver_kw)


def test_a_drifting_solve_stops_at_the_first_step_past_the_tolerance():
    """The 179% configuration above, at the default: it stops, and says where and by how much."""
    m0, drift = _inputs(5.0)
    warn = _solver(0.3)
    masses = np.array([warn.problem.geometry.integrate(row) for row in warn.solve_fp_system(m0, drift)])
    first = int(np.argmax(np.abs(masses / masses[0] - 1.0) > 1e-2))
    assert first > 0, "the warn-mode run must cross the tolerance for this test to mean anything"
    with pytest.raises(ValueError) as exc:
        _build(0.3).solve_fp_system(m0, drift)
    message = str(exc.value)
    assert f"at step {first}" in message, message
    assert "mass ratio" in message
    assert "mass_drift" in message, "the message must name the keyword that keeps the solve going"


def test_a_conserving_configuration_passes():
    """cos(2 pi x) is symmetric about x = 1/2, so the boundary-stencil leaks at the two walls add rather
    than cancel (an antisymmetric mode such as cos(pi x) conserves by symmetry alone). Pure diffusion,
    41 points, Nt 160: 0.44% (1.075% at 21 / Nt 40, 0.20% at 81 / Nt 640 -- O(h))."""
    grid, x, solver = _diffusion_solver(41, 160)
    m0 = 1 + 0.5 * np.cos(2 * np.pi * x)
    m0 /= grid.integrate(m0)
    result = solver.solve_fp_system(m0, np.zeros((161, 41)))
    assert abs(grid.integrate(result[-1]) - 1.0) < 1e-2


def test_a_density_that_leaks_mass_stops():
    """The operator leaks at the boundary stencil even without drift, and not always at O(h): a bump at
    x = 0.3 loses 2.7% at 81 points (3.3% at 21, 2.6% at 41; Nt 640 / 40 / 160). A LOSS, so the gate
    must take |drift|."""
    grid, x, solver = _diffusion_solver(81, 640)
    m0 = np.exp(-50 * (x - 0.3) ** 2)
    m0 /= grid.integrate(m0)
    with pytest.raises(ValueError, match=r"mass ratio 0\.98"):
        solver.solve_fp_system(m0, np.zeros((641, 81)))


def test_a_source_that_moves_the_mass_is_not_stopped():
    """Under a source the law is d/dt (integral of m) = integral of S: a constant source adds T*S*|domain|
    of mass on purpose, and the budget counts it, so a correct solve passes and lands on the budget."""
    grid, x, solver = _diffusion_solver(21, 40)
    m0 = 1 + 0.5 * np.cos(np.pi * x)
    m0 /= grid.integrate(m0)
    result = solver.solve_fp_system(
        m0, np.zeros((41, 21)), source_term=lambda t, xx: np.full(np.asarray(xx).shape[0], 0.1)
    )
    assert grid.integrate(result[-1]) == pytest.approx(1.0 + T * 0.1, abs=1e-9)


def test_the_budget_closes_on_a_pure_source_to_round_off():
    """The budget must add S_k by the integrator's own rule: forward Euler at t_k. A constant density
    under a spatially constant S(t) = 1 + t is moved by the source alone (the GFDM Laplacian of a
    constant is 1.9e-13), so the budget closes to round-off and a 1e-12 tolerance passes. Summing S at
    t_{k+1} instead is off by dt (S(T) - S(0)) = 6.25e-3 of a 1.6 mass and must raise."""
    grid, _x, solver = _diffusion_solver(21, 40, mass_drift_tolerance=1e-12)
    result = solver.solve_fp_system(
        np.ones(21), np.zeros((41, 21)), source_term=lambda t, xx: np.full(np.asarray(xx).shape[0], 1.0 + t)
    )
    dt = T / 40
    assert grid.integrate(result[-1]) == pytest.approx(1.0 + sum(dt * (1.0 + k * dt) for k in range(40)), abs=1e-12)


@pytest.mark.parametrize("on_grid", [True, False], ids=["grid_nodes", "jittered"])
def test_a_draining_source_is_not_stopped(on_grid):
    """A source that removes 95% of the mass is legitimate. On grid nodes the budget follows it. On
    scattered points the lower bound's reference is sum|m_0| minus the most the source can drain, 5% of
    sum|m_0| here, and the drained density sits at that reference."""
    from mfgarchon.utils.numerical import gross_mass_excursion

    x0 = np.linspace(0, 1, 21)
    if not on_grid:
        x0[1:-1] += 1e-3 * np.random.default_rng(0).uniform(-1, 1, 19)
    _, x, solver = _diffusion_solver(21, 40, points=None if on_grid else x0)
    m0 = 1 + 0.5 * np.cos(np.pi * x)
    result = solver.solve_fp_system(m0, np.zeros((41, 21)), source_term=lambda t, xx: -0.95 * m0 / T)
    if not on_grid:
        final = float(np.abs(result[-1]).sum())
        assert gross_mass_excursion(final, float(np.abs(m0).sum())) is not None, (
            "against sum|m_0| alone this level is below the band, so the test reaches the drain-adjusted reference"
        )


def test_on_scattered_points_only_the_gross_check_runs_and_it_is_said_once(mfg_caplog):
    """No measure, so no conservation check -- said once, at construction, not once per solve."""
    import logging

    rng = np.random.default_rng(0)
    points = np.sort(np.concatenate([[0.0, 1.0], rng.uniform(0.02, 0.98, 19)]))
    with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.fp_solvers.fp_gfdm"):
        _, x, solver = _diffusion_solver(21, 40, points=points)
        m0 = np.exp(-50 * (x - 0.3) ** 2)
        solver.solve_fp_system(m0, np.zeros((41, 21)))
        solver.solve_fp_system(m0, np.zeros((41, 21)))
    messages = [m for m in mfg_caplog.messages if "measure" in m]
    assert len(messages) == 1, messages
    assert "only a gross check on sum|m| runs" in messages[0]


def _blowup(**solver_kw):
    """Review 2 of #2526: 21 nodes jittered by 1e-3, sigma 0.1, drift 25 (x - 1/2)^2, Nt = 2560. Left to
    run (mass_drift="warn") it returns a density of sum|m| 2.53e+09 x the initial, finite and positive."""
    n, n_t = 21, 2560
    x = np.linspace(0, 1, n)
    x[1:-1] += 1e-3 * np.random.default_rng(0).uniform(-1, 1, n - 2)
    solver = FPGFDMSolver(_build(0.1, Nt=n_t).problem, collocation_points=x.reshape(-1, 1), **solver_kw)
    m0 = np.exp(-30 * (x - 0.5) ** 2)
    return solver, m0 / m0.sum(), np.tile(25.0 * (x - 0.5) ** 2, (n_t + 1, 1))


def test_a_blow_up_on_scattered_points_is_stopped_by_the_gross_check():
    import re

    solver, m0, drift = _blowup()
    with pytest.raises(ValueError, match="above the bound") as exc:
        solver.solve_fp_system(m0, drift)
    factor = float(re.search(r"sum\|m\| is (\S+) times", str(exc.value)).group(1))
    assert factor > 10.0, "print the factor with enough digits to be past the band edge it crossed"


@pytest.mark.parametrize("source", [0.0, 0.01], ids=["source_zero", "source_0.01"])
@pytest.mark.parametrize("on_grid", [True, False], ids=["grid_nodes", "jittered"])
def test_a_source_term_does_not_silence_the_check(on_grid, source):
    """Review 3 of #2526: "has a source" exempted every check, so these four returned ~2.5e+09 x their
    mass with no signal, where main warned. Grid nodes: the budget. Jittered: the gross upper bound."""
    if on_grid:
        solver = _build(0.1, Nt=2560)
        m0, drift = _inputs(25.0)
        drift = np.tile(drift[0], (2561, 1))
    else:
        solver, m0, drift = _blowup()
    with pytest.raises(ValueError, match="to its budget" if on_grid else "above the bound"):
        solver.solve_fp_system(m0, drift, source_term=lambda t, xx: np.full(np.asarray(xx).shape[0], source))


def _r4(on_grid, **solver_kw):
    """Review 4 of #2526: the real operator under an outward drift 10 (x - 1/2) drains sum|m| to 0.019 of
    its start with nothing going negative (sigma 0.3, Nt 160, 21 points)."""
    x = np.linspace(0, 1, 21)
    if not on_grid:
        x[1:-1] += 1e-3 * np.random.default_rng(0).uniform(-1, 1, 19)
    solver = FPGFDMSolver(_build(0.3, Nt=160).problem, collocation_points=x.reshape(-1, 1), **solver_kw)
    return solver, np.exp(-30 * (x - 0.5) ** 2), np.tile(10.0 * (x - 0.5), (161, 1))


def _r3(on_grid, **solver_kw):
    if on_grid:
        solver = _build(0.1, Nt=2560, **solver_kw)
        m0, drift = _inputs(25.0)
        return solver, m0, np.tile(drift[0], (2561, 1))
    return _blowup(**solver_kw)


@pytest.mark.parametrize("mode", ["raise", "warn"])
@pytest.mark.parametrize("on_grid", [True, False], ids=["grid_nodes", "jittered"])
@pytest.mark.parametrize("case", [_r3, _r4], ids=["review3_blowup", "review4_vanish"])
def test_no_source_and_a_zero_source_are_one_case(case, on_grid, mode):
    """The class, not an instance: S = 0 is what "no source" means, so the gate must not tell them apart.
    Reviews 3 and 4 each found a check that a source identically zero switched off."""
    import logging

    solver_none, m0, drift = case(on_grid, mass_drift=mode)
    solver_zero, _, _ = case(on_grid, mass_drift=mode)
    outcomes = []
    for solver, kwargs in (
        (solver_none, {}),
        (solver_zero, {"source_term": lambda t, xx: np.zeros(np.asarray(xx).shape[0])}),
    ):
        lines: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda record, lines=lines: lines.append(record.getMessage())
        logger = logging.getLogger("mfgarchon.alg.numerical.fp_solvers.fp_gfdm")
        logger.addHandler(handler)
        try:
            result = ("returned", np.asarray(solver.solve_fp_system(m0, drift, **kwargs)))
        except ValueError as exc:
            result = ("raised", str(exc))
        finally:
            logger.removeHandler(handler)
        outcomes.append((result, lines))
    (r_none, l_none), (r_zero, l_zero) = outcomes
    assert r_none[0] == r_zero[0]
    if r_none[0] == "raised":
        assert r_none[1] == r_zero[1]
    else:
        np.testing.assert_array_equal(r_none[1], r_zero[1])
    assert l_none == l_zero
    if mode == "raise":
        assert r_none[0] == "raised", "every one of these four setups is a failed solve the gate must stop"


def test_a_small_positive_source_does_not_waive_the_lower_bound():
    """Review 4: under source 0.01 the drained density was not reported. A positive source cannot
    remove mass, so the lower bound keeps its full reference."""
    solver, m0, drift = _r4(on_grid=False)
    with pytest.raises(ValueError, match="below the bound"):
        solver.solve_fp_system(m0, drift, source_term=lambda t, xx: np.full(np.asarray(xx).shape[0], 0.01))


def test_the_drift_is_measured_against_the_source_scale():
    """A source adding 20x the mass carries the operator's leak with it: |mass - budget| is 3.7% of the
    initial mass but 0.18% of the scale initial mass + |source| added, which is what the tolerance is on."""
    grid, x, solver = _diffusion_solver(41, 160)
    m0 = 1 + 0.5 * np.cos(2 * np.pi * x)
    m0 /= grid.integrate(m0)
    solver.solve_fp_system(m0, np.zeros((161, 41)), source_term=lambda t, xx: 20.0 * m0 / T)


def test_the_gross_upper_reference_counts_the_positive_source():
    """On scattered points a source adding 20x the mass takes sum|m| to 21x its start, legitimately; the
    upper bound is 10 x (sum|m_0| + the positive source added), not 10 x sum|m_0|."""
    x0 = np.linspace(0, 1, 21)
    x0[1:-1] += 1e-3 * np.random.default_rng(0).uniform(-1, 1, 19)
    _, x, solver = _diffusion_solver(21, 40, points=x0)
    m0 = 1 + 0.5 * np.cos(np.pi * x)
    solver.solve_fp_system(m0, np.zeros((41, 21)), source_term=lambda t, xx: 20.0 * m0 / T)


def test_the_gross_sum_takes_absolute_values():
    """One step of explicit diffusion at dt*D/h^2 = 50 turns a spike into an alternating-sign field: the
    signed sum is 4.72, inside the band, while sum|m| is 142. The gross check must see that before the
    clip does."""
    x = np.linspace(0, 1, 21)
    x[1:-1] += 1e-3 * np.random.default_rng(0).uniform(-1, 1, 19)
    solver = FPGFDMSolver(_build(1.0, Nt=2).problem, collocation_points=x.reshape(-1, 1))
    m0 = np.zeros(21)
    m0[10] = 1.0
    with pytest.raises(ValueError, match="above the bound"):
        solver.solve_fp_system(m0, np.zeros((3, 21)))


def test_the_gross_baseline_takes_absolute_values():
    """A sign-indefinite m0 (sum 0.48, sum|m| 9.04): against its cancelled sum the first step would look
    like an 18x blow-up. Against sum|m0| it is in band, and the clip gate is what stops it."""
    x = np.linspace(0, 1, 21)
    x[1:-1] += 1e-3 * np.random.default_rng(0).uniform(-1, 1, 19)
    solver = FPGFDMSolver(_build(0.3, Nt=10).problem, collocation_points=x.reshape(-1, 1))
    m0 = np.exp(-50 * (x - 0.3) ** 2) - 0.9 * np.exp(-50 * (x - 0.7) ** 2)
    with pytest.raises(ValueError) as exc:
        solver.solve_fp_system(m0, np.zeros((11, 21)))
    assert "Clipping it to zero" in str(exc.value)
    assert "gross" not in str(exc.value)


def test_in_warn_mode_the_gross_check_warns_and_returns(mfg_caplog):
    import logging

    solver, m0, drift = _blowup(mass_drift="warn")
    with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.fp_solvers.fp_gfdm"):
        solver.solve_fp_system(m0, drift)
    assert any("gross blow-up check" in m and "Not a conservation check" in m for m in mfg_caplog.messages)


def test_in_warn_mode_a_vanishing_density_is_reported_too(mfg_caplog):
    """The worst excursion is the farthest from the band in either direction. A stand-in operator
    (dm/dt = -c m) decays sum|m| to 0.0084 of its start; tracking only growth would never report it."""
    import logging

    x = np.linspace(0, 1, 21)
    x[1:-1] += 1e-3 * np.random.default_rng(0).uniform(-1, 1, 19)
    solver = FPGFDMSolver(_build(0.3, Nt=40).problem, collocation_points=x.reshape(-1, 1), mass_drift="warn")
    solver.gfdm_operator.laplacian = lambda m: -200.0 * np.asarray(m)
    with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.fp_solvers.fp_gfdm"):
        solver.solve_fp_system(np.exp(-30 * (x - 0.5) ** 2), np.zeros((41, 21)))
    assert any("gross blow-up check" in m and "0.00844" in m for m in mfg_caplog.messages)


def test_an_exact_mode_on_a_strongly_non_uniform_cloud_passes_the_gross_check():
    """Spacing ratio 3.73 (x = (4^s - 1)/3): the unweighted sum|m| moves only to 0.976 of its start, far
    inside [0.1, 10]. That pins this specimen only: the band's lower edge sits well below 0.976. It is
    not a proof that no adaptive cloud false-alarms."""
    s_ = np.linspace(0, 1, 21)
    points = (np.exp(np.log(4.0) * s_) - 1) / 3.0
    _, x, solver = _diffusion_solver(21, 200, points=points)
    solver.solve_fp_system(1 + 0.5 * np.cos(2 * np.pi * x), np.zeros((201, 21)))


def test_the_2d_measure_is_the_geometrys_own_integral():
    """C-order points on an 11 x 7 grid: the measure is geometry.integrate on the reshaped array, and on
    a field asymmetric in both axes it equals the nested trapezoid. Square cells, so no stencil degenerates."""
    from mfgarchon.geometry.boundary import no_flux_bc as nf

    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[11, 7], boundary_conditions=nf(dimension=2))
    problem = MFGProblem(
        geometry=grid,
        Nt=4,
        T=T,
        volatility=0.3,
        components=MFGComponents(
            m_initial=lambda x, y: np.full_like(np.asarray(x, dtype=float), 1 / 0.6),
            u_terminal=lambda x, y: 0.0,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        ),
    )
    X, Y = np.meshgrid(*grid.coordinates, indexing="ij")
    solver = FPGFDMSolver(problem, collocation_points=np.column_stack([X.ravel(), Y.ravel()]))
    f = np.exp(X) * (1 + Y**2)
    nested = np.trapezoid(np.trapezoid(f, grid.coordinates[1], axis=1), grid.coordinates[0])
    assert solver._mass_measure is not None
    assert solver._mass_measure(f.ravel()) == pytest.approx(nested, rel=1e-12)


def test_the_remedy_does_not_tell_a_stabilised_solve_to_stabilise_either():
    """The mass-drift remedy follows upwind_scheme, as the clip gate's does."""
    m0, drift = _inputs(5.0)
    with pytest.raises(ValueError) as exc:
        _build(0.3, upwind_scheme="linear").solve_fp_system(m0, drift)
    message = str(exc.value)
    assert "upwind_scheme='linear' reduces the leak without removing it" in message
    assert "'linear' or 'exponential' reduces" not in message


def test_the_1822_surface_runs_fp_gfdm_in_warn_mode():
    """That surface measures the BC residual; an xfail(raises=ValueError) there cannot tell this gate
    from the clip gate, so the kwarg it injects is pinned here."""
    from test_periodic_capability_invariant_1822 import _solver_kwargs

    kwargs = _solver_kwargs(FPGFDMSolver, np.linspace(0.0, 1.0, 11), 11)
    assert kwargs.get("mass_drift") == "warn"


def test_a_non_finite_density_stops_whatever_mass_drift_says():
    m0, _ = _inputs(5.0)
    drift = np.full((NT + 1, N), np.inf)
    with pytest.raises(ValueError, match="not finite"):
        _solver(0.3).solve_fp_system(m0, drift)


def test_the_tolerance_is_the_opt_out():
    """mass_drift_tolerance is the documented route: the 179% run (2.44 under the measure) passes at 2."""
    m0, drift = _inputs(5.0)
    _build(0.3, mass_drift_tolerance=2.0).solve_fp_system(m0, drift)


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 0.0])
def test_an_invalid_tolerance_is_refused(bad):
    with pytest.raises(ValueError, match="mass_drift_tolerance"):
        _build(0.3, mass_drift_tolerance=bad)


def test_mass_drift_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mass_drift"):
        _build(0.3, mass_drift="ignore")

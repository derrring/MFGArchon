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
# collocation points are the grid's nodes. An unweighted sum is a different functional: on an exact
# solution its drift reached 1.75% at 21 uniform points, and the first version of this gate measured it.


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
    """Not symmetric about x = 1/2, so the boundary stencils on the two walls do not cancel: cos(2 pi x),
    pure diffusion, 41 points, drifts 0.44% (1.08% at 21 points, 0.20% at 81 -- O(h))."""
    grid, x, solver = _diffusion_solver(41, 160)
    m0 = 1 + 0.5 * np.cos(2 * np.pi * x)
    m0 /= grid.integrate(m0)
    result = solver.solve_fp_system(m0, np.zeros((161, 41)))
    assert abs(grid.integrate(result[-1]) - 1.0) < 1e-2


def test_a_density_that_leaks_mass_stops():
    """The operator leaks at the boundary stencil even without drift, and not always at O(h): a bump at
    x = 0.3 loses 2.7% at 81 points (3.3% at 21, 2.6% at 41). A LOSS, so the gate must take |drift|."""
    grid, x, solver = _diffusion_solver(81, 640)
    m0 = np.exp(-50 * (x - 0.3) ** 2)
    m0 /= grid.integrate(m0)
    with pytest.raises(ValueError, match=r"mass ratio 0\.98"):
        solver.solve_fp_system(m0, np.zeros((641, 81)))


def test_a_source_that_moves_the_mass_is_not_stopped():
    """Mass is conserved only without a source, so the gate does not apply with one: a constant
    source adds T*S*|domain| of mass on purpose, and stopping that would fail a correct solve."""
    grid, x, solver = _diffusion_solver(21, 40)
    m0 = 1 + 0.5 * np.cos(np.pi * x)
    m0 /= grid.integrate(m0)
    result = solver.solve_fp_system(
        m0, np.zeros((41, 21)), source_term=lambda t, xx: np.full(np.asarray(xx).shape[0], 0.1)
    )
    assert grid.integrate(result[-1]) > 1.03, "the source should have added mass"


def test_on_scattered_points_the_check_is_not_performed_and_says_so(mfg_caplog):
    """No measure, so no mass check and no drift warning of the wrong functional -- once, a statement."""
    import logging

    rng = np.random.default_rng(0)
    points = np.sort(np.concatenate([[0.0, 1.0], rng.uniform(0.02, 0.98, 19)]))
    _, x, solver = _diffusion_solver(21, 40, points=points)
    m0 = np.exp(-50 * (x - 0.3) ** 2)
    with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.fp_solvers.fp_gfdm"):
        solver.solve_fp_system(m0, np.zeros((41, 21)))
    messages = [m for m in mfg_caplog.messages if "mass" in m]
    assert len(messages) == 1, messages
    assert "not performed" in messages[0]


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

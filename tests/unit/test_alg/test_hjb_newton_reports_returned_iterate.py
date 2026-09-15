(
    r"""A non-converged inner HJB Newton must REPORT the residual of the iterate it returns (#1745).

The warning printed the PREVIOUS iterate's residual while returning the current one -- measured,
up to 8.5x better than the truth. A diagnostic that is accurate about a quantity nobody receives
is the shape of silent-wrong this issue exists to remove.

Returning the best-seen iterate instead was tried and REVERTED, and the reason is worth keeping:
it made 90 of 96 inner solves on the multi-population fixture return their input unchanged. On a
configuration where Newton never improves on its starting point, "the best iterate" IS the
starting point, so the solve becomes an identity map -- which silently erased the cross-coupling
that `test_hjb_sees_cross_density_bug_1157` exists to detect. Handing back an unmoved iterate is
a worse failure than handing back a moved one, because nothing downstream can tell.
"""
    ""
)

from __future__ import annotations

import re
import warnings

import numpy as np

from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _stiff_problem():
    """sigma=0.05 on 21 points, solved below with a Newton budget of 2: the inner solve cannot converge.

    A configuration where every inner solve converges cannot exercise the non-converged return
    path at all, and that path is what this file covers. The default budget converged here once the
    non-decrease guard went (#1878), so the budget is what keeps the path exercised now.
    """
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(
        geometry=grid,
        Nt=10,
        T=1.0,
        sigma=0.05,
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


def _solve(problem, max_iterations):
    return problem.solve(
        hjb_solver=HJBFDMSolver(problem, max_newton_iterations=2),
        fp_solver=FPFDMSolver(problem),
        max_iterations=max_iterations,
        verbose=False,
    )


def _solve_capturing_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _solve(_stiff_problem(), max_iterations=3)
    messages = [str(w.message) for w in caught if "did not converge" in str(w.message)]
    residuals = [float(m) for msg in messages for m in re.findall(r"residual ([0-9.e+-]+) against", msg)]
    return result, residuals


def test_the_configuration_still_exercises_the_non_converged_path():
    """Positive control. If this stops warning, every assertion below is vacuous.

    It would stop warning if the scheme improved -- which would be good news, and would mean this
    file needs a new configuration rather than a relaxed assertion.
    """
    _, residuals = _solve_capturing_warnings()
    assert residuals, "no inner solve failed to converge; this file no longer tests anything"


def test_the_returned_value_function_is_finite_and_does_not_diverge():
    """The returned U is not required to be a root -- the warning says it is not.

    It is required not to be a blow-up, which is what a Newton loop handing back iterates that never
    converge could eventually produce. A blow-up keeps growing with the sweep count; a large-but-settled answer
    stops. That is what is asserted, because it is what "diverged" means.

    ~~``abs(U).max() < 1e3``~~ [CHANGED 2026-08-12] was a magnitude threshold, and it was calibrated
    to one fixture rather than bounding anything. Measured on `main` at the time it was replaced, it
    is already violated by configurations it was never run on -- ``2.945e+03`` at Nx=41/10 sweeps and
    ``1.652e+03`` at this very Nx with 10 sweeps instead of 3 -- so it passed by choice of sweep
    count, not because 1e3 held.

    What moved it here: #1900 removed a post-solve overwrite that forced ``u[0] = u[1]`` at both
    walls on every timestep, artificially flattening the solution there. Without that clamp the value
    function is legitimately larger on this stiff fixture -- and it still settles
    (``1.625e+04 -> 1.622e+04 -> 1.624e+04`` at Nx=81 over 5/10/15 sweeps), while the outer
    iteration converges better than before (``err_U`` 4.5e-03 -> 2.15e-04 at 15 sweeps).
    """
    result, residuals = _solve_capturing_warnings()
    U = np.asarray(result.U, dtype=float)
    assert np.isfinite(U).all(), "the returned value function contains non-finite entries"

    # Divergence, not magnitude: run the same fixture for 4x the sweeps and require the norm to
    # settle rather than grow with the iteration count.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        longer = _solve(_stiff_problem(), max_iterations=12)
    short_norm = float(np.abs(U).max())
    long_norm = float(np.abs(np.asarray(longer.U, dtype=float)).max())
    assert np.isfinite(long_norm), "the value function is non-finite after 12 sweeps"
    assert long_norm < 8.0 * max(short_norm, 1.0), (
        f"||U||_inf grew from {short_norm:.3e} at 3 sweeps to {long_norm:.3e} at 12: the loop is "
        f"returning a diverging iterate rather than a settled one, while reporting a residual of at "
        f"most {max(residuals):.3e}"
    )


def test_a_stopped_solve_reports_the_iterate_it_is_returning():
    """Drive the loop with a controlled residual sequence and read what the warning says.

    Integration-level thresholds do NOT pin this -- tried twice. A magnitude threshold passes
    either way (both iterates are large on the stiff fixture), and comparing populations of
    warned-vs-all residuals compares different sets. The difference is visible only inside the
    loop, so this drives the loop directly.

    Sequence 10, 1, 5, 7 with a budget of 3 steps: three steps are taken, and the fourth call only
    measures the iterate they reached. That iterate is returned, so the warning must say 7 after 3
    steps. Reporting the previous residual says 5, an iterate the caller never receives; returning an
    iterate without measuring it reports a residual nobody computed.
    """
    import mfgarchon.alg.numerical.hjb_solvers.base_hjb as bh

    sequence = iter([10.0, 1.0, 5.0, 7.0])
    original = bh.newton_hjb_step

    def fake(U_current, *args, **kwargs):
        return np.asarray(U_current, dtype=float) + 1.0, 0.0, next(sequence)

    problem = _stiff_problem()
    bh.newton_hjb_step = fake
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            returned = bh.solve_hjb_timestep_newton(
                U_n_plus_1_from_hjb_step=np.zeros(21),
                U_k_n_from_prev_picard=np.zeros(21),
                M_density_at_n_plus_1=np.ones(21),
                problem=problem,
                t_idx_n=0,
                max_newton_iterations=3,
                newton_tolerance=1e-12,
            )
    finally:
        bh.newton_hjb_step = original

    messages = [str(w.message) for w in caught if "did not converge" in str(w.message)]
    assert messages, "a solve that spent its budget above tolerance did not report it"
    reported = float(re.findall(r"residual ([0-9.e+-]+) against", messages[0])[0])
    assert reported == 7.0, (
        f"warning reported {reported}, not the returned iterate's residual. The iterate being "
        f"returned has residual 7.0 -- the report must describe what the caller receives"
    )
    assert "after 3 Newton steps" in messages[0], messages[0]
    np.testing.assert_array_equal(returned, np.full(21, 3.0))


def test_a_solve_stopped_by_a_non_finite_step_counts_the_steps_it_took():
    """The warning counts the steps the returned iterate took, not the budget (#1878).

    The second step comes back non-finite with a budget of 5, so the returned iterate is one step from
    the start. The warning used to print the budget whatever stopped the solve.
    """
    import mfgarchon.alg.numerical.hjb_solvers.base_hjb as bh

    calls = {"n": 0}
    original = bh.newton_hjb_step

    def fake(U_current, *args, **kwargs):
        calls["n"] += 1
        step = np.full(21, np.nan) if calls["n"] == 2 else np.asarray(U_current, dtype=float) + 1.0
        return step, 0.0, 10.0

    problem = _stiff_problem()
    bh.newton_hjb_step = fake
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bh.solve_hjb_timestep_newton(
                U_n_plus_1_from_hjb_step=np.zeros(21),
                U_k_n_from_prev_picard=np.zeros(21),
                M_density_at_n_plus_1=np.ones(21),
                problem=problem,
                t_idx_n=0,
                max_newton_iterations=5,
                newton_tolerance=1e-12,
            )
    finally:
        bh.newton_hjb_step = original

    messages = [str(w.message) for w in caught if "did not converge" in str(w.message)]
    assert messages, "a solve stopped by a non-finite step did not report it"
    assert "non-finite" in messages[0], messages[0]
    assert "after 1 Newton steps" in messages[0], messages[0]

"""The residual of `x = G(x)` is ||G(x) - x||, and damping must not scale it (#1684 item 6).

`FixedPointSolver` measured ||x_updated - x_current||, where x_updated is the value AFTER
under-relaxation. That equals `relaxation * ||G(x) - x||`, so the reported residual shrinks with
the damping factor while the actual fixed-point error does not. Turning damping down made
anything converge.

Measured on `G(x) = 0.9x + 1` (fixed point 10) from x0 = 0 at tolerance 1e-2, before the fix:

    relaxation   converged   iters   reported     x returned   true |G(x)-x|
       0.1         False      200    1.353e-02     8.660203      0.133980
       0.01        True         2    9.990e-03     0.019990      0.998001
       0.001       True         1    1.000e-03     0.001000      0.999900

Success at x = 0.02 and failure at x = 8.66, on the same problem. The 0.001 run declared
convergence after ONE iteration, having barely moved off the initial guess.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.utils.numerical.nonlinear_solvers import FixedPointSolver

# A contraction with a fixed point far from the start, so "barely moved" and "converged" are
# unmistakably different states.
A, B, X_STAR, X0 = 0.9, 1.0, 10.0, 0.0


def _G(x):
    return A * x + B


@pytest.mark.parametrize("relaxation", [1.0, 0.5, 0.1, 0.01, 0.001])
def test_the_first_residual_is_undamped_at_every_damping(relaxation: float):
    """The law, pinned where the iterate is known exactly.

    The solver measures the residual at the iterate it is LEAVING, not the one it returns, so
    the only point at which the expected value is known without reconstructing the update is the
    first: x_current is x0, and |G(x0) - x0| = |1 - 0| = 1 whatever the damping.

    Under the defect this came back as `relaxation` -- 1e-3 at relaxation 0.001, three orders
    below the truth.
    """
    solver = FixedPointSolver(relaxation=relaxation, tolerance=1e-300, max_iterations=1, norm_type="absolute")
    _x, info = solver.solve(_G, np.array(X0))

    expected = abs(_G(X0) - X0)
    assert info.residual_history[0] == pytest.approx(expected, rel=1e-12), (
        f"relaxation={relaxation}: first residual {info.residual_history[0]:.6e} against |G(x0)-x0| "
        f"= {expected:.6e} -- a ratio of {info.residual_history[0] / expected:.6f}"
    )


def test_heavier_damping_cannot_buy_convergence():
    """The behavioural half: the defect made `converged` a function of the damping factor.

    Under it, relaxation 0.01 returned True at x = 0.02 while relaxation 0.1 returned False at
    x = 8.66. Whatever the iteration budget, a run that stops nearer the fixed point must not be
    the one that reports failure.
    """
    results = {}
    for relaxation in (0.1, 0.01, 0.001):
        solver = FixedPointSolver(relaxation=relaxation, tolerance=1e-2, max_iterations=200, norm_type="absolute")
        x, info = solver.solve(_G, np.array(X0))
        results[relaxation] = (info.converged, abs(float(np.atleast_1d(x)[0]) - X_STAR))

    for relaxation, (converged, distance) in results.items():
        assert not converged, (
            f"relaxation={relaxation} reported convergence {distance:.3f} away from the fixed "
            f"point, within a 200-iteration budget it cannot make in"
        )

    # And the ordering is the honest one: less damping gets closer in the same budget.
    assert results[0.1][1] < results[0.01][1] < results[0.001][1]


def test_undamped_and_damped_agree_on_the_residual_at_the_same_point():
    """Relaxation changes the PATH, not the measurement.

    Two solvers stopped at the same iterate must report the same residual. Under the defect they
    differed by the ratio of their damping factors -- which is what made the number meaningless
    as a comparison between runs.
    """
    one_step = {}
    for relaxation in (1.0, 0.25):
        solver = FixedPointSolver(relaxation=relaxation, tolerance=1e-300, max_iterations=1, norm_type="absolute")
        _x, info = solver.solve(_G, np.array(X0))
        one_step[relaxation] = info.residual_history[0]

    # Both start from x0, so both measure |G(x0) - x0| = |1 - 0| = 1 regardless of damping.
    assert one_step[1.0] == pytest.approx(abs(_G(X0) - X0))
    assert one_step[0.25] == pytest.approx(one_step[1.0]), (
        f"the first-iterate residual moved with damping: {one_step[0.25]:.6e} vs {one_step[1.0]:.6e}"
    )

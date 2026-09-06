"""`flux_diagnostics.compute_mass_conservation_error` measures the grid measure (Issue #2260).

The function computed `sum(M, axis=spatial) * prod(spacing)` -- the rectangle rule -- and
published `is_conservative: drift_pct < 1.0` from it. #2145 established that the rectangle rule
is not the mass on this library's endpoint-inclusive grid: the two boundary nodes each own half a
cell, not a full one, so the rectangle rule over-counts by `spacing * (M[..., 0] + M[..., -1]) / 2`
per axis. Before #2258 the SL/CN family conserved `sum(m)` by construction, so the bug agreed with
the family it was measuring by accident; after #2258 moved that family onto the grid measure, the
disagreement became every SL solve.

The measured incident: `FPSLSolver`, driven transport, exactly conservative in the grid measure
(drift 2.021e-14), reported by this function as 2.6692% drift and `is_conservative=False`.

Oracle discipline: the "expected" values below come from `scipy.integrate.trapezoid`, called
directly on hand-built coordinates -- not from `quadrature_weights_nd`, which is what the fixed
function now calls internally. Comparing against the same owner the fix uses would make the
oracle tautological; `scipy.integrate.trapezoid` is an independent implementation of the same
mathematical object (see `test_mass_conservation_error_1672.py` for the same discipline applied to
`SolverResult.mass_conservation_error`, a different code path this issue does not touch).
"""

from __future__ import annotations

import pytest

import numpy as np
from scipy.integrate import trapezoid

from mfgarchon.utils.numerical.flux_diagnostics import compute_mass_conservation_error


def _wall_marching_trajectory(nx: int, nt: int, wall_from: float, wall_to: float) -> tuple[np.ndarray, float]:
    """A density trajectory exactly conservative in the TRAPEZOID measure, with mass concentrating
    at the wall over time -- the shape that makes rectangle and trapezoid disagree.

    Uniform in the interior, only the last node varies; each slice is rescaled by an
    independently-computed trapezoid integral so its true mass is exactly 1. A density that is
    ~0 at both walls (the naive way to build a "conservative" fixture) makes rectangle and
    trapezoid agree by construction, since they differ only in the weight given to the boundary
    nodes -- this fixture is deliberately NOT that case.
    """
    x = np.linspace(0.0, 1.0, nx)
    M = np.zeros((nt, nx))
    for t, wall_value in enumerate(np.linspace(wall_from, wall_to, nt)):
        profile = np.ones(nx)
        profile[-1] = wall_value
        M[t] = profile / trapezoid(profile, x)
    return M, x[1] - x[0]


class TestTheGridMeasureIsWhatIsReported:
    def test_a_trajectory_exactly_conservative_in_the_grid_measure_is_reported_as_conservative(self):
        """The external oracle. Every slice has trapezoid-mass exactly 1 by construction
        (verified independently below); the function must agree to within its own rounding,
        not to within the ~5% the rectangle rule would have reported on this fixture."""
        M, h = _wall_marching_trajectory(nx=21, nt=6, wall_from=1e-3, wall_to=6.65)
        x = np.linspace(0.0, 1.0, 21)

        oracle_mass = np.array([trapezoid(row, x) for row in M])
        assert oracle_mass == pytest.approx(1.0, abs=1e-12), "the fixture construction itself is broken"

        result = compute_mass_conservation_error(M, spacing=h)
        assert result["mass_drift_percent"] < 1e-9, (
            f"a trajectory exactly conservative in the grid measure reported "
            f"{result['mass_drift_percent']!r}% drift (#2260)"
        )
        assert result["is_conservative"] is True or bool(result["is_conservative"]) is True

    def test_the_pre_fix_rectangle_rule_would_have_flagged_this_same_trajectory(self):
        """A labelled defect pin: demonstrates what #2260 WAS, on the identical fixture the
        oracle test above proves is fine. If a future edit reintroduces the rectangle rule
        (`sum(M, axis=1) * h` in place of the grid-measure weights), the oracle test above
        starts failing -- this test exists to say WHY, by reproducing the wrong number here
        rather than only in a commit message.

        Retirement condition: this trips (by construction, it always passes -- it is testing
        the OLD formula directly, not the library) only in the sense that it documents a
        number nobody should see again from the library's own function. Delete it if a second
        rectangle-rule regression test elsewhere in the suite already covers this fixture.
        """
        M, h = _wall_marching_trajectory(nx=21, nt=6, wall_from=1e-3, wall_to=6.65)
        rectangle_mass = np.sum(M, axis=1) * h
        rectangle_drift_pct = abs(rectangle_mass[-1] - rectangle_mass[0]) / rectangle_mass[0] * 100
        assert rectangle_drift_pct > 1.0, (
            "the rectangle rule no longer disagrees with the trapezoid rule on this fixture -- "
            "the fixture has stopped expressing the defect #2260 was about"
        )

    def test_a_genuinely_non_conservative_trajectory_is_still_flagged(self):
        """The positive control for the fix itself: it must not have become so lenient that it
        stops detecting a real defect. Without this, a fix that always returns
        `is_conservative=True` would pass the oracle test above vacuously."""
        nx, h = 21, 1.0 / 20
        M = np.tile(np.ones(nx), (6, 1))
        M[-1] *= 2.0  # mass genuinely doubles

        result = compute_mass_conservation_error(M, spacing=h)
        assert result["mass_drift_percent"] > 50.0
        assert not result["is_conservative"]

    def test_the_measure_names_itself(self):
        """#2260's own closing question: 'is_conservative should say which measure it is a
        verdict about, or stop being a verdict.'"""
        M = np.tile(np.ones(11), (3, 1))
        result = compute_mass_conservation_error(M, spacing=0.1)
        assert result["measure"] == "grid"

    def test_two_dimensional_trajectories_use_the_same_measure(self):
        """The function accepts a spacing tuple for nD; the fix must not be 1D-only. A uniform
        field on `[0,1]^2` has known trapezoid mass 1.0 by direct construction (`1/((nx-1)(ny-1))`
        per cell contribution is not how trapezoid weights work at the corners, so this fixture
        is built from the same `_wall_marching_trajectory` logic collapsed to a product of two
        1D profiles, each independently verified by `scipy.integrate.trapezoid`)."""
        nx, ny = 11, 9
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(0.0, 1.0, ny)
        hx, hy = x[1] - x[0], y[1] - y[0]

        profile_x = np.ones(nx)
        profile_y = np.ones(ny)
        mass_x = trapezoid(profile_x, x)
        mass_y = trapezoid(profile_y, y)
        field = np.outer(profile_x / mass_x, profile_y / mass_y)  # oracle mass exactly 1

        M = np.tile(field, (4, 1, 1))
        result = compute_mass_conservation_error(M, spacing=(hx, hy))
        assert result["mass_drift_percent"] < 1e-9
        assert result["initial_mass"] == pytest.approx(1.0, rel=1e-9)

r"""The U convergence metric must ignore an additive function of time (#1873).

`u` reaches the density only through `grad(u)`; the drift is `-grad(u)/c`. So adding a
constant to a whole time slice changes nothing the coupled system can observe -- but it
changes `||u||` and `||du||`, and on a real problem it dominates both. Measured on the 1-D
smoke fixture: 99.77% of `||U||`'s energy is that mode, and 99.71% of the per-sweep change.

Both directions of the error matter, which is why this is measured rather than argued. The
mode inflates the ABSOLUTE error, so a converged solve is reported as not converged. It
also inflates the DENOMINATOR of the relative error, so a solve is reported as converged
for a reason unrelated to the solve: at the sweep where the raw relative error first passed
1e-6, the gauge-free part was still moving at 8.0e-6 and the drift field at 9.7e-6.

M is deliberately not projected. Its level is mass -- observable, conserved, and with no
additive freedom to remove -- so one test here pins that the asymmetry is intended.

No test name in this file may contain "large", "slow" or "benchmark": tests/conftest.py:100
marks any such test slow and the gate excludes slow. The first version of the pure-gauge
case below was called "..._not_a_large_one" and was silently absent from the authoritative
suite -- in the same change that filed #1875 for that rule.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.utils.convergence.convergence_metrics import calculate_l2_convergence_metrics

NT, NX = 11, 21
DX, DT = 1.0 / (NX - 1), 1.0 / (NT - 1)
RNG = np.random.default_rng(1873)


def _fields():
    U_old = RNG.standard_normal((NT, NX))
    U_new = U_old + 1e-3 * RNG.standard_normal((NT, NX))
    M_old = np.abs(RNG.standard_normal((NT, NX))) + 0.1
    M_new = M_old + 1e-3 * RNG.standard_normal((NT, NX))
    return U_new, U_old, M_new, M_old


@pytest.mark.parametrize(("offset_scale", "tolerance"), [(1.0, 1e-9), (1e3, 1e-9), (1e6, 1e-6)])
def test_the_u_error_is_unchanged_by_an_additive_function_of_time(offset_scale, tolerance):
    """A per-slice constant is invisible to the density, so it must be invisible here.

    The offsets differ between the two iterates as well as across time -- a gauge that
    only moved with the iteration would be caught by a metric that merely subtracted a
    single global constant.

    The tolerance loosens at 1e6 because the projection is subtraction, and subtraction of
    a 1e+06 offset from an O(1) field spends six of float64's sixteen digits. That is a real
    limit of removing the mode this way rather than never forming it: invariance holds to
    about `offset_scale * eps / ||U||`. It is recorded here rather than hidden by dropping
    the case, because a caller whose value function genuinely carries an offset that large
    should know the metric goes soft, not silently wrong.
    """
    U_new, U_old, M_new, M_old = _fields()
    base = calculate_l2_convergence_metrics(U_new, U_old, M_new, M_old, DX, DT)

    off_new = offset_scale * RNG.standard_normal((NT, 1))
    off_old = offset_scale * RNG.standard_normal((NT, 1))
    shifted = calculate_l2_convergence_metrics(U_new + off_new, U_old + off_old, M_new, M_old, DX, DT)

    assert shifted["l2distu_abs"] == pytest.approx(base["l2distu_abs"], rel=tolerance), (
        f"an offset of scale {offset_scale:g} moved the absolute U error"
    )
    assert shifted["l2distu_rel"] == pytest.approx(base["l2distu_rel"], rel=tolerance), (
        f"an offset of scale {offset_scale:g} moved the relative U error"
    )


def test_a_pure_gauge_change_is_zero_error_not_a_big_one():
    """Two iterates differing ONLY by a per-slice constant have converged in U.

    This is the sharp form: the raw metric would report an error of the size of the offset
    -- here 1e+03 -- on a pair whose drift fields are identical.
    """
    U_old = RNG.standard_normal((NT, NX))
    U_new = U_old + 1e3 * RNG.standard_normal((NT, 1))
    M = np.abs(RNG.standard_normal((NT, NX))) + 0.1

    m = calculate_l2_convergence_metrics(U_new, U_old, M, M, DX, DT)

    assert m["l2distu_abs"] < 1e-9, f"a pure gauge change reported {m['l2distu_abs']:.3e}"
    assert np.allclose(np.gradient(U_new, axis=1), np.gradient(U_old, axis=1)), "the fixture is not a pure gauge change"


def test_a_real_change_still_registers():
    """The projection must not swallow the signal along with the gauge.

    Without this, returning a zero matrix would satisfy every assertion above.
    """
    U_old = RNG.standard_normal((NT, NX))
    U_new = U_old + 1e-2 * np.sin(np.linspace(0, np.pi, NX))[None, :]
    M = np.abs(RNG.standard_normal((NT, NX))) + 0.1

    m = calculate_l2_convergence_metrics(U_new, U_old, M, M, DX, DT)

    assert m["l2distu_abs"] > 1e-4, f"a spatially varying change of 1e-2 reported {m['l2distu_abs']:.3e}"


def test_the_density_metric_is_not_projected():
    """M's level is mass. Shifting it is a real change and must be reported as one.

    The asymmetry between the two variables is the point, not an oversight: there is no
    additive freedom in `m` to remove.
    """
    U = RNG.standard_normal((NT, NX))
    M_old = np.abs(RNG.standard_normal((NT, NX))) + 0.1
    M_new = M_old + 0.5  # a uniform shift in density is a change in mass

    m = calculate_l2_convergence_metrics(U, U, M_new, M_old, DX, DT)

    assert m["l2distm_abs"] > 1e-2, f"a uniform density shift reported {m['l2distm_abs']:.3e}"


def test_the_projection_survives_a_two_dimensional_field():
    """The nD path passes (Nt, Nx, Ny); the mean must be over space, not over one axis."""
    U_old = RNG.standard_normal((NT, 7, 5))
    U_new = U_old + 1e-3 * RNG.standard_normal((NT, 7, 5))
    M = np.abs(RNG.standard_normal((NT, 7, 5))) + 0.1

    base = calculate_l2_convergence_metrics(U_new, U_old, M, M, DX, DT)
    shifted = calculate_l2_convergence_metrics(U_new + 1e4 * RNG.standard_normal((NT, 1, 1)), U_old, M, M, DX, DT)

    assert shifted["l2distu_abs"] == pytest.approx(base["l2distu_abs"], rel=1e-9)

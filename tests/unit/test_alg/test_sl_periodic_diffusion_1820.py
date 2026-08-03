"""The periodic Crank-Nicolson step against the analytic heat kernel. Issue #1820.

Oracle: for `u(0, x) = sin(k x)` on a periodic domain, diffusion with coefficient `D` gives
`u(t, x) = exp(-D k^2 t) sin(k x)` exactly. Computed independently of the scheme, so it stays
discriminating no matter how the SL family is later consolidated.

This is here because the **seam alone was not enough**. `_crank_nicolson_periodic_1d` wrapped all
`N` nodes -- taking `U[N-1]` as node 0's left neighbour -- while `TensorProductGrid` is
endpoint-inclusive, so those two entries are the same physical point and the stencil reached a
neighbour at distance 0 instead of `dx`. That is a different operator, not a rounding difference:
max error against the kernel was `1.19e-01`, and it did not shrink under refinement.

`InterpolationApplicator.enforce_values` runs after the diffusion step and now identifies the two
endpoints correctly, which drove the end-to-end seam to `2.4e-16` **while the interior stayed wrong
by 3.0e-02**. A seam assertion alone would have called that fixed. Hence an accuracy oracle.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import solve_crank_nicolson_diffusion_1d

SIGMA = 0.3
DT = 0.05
D = SIGMA**2 / 2.0
K = 2 * np.pi


def _run(nx: int):
    x = np.linspace(0.0, 1.0, nx)
    u0 = np.sin(K * x)
    out = solve_crank_nicolson_diffusion_1d(u0.copy(), DT, SIGMA, x, bc_type="periodic")
    exact = np.exp(-D * K * K * DT) * u0
    return out, exact


def test_the_periodic_step_matches_the_analytic_heat_kernel():
    """The mode decays by exp(-D k^2 t), and nothing else about it changes."""
    out, exact = _run(21)
    assert np.abs(out - exact).max() < 5e-3, (
        "periodic Crank-Nicolson does not reproduce the analytic decay of a single Fourier mode"
    )


def test_the_error_falls_under_refinement():
    """The property the wrapped-N form did not have: its 1.19e-01 was a fixed operator error.

    Asserted as a decrease rather than a rate -- at these step sizes the temporal error is not
    negligible, so a clean O(dx^2) claim would be over-reading the numbers.
    """
    errors = [float(np.abs(o - e).max()) for o, e in (_run(n) for n in (21, 41, 81))]
    assert errors[1] < errors[0], f"refinement did not help: {errors}"
    assert errors[2] < errors[1], f"refinement did not help: {errors}"


def test_the_step_leaves_no_seam():
    """x_min and x_max are one point, so the step must not separate them."""
    out, _ = _run(21)
    assert abs(out[0] - out[-1]) < 1e-14


def test_a_constant_is_unchanged():
    """Diffusion of a constant is the constant -- the cheapest way to catch a stencil that
    reaches the wrong neighbour, since every interior row must sum its off-diagonals to the
    diagonal for this to hold."""
    x = np.linspace(0.0, 1.0, 21)
    out = solve_crank_nicolson_diffusion_1d(np.full(21, 2.5), DT, SIGMA, x, bc_type="periodic")
    np.testing.assert_allclose(out, 2.5, atol=1e-12)


def test_too_few_nodes_is_refused():
    """Two nodes on an endpoint-inclusive periodic grid is one degree of freedom, not a domain."""
    with pytest.raises(ValueError, match="at least 3 nodes"):
        solve_crank_nicolson_diffusion_1d(np.array([1.0, 1.0]), DT, SIGMA, np.array([0.0, 1.0]), bc_type="periodic")

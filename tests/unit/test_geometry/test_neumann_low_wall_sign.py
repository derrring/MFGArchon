"""
Pinning test for Issue #1262: linear-reflection nonzero-Neumann low-wall sign.

The order<=2 path must encode du/dn = v (outward normal), not du/dx = v.
At the LOW wall (outward normal -x): ghost = mirror + 2*dx*v.
At the HIGH wall (outward normal +x): ghost = mirror + 2*dx*v.
Both walls agree; both must agree with the order>2 polynomial path.

The mirror is the node beside the wall node and the step is 2*dx since #1935, which made the ghost
node-centred; it was the wall node itself and dx before. The sign law #1262 pins is the same.
"""

import numpy as np

from mfgarchon.geometry.boundary import neumann_bc
from mfgarchon.geometry.boundary.applicator_fdm import PreallocatedGhostBuffer


def _make_buffer(v: float, order: int, n: int = 10) -> PreallocatedGhostBuffer:
    """Create a 1D ghost buffer with Neumann BC du/dn = v."""
    bc = neumann_bc(value=v, dimension=1)
    buf = PreallocatedGhostBuffer(
        interior_shape=(n,),
        boundary_conditions=bc,
        domain_bounds=np.array([[0.0, 1.0]]),
        order=order,
    )
    return buf


class TestNeumannLowWallSign:
    """Issue #1262 — low-wall ghost must be mirror + 2*dx*v, not mirror - 2*dx*v."""

    def test_order2_low_ghost_plus_dx_v(self):
        """
        For nonzero v, order=2 low-wall ghost = mirror + 2*dx*v.

        Buggy code: -=  =>  ghost = mirror - 2*dx*v  (FAILS).
        Fixed code: +=  =>  ghost = mirror + 2*dx*v  (PASSES).
        """
        v = 2.5
        buf = _make_buffer(v=v, order=2)
        buf.interior[:] = np.ones(10) * 3.0
        buf.update_ghosts(time=0.0)

        n = 10
        dx = 1.0 / (n - 1)  # domain [0,1], n points
        mirror = buf.interior[1]  # 3.0
        expected_lo = mirror + 2 * dx * v
        actual_lo = buf.padded[0]

        assert np.isclose(actual_lo, expected_lo), (
            f"Low ghost = {actual_lo:.6f}, expected {expected_lo:.6f} "
            f"(mirror + 2*dx*v = {mirror} + 2*{dx}*{v}). "
            f"Wrong sign would give {mirror - 2 * dx * v:.6f}."
        )

    def test_order2_high_ghost_plus_dx_v(self):
        """High-wall ghost = mirror + 2*dx*v (was already correct, must not regress)."""
        v = 2.5
        buf = _make_buffer(v=v, order=2)
        buf.interior[:] = np.ones(10) * 3.0
        buf.update_ghosts(time=0.0)

        n = 10
        dx = 1.0 / (n - 1)
        mirror = buf.interior[-2]  # 3.0
        expected_hi = mirror + 2 * dx * v
        actual_hi = buf.padded[-1]

        assert np.isclose(actual_hi, expected_hi), f"High ghost = {actual_hi:.6f}, expected {expected_hi:.6f}."

    def test_order2_and_order4_low_ghost_agree(self):
        """
        order=2 and order=4 must give the same low-wall ghost for a linear field.

        Linear data u(x) = c + a*x satisfies du/dn = v at the low wall iff
        du/dx|_{x_lo} = -v (outward normal -x). For such a field polynomial
        extrapolation is exact, so order=2 and order=4 must agree to machine eps.

        Buggy order=2 (- sign) would give the mirror minus 2*dx*v instead of plus,
        diverging from the order=4 result.
        """
        v = 1.5
        n = 10
        dx = 1.0 / (n - 1)

        # Linear field: u(x) = 5.0 - v*x  =>  du/dx = -v  =>  du/dn(low) = +v
        # Grid points at the nodes x = 0, dx, 2*dx, ..., the wall being one (#1935).
        x_interior = np.linspace(0.0, 1.0, n)
        u_interior = 5.0 - v * x_interior  # du/dx = -v => du/dn = +v at low wall

        buf2 = _make_buffer(v=v, order=2)
        buf4 = _make_buffer(v=v, order=4, n=n)

        buf2.interior[:] = u_interior
        buf4.interior[:] = u_interior

        buf2.update_ghosts(time=0.0)
        buf4.update_ghosts(time=0.0)

        lo2 = buf2.padded[0]
        lo4 = buf4.padded[0]

        assert np.isclose(lo2, lo4, atol=1e-10), (
            f"order=2 low ghost = {lo2:.8f}, order=4 = {lo4:.8f}. "
            f"They must agree for a linear field. "
            f"Wrong sign gives {u_interior[1] - 2 * dx * v:.8f} instead of "
            f"{u_interior[1] + 2 * dx * v:.8f}."
        )

    def test_zero_flux_neumann_unchanged(self):
        """v=0 must leave the ghost identical to the pure mirror about the wall node (regression guard)."""
        buf = _make_buffer(v=0.0, order=2)
        buf.interior[:] = np.arange(1, 11, dtype=np.float64)
        buf.update_ghosts(time=0.0)

        assert buf.padded[0] == buf.interior[1], "v=0 low ghost must mirror interior[1]"
        assert buf.padded[-1] == buf.interior[-2], "v=0 high ghost must mirror interior[-2]"

    def test_negative_v_low_ghost(self):
        """Sign must be consistent for v < 0."""
        v = -3.0
        buf = _make_buffer(v=v, order=2)
        buf.interior[:] = np.ones(10) * 7.0
        buf.update_ghosts(time=0.0)

        n = 10
        dx = 1.0 / (n - 1)
        expected_lo = buf.interior[1] + 2 * dx * v  # 7.0 + 2*dx*(-3.0) = 7.0 - 6*dx
        assert np.isclose(buf.padded[0], expected_lo), (
            f"Negative-v: low ghost = {buf.padded[0]:.6f}, expected {expected_lo:.6f}."
        )

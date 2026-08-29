#!/usr/bin/env python3
"""Issue #1428: the MFG-coupled advection helper compute_advection_term_nd must support the
discretely-conservative finite-volume divergence (mass_conservative=True), matching the
direct-drift helper and every other FP path (#1184). Previously it had no such option, so the
tensor-diffusion MFG-coupled explicit FP branch leaked mass at no-flux walls.

Conservation property: for a conservative FV divergence with zero advective flux through no-flux
walls, the discrete divergence sums (integrates) to ~0 — interior fluxes telescope and boundary
fluxes are zero. The non-conservative node divergence does NOT sum to zero (the leak).
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_fdm_advection import compute_advection_term_nd
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc
from mfgarchon.utils.numerical.quadrature import quadrature_weights_1d


def _setup_2d(n=12):
    """2D density driven against a wall by a strong U-gradient (the leak-prone configuration)."""
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    # Density bump near the left/bottom; value function pushing it toward the wall.
    M = np.exp(-30 * ((X - 0.25) ** 2 + (Y - 0.25) ** 2)) + 0.05
    U = 0.5 * ((X - 0.0) ** 2 + (Y - 0.0) ** 2)  # grad U points away from corner → drift toward it
    dx = xs[1] - xs[0]
    return M, U, (dx, dx)


class TestCoupledAdvectionConservation:
    def test_conservative_branch_sums_to_zero_under_no_flux(self):
        M, U, spacing = _setup_2d()
        bc = no_flux_bc(dimension=2)
        adv = compute_advection_term_nd(M, U, 1.0, spacing, 2, bc, mass_conservative=True)
        # Against the CONTROL VOLUMES (#2145), not a bare sum. The divergence theorem gives
        # `int div(alpha m) = 0` when the wall flux vanishes, and the discrete integral on an
        # endpoint-inclusive grid is `sum_i w_i (div)_i`, with the wall nodes owning half a cell per
        # axis and the corner a quarter. A bare sum weights every node equally, so it tests a
        # different functional -- the one the wall rows used to telescope against while the scheme
        # lost real mass.
        w1 = quadrature_weights_1d(np.arange(M.shape[0], dtype=float) * spacing[0])
        total = float(np.sum(np.multiply.outer(w1, w1) * adv))
        assert abs(total) < 1e-10, (
            f"conservative FV advection must integrate to ~0 under no-flux (zero wall flux), got {total:.3e}"
        )

    def test_nonconservative_default_leaks(self):
        """The default (node divergence) does NOT conserve — pins the contrast so a regression that
        silently makes the default conservative (changing baseline) is caught."""
        M, U, spacing = _setup_2d()
        bc = no_flux_bc(dimension=2)
        adv = compute_advection_term_nd(M, U, 1.0, spacing, 2, bc, mass_conservative=False)
        total = float(np.sum(adv))
        assert abs(total) > 1e-8, (
            f"node-divergence default expected to leak (nonzero sum); got {total:.3e} — "
            f"if this is ~0 the default silently changed."
        )

    def test_conservative_branch_accepts_explicit_periodic_bc(self):
        """Issue #1428: an explicit periodic BoundaryConditions object (not bc=None) must be
        accepted by the conservative-FV mode (wrap-face telescoping) and integrate to ~0 — it
        previously raised NotImplementedError because only bc=None counted as periodic."""
        M, U, spacing = _setup_2d()
        bc = periodic_bc(dimension=2)
        adv = compute_advection_term_nd(M, U, 1.0, spacing, 2, bc, mass_conservative=True)
        total = float(np.sum(adv))
        assert abs(total) < 1e-10, f"conservative FV advection must integrate to ~0 under periodic, got {total:.3e}"

    def test_default_is_nonconservative_byte_identical(self):
        """Omitting mass_conservative must equal mass_conservative=False (baseline preserved)."""
        M, U, spacing = _setup_2d()
        bc = no_flux_bc(dimension=2)
        a_default = compute_advection_term_nd(M, U, 1.0, spacing, 2, bc)
        a_false = compute_advection_term_nd(M, U, 1.0, spacing, 2, bc, mass_conservative=False)
        assert np.array_equal(a_default, a_false)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

#!/usr/bin/env python3
"""Issue #1430 (Strand C — cross-path pinning) / #1428: the two FP-FDM advection helpers must
compute the IDENTICAL advection term for the same drift.

``compute_advection_term_nd(M, U, coupling, ...)`` is the MFG-coupled path: it derives the drift
``alpha[d] = -coupling * d_d U`` from the HJB value function, then assembles ``div(alpha * m)``.
``compute_advection_from_drift_nd(M, drift, ...)`` is the direct-drift path: it takes ``alpha``
already computed. Both wrap the SAME ``AdvectionOperator(form="divergence", scheme, bc,
mass_conservative)`` — they differ ONLY in how the velocity field is obtained. So for
``drift[d] = -coupling * np.gradient(U, dx, axis=d)`` the two must agree byte-for-byte.

This is exactly the shared logic that drifted in Issue #1428/#1438: the conservative-FV divergence
(#1184, ``mass_conservative=True`` — zeroes advective flux through no-flux walls) was wired into the
direct-drift path but NOT the MFG-coupled path, so the tensor-diffusion MFG-coupled explicit FP
branch alone leaked mass. ``test_advection_coupled_conservation.py`` pins that the coupled helper now
*conserves*; this pins that the two helpers *agree* — so a future fix or scheme change applied to one
path but not its sibling (the #1428/#1438 bug class) fails CI. Generalizes the #1422 byte-identical
pinning pattern across the coupled-vs-drift fork, the FP analog of the 1D≡nD HJB pin (#1439).

Why B0 (no baseline risk): pure test addition validating existing (post-#1438) behavior; no
production code is touched. Verified passing on current main.

Refs #1430, #1428, #1184, #1071.
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_fdm_advection import (
    compute_advection_from_drift_nd,
    compute_advection_term_nd,
)
from mfgarchon.geometry.boundary import no_flux_bc


def _drift_from_u(U: np.ndarray, coupling: float, spacing: tuple[float, ...], ndim: int) -> np.ndarray:
    """The exact drift compute_advection_term_nd derives internally: alpha[d] = -coupling * d_d U,
    stacked as the (ndim, *shape) velocity field."""
    return np.stack([-coupling * np.gradient(U, spacing[d], axis=d) for d in range(ndim)], axis=0)


def _setup_1d(n=24):
    x = np.linspace(0.0, 1.0, n)
    M = np.exp(-30 * (x - 0.3) ** 2) + 0.05
    U = 0.5 * (x - 0.0) ** 2  # nonzero gradient → nonzero drift
    return M, U, (x[1] - x[0],)


def _setup_2d(n=12):
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    M = np.exp(-30 * ((X - 0.25) ** 2 + (Y - 0.25) ** 2)) + 0.05
    U = 0.5 * (X**2 + Y**2)  # grad U drives density toward the corner (the leak-prone case)
    dx = xs[1] - xs[0]
    return M, U, (dx, dx)


class TestCoupledVsDriftAdvectionAgreement:
    """The MFG-coupled (from-U) and direct-drift FP advection helpers must agree byte-for-byte."""

    @pytest.mark.parametrize("mass_conservative", [False, True])
    @pytest.mark.parametrize("dim", [1, 2])
    def test_coupled_equals_drift_byte_identical(self, dim, mass_conservative):
        coupling = 0.7
        M, U, spacing = _setup_1d() if dim == 1 else _setup_2d()
        bc = no_flux_bc(dimension=dim)
        drift = _drift_from_u(U, coupling, spacing, dim)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a_coupled = compute_advection_term_nd(M, U, coupling, spacing, dim, bc, mass_conservative=mass_conservative)
            a_drift = compute_advection_from_drift_nd(
                M, drift, spacing, dim, bc=bc, mass_conservative=mass_conservative
            )

        # Not vacuous: the advection must actually be nonzero for this driven density.
        assert np.max(np.abs(a_coupled)) > 1e-6, "advection is ~0 — fixture not exercising the operator"
        # Byte-identical: the two paths wrap the same operator; only the drift source differs.
        assert np.array_equal(a_coupled, a_drift), (
            f"coupled (from-U) and direct-drift FP advection disagree (dim={dim}, "
            f"mass_conservative={mass_conservative}): max|diff| = "
            f"{float(np.max(np.abs(a_coupled - a_drift))):.3e}. The two helpers must wrap the same "
            f"divergence/upwind/conservation logic — a fix applied to one path but not its sibling is "
            f"the Issue #1428/#1438 cross-path bug class."
        )

    def test_conservative_and_nonconservative_actually_differ(self):
        """Guard the pin's relevance: mass_conservative does change the result (so the byte-identity
        above is asserted on a live distinction, not a no-op flag). The #1184/#1428 conservative-FV
        divergence differs from the node divergence at no-flux walls."""
        coupling = 0.7
        M, U, spacing = _setup_2d()
        bc = no_flux_bc(dimension=2)
        a_false = compute_advection_term_nd(M, U, coupling, spacing, 2, bc, mass_conservative=False)
        a_true = compute_advection_term_nd(M, U, coupling, spacing, 2, bc, mass_conservative=True)
        assert not np.array_equal(a_false, a_true), (
            "mass_conservative=True/False produced identical results — the flag is a no-op here, so "
            "the cross-path byte-identity test would not be exercising the conservative branch."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

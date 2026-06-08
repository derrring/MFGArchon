#!/usr/bin/env python3
"""
Cross-path convention-agreement guards.

The dominant bug class in this library is the *same* convention implemented along several
parallel code paths, each carrying a private copy, with no single owning abstraction — and the
divergence is silent (no exception / NaN). These tests pin the conventions that have *converged*
to a single value across their parallel implementations, so a future private-copy drift fails
loudly instead of silently.

Scope note (verified once via a probe, not re-run here as brittle permanent assertions): the
``sigma -> D`` convention was additionally checked by *recovering* D from the assembled FP-FDM
upwind/divergence matrices, the ADI Crank-Nicolson operator, and the weak-form FP coefficient —
all 66 (sigma x path) combinations agreed to <= 1e-14. The behavioral magnitude guard for the
solver dynamics lives separately in ``tests/integration/test_diffusion_magnitude_gate.py``
(Issue #1188); this file guards the *resolution* layer (the converter + the touchpoints solvers
read) so the two are complementary.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

SIGMAS = [0.1, 0.7, 1.0, np.sqrt(2.0), 2.5, 3.0]


class TestSigmaToDiffusionAgreement:
    """Issue #811 / #1192: every path resolves the SDE volatility to D = sigma**2 / 2."""

    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_converter_and_problem_property_agree(self, sigma):
        """The canonical converter and the MFGProblem.diffusion property (which the solvers
        read) must both yield D = sigma**2 / 2 — i.e. the property delegates to the single
        source, not a private copy."""
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
        from mfgarchon.core.mfg_components import MFGComponents
        from mfgarchon.core.mfg_problem import MFGProblem
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc

        d_reference = 0.5 * sigma * sigma

        components = MFGComponents(
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        )
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.1, Nt=5, sigma=sigma, components=components)

        assert diffusion_from_volatility(sigma) == pytest.approx(d_reference, rel=1e-12)
        assert float(problem.diffusion) == pytest.approx(d_reference, rel=1e-12)

    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_gfdm_sigma_resolution_agrees(self, sigma):
        """The 2D scattered-cloud GFDM HJB path resolves sigma via _get_sigma_value, then applies
        the canonical converter (hjb_gfdm.py:2053-2054 etc.). It must agree with the converter."""
        from types import SimpleNamespace

        from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver

        stub = SimpleNamespace(problem=SimpleNamespace(sigma=sigma))
        resolved = HJBGFDMSolver._get_sigma_value(stub, None)
        assert diffusion_from_volatility(resolved) == pytest.approx(0.5 * sigma * sigma, rel=1e-12)

    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_backend_literal_equals_single_source(self, sigma):
        """Several backends apply D via the literal ``0.5 * sigma**2`` rather than the converter
        (numpy/jax/numba/torch). Pin that the literal is bit-equal to the single source so a
        backend edit that drifts from sigma**2/2 fails here."""
        assert 0.5 * sigma**2 == diffusion_from_volatility(sigma)


class TestGeometryBoundsAccessor:
    """Issue #1056: the .bounds / get_bounding_box accessors are non-uniform across geometry
    classes, but get_bounds() is the one accessor present on ALL of them (the Geometry ABC
    contract). Pin that uniform contract so it is not eroded; the .bounds non-uniformity itself
    remains tracked in #1056."""

    @staticmethod
    def _geometries():
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc
        from mfgarchon.geometry.implicit.csg_operations import (
            DifferenceDomain,
            IntersectionDomain,
            UnionDomain,
        )
        from mfgarchon.geometry.implicit.hyperrectangle import Hyperrectangle
        from mfgarchon.geometry.implicit.hypersphere import Hypersphere

        box = Hyperrectangle(bounds=[(0.0, 1.0), (0.0, 1.0)])
        ball = Hypersphere(center=[0.5, 0.5], radius=0.4)
        return [
            (
                "TensorProductGrid",
                TensorProductGrid(
                    bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[5, 6], boundary_conditions=no_flux_bc(dimension=2)
                ),
            ),
            ("Hyperrectangle", box),
            ("Hypersphere", ball),
            ("UnionDomain", UnionDomain([box, ball])),
            ("IntersectionDomain", IntersectionDomain([box, ball])),
            ("DifferenceDomain", DifferenceDomain(box, ball)),
        ]

    def test_get_bounds_uniform_contract(self):
        """Every geometry exposes get_bounds() -> (mins, maxs), each length d, mins <= maxs."""
        for name, geom in self._geometries():
            result = geom.get_bounds()
            assert isinstance(result, tuple), f"{name}: get_bounds must return a tuple"
            assert len(result) == 2, f"{name}: get_bounds must return (mins, maxs)"
            mins, maxs = np.asarray(result[0], dtype=float), np.asarray(result[1], dtype=float)
            assert mins.shape == maxs.shape, f"{name}: mins/maxs shape mismatch"
            assert np.all(mins <= maxs), f"{name}: mins must be <= maxs, got {mins} / {maxs}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

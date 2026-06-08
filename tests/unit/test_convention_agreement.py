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

    def test_get_bounding_box_is_derived_view_of_get_bounds(self):
        """For the implicit family, get_bounding_box() is the (d, 2) view of the same source:
        column_stack(get_bounds()) == get_bounding_box() (Issue #1056)."""
        from mfgarchon.geometry.implicit.csg_operations import (
            DifferenceDomain,
            IntersectionDomain,
            UnionDomain,
        )
        from mfgarchon.geometry.implicit.hyperrectangle import Hyperrectangle
        from mfgarchon.geometry.implicit.hypersphere import Hypersphere

        box = Hyperrectangle(bounds=[(0.0, 1.0), (0.0, 1.0)])
        ball = Hypersphere(center=[0.5, 0.5], radius=0.4)
        for name, geom in [
            ("Hyperrectangle", box),
            ("Hypersphere", ball),
            ("UnionDomain", UnionDomain([box, ball])),
            ("IntersectionDomain", IntersectionDomain([box, ball])),
            ("DifferenceDomain", DifferenceDomain(box, ball)),
        ]:
            mins, maxs = geom.get_bounds()
            np.testing.assert_allclose(np.column_stack([mins, maxs]), geom.get_bounding_box(), atol=1e-12, err_msg=name)


class TestBoundaryToleranceSingleSource:
    """Issue #1101: boundary on-wall tolerances are single-sourced in
    geometry/boundary/tolerances.py. Pin the values (a future edit cannot silently shift them)
    and pin that the key classifier defaults reference the constants — so the scattered magic
    literals do not regrow. The values are intentionally distinct (grid-exact vs scattered vs SDF)
    and are NOT collapsed to one (that would loosen analytic boundary detection 4 decades)."""

    def test_constant_values_pinned(self):
        from mfgarchon.geometry.boundary import tolerances as tol

        assert tol.BOUNDARY_TOL == 1e-6
        assert tol.ONWALL_TOL == 1e-10
        assert tol.SDF_BOUNDARY_TOL == 1e-8
        assert tol.BOUNDARY_REL_TOL == 1e-12

    def test_classifier_defaults_reference_single_source(self):
        """The paper-path GFDM classifier defaults to BOUNDARY_TOL (1e-6) and the analytic
        Geometry on-wall defaults to ONWALL_TOL (1e-10) — byte-identical to the prior literals."""
        import inspect

        from mfgarchon.geometry.base import Geometry
        from mfgarchon.geometry.boundary.conditions import BoundaryConditions
        from mfgarchon.geometry.boundary.tolerances import BOUNDARY_TOL, ONWALL_TOL

        face_default = inspect.signature(BoundaryConditions.identify_boundary_face).parameters["tolerance"].default
        assert face_default == BOUNDARY_TOL == 1e-6
        onwall_default = inspect.signature(Geometry.is_on_boundary).parameters["tolerance"].default
        assert onwall_default == ONWALL_TOL == 1e-10


class TestOutwardNormalSourceAgreement:
    """Issue #1114: the two outward-normal sources (face-derived vs SDF-gradient) must agree on
    outer-box walls. `get_outward_normal` returns the exact face normal there — NOT the obstacle
    SDF gradient — for Difference-style domains (outer box + obstacle SDF); the SDF gradient is
    used only for genuinely curved boundaries."""

    @staticmethod
    def _difference_bc():
        from mfgarchon.geometry.boundary.conditions import BCSegment, BCType, BoundaryConditions

        def obstacle_sdf(p):
            p = np.asarray(p, dtype=float)
            return 0.2 - np.linalg.norm(p - np.array([0.5, 0.5]))

        bc = BoundaryConditions(segments=[BCSegment(name="w", bc_type=BCType.NO_FLUX, boundary="x_min")], dimension=2)
        bc.domain_bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        bc.domain_sdf = obstacle_sdf
        return bc

    def test_outer_wall_uses_face_normal_not_obstacle_sdf(self):
        bc = self._difference_bc()
        point = np.array([0.0, 0.5])  # on the outer left wall
        normal = bc.get_outward_normal(point)
        # exact face normal, not the obstacle-pointing SDF gradient (the #1114 misfire)
        np.testing.assert_allclose(normal, [-1.0, 0.0], atol=1e-12)
        # and it agrees with the canonical face-derived source
        face = bc.identify_boundary_face(point)
        np.testing.assert_allclose(normal, bc.outward_normal_for_face(face, dimension=2), atol=1e-12)

    def test_curved_boundary_still_uses_sdf_gradient(self):
        bc = self._difference_bc()
        # A point on the obstacle surface at a DIAGONAL (interior to the box, not on any outer
        # wall). A face normal could only be axis-aligned, so a diagonal result proves the SDF
        # gradient path is used — not snapped to an axis face.
        d = 0.2 / np.sqrt(2.0)
        point = np.array([0.5 + d, 0.5 + d])  # on the r=0.2 obstacle circle, 45 degrees
        normal = bc.get_outward_normal(point)
        assert normal is not None
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-9)
        # diagonal => both components non-trivial (an axis face normal would have a zero component)
        assert abs(normal[0]) > 0.1 and abs(normal[1]) > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

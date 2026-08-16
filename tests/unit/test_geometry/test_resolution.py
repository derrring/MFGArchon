"""Tests for BC resolution layer (Layer 2, Issue #848 Phase 3)."""

from __future__ import annotations

from typing import Any

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCResolver,
    BCSegment,
    BCType,
    BoundaryConditions,
    FPResolver,
    HJBResolver,
    MathBCType,
    ResolvedBC,
    resolve_bc,
)
from mfgarchon.geometry.boundary.conditions import (
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
    robin_bc,
)
from mfgarchon.geometry.boundary.resolution import (
    resolved_bc_to_calculator,
    to_boundary_conditions,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hjb_resolver() -> HJBResolver:
    return HJBResolver()


@pytest.fixture
def fp_resolver() -> FPResolver:
    return FPResolver()


@pytest.fixture
def empty_state() -> dict[str, Any]:
    return {}


class MockProvider:
    """Mock BCValueProvider for testing."""

    def __init__(self, return_value: float = 0.42):
        self._return_value = return_value

    def compute(self, state: dict[str, Any]) -> float:
        return self._return_value


# =============================================================================
# Protocol conformance
# =============================================================================


class TestProtocolConformance:
    def test_hjb_resolver_is_bc_resolver(self):
        assert isinstance(HJBResolver(), BCResolver)

    def test_fp_resolver_is_bc_resolver(self):
        assert isinstance(FPResolver(), BCResolver)


# =============================================================================
# HJBResolver
# =============================================================================


class TestHJBResolver:
    def test_no_flux_resolves_to_neumann(self, hjb_resolver, empty_state):
        bc = no_flux_bc(dimension=1)
        results = resolve_bc(bc, hjb_resolver, empty_state)
        assert len(results) >= 1
        for r in results:
            assert r.math_type == MathBCType.NEUMANN
            assert r.value == 0.0
            assert r.original_bc_type == BCType.NO_FLUX

    def test_reflecting_resolves_to_neumann(self, hjb_resolver, empty_state):
        seg = BCSegment(name="wall", bc_type=BCType.REFLECTING, value=0.0)
        bc = BoundaryConditions(segments=[seg], dimension=1)
        results = resolve_bc(bc, hjb_resolver, empty_state)
        assert results[0].math_type == MathBCType.NEUMANN
        assert results[0].value == 0.0

    def test_dirichlet_passthrough(self, hjb_resolver, empty_state):
        bc = dirichlet_bc(1.5, dimension=1)
        results = resolve_bc(bc, hjb_resolver, empty_state)
        assert results[0].math_type == MathBCType.DIRICHLET
        assert results[0].value == 1.5

    def test_neumann_passthrough(self, hjb_resolver, empty_state):
        bc = neumann_bc(0.5, dimension=1)
        results = resolve_bc(bc, hjb_resolver, empty_state)
        assert results[0].math_type == MathBCType.NEUMANN
        assert results[0].value == 0.5

    def test_robin_passthrough(self, hjb_resolver, empty_state):
        bc = robin_bc(alpha=1.0, beta=2.0, value=0.5, dimension=1)
        results = resolve_bc(bc, hjb_resolver, empty_state)
        assert results[0].math_type == MathBCType.ROBIN
        assert results[0].value == 0.5
        assert results[0].alpha == 1.0
        assert results[0].beta == 2.0

    def test_periodic_passthrough(self, hjb_resolver, empty_state):
        bc = periodic_bc(dimension=1)
        results = resolve_bc(bc, hjb_resolver, empty_state)
        assert results[0].math_type == MathBCType.PERIODIC

    def test_extrapolation_linear_passthrough(self, hjb_resolver, empty_state):
        seg = BCSegment(name="far", bc_type=BCType.EXTRAPOLATION_LINEAR, value=0.0)
        bc = BoundaryConditions(segments=[seg], dimension=1)
        results = resolve_bc(bc, hjb_resolver, empty_state)
        assert results[0].math_type == MathBCType.EXTRAPOLATION_LINEAR

    def test_provider_on_no_flux_resolves_to_robin(self, hjb_resolver):
        """AdjointConsistentProvider pattern: NO_FLUX with provider -> Robin."""
        provider = MockProvider(return_value=0.42)
        seg = BCSegment(name="ac_wall", bc_type=BCType.NO_FLUX, value=provider)
        bc = BoundaryConditions(segments=[seg], dimension=1)
        state = {"m_current": np.ones(10), "geometry": None}
        results = resolve_bc(bc, hjb_resolver, state)
        assert results[0].math_type == MathBCType.ROBIN
        assert results[0].value == pytest.approx(0.42)
        assert results[0].alpha == 0.0
        assert results[0].beta == 1.0


# =============================================================================
# FPResolver
# =============================================================================


class TestFPResolver:
    @pytest.mark.parametrize("bc_type", [BCType.NO_FLUX, BCType.REFLECTING])
    def test_an_impermeable_wall_resolves_to_the_robin_coefficients_of_the_flux_condition(self, fp_resolver, bc_type):
        """RE-POINTED 2026-08-16. This asserted ``MathBCType.ZERO_FLUX``, which was the intent
        passed through under a second name -- its own enum comment read "needs drift+diffusion
        for calculator", so Layer 3 had to dispatch on it again.

        ``J.n = 0`` with ``J = v*m - D*grad(m)`` is ``v_n*m - D*d_n m = 0``, i.e. Robin with
        ``alpha = v_n``, ``beta = -D``, ``g = 0``. The resolution is now that arithmetic.
        """
        seg = BCSegment(name="wall", bc_type=bc_type, value=0.0)
        bc = BoundaryConditions(segments=[seg], dimension=1)

        r = resolve_bc(bc, fp_resolver, {"drift": 0.6, "diffusion": 0.35})[0]

        assert r.math_type == MathBCType.ROBIN
        assert r.original_bc_type == bc_type
        assert r.alpha == pytest.approx(0.6), "alpha is the outward normal drift"
        assert r.beta == pytest.approx(-0.35), "beta is -D"
        assert r.value == pytest.approx(0.0), "the flux condition is homogeneous"

    def test_an_impermeable_wall_without_the_coefficients_refuses(self, fp_resolver, empty_state):
        """A defaulted drift makes the wall pure Neumann, which conserves mass at a tangential
        wall and leaks at every other one -- a wrong answer that still converges. The message
        must name the missing key, since the caller has two to supply."""
        seg = BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0)
        bc = BoundaryConditions(segments=[seg], dimension=1)

        with pytest.raises(ValueError, match="drift"):
            resolve_bc(bc, fp_resolver, empty_state)
        with pytest.raises(ValueError, match="diffusion"):
            resolve_bc(bc, fp_resolver, {"drift": 0.6})

    def test_the_drift_may_be_a_field(self, fp_resolver):
        """The reason ``ResolvedBC`` stopped declaring its coefficients ``float``: the outward
        normal drift varies along the wall and is recomputed each Picard iterate."""
        seg = BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0)
        bc = BoundaryConditions(segments=[seg], dimension=2)
        drift = np.array([0.0, 0.6, -0.6, 2.5])

        r = resolve_bc(bc, fp_resolver, {"drift": drift, "diffusion": 0.35})[0]

        np.testing.assert_allclose(r.alpha, drift)

    def test_dirichlet_passthrough(self, fp_resolver, empty_state):
        bc = dirichlet_bc(0.0, dimension=1)
        results = resolve_bc(bc, fp_resolver, empty_state)
        assert results[0].math_type == MathBCType.DIRICHLET

    def test_periodic_passthrough(self, fp_resolver, empty_state):
        bc = periodic_bc(dimension=1)
        results = resolve_bc(bc, fp_resolver, empty_state)
        assert results[0].math_type == MathBCType.PERIODIC


# =============================================================================
# resolve_bc()
# =============================================================================


class TestResolveBC:
    def test_empty_segments(self, hjb_resolver, empty_state):
        bc = BoundaryConditions(segments=[], dimension=1)
        results = resolve_bc(bc, hjb_resolver, empty_state)
        assert results == []

    def test_mixed_bc_resolves_each_segment(self, hjb_resolver, empty_state):
        """Mixed BCs: each segment resolved independently."""
        segments = [
            BCSegment(
                name="left",
                bc_type=BCType.DIRICHLET,
                value=1.0,
                boundary="x_min",
            ),
            BCSegment(
                name="right",
                bc_type=BCType.NO_FLUX,
                value=0.0,
                boundary="x_max",
            ),
        ]
        bc = BoundaryConditions(segments=segments, dimension=1)
        results = resolve_bc(bc, hjb_resolver, empty_state)
        assert len(results) == 2
        assert results[0].math_type == MathBCType.DIRICHLET
        assert results[0].value == 1.0
        assert results[1].math_type == MathBCType.NEUMANN
        assert results[1].value == 0.0

    def test_none_state_defaults_to_empty(self, hjb_resolver):
        bc = no_flux_bc(dimension=1)
        results = resolve_bc(bc, hjb_resolver, None)
        assert len(results) >= 1

    def test_segment_names_preserved(self, fp_resolver, empty_state):
        seg = BCSegment(name="my_wall", bc_type=BCType.NO_FLUX, value=0.0)
        bc = BoundaryConditions(segments=[seg], dimension=1)
        results = resolve_bc(bc, fp_resolver, {"drift": 0.0, "diffusion": 1.0})
        assert results[0].segment_name == "my_wall"


# =============================================================================
# resolved_bc_to_calculator()
# =============================================================================


class TestResolvedBCToCalculator:
    def test_2d_shape(self):
        rbc = ResolvedBC(MathBCType.NEUMANN, value=0.0)
        topo, _calc = resolved_bc_to_calculator(rbc, shape=(50, 50))
        assert topo.dimension == 2


# =============================================================================
# to_boundary_conditions()
# =============================================================================


class TestToBoundaryConditions:
    def test_round_trip_preserves_types(self):
        resolved = [
            ResolvedBC(MathBCType.NEUMANN, 0.0, segment_name="left"),
            ResolvedBC(MathBCType.DIRICHLET, 1.0, segment_name="right"),
        ]
        bc = to_boundary_conditions(resolved, dimension=1)
        assert len(bc.segments) == 2
        assert bc.segments[0].bc_type == BCType.NEUMANN
        assert bc.segments[1].bc_type == BCType.DIRICHLET
        assert bc.segments[1].value == 1.0

    def test_empty_resolved_returns_default(self):
        bc = to_boundary_conditions([], dimension=1)
        assert bc.dimension == 1
        assert len(bc.segments) > 0  # Default no_flux_bc has segments

    def test_dimension_propagated(self):
        resolved = [ResolvedBC(MathBCType.PERIODIC, segment_name="all")]
        bc = to_boundary_conditions(resolved, dimension=2)
        assert bc.dimension == 2


# =============================================================================
# ResolvedBC dataclass
# =============================================================================


class TestResolvedBC:
    def test_defaults(self):
        rbc = ResolvedBC(MathBCType.DIRICHLET)
        assert rbc.value == 0.0
        assert rbc.alpha == 1.0
        assert rbc.beta == 0.0
        assert rbc.segment_name == ""
        assert rbc.original_bc_type is None

"""Unit tests for `mfgarchon.config.mfg_methods` (Issue #1010 B2).

Covers the 14 leaf method configs + composite HJB/FP configs. Focus on:
- Default instantiation works for every class
- Literal/enum fields reject invalid values
- Range-constrained numeric fields enforce bounds
- Model-level validators (`@model_validator(mode="after")`) fire correctly
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mfgarchon.config import (
    BoundaryAccuracyConfig,
    DerivativeConfig,
    FDMConfig,
    FPConfig,
    GFDMConfig,
    HJBConfig,
    NeighborhoodConfig,
    NetworkConfig,
    NewtonConfig,
    ParticleConfig,
    QPConfig,
    SLConfig,
    WENOConfig,
)

# FEMConfig lives in mfg_methods but is not re-exported from mfgarchon.config
from mfgarchon.config.mfg_methods import FEMConfig


class TestNewtonConfig:
    def test_defaults(self):
        c = NewtonConfig()
        assert c.max_iterations == 10
        assert c.tolerance == 1e-6
        assert c.relaxation == 1.0

    def test_relaxation_range(self):
        NewtonConfig(relaxation=0.01)
        NewtonConfig(relaxation=1.0)
        with pytest.raises(ValidationError):
            NewtonConfig(relaxation=0.0)
        with pytest.raises(ValidationError):
            NewtonConfig(relaxation=1.5)

    def test_tolerance_positive(self):
        with pytest.raises(ValidationError):
            NewtonConfig(tolerance=0.0)
        with pytest.raises(ValidationError):
            NewtonConfig(tolerance=-1e-6)


class TestFDMConfig:
    def test_defaults(self):
        c = FDMConfig()
        assert c.scheme == "upwind"
        assert c.time_stepping == "implicit"

    @pytest.mark.parametrize("scheme", ["central", "upwind", "lax_friedrichs"])
    def test_valid_schemes(self, scheme):
        assert FDMConfig(scheme=scheme).scheme == scheme

    def test_invalid_scheme(self):
        with pytest.raises(ValidationError):
            FDMConfig(scheme="godunov")


class TestFEMConfig:
    def test_defaults(self):
        c = FEMConfig()
        assert c.element_order == 1
        assert c.quadrature_order == 2 * 1 + 1

    def test_element_order_auto_quadrature(self):
        c = FEMConfig(element_order=2)
        assert c.quadrature_order == 2 * 2 + 1

    def test_element_order_enum(self):
        with pytest.raises(ValidationError):
            FEMConfig(element_order=3)


class TestQPConfig:
    def test_defaults(self):
        c = QPConfig()
        assert c.optimization_level == "none"
        assert c.solver == "osqp"
        assert c.warm_start is True
        assert c.constraint_mode == "indirect"

    @pytest.mark.parametrize("level", ["none", "auto", "always"])
    def test_optimization_levels(self, level):
        assert QPConfig(optimization_level=level).optimization_level == level


class TestNeighborhoodConfig:
    def test_defaults(self):
        c = NeighborhoodConfig()
        assert c.mode == "hybrid"
        assert c.k_neighbors is None
        assert c.adaptive is False
        assert c.max_delta_multiplier == 5.0

    def test_max_delta_multiplier_must_exceed_one(self):
        with pytest.raises(ValidationError):
            NeighborhoodConfig(max_delta_multiplier=1.0)
        NeighborhoodConfig(max_delta_multiplier=1.01)


class TestDerivativeConfig:
    def test_defaults(self):
        c = DerivativeConfig()
        assert c.method == "taylor"
        assert c.rbf_kernel == "phs3"
        assert c.rbf_poly_degree == 2

    def test_poly_degree_non_negative(self):
        DerivativeConfig(rbf_poly_degree=0)
        with pytest.raises(ValidationError):
            DerivativeConfig(rbf_poly_degree=-1)


class TestBoundaryAccuracyConfig:
    def test_defaults(self):
        c = BoundaryAccuracyConfig()
        assert c.local_coordinate_rotation is False
        assert c.ghost_nodes is False
        assert c.wind_dependent_bc is False

    def test_wind_dependent_bc_requires_ghost_nodes(self):
        with pytest.raises(ValidationError, match="wind_dependent_bc=True requires ghost_nodes=True"):
            BoundaryAccuracyConfig(wind_dependent_bc=True, ghost_nodes=False)

    def test_wind_dependent_bc_with_ghost_nodes_ok(self):
        c = BoundaryAccuracyConfig(wind_dependent_bc=True, ghost_nodes=True)
        assert c.wind_dependent_bc is True


class TestGFDMConfig:
    def test_defaults(self):
        c = GFDMConfig()
        assert c.delta == 0.1
        assert c.taylor_order == 2
        assert c.weight_function == "wendland"
        assert c.congestion_mode == "additive"
        assert isinstance(c.qp, QPConfig)
        assert isinstance(c.neighborhood, NeighborhoodConfig)
        assert isinstance(c.derivative, DerivativeConfig)
        assert isinstance(c.boundary_accuracy, BoundaryAccuracyConfig)

    def test_taylor_order_bounds(self):
        GFDMConfig(taylor_order=1)
        GFDMConfig(taylor_order=2)
        with pytest.raises(ValidationError):
            GFDMConfig(taylor_order=0)
        with pytest.raises(ValidationError):
            GFDMConfig(taylor_order=3)

    def test_delta_positive(self):
        with pytest.raises(ValidationError):
            GFDMConfig(delta=0.0)


class TestSLConfig:
    def test_defaults(self):
        c = SLConfig()
        assert c.interpolation_method == "cubic"
        assert c.rk_order == 2
        assert c.cfl_number == 0.5

    def test_cfl_range(self):
        SLConfig(cfl_number=0.01)
        SLConfig(cfl_number=1.0)
        with pytest.raises(ValidationError):
            SLConfig(cfl_number=0.0)
        with pytest.raises(ValidationError):
            SLConfig(cfl_number=1.5)

    def test_rk_order_enum(self):
        for k in (1, 2, 3, 4):
            assert SLConfig(rk_order=k).rk_order == k
        with pytest.raises(ValidationError):
            SLConfig(rk_order=5)


class TestWENOConfig:
    def test_defaults_roundtrip(self):
        c = WENOConfig()
        rebuilt = WENOConfig(**c.model_dump())
        assert rebuilt == c


class TestParticleConfig:
    def test_defaults_and_instantiation(self):
        c = ParticleConfig()
        assert c is not None
        assert type(c).__module__ == "mfgarchon.config.mfg_methods"


class TestNetworkConfig:
    def test_defaults_and_instantiation(self):
        c = NetworkConfig()
        assert c is not None


class TestHJBConfig:
    def test_defaults(self):
        c = HJBConfig()
        assert c.method in {"fdm", "fem", "gfdm", "sl", "weno"}


class TestFPConfig:
    def test_defaults(self):
        c = FPConfig()
        assert c.method in {"fdm", "fem", "particle", "network"}

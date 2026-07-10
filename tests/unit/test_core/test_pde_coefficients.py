"""
Unit tests for PDE coefficient handling utilities.

Tests CoefficientField abstraction for scalar, array, and callable coefficients.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.pde_coefficients import (
    CoefficientField,
    fp_drift_coefficient,
    get_spatial_grid,
    resolve_diffusion_source,
)


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


class TestCoefficientFieldScalar:
    """Test CoefficientField with scalar coefficients."""

    def test_none_returns_default(self):
        """Test that None field returns default value."""
        field = CoefficientField(None, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        result = field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

        assert result == 0.1
        assert field.is_constant()
        assert not field.is_callable()
        assert not field.is_array()

    def test_scalar_float(self):
        """Test scalar float coefficient."""
        field = CoefficientField(0.05, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        result = field.evaluate_at(timestep_idx=5, grid=grid, density=density, dt=0.01)

        assert result == 0.05
        assert field.is_constant()

    def test_scalar_int(self):
        """Test scalar int coefficient (should be converted to float)."""
        field = CoefficientField(2, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        result = field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

        assert result == 2.0
        assert isinstance(result, float)


class TestCoefficientFieldArray:
    """Test CoefficientField with array coefficients."""

    def test_spatially_varying_1d(self):
        """Test spatially varying diffusion in 1D."""
        # Diffusion increases linearly from 0.05 to 0.15
        sigma_spatial = np.linspace(0.05, 0.15, 11)

        field = CoefficientField(sigma_spatial, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        result = field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

        np.testing.assert_array_equal(result, sigma_spatial)
        assert field.is_array()
        assert not field.is_callable()

    def test_spatiotemporal_1d(self):
        """Test spatiotemporal diffusion in 1D."""
        Nt, Nx = 20, 11
        # Diffusion varies in both space and time
        sigma_st = np.random.uniform(0.05, 0.15, (Nt, Nx))

        field = CoefficientField(sigma_st, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, Nx)
        density = np.ones(Nx)

        # Extract at timestep 5
        result = field.evaluate_at(timestep_idx=5, grid=grid, density=density, dt=0.01)

        np.testing.assert_array_equal(result, sigma_st[5, :])

    def test_spatially_varying_2d(self):
        """Test spatially varying diffusion in 2D."""
        shape = (10, 10)
        sigma_spatial = np.random.uniform(0.05, 0.15, shape)

        field = CoefficientField(sigma_spatial, default_value=0.1, field_name="diffusion", dimension=2)

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        grid = (x, y)
        density = np.ones(shape)

        result = field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

        np.testing.assert_array_equal(result, sigma_spatial)

    def test_spatiotemporal_2d(self):
        """Test spatiotemporal diffusion in 2D."""
        Nt = 15
        shape = (10, 10)
        sigma_st = np.random.uniform(0.05, 0.15, (Nt, 10, 10))

        field = CoefficientField(sigma_st, default_value=0.1, field_name="diffusion", dimension=2)

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        grid = (x, y)
        density = np.ones(shape)

        result = field.evaluate_at(timestep_idx=7, grid=grid, density=density, dt=0.01)

        np.testing.assert_array_equal(result, sigma_st[7, :, :])

    def test_array_wrong_spatial_shape(self):
        """Test that wrong spatial shape raises error."""
        sigma_spatial = np.ones(15)  # Wrong size

        field = CoefficientField(sigma_spatial, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        with pytest.raises(ValueError, match=r"has shape.*expected"):
            field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

    def test_array_wrong_dimensions(self):
        """Test that wrong number of dimensions raises error."""
        sigma = np.ones((10, 10, 10))  # 3D array for 1D problem

        field = CoefficientField(sigma, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        with pytest.raises(ValueError, match=r"must have.*dimensions"):
            field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)


class TestCoefficientFieldCallable:
    """Test CoefficientField with callable coefficients."""

    def test_callable_scalar_return(self):
        """Test callable returning scalar."""

        def constant_diffusion(t, x, m):
            return 0.05

        field = CoefficientField(constant_diffusion, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        result = field.evaluate_at(timestep_idx=5, grid=grid, density=density, dt=0.01)

        assert result == 0.05
        assert field.is_callable()
        assert not field.is_constant()

    def test_callable_array_return(self):
        """Test callable returning array."""

        def porous_medium(t, x, m):
            return 0.1 * m

        field = CoefficientField(porous_medium, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.linspace(0.5, 1.5, 11)

        result = field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

        expected = 0.1 * density
        np.testing.assert_array_almost_equal(result, expected)

    def test_callable_time_dependent(self):
        """Test callable using time parameter."""

        def time_varying_diffusion(t, x, m):
            return 0.1 + 0.05 * t

        field = CoefficientField(time_varying_diffusion, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)
        dt = 0.01

        # At t=0 (timestep 0)
        result0 = field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=dt)
        assert result0 == pytest.approx(0.1)

        # At t=0.1 (timestep 10): t = 10 * 0.01 = 0.1
        result10 = field.evaluate_at(timestep_idx=10, grid=grid, density=density, dt=dt)
        assert result10 == pytest.approx(0.1 + 0.05 * 0.1)  # 0.105

    def test_callable_spatial_dependent(self):
        """Test callable using spatial coordinates."""

        def spatial_diffusion(t, x, m):
            # Diffusion increases with x
            return 0.05 + 0.1 * x

        field = CoefficientField(spatial_diffusion, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        result = field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

        expected = 0.05 + 0.1 * grid
        np.testing.assert_array_almost_equal(result, expected)

    def test_callable_density_dependent(self):
        """Test callable using density."""

        def crowd_dynamics(t, x, m):
            m_max = np.max(m) if np.max(m) > 0 else 1.0
            return 0.05 + 0.1 * (1 - m / m_max)

        field = CoefficientField(crowd_dynamics, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.linspace(0.5, 1.5, 11)

        result = field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

        m_max = np.max(density)
        expected = 0.05 + 0.1 * (1 - density / m_max)
        np.testing.assert_array_almost_equal(result, expected)

    def test_callable_wrong_shape(self):
        """Test callable returning wrong shape raises error."""

        def wrong_shape(t, x, m):
            return np.ones(5)  # Wrong size

        field = CoefficientField(wrong_shape, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        with pytest.raises(ValueError, match=r"returned array with shape.*expected"):
            field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

    def test_callable_wrong_type(self):
        """Test callable returning wrong type raises error."""

        def wrong_type(t, x, m):
            return "invalid"

        field = CoefficientField(wrong_type, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        with pytest.raises(TypeError, match="must return float or ndarray"):
            field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

    def test_callable_nan_detection(self):
        """Test callable returning NaN raises error."""

        def nan_diffusion(t, x, m):
            result = np.ones_like(m)
            result[5] = np.nan
            return result

        field = CoefficientField(nan_diffusion, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        with pytest.raises(ValueError, match="returned NaN or Inf"):
            field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

    def test_callable_inf_detection(self):
        """Test callable returning Inf raises error."""

        def inf_diffusion(t, x, m):
            result = np.ones_like(m)
            result[3] = np.inf
            return result

        field = CoefficientField(inf_diffusion, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        with pytest.raises(ValueError, match="returned NaN or Inf"):
            field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)


class TestGetSpatialGrid:
    """Test get_spatial_grid utility function."""

    def test_geometry_based_api_1d(self):
        """Test grid extraction with geometry-based API (1D)."""
        domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=domain, T=1.0, Nt=50, sigma=0.1, components=_default_components())

        grid = get_spatial_grid(problem)

        assert isinstance(grid, np.ndarray)
        assert len(grid) == 51
        np.testing.assert_array_almost_equal(grid, np.linspace(0, 1, 51))

    def test_legacy_api_1d(self):
        """Test grid extraction with legacy API (1D)."""
        # This test now uses Geometry-First API instead of deprecated legacy API
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, sigma=0.1, components=_default_components())

        grid = get_spatial_grid(problem)

        assert isinstance(grid, np.ndarray)
        assert len(grid) == 51  # 51 grid points
        np.testing.assert_array_almost_equal(grid, np.linspace(0, 1, 51))

    def test_missing_geometry_raises_error(self):
        """Test that problem without geometry raises error."""

        # Create a minimal problem-like object without geometry
        class MinimalProblem:
            pass

        problem = MinimalProblem()

        with pytest.raises(AttributeError, match="must have geometry"):
            get_spatial_grid(problem)


class TestCoefficientFieldEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_diffusion(self):
        """Test zero diffusion coefficient."""
        field = CoefficientField(0.0, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        result = field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

        assert result == 0.0

    def test_negative_diffusion_allowed(self):
        """Test that negative diffusion is allowed (validation elsewhere)."""
        field = CoefficientField(-0.1, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        result = field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

        assert result == -0.1

    def test_very_large_diffusion(self):
        """Test very large diffusion coefficient."""
        field = CoefficientField(1e6, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        result = field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

        assert result == 1e6

    def test_callable_with_no_dt(self):
        """Test callable evaluation when dt=None (uses timestep index as time)."""

        def time_diffusion(t, x, m):
            return 0.1 * t  # t will be timestep index

        field = CoefficientField(time_diffusion, default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        result = field.evaluate_at(timestep_idx=5, grid=grid, density=density, dt=None)

        assert result == 0.5  # 0.1 * 5

    def test_invalid_field_type(self):
        """Test invalid field type raises error."""
        field = CoefficientField("invalid", default_value=0.1, field_name="diffusion", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        with pytest.raises(TypeError, match="must be None, float, ndarray, or Callable"):
            field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)

    def test_field_name_in_error_messages(self):
        """Test that field_name appears in error messages."""

        def wrong_shape(t, x, m):
            return np.ones(5)

        field = CoefficientField(wrong_shape, default_value=0.1, field_name="my_custom_field", dimension=1)

        grid = np.linspace(0, 1, 11)
        density = np.ones(11)

        with pytest.raises(ValueError, match="my_custom_field"):
            field.evaluate_at(timestep_idx=0, grid=grid, density=density, dt=0.01)


class TestScalarDiffusionFromVolatility:
    """Issue #811 / FEM survey: the weak-form / FEM family's single scalar D = sigma^2/2,
    single-sourced via diffusion_from_volatility. Byte-identical to the prior inline copies
    (None -> 0.5*sigma^2, scalar -> 0.5*v^2, array -> 0.5*mean(v)^2); the array case warns
    because a scalar-D solver cannot represent a spatially-varying field."""

    def test_none_uses_fallback_sigma(self):
        from mfgarchon.utils.pde_coefficients import scalar_diffusion_from_volatility

        assert scalar_diffusion_from_volatility(None, 0.3) == pytest.approx(0.5 * 0.3**2, rel=1e-12)

    def test_scalar_is_converted(self):
        from mfgarchon.utils.pde_coefficients import scalar_diffusion_from_volatility

        for v in (0.1, 0.7, 2.5):
            assert scalar_diffusion_from_volatility(v, 0.3) == pytest.approx(0.5 * v**2, rel=1e-12)

    def test_array_collapses_to_mean_with_warning(self):
        from mfgarchon.utils.pde_coefficients import scalar_diffusion_from_volatility

        arr = np.array([0.2, 0.4, 0.6])
        with pytest.warns(UserWarning, match="collapsed to its mean"):
            d = scalar_diffusion_from_volatility(arr, 0.3)
        # byte-identical to the prior inline `0.5 * mean(arr)**2`
        assert d == pytest.approx(0.5 * float(np.mean(arr)) ** 2, rel=1e-12)


class TestFpDriftCoefficient:
    """Issue #1420: fp_drift_coefficient single-sources 1/control_cost from a quadratic-MINIMIZE
    SeparableHamiltonian, falls back to coupling_coefficient otherwise, and fails loud when neither
    is available (V1 — no silent 1.0 fallback for a malformed/duck-typed problem)."""

    @pytest.mark.parametrize("control_cost", [0.5, 1.0, 2.0])
    def test_sources_inverse_control_cost_from_quadratic_hamiltonian(self, control_cost):
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
        comp = MFGComponents(
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=control_cost)),
        )
        prob = MFGProblem(geometry=grid, components=comp, T=0.2, Nt=2, sigma=0.1)
        # the quadratic-Sep-H path wins over the default coupling_coefficient (0.5)
        assert fp_drift_coefficient(prob) == pytest.approx(1.0 / control_cost)

    def test_falls_back_to_coupling_coefficient_without_quadratic_hamiltonian(self):
        class _Obj:
            hamiltonian_class = None
            coupling_coefficient = 0.7

        assert fp_drift_coefficient(_Obj()) == pytest.approx(0.7)

    def test_fails_loud_when_no_hamiltonian_and_no_coupling(self):
        # V1: a duck-typed problem with neither a quadratic SeparableHamiltonian nor
        # coupling_coefficient must raise, not silently return 1.0.
        class _Bare:
            hamiltonian_class = None

        with pytest.raises(ValueError, match="Cannot determine the FP drift coefficient"):
            fp_drift_coefficient(_Bare())

    def test_maximize_quadratic_separable_h_fails_loud_not_coupling_fallback(self):
        # Issue #1542 / RFC #1574 Phase 0: a MAXIMIZE-quadratic SeparableHamiltonian reaches this
        # function (the router gates on is_smooth() alone) but `-c*grad(U)` is the wrong drift for it.
        # It must fail loud, NOT silently fall back to coupling_coefficient (default 0.5 => wrong-sign
        # downhill drift). The discriminating check: coupling_coefficient IS set, so a revert to the
        # old fallback would return 0.5 and this test would catch it.
        from mfgarchon.core.hamiltonian import OptimizationSense

        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
        comp = MFGComponents(
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(lambda_=2.0, sense=OptimizationSense.MAXIMIZE)
            ),
        )
        prob = MFGProblem(geometry=grid, components=comp, T=0.2, Nt=2, sigma=0.1)
        assert getattr(prob, "coupling_coefficient", None) is not None  # the trap the old code fell into
        with pytest.raises(NotImplementedError, match="quadratic-MINIMIZE"):
            fp_drift_coefficient(prob)


class TestResolveDiffusionSource:
    """Issue #1412: the shared single-source volatility (sigma) lookup that HJB and FP
    solvers consume — generalizing HJBGFDMSolver._resolve_diffusion_source so an override
    is resolved identically on every path (no per-solver private copy)."""

    _POINTS = np.linspace(0.0, 1.0, 11).reshape(-1, 1)  # x in [0,1], center = 0.5

    def test_scalar_returns_float(self):
        assert resolve_diffusion_source(0.3) == pytest.approx(0.3)
        assert isinstance(resolve_diffusion_source(0.3), float)

    def test_array_per_point_indexes(self):
        arr = np.linspace(0.2, 0.8, 11)
        assert resolve_diffusion_source(arr, index=0) == pytest.approx(0.2)
        assert resolve_diffusion_source(arr, index=10) == pytest.approx(0.8)

    def test_array_batch_collapses_to_mean(self):
        """Batch path (index=None) collapses an array to its mean — MFGProblem's own
        array -> scalar convention, so the batch and per-point paths stay consistent."""
        arr = np.array([0.2, 0.4, 0.6])
        assert resolve_diffusion_source(arr, index=None) == pytest.approx(0.4)

    def test_array_out_of_range_index_falls_to_mean(self):
        arr = np.array([0.2, 0.4, 0.6])
        assert resolve_diffusion_source(arr, index=99) == pytest.approx(0.4)

    def test_callable_per_point_evaluates_at_point(self):
        def sigma(x):
            return 0.5 + 0.5 * float(np.atleast_1d(x)[0])

        assert resolve_diffusion_source(sigma, index=0, points=self._POINTS) == pytest.approx(0.5)
        assert resolve_diffusion_source(sigma, index=10, points=self._POINTS) == pytest.approx(1.0)

    def test_callable_batch_evaluates_at_domain_center(self):
        """Batch path evaluates the callable at the domain center (mean of points) — NOT a
        hardcoded placeholder (the #1316 sigma=1.0 regression class this single source kills)."""

        def sigma(x):
            return 0.5 + 0.5 * float(np.atleast_1d(x)[0])

        assert resolve_diffusion_source(sigma, index=None, points=self._POINTS) == pytest.approx(0.75)

    def test_callable_without_points_fails_loud(self):
        """A callable source with no points cannot be evaluated; refusing to substitute a
        placeholder sigma is the fail-loud contract (Issue #1412)."""
        with pytest.raises(ValueError, match="callable volatility source needs `points`"):
            resolve_diffusion_source(lambda x: 0.3, index=None, points=None)

    def test_matches_hjb_gfdm_private_adapter(self):
        """Convention agreement — HJBGFDMSolver._resolve_diffusion_source must now be a thin
        adapter over the shared function (same scalar for the same source/point)."""
        import warnings

        from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=grid, T=0.2, Nt=10, sigma=0.3, components=_default_components())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = HJBGFDMSolver(problem, collocation_points=self._POINTS, delta=0.3)

        arr = np.linspace(0.2, 0.8, 11)

        def sigma_cb(x):
            return 0.5 + 0.5 * float(np.atleast_1d(x)[0])

        for source, idx in [(0.42, None), (arr, 3), (arr, None), (sigma_cb, 5), (sigma_cb, None)]:
            assert solver._resolve_diffusion_source(source, idx) == pytest.approx(
                resolve_diffusion_source(source, index=idx, points=solver.collocation_points)
            )

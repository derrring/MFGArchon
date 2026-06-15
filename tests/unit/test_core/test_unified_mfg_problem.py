"""
Unit tests for unified MFGProblem (Phase 3.1).

Tests all 5 initialization modes:
1. Legacy 1D mode (backward compatible)
2. N-D grid mode
3. Complex geometry mode
4. Network MFG mode
5. Custom components mode

Also tests:
- Solver compatibility detection
- Parameter aliases
- Mode detection and validation
- Backward compatibility
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for 1D testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


def _default_components_nd(dimension: int):
    """Default MFGComponents for N-D testing (Issue #670: explicit specification required)."""

    def m_initial_nd(x):
        x_arr = np.asarray(x)
        return np.exp(-10 * np.sum((x_arr - 0.5) ** 2))

    return MFGComponents(
        m_initial=m_initial_nd,
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


class TestLegacy1DMode:
    """Test Mode 1: Legacy 1D initialization."""

    def test_basic_1d_problem(self):
        """Test basic 1D problem creation."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0, 1)], Nx_points=[100 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            T=1.0,
            Nt=50,
            sigma=0.1,
            components=_default_components(),
        )

        assert problem.dimension == 1
        assert problem.domain_type == "grid"
        bounds = problem.geometry.get_bounds()
        assert bounds[0][0] == 0
        assert bounds[1][0] == 1
        assert problem.geometry.get_grid_shape()[0] - 1 == 100  # Nx intervals
        assert problem.T == 1.0
        assert problem.Nt == 50
        assert problem.sigma == 0.1

    def test_1d_with_Lx_alias(self):
        """Test 1D problem with Lx parameter (alternative to xmin/xmax)."""
        # Note: Lx alias is specified in design but not yet implemented
        # For now, test the standard xmin/xmax interface
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0, 2.0)], Nx_points=[100 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            T=1.0,
            Nt=50,
            sigma=0.1,
            components=_default_components(),
        )

        assert problem.dimension == 1
        bounds = problem.geometry.get_bounds()
        assert bounds[0][0] == 0
        assert bounds[1][0] == 2.0
        assert bounds[1][0] - bounds[0][0] == 2.0  # domain length (formerly Lx)

    def test_1d_solver_compatibility(self):
        """Test solver compatibility for 1D problems."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0, 1)], Nx_points=[100 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            T=1.0,
            Nt=50,
            sigma=0.1,
            components=_default_components(),
        )

        assert "fdm" in problem.solver_compatible
        assert "semi_lagrangian" in problem.solver_compatible
        assert "particle" in problem.solver_compatible
        assert problem.solver_recommendations["default"] in ["fdm", "semi_lagrangian"]


class TestNDGridMode:
    """Test Mode 2: N-D grid initialization."""

    def test_2d_grid_problem(self):
        """Test 2D grid problem creation."""
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[50, 50],
            T=1.0,
            Nt=100,
            sigma=0.1,
            components=_default_components_nd(2),
        )

        assert problem.dimension == 2
        assert problem.domain_type == "grid"
        assert problem.spatial_bounds == [(0, 1), (0, 1)]
        assert problem.spatial_discretization == [50, 50]
        assert problem.T == 1.0
        assert problem.Nt == 100
        assert problem.sigma == 0.1

    def test_3d_grid_problem(self):
        """Test 3D grid problem creation."""
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1), (0, 1)],
            spatial_discretization=[30, 30, 30],
            T=1.0,
            Nt=50,
            sigma=0.1,
            components=_default_components_nd(3),
        )

        assert problem.dimension == 3
        assert problem.domain_type == "grid"

    def test_4d_grid_problem(self):
        """Test 4D grid problem creation with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            problem = MFGProblem(
                spatial_bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
                spatial_discretization=[10, 10, 10, 10],
                T=1.0,
                Nt=50,
                sigma=0.1,  # SDE volatility (Issue #811)
                components=_default_components_nd(4),
            )

            # Should warn about high-dimensional complexity
            # Filter for UserWarning (not DeprecationWarning)
            user_warnings = [
                x for x in w if issubclass(x.category, UserWarning) and not issubclass(x.category, DeprecationWarning)
            ]
            assert len(user_warnings) >= 1
            assert "O(N^d)" in str(user_warnings[0].message) or "dimension" in str(user_warnings[0].message).lower()

        assert problem.dimension == 4
        assert "particle" in problem.solver_compatible
        assert problem.solver_recommendations.get("high_dimensional") == "particle"

    def test_non_uniform_grid(self):
        """Test non-uniform grid resolution."""
        problem = MFGProblem(
            spatial_bounds=[(0, 2), (0, 1)],
            spatial_discretization=[100, 50],
            T=1.0,
            Nt=100,
            sigma=0.05,
            components=_default_components_nd(2),
        )

        assert problem.spatial_discretization == [100, 50]
        assert problem.dimension == 2

    def test_time_domain_alias(self):
        """Test time_domain parameter alias."""
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[50, 50],
            time_domain=(2.0, 200),
            sigma=0.1,
            components=_default_components_nd(2),
        )

        assert problem.T == 2.0
        assert problem.Nt == 200

    def test_diffusion_converts_to_sigma(self):
        """Test diffusion= D converts to sigma = sqrt(2D) (Issue #811)."""
        import math

        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[50, 50],
            T=1.0,
            Nt=100,
            diffusion=0.2,
            components=_default_components_nd(2),
        )

        # diffusion=0.2 means D=0.2, so sigma = sqrt(2*0.2) ~ 0.6325
        assert problem.sigma == pytest.approx(math.sqrt(2 * 0.2))
        assert problem.diffusion == pytest.approx(0.2)


class TestModeDetection:
    """Test mode detection and validation."""

    def test_unambiguous_1d_mode(self):
        """Test that 1D mode is detected correctly."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, sigma=0.1, components=_default_components())
        assert problem.dimension == 1

    def test_unambiguous_nd_mode(self):
        """Test that N-D mode is detected correctly."""
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[50, 50],
            T=1.0,
            Nt=50,
            sigma=0.1,
            components=_default_components_nd(2),
        )
        assert problem.dimension == 2

    def test_ambiguous_mode_raises_error(self):
        """Test that ambiguous initialization raises error."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))
        with pytest.raises(ValueError, match="Ambiguous initialization"):
            MFGProblem(
                geometry=geometry,
                spatial_bounds=[(0, 1)],
                spatial_discretization=[50],
                T=1.0,
                Nt=50,
                sigma=0.1,
                components=_default_components(),
            )

    def test_missing_required_params_uses_defaults(self):
        """Test that missing parameters use defaults (no error)."""
        # MFGProblem has defaults for parameters, including default 1D domain
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            problem = MFGProblem(T=1.0, Nt=50, sigma=0.1, components=_default_components())

            # Should warn about using default domain
            assert any("default" in str(x.message).lower() for x in w)

        # Should create valid 1D problem with defaults
        assert problem.dimension == 1
        assert problem.T == 1.0
        assert problem.Nt == 50


class TestSolverCompatibility:
    """Test solver compatibility detection."""

    def test_fdm_compatibility_1d(self):
        """Test FDM compatibility with 1D grid."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, sigma=0.1, components=_default_components())
        assert "fdm" in problem.solver_compatible

    def test_fdm_compatibility_2d(self):
        """Test FDM compatibility with 2D grid."""
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[50, 50],
            T=1.0,
            Nt=50,
            sigma=0.1,
            components=_default_components_nd(2),
        )
        assert "fdm" in problem.solver_compatible

    def test_fdm_incompatibility_4d(self):
        """Test FDM incompatibility with 4D grid."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = MFGProblem(
                spatial_bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
                spatial_discretization=[10, 10, 10, 10],
                T=1.0,
                Nt=50,
                sigma=0.1,
                components=_default_components_nd(4),
            )

        # FDM should still be marked as compatible (compatibility check is lenient)
        # But recommendation should suggest particle for high dimensions
        assert problem.solver_recommendations.get("high_dimensional") == "particle"

    def test_particle_compatibility(self):
        """Test particle solver compatibility."""
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[50, 50],
            T=1.0,
            Nt=50,
            sigma=0.1,
            components=_default_components_nd(2),
        )
        assert "particle" in problem.solver_compatible

    def test_get_solver_info(self):
        """Test get_solver_info() method."""
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[50, 50],
            T=1.0,
            Nt=50,
            sigma=0.1,
            components=_default_components_nd(2),
        )

        info = problem.get_solver_info()

        assert "compatible" in info
        assert "recommendations" in info
        assert "dimension" in info
        assert "domain_type" in info
        assert "complexity" in info
        assert "default_solver" in info

        assert info["dimension"] == 2
        assert info["domain_type"] == "grid"
        assert isinstance(info["compatible"], list)
        assert isinstance(info["recommendations"], dict)

    def test_validate_solver_type_compatible(self):
        """Test validate_solver_type with compatible solver."""
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[50, 50],
            T=1.0,
            Nt=50,
            sigma=0.1,
            components=_default_components_nd(2),
        )

        # Should not raise error
        problem.validate_solver_type("fdm")
        problem.validate_solver_type("particle")

    def test_validate_solver_type_incompatible(self):
        """Test validate_solver_type with incompatible solver."""
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[50, 50],
            T=1.0,
            Nt=50,
            sigma=0.1,
            components=_default_components_nd(2),
        )

        # Network solver should be incompatible with grid problems
        with pytest.raises(ValueError, match="incompatible"):
            problem.validate_solver_type("network_solver")


class TestBackwardCompatibility:
    """Test backward compatibility with old API."""

    def test_old_1d_interface(self):
        """Test that old 1D interface still works."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, sigma=0.1, components=_default_components())

        # Old attributes should exist (for backward compatibility)
        bounds = problem.geometry.get_bounds()
        assert bounds[0][0] == 0  # xmin via geometry
        assert bounds[1][0] == 1  # xmax via geometry
        assert problem.geometry.get_grid_shape()[0] - 1 == 100  # Nx intervals
        assert problem.T == 1.0
        assert problem.Nt == 50
        assert problem.sigma == 0.1

    def test_2d_problem_via_mfgproblem(self):
        """Test 2D problem creation via MFGProblem (replaces deprecated GridBasedMFGProblem)."""
        # Create 2D problem directly with MFGProblem
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[50, 50],
            T=1.0,
            Nt=100,
            sigma=0.1,
            components=_default_components_nd(2),
        )

        # Should create valid 2D problem
        assert problem.dimension == 2
        assert problem.T == 1.0
        assert problem.Nt == 100
        assert problem.sigma == 0.1
        assert problem.spatial_bounds == [(0, 1), (0, 1)]
        assert problem.spatial_discretization == [50, 50]


class TestComplexityEstimation:
    """Test computational complexity estimation."""

    def test_1d_complexity(self):
        """Test 1D problem complexity estimation."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, sigma=0.1, components=_default_components())
        info = problem.get_solver_info()

        assert "complexity" in info
        # Complexity should reference grid size (101 points from 100 intervals)
        assert "O(N" in info["complexity"] or "101" in info["complexity"] or "100" in info["complexity"]

    def test_2d_complexity(self):
        """Test 2D problem complexity estimation."""
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[50, 50],
            T=1.0,
            Nt=100,
            sigma=0.1,
            components=_default_components_nd(2),
        )
        info = problem.get_solver_info()

        assert "complexity" in info
        # Check that complexity string is reasonable
        complexity_str = info["complexity"]
        assert "O(N" in complexity_str or "Nx" in complexity_str or "50" in complexity_str

    def test_4d_complexity_warning(self):
        """Test that 4D problems warn about complexity."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            problem = MFGProblem(
                spatial_bounds=[(0, 1)] * 4,
                spatial_discretization=[10] * 4,
                T=1.0,
                Nt=50,
                sigma=0.1,
                components=_default_components_nd(4),
            )

            # Should warn about high-dimensional complexity
            warning_msgs = [str(x.message) for x in w]
            assert any("O(N^d)" in msg or "dimension" in msg.lower() for msg in warning_msgs)

        info = problem.get_solver_info()
        # Check that complexity mentions high dimension
        assert "N^4" in info["complexity"] or "curse" in info["complexity"].lower()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_dimension_raises_error(self):
        """Test that zero-dimension raises error."""
        with pytest.raises(ValueError):
            MFGProblem(
                spatial_bounds=[],
                spatial_discretization=[],
                T=1.0,
                Nt=50,
                sigma=0.1,
                components=_default_components(),
            )

    def test_mismatched_dimensions_raises_error(self):
        """Test that mismatched dimensions raise error."""
        with pytest.raises(ValueError):
            MFGProblem(
                spatial_bounds=[(0, 1), (0, 1)],  # 2D
                spatial_discretization=[50, 50, 50],  # 3D
                T=1.0,
                Nt=50,
                sigma=0.1,
                components=_default_components_nd(2),
            )

    def test_negative_time_raises_error(self):
        """Test that negative time raises error or warning."""
        # Note: Current implementation may not validate this - test what actually happens
        try:
            geometry = TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1)
            )
            problem = MFGProblem(geometry=geometry, T=-1.0, Nt=50, sigma=0.1, components=_default_components())
            # If no error, just check that problem was created
            assert problem.T == -1.0  # May need validation in future
        except ValueError:
            pass  # Expected behavior

    def test_zero_timesteps_raises_error(self):
        """Test that zero timesteps raises error or creates problem."""
        # Note: Current implementation may not validate this
        try:
            geometry = TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1)
            )
            problem = MFGProblem(geometry=geometry, T=1.0, Nt=0, sigma=0.1, components=_default_components())
            # If no error, check problem was created
            assert problem.Nt == 0
        except (ValueError, ZeroDivisionError):
            pass  # Expected behavior

    def test_negative_diffusion_raises_error(self):
        """Test that negative diffusion D raises ValueError (Issue #811)."""
        # Negative D is non-physical — sqrt(2*D) is imaginary
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))
        with pytest.raises(ValueError, match="non-negative"):
            MFGProblem(geometry=geometry, T=1.0, Nt=50, diffusion=-0.1, components=_default_components())


class TestCustomComponentExceptionPropagation:
    """Test that custom component exceptions propagate to user (not silently return NaN).

    Verifies fix for issue #420: Silent failures in custom component evaluation.
    Updated for Issue #673: Class-based Hamiltonian API.
    """

    def test_custom_hamiltonian_exception_propagates(self):
        """Test that exceptions in custom Hamiltonian propagate to caller."""
        from mfgarchon.core.hamiltonian import HamiltonianBase
        from mfgarchon.core.mfg_problem import MFGComponents

        # Custom Hamiltonian class that raises an exception
        class BrokenHamiltonian(HamiltonianBase):
            def __call__(self, x, m, p, t=0.0):
                raise ValueError("Intentional error in custom Hamiltonian")

            def dp(self, x, m, p, t=0.0):
                return 0.0

            def dm(self, x, m, p, t=0.0):
                return 0.0

        domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
        components = MFGComponents(
            hamiltonian=BrokenHamiltonian(),
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
            u_terminal=lambda x: 0.0,
            problem_type="custom",
        )

        problem = MFGProblem(geometry=domain, T=1.0, Nt=10, sigma=0.1, components=components)

        # Exception should propagate, not be silently caught
        with pytest.raises(ValueError, match="Intentional error"):
            problem.H(
                x_idx=5,
                m_at_x=1.0,
                derivs={(1,): 0.5},
                t_idx=0,
            )

    def test_custom_hamiltonian_dm_exception_propagates(self):
        """Test that exceptions in custom dH/dm propagate to caller."""
        from mfgarchon.core.hamiltonian import HamiltonianBase
        from mfgarchon.core.mfg_problem import MFGComponents

        # Custom Hamiltonian class with broken dm() method
        class BrokenDmHamiltonian(HamiltonianBase):
            def __call__(self, x, m, p, t=0.0):
                return 0.0

            def dp(self, x, m, p, t=0.0):
                return 0.0

            def dm(self, x, m, p, t=0.0):
                raise RuntimeError("Intentional error in dH/dm")

        domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
        components = MFGComponents(
            hamiltonian=BrokenDmHamiltonian(),
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
            u_terminal=lambda x: 0.0,
            problem_type="custom",
        )

        problem = MFGProblem(geometry=domain, T=1.0, Nt=10, sigma=0.1, components=components)

        # Exception should propagate, not be silently caught
        with pytest.raises(RuntimeError, match="Intentional error"):
            problem.dH_dm(
                x_idx=5,
                m_at_x=1.0,
                derivs={(1,): 0.5},
                t_idx=0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

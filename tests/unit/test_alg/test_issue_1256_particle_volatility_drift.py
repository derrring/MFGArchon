"""
Pinning tests for Issue #1256: FPParticleSolver volatile-field shape validation
and drift_is_precomputed on 1D paths.

(A) Anisotropic (d,d) volatility matrix routed through spatial interpolator -> silent garbage.
    Part 1 fix (Issue #1256): the nD CPU path now IMPLEMENTS anisotropic Σ via the
    per-particle increment ΔX = Σ(x_p) @ dW_p (constant (d,d) and spatial (*grid,d,d)).
    Shape validation still rejects (d,), wrong-grid spatial fields, and mismatched
    matrix-trailing arrays. Quantitative covariance validation lives in
    test_fp_particle_anisotropic_sigma_1256.py.
(B) drift_is_precomputed=True silently ignored on 1D CPU/GPU paths (always differentiates U).
    Still deferred (Refs #1256): raise NotImplementedError when drift_is_precomputed=True
    and dimension==1.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc, periodic_bc


def _hamiltonian():
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _1d_problem():
    """Minimal 1D MFGProblem for issue #1256 tests."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[10],
        boundary_conditions=neumann_bc(dimension=1),
    )
    return MFGProblem(
        geometry=geometry,
        T=0.1,
        Nt=5,
        sigma=0.1,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=_hamiltonian(),
        ),
    )


def _2d_problem():
    """Minimal 2D MFGProblem for issue #1256 tests."""

    def m_initial_2d(x):
        x_arr = np.asarray(x)
        return np.exp(-10 * np.sum((x_arr - 0.5) ** 2))

    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[10, 10],
        boundary_conditions=periodic_bc(dimension=2),
    )
    return MFGProblem(
        geometry=geometry,
        T=0.1,
        Nt=5,
        sigma=0.1,
        components=MFGComponents(
            m_initial=m_initial_2d,
            u_terminal=lambda x: 0.0,
            hamiltonian=_hamiltonian(),
        ),
    )


class TestIssue1256VolatilityShapeValidation:
    """Bug (A) Part 1: (d,d) anisotropic Sigma is now SUPPORTED on the nD CPU path;
    only genuinely-unsupported array shapes still raise."""

    def test_2d_diagonal_sigma_matrix_now_supported(self):
        """
        Issue #1256 Part 1: a (d,d) anisotropic Σ on the nD CPU path is implemented.

        volatility_field=np.diag([s1, s2]) is a (2,2) matrix passed to a 2D problem.
        Before Part 1 (#1274): raised ValueError (fail-loud placeholder).
        After Part 1: runs the per-particle ΔX = Σ @ dW increment and returns a finite
        density. (Quantitative covariance correctness is in the dedicated validation file.)
        """
        problem = _2d_problem()
        assert problem.dimension == 2, "problem must be 2D for this test"

        solver = FPParticleSolver(problem, num_particles=50)

        grid_shape = problem.geometry.get_grid_shape()
        m0 = np.ones(grid_shape) / np.prod(grid_shape)

        # Callable drift — routes through _solve_fp_system_callable_drift
        def zero_drift(t, x, m):
            return np.zeros_like(x)

        anisotropic_sigma = np.diag([0.1, 0.2])  # shape (2,2) — anisotropic noise matrix

        result = solver.solve_fp_system(
            M_initial=m0,
            drift_field=zero_drift,
            volatility_field=anisotropic_sigma,
            show_progress=False,
        )
        assert result.ndim == 3  # (time, Nx, Ny)
        assert result.shape[1:] == grid_shape
        assert np.all(np.isfinite(result))

    def test_mismatched_matrix_trailing_array_raises_valueerror(self):
        """
        A matrix-trailing array whose leading dims do not match the grid is unsupported.

        E.g. (3, 3, 2, 2) on a (10, 10) grid: trailing (2,2) is a Σ matrix but the leading
        (3,3) is neither the grid nor empty. Must fail loud (not silently interpolated).
        """
        problem = _2d_problem()
        solver = FPParticleSolver(problem, num_particles=50)
        grid_shape = problem.geometry.get_grid_shape()
        m0 = np.ones(grid_shape) / np.prod(grid_shape)

        def zero_drift(t, x, m):
            return np.zeros_like(x)

        bad = np.zeros((3, 3, 2, 2))  # leading (3,3) != grid (10,10)

        with pytest.raises(ValueError, match="leading dims"):
            solver.solve_fp_system(
                M_initial=m0,
                drift_field=zero_drift,
                volatility_field=bad,
                show_progress=False,
            )

    def test_per_axis_sigma_array_raises_valueerror(self):
        """
        Per-axis (d,) volatility array must be rejected with a clear message.

        Before the fix: hits a cryptic broadcast error inside the interpolator.
        After the fix: raises ValueError with a message about ambiguous shape.
        """
        problem = _2d_problem()
        assert problem.dimension == 2, "problem must be 2D for this test"

        solver = FPParticleSolver(problem, num_particles=50)

        grid_shape = problem.geometry.get_grid_shape()
        m0 = np.ones(grid_shape) / np.prod(grid_shape)

        def zero_drift(t, x, m):
            return np.zeros_like(x)

        per_axis_sigma = np.array([0.1, 0.2])  # shape (2,) — ambiguous/unsupported

        with pytest.raises(ValueError, match="ambiguous"):
            solver.solve_fp_system(
                M_initial=m0,
                drift_field=zero_drift,
                volatility_field=per_axis_sigma,
                show_progress=False,
            )

    def test_wrong_shape_spatial_sigma_raises_valueerror(self):
        """
        Spatial sigma array with shape != grid_shape must be rejected.

        After the fix: raises ValueError with a shape-mismatch message.
        """
        problem = _2d_problem()
        assert problem.dimension == 2, "problem must be 2D for this test"

        solver = FPParticleSolver(problem, num_particles=50)

        grid_shape = problem.geometry.get_grid_shape()
        m0 = np.ones(grid_shape) / np.prod(grid_shape)

        def zero_drift(t, x, m):
            return np.zeros_like(x)

        # Wrong spatial shape (5x5 instead of grid_shape=(10,10))
        wrong_shape_sigma = np.full((5, 5), 0.1)

        with pytest.raises(ValueError, match="grid shape"):
            solver.solve_fp_system(
                M_initial=m0,
                drift_field=zero_drift,
                volatility_field=wrong_shape_sigma,
                show_progress=False,
            )

    def test_valid_spatial_sigma_array_accepted(self):
        """
        A spatial sigma array with shape == grid_shape should still be accepted.

        Regression guard: the fix must not break valid usage.
        """
        problem = _2d_problem()
        assert problem.dimension == 2, "problem must be 2D for this test"

        solver = FPParticleSolver(problem, num_particles=50)

        grid_shape = problem.geometry.get_grid_shape()
        m0 = np.ones(grid_shape) / np.prod(grid_shape)

        def zero_drift(t, x, m):
            return np.zeros_like(x)

        # sigma field matching grid_shape exactly — should be accepted
        valid_sigma = np.full(grid_shape, 0.1)

        result = solver.solve_fp_system(
            M_initial=m0,
            drift_field=zero_drift,
            volatility_field=valid_sigma,
            show_progress=False,
        )
        assert result.ndim == 3  # (time, Nx, Ny)
        assert result.shape[1:] == grid_shape
        assert np.all(np.isfinite(result))


class TestIssue1256DriftIsPrecomputed1D:
    """Bug (B): drift_is_precomputed=True silently ignored on 1D paths."""

    def test_1d_drift_is_precomputed_raises_not_implemented(self):
        """
        Pinning test for Issue #1256 bug B.

        Before the fix: self._drift_is_precomputed is stored but never read in the 1D
        CPU/GPU paths; they unconditionally differentiate drift_field as U, producing wrong
        drift or a crash.
        After the fix: raises NotImplementedError with a message referencing #1256.
        """
        problem = _1d_problem()
        assert problem.dimension == 1, "problem must be 1D for this test"

        solver = FPParticleSolver(problem, num_particles=50)

        Nx = problem.geometry.num_spatial_points
        Nt = problem.Nt

        m0 = np.ones(Nx) / Nx
        # Precomputed velocity array alpha(t,x) — zero constant drift
        alpha_precomputed = np.zeros((Nt, Nx))

        with pytest.raises(NotImplementedError, match="drift_is_precomputed"):
            solver.solve_fp_system(
                M_initial=m0,
                drift_field=alpha_precomputed,
                drift_is_precomputed=True,
                show_progress=False,
            )

    def test_1d_drift_is_precomputed_false_still_works(self):
        """
        drift_is_precomputed=False (default) on 1D must still work after the fix.

        Regression guard: the fail-loud check must only fire for True.
        """
        problem = _1d_problem()
        assert problem.dimension == 1, "problem must be 1D for this test"

        solver = FPParticleSolver(problem, num_particles=50)

        Nx = problem.geometry.num_spatial_points
        Nt = problem.Nt

        m0 = np.ones(Nx) / Nx
        # Value function U (not precomputed alpha) — default VALUE_FUNCTION path
        U = np.zeros((Nt, Nx))

        result = solver.solve_fp_system(
            M_initial=m0,
            drift_field=U,
            drift_is_precomputed=False,
            show_progress=False,
        )
        assert result.shape == (Nt + 1, Nx)
        assert np.all(np.isfinite(result))

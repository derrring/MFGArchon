"""
Pinning tests for Issue #1079: anisotropic sigma-tensor silent drops.

Three sites were identified. Two remain open after earlier partial fixes:

  Site 3 — GFDM (joint_socp path): e_lap cross-derivative target is zero, so
    a full (d,d) sigma tensor passed to HJBGFDMSolver is silently collapsed to
    a scalar mean in MFGProblem.sigma (via np.mean over all tensor entries), then
    used as an isotropic coefficient. No error, no warning.
    Fix: raise NotImplementedError at HJBGFDMSolver construction.

  Site 4 — HJB-SL-ADI: the convention is sigma_tensor = covariance matrix
    (sigma*sigma^T). Diagonal ADI and explicit cross-derivative both assume
    symmetry. A non-symmetric tensor silently produces wrong results.
    Fix: raise ValueError if sigma_tensor is not symmetric.

Both tests FAIL on pre-fix code (silent wrong result) and PASS after the fix.
Refs #1079.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import adi_diffusion_step

# =============================================================================
# Site 4 — HJB-SL-ADI: non-symmetric sigma raises ValueError
# =============================================================================


class TestADIAsymmetricSigmaRaises:
    """Issue #1079, Site 4: adi_diffusion_step must reject non-symmetric sigma_tensor.

    Convention: sigma_tensor = covariance (sigma*sigma^T), which is always
    symmetric. An asymmetric tensor means the caller is violating the convention;
    letting it through silently produces wrong cross-derivative coefficients.

    FAILS on pre-fix code (no symmetry check) and PASSES after fix (ValueError raised).
    """

    def test_asymmetric_2d_sigma_raises_value_error(self) -> None:
        """Non-symmetric (d,d) sigma_tensor must raise ValueError, not silently proceed."""
        Nx, Ny = 8, 8
        dx, dy = 0.1, 0.1
        grid_shape = (Nx, Ny)
        spacing = np.array([dx, dy])
        U = np.random.default_rng(0).standard_normal(grid_shape)

        # Intentionally asymmetric: sigma[0,1] != sigma[1,0]
        sigma_asym = np.array([[0.1, 0.05], [0.03, 0.1]])
        assert not np.allclose(sigma_asym, sigma_asym.T), "Setup: tensor must be asymmetric"

        with pytest.raises(ValueError, match="symmetric"):
            adi_diffusion_step(U, dt=0.01, sigma=sigma_asym, spacing=spacing, grid_shape=grid_shape)

    def test_symmetric_off_diagonal_does_not_raise(self) -> None:
        """Symmetric (d,d) sigma with off-diagonal must NOT raise."""
        Nx, Ny = 8, 8
        dx, dy = 0.1, 0.1
        grid_shape = (Nx, Ny)
        spacing = np.array([dx, dy])
        U = np.random.default_rng(1).standard_normal(grid_shape)

        sigma_sym = np.array([[0.1, 0.04], [0.04, 0.1]])
        assert np.allclose(sigma_sym, sigma_sym.T), "Setup: tensor must be symmetric"

        # Should not raise; cross-derivative is applied
        U_out = adi_diffusion_step(U, dt=0.01, sigma=sigma_sym, spacing=spacing, grid_shape=grid_shape)
        assert U_out.shape == grid_shape

    def test_symmetric_off_diagonal_cross_term_applied(self) -> None:
        """With a symmetric off-diagonal sigma, the result must differ from diagonal-only.

        This confirms the cross-derivative is actually applied (not silently dropped).
        Uses a quadratic initial condition so the cross-derivative is nonzero.
        """
        Nx, Ny = 12, 12
        dx, dy = 0.1, 0.1
        grid_shape = (Nx, Ny)
        spacing = np.array([dx, dy])
        xs = np.arange(Nx) * dx
        ys = np.arange(Ny) * dy
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        # u(x,y) = x*y has d^2u/dx dy = 1 everywhere (nonzero cross-derivative)
        U = X * Y

        # Diagonal sigma: no cross term
        sigma_diag = np.array([[0.1, 0.0], [0.0, 0.1]])
        U_diag = adi_diffusion_step(U.copy(), dt=0.01, sigma=sigma_diag, spacing=spacing, grid_shape=grid_shape)

        # Full symmetric sigma with nonzero off-diagonal: cross term is applied
        b = 0.05
        sigma_full = np.array([[0.1, b], [b, 0.1]])
        U_full = adi_diffusion_step(U.copy(), dt=0.01, sigma=sigma_full, spacing=spacing, grid_shape=grid_shape)

        # Results must differ at interior points (cross-derivative contribution nonzero)
        interior = np.s_[1:-1, 1:-1]
        max_diff = np.max(np.abs(U_full[interior] - U_diag[interior]))
        assert max_diff > 1e-10, (
            f"Off-diagonal sigma must produce a different result from diagonal-only sigma "
            f"(cross-derivative should be applied). max_diff = {max_diff:.3e}"
        )


# =============================================================================
# Site 3 — GFDM (HJBGFDMSolver): full-tensor sigma raises NotImplementedError
# =============================================================================


class TestGFDMFullTensorSigmaRaises:
    """Issue #1079, Site 3: HJBGFDMSolver must refuse a full (d,d) sigma tensor.

    The GFDM Laplacian stencil target (e_lap in joint_socp.py) has zero weight on
    the cross-derivative column. If problem.volatility_field is a (d,d) tensor, the
    solver would silently use only an isotropic approximation derived from the scalar
    MFGProblem.sigma, dropping all cross-derivative D_ij d^2u/dx_i dx_j terms.

    MFGProblem construction validates that volatility_field matches the spatial grid
    shape (Issue #687), so direct construction with sigma=tensor already fails.
    However, the volatility_field can be a tensor when:
      - Set programmatically after problem construction
      - Received through other internal paths that bypass construction validation

    The HJBGFDMSolver guard catches this case at construction time and provides
    a clear, solver-specific NotImplementedError (Issue #1079).

    FAILS on pre-fix code (no error from HJBGFDMSolver) and PASSES after fix
    (NotImplementedError raised from __init__).
    """

    @staticmethod
    def _make_2d_problem_scalar_sigma():
        """Build a minimal 2D MFGProblem with scalar sigma (passes MFGProblem validation)."""
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
        from mfgarchon.core.mfg_components import MFGComponents
        from mfgarchon.core.mfg_problem import MFGProblem
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc

        ham = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        )
        components = MFGComponents(
            m_initial=lambda x: np.exp(-5.0 * float(np.sum((np.asarray(x) - 0.5) ** 2))),
            u_terminal=lambda x: 0.0,
            hamiltonian=ham,
        )
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        return MFGProblem(
            geometry=domain,
            T=0.1,
            Nt=5,
            sigma=0.1,
            components=components,
        )

    @staticmethod
    def _make_2d_collocation_points():
        """Build a small 2D collocation grid."""
        xs = np.linspace(0.0, 1.0, 10)
        ys = np.linspace(0.0, 1.0, 10)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return np.column_stack([X.ravel(), Y.ravel()])

    def test_full_tensor_volatility_field_raises_not_implemented(self) -> None:
        """HJBGFDMSolver construction raises NotImplementedError when problem.volatility_field
        is a (d,d) tensor.

        Reproduces the silent-drop path: problem is built with scalar sigma (valid), then
        volatility_field is overwritten with a full tensor before the solver is created.
        On pre-fix code, HJBGFDMSolver.__init__ does not check volatility_field shape →
        it silently uses the scalar MFGProblem.sigma mean, dropping cross-derivative terms.
        After the fix, NotImplementedError is raised immediately at construction.
        """
        from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

        problem = self._make_2d_problem_scalar_sigma()
        coll = self._make_2d_collocation_points()

        # Simulate the condition that was silently wrong: volatility_field is a (d,d) tensor
        sigma_tensor = np.array([[0.1, 0.04], [0.04, 0.1]])
        problem.volatility_field = sigma_tensor  # bypass MFGProblem construction validation

        # Pre-fix: HJBGFDMSolver.__init__ has no guard → no error → wrong isotropic solve.
        # Post-fix: NotImplementedError at construction (fail-loud, Issue #1079).
        with pytest.raises(NotImplementedError, match="full-tensor"):
            HJBGFDMSolver(problem, coll)

    def test_scalar_sigma_gfdm_does_not_raise(self) -> None:
        """Scalar sigma must not trigger the tensor guard."""
        from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

        problem = self._make_2d_problem_scalar_sigma()
        coll = self._make_2d_collocation_points()

        # volatility_field is a float scalar — should not raise
        solver = HJBGFDMSolver(problem, coll)
        assert solver is not None

    def test_1d_array_sigma_gfdm_does_not_raise(self) -> None:
        """1-D per-axis diagonal sigma array must not trigger the tensor guard."""
        from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
        from mfgarchon.core.mfg_components import MFGComponents
        from mfgarchon.core.mfg_problem import MFGProblem
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc

        ham = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        )
        components = MFGComponents(
            m_initial=lambda x: np.exp(-5.0 * float(np.sum((np.asarray(x) - 0.5) ** 2))),
            u_terminal=lambda x: 0.0,
            hamiltonian=ham,
        )
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        problem = MFGProblem(
            geometry=domain,
            T=0.1,
            Nt=5,
            sigma=0.1,
            components=components,
        )
        coll = self._make_2d_collocation_points()

        # Simulate a 1-D diagonal sigma stored in volatility_field
        problem.volatility_field = np.array([0.1, 0.08])  # ndim == 1, OK

        # Should not raise: 1-D is diagonal per-axis, not a full tensor
        solver = HJBGFDMSolver(problem, coll)
        assert solver is not None

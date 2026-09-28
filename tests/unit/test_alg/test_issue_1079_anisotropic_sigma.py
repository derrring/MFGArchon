"""
Pinning tests for Issue #1079: anisotropic sigma-tensor silent drops.

Three sites were identified. Two remain open after earlier partial fixes:

  Site 3 — GFDM (joint_socp path): e_lap cross-derivative target is zero, so
    a full (d,d) sigma tensor passed to HJBGFDMSolver is silently collapsed to
    a scalar mean in MFGProblem.sigma (via np.mean over all tensor entries), then
    used as an isotropic coefficient. No error, no warning.
    Fix: raise NotImplementedError at HJBGFDMSolver construction.

  Site 4 — HJB-SL-ADI: the (d,d) sigma is the SYMMETRIC standard-deviation matrix S
    (the symmetric square root of the covariance), D = 1/2 S S^T (RFC #1596, re-founding
    the original #1079 "covariance" reading). Diagonal ADI and explicit cross-derivative
    both require symmetry. A non-symmetric tensor is caller confusion and silently produces
    wrong results. Fix: reject a non-symmetric (d,d) sigma (validate_symmetric_psd).

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
    """Issue #1079 Site 4, re-founded by RFC #1596: adi_diffusion_step must reject a
    non-symmetric (d,d) sigma.

    Convention (RFC #1596): the (d,d) sigma is the SYMMETRIC standard-deviation matrix S
    (the symmetric square root of the covariance), so D = 1/2 S S^T. An asymmetric tensor
    is caller confusion (a Cholesky factor, a raw covariance, or a malformed matrix), which
    would silently produce wrong cross-derivative coefficients. The raise is UNCHANGED from
    the original #1079 fix; only its rationale flips from "covariance must be symmetric" to
    "the volatility std-dev matrix must be symmetric" (PR #1548 had wrongly *removed* this
    raise; RFC #1596 keeps it and routes the squaring through the single-source converter).
    """

    def test_asymmetric_2d_volatility_rejected(self) -> None:
        """Non-symmetric (d,d) sigma must raise ValueError, not silently proceed (RFC #1596)."""
        Nx, Ny = 8, 8
        dx, dy = 0.1, 0.1
        grid_shape = (Nx, Ny)
        spacing = np.array([dx, dy])
        U = np.random.default_rng(0).standard_normal(grid_shape)

        # Intentionally asymmetric: sigma[0,1] != sigma[1,0]
        sigma_asym = np.array([[0.1, 0.05], [0.03, 0.1]])
        assert not np.allclose(sigma_asym, sigma_asym.T), "Setup: tensor must be asymmetric"

        with pytest.raises(ValueError, match="symmetric"):
            adi_diffusion_step(U, dt=0.01, volatility=sigma_asym, spacing=spacing, grid_shape=grid_shape)

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
        U_out = adi_diffusion_step(U, dt=0.01, volatility=sigma_sym, spacing=spacing, grid_shape=grid_shape)
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
        U_diag = adi_diffusion_step(U.copy(), dt=0.01, volatility=sigma_diag, spacing=spacing, grid_shape=grid_shape)

        # Full symmetric sigma with nonzero off-diagonal: cross term is applied
        b = 0.05
        sigma_full = np.array([[0.1, b], [b, 0.1]])
        U_full = adi_diffusion_step(U.copy(), dt=0.01, volatility=sigma_full, spacing=spacing, grid_shape=grid_shape)

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
    def _make_2d_problem_scalar_sigma(**volatility):
        """Build a minimal 2D MFGProblem, with a scalar volatility of 0.1 unless one is passed."""
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
            components=components,
            **(volatility or {"volatility": 0.1}),
        )

    @staticmethod
    def _make_2d_collocation_points():
        """Build a small 2D collocation grid."""
        xs = np.linspace(0.0, 1.0, 10)
        ys = np.linspace(0.0, 1.0, 10)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return np.column_stack([X.ravel(), Y.ravel()])

    def test_full_tensor_volatility_field_raises_not_implemented(self) -> None:
        """HJBGFDMSolver construction raises NotImplementedError for a (d,d) tensor volatility.

        The GFDM Laplacian drops the cross-derivative terms, so solving a tensor volatility as if it
        were isotropic is refused at construction (fail-loud, Issue #1079). The tensor reaches the
        problem through its constructor, MFGProblem(volatility=S, volatility_kind="tensor").
        """
        from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

        problem = self._make_2d_problem_scalar_sigma(
            volatility=np.array([[0.1, 0.04], [0.04, 0.1]]), volatility_kind="tensor"
        )
        coll = self._make_2d_collocation_points()

        with pytest.raises(NotImplementedError, match="tensor volatility"):
            HJBGFDMSolver(problem, coll)

    def test_scalar_sigma_gfdm_does_not_raise(self) -> None:
        """Scalar sigma must not trigger the tensor guard."""
        from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

        problem = self._make_2d_problem_scalar_sigma()
        coll = self._make_2d_collocation_points()

        # volatility_field is a float scalar — should not raise
        solver = HJBGFDMSolver(problem, coll)
        assert solver is not None

    def test_a_field_volatility_gfdm_does_not_raise(self) -> None:
        """A non-tensor array (a per-point field) must not trigger the tensor guard."""
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
            volatility=np.full((10, 10), 0.1),
            volatility_kind="field",
            components=components,
        )
        coll = self._make_2d_collocation_points()

        # Should not raise: a per-point field is not a tensor
        solver = HJBGFDMSolver(problem, coll)
        assert solver is not None

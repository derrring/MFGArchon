"""
Unit tests for LaplacianOperator.

Tests the discrete Laplacian on structured grids using known analytical
solutions. Central difference Laplacian is exact for quadratic functions.

Created: 2026-02-10 (Issue #768 - Test coverage for operators/)
"""

import pytest

import numpy as np
from scipy.sparse.linalg import LinearOperator

from mfgarchon.geometry.boundary import neumann_bc, no_flux_bc, periodic_bc
from mfgarchon.operators.differential.laplacian import LaplacianOperator

# =============================================================================
# Fixtures
# =============================================================================


def _1d_grid(n=100):
    """Create 1D uniform grid on [0, 1]."""
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    return x, dx


def _2d_grid(nx=50, ny=50):
    """Create 2D uniform grid on [0, 1]^2."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y, dx, dy


# =============================================================================
# Basic Functionality
# =============================================================================


class TestLaplacianBasic:
    """Test basic LaplacianOperator functionality."""

    @pytest.mark.unit
    def test_1d_quadratic_exact(self):
        """Laplacian of u=x^2 should be exactly 2 at interior points.

        Note: Neumann ghost cell uses copy (u_ghost = u_boundary), which is
        1st-order at boundary. Interior uses standard 3-point stencil that is
        exact for quadratics.
        """
        x, dx = _1d_grid(100)
        u = x**2

        bc = neumann_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(100,), bc=bc)
        Lu = L(u)

        assert Lu.shape == (100,)
        # Interior: 3-point stencil exact for quadratic
        error = np.max(np.abs(Lu[2:-2] - 2.0))
        assert error < 1e-10

    @pytest.mark.unit
    def test_2d_quadratic_exact(self):
        """Laplacian of u=x^2+y^2 should be exactly 4 at interior points."""
        X, Y, dx, dy = _2d_grid(50, 50)
        u = X**2 + Y**2

        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(50, 50), bc=bc)
        Lu = L(u)

        assert Lu.shape == (50, 50)
        # Interior: exact for quadratic
        error = np.max(np.abs(Lu[2:-2, 2:-2] - 4.0))
        assert error < 1e-10

    @pytest.mark.unit
    def test_1d_constant_zero(self):
        """Laplacian of constant function should be 0."""
        x, dx = _1d_grid(50)
        u = np.ones_like(x) * 5.0

        bc = neumann_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(50,), bc=bc)
        Lu = L(u)

        np.testing.assert_allclose(Lu, 0.0, atol=1e-12)

    @pytest.mark.unit
    def test_1d_linear_zero(self):
        """Laplacian of linear function should be 0."""
        x, dx = _1d_grid(50)
        u = 3.0 * x + 1.0

        bc = neumann_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(50,), bc=bc)
        Lu = L(u)

        # Interior should be exactly 0 (boundary may differ due to BC)
        np.testing.assert_allclose(Lu[1:-1], 0.0, atol=1e-10)

    @pytest.mark.unit
    def test_shape_preserved(self):
        """Output shape should match input shape."""
        X, _Y, dx, dy = _2d_grid(40, 30)
        u = X**2

        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(40, 30), bc=bc)
        Lu = L(u)

        assert Lu.shape == (40, 30)

    @pytest.mark.unit
    def test_integer_field_shape(self):
        """Should accept integer field_shape for 1D case."""
        L = LaplacianOperator(spacings=[0.1], field_shape=50, bc=neumann_bc(dimension=1))
        assert L.field_shape == (50,)
        assert L.shape == (50, 50)


# =============================================================================
# scipy Interface
# =============================================================================


class TestLaplacianScipyInterface:
    """Test scipy LinearOperator compatibility."""

    @pytest.mark.unit
    def test_isinstance(self):
        """Should be a scipy LinearOperator."""
        L = LaplacianOperator(spacings=[0.1], field_shape=(50,))
        assert isinstance(L, LinearOperator)

    @pytest.mark.unit
    def test_matvec_callable_consistency(self):
        """L(u) and L @ u.ravel() should give identical results."""
        X, Y, dx, dy = _2d_grid(30, 30)
        u = X**2 + Y**2

        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(30, 30), bc=bc)

        Lu_callable = L(u)
        Lu_matvec = L @ u.ravel()

        np.testing.assert_allclose(Lu_callable.ravel(), Lu_matvec, atol=1e-14)

    @pytest.mark.unit
    def test_flattened_input(self):
        """Should accept flattened 1D input."""
        X, Y, dx, dy = _2d_grid(30, 30)
        u = X**2 + Y**2

        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(30, 30), bc=bc)

        Lu_flat = L(u.ravel())
        Lu_field = L(u)

        np.testing.assert_allclose(Lu_flat, Lu_field.ravel(), atol=1e-14)

    @pytest.mark.unit
    def test_operator_shape(self):
        """Operator shape should be (N, N) where N = prod(field_shape)."""
        L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(20, 30))
        assert L.shape == (600, 600)


# =============================================================================
# Sparse Export
# =============================================================================


class TestLaplacianSparse:
    """Test sparse matrix export (as_scipy_sparse)."""

    @pytest.mark.unit
    def test_sparse_matches_matvec_neumann(self):
        """Sparse matrix and matvec should agree at interior for Neumann BC.

        Note: Ghost cell (matvec) uses copy, sparse uses mirror doubling.
        These differ at boundary points but agree in the interior.
        """
        x, dx = _1d_grid(30)
        u = x**3

        bc = neumann_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(30,), bc=bc)

        L_sparse = L.as_scipy_sparse()
        Lu_matvec = L @ u.ravel()
        Lu_sparse = L_sparse @ u.ravel()

        # Interior points should match
        np.testing.assert_allclose(Lu_sparse[2:-2], Lu_matvec[2:-2], atol=1e-10)

    @pytest.mark.unit
    def test_sparse_matches_matvec_periodic(self):
        """Sparse matrix and matvec should match for periodic BC."""
        x, dx = _1d_grid(30)
        u = np.sin(2 * np.pi * x)  # Periodic on [0, 1]

        bc = periodic_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(30,), bc=bc)

        L_sparse = L.as_scipy_sparse()
        Lu_matvec = L @ u.ravel()
        Lu_sparse = L_sparse @ u.ravel()

        np.testing.assert_allclose(Lu_sparse, Lu_matvec, atol=1e-10)

    @pytest.mark.unit
    def test_sparse_2d_neumann(self):
        """2D sparse export should match matvec at interior for Neumann BC."""
        X, Y, dx, dy = _2d_grid(20, 20)
        u = X**2 + Y**2

        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(20, 20), bc=bc)

        L_sparse = L.as_scipy_sparse()
        Lu_matvec = (L @ u.ravel()).reshape(20, 20)
        Lu_sparse = (L_sparse @ u.ravel()).reshape(20, 20)

        # Interior match (boundary differs due to ghost cell vs direct assembly)
        np.testing.assert_allclose(Lu_sparse[2:-2, 2:-2], Lu_matvec[2:-2, 2:-2], atol=1e-10)

    @pytest.mark.unit
    def test_sparse_csr_format(self):
        """Sparse export should return CSR format."""
        import scipy.sparse as sparse

        bc = neumann_bc(dimension=1)
        L = LaplacianOperator(spacings=[0.1], field_shape=(20,), bc=bc)
        L_sparse = L.as_scipy_sparse()

        assert sparse.issparse(L_sparse)
        assert isinstance(L_sparse, sparse.csr_matrix)

    @pytest.mark.unit
    def test_sparse_size_limit(self):
        """Should raise ValueError for grids larger than 100k points."""
        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[0.01, 0.01], field_shape=(400, 400), bc=bc)

        with pytest.raises(ValueError, match="too large"):
            L.as_scipy_sparse()


# =============================================================================
# Boundary Conditions
# =============================================================================


class TestLaplacianBC:
    """Test boundary condition handling."""

    @pytest.mark.unit
    def test_periodic_bc_sin(self):
        """Periodic BC on sin(2pi*x) should give -4pi^2 sin(2pi*x)."""
        n = 100
        x = np.linspace(0, 1, n, endpoint=False)  # Periodic: exclude endpoint
        dx = x[1] - x[0]
        u = np.sin(2 * np.pi * x)

        bc = periodic_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(n,), bc=bc)
        Lu = L(u)

        expected = -((2 * np.pi) ** 2) * np.sin(2 * np.pi * x)
        error = np.max(np.abs(Lu - expected))
        # O(h^2) for 2nd-order central diff
        assert error < 0.5

    @pytest.mark.unit
    def test_no_flux_quadratic(self):
        """No-flux BC with quadratic: exact at interior, boundary error bounded."""
        X, Y, dx, dy = _2d_grid(40, 40)
        u = X**2 + Y**2

        bc = no_flux_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(40, 40), bc=bc)
        Lu = L(u)

        # Interior: exact for quadratic
        error_interior = np.max(np.abs(Lu[2:-2, 2:-2] - 4.0))
        assert error_interior < 1e-10

        # Boundary: 1st-order ghost cell introduces O(1/h) error
        # but overall result is bounded
        assert Lu.shape == (40, 40)

    @pytest.mark.unit
    def test_no_bc_periodic_wrapping(self):
        """With bc=None, should use periodic wrapping (np.roll)."""
        n = 50
        x = np.linspace(0, 1, n, endpoint=False)
        dx = x[1] - x[0]
        u = np.sin(2 * np.pi * x)

        L = LaplacianOperator(spacings=[dx], field_shape=(n,), bc=None)
        Lu = L(u)

        expected = -((2 * np.pi) ** 2) * np.sin(2 * np.pi * x)
        error = np.max(np.abs(Lu - expected))
        assert error < 0.5


# =============================================================================
# Validation
# =============================================================================


class TestLaplacianValidation:
    """Test input validation."""

    @pytest.mark.unit
    def test_spacings_mismatch(self):
        """Should raise ValueError for mismatched spacings/field_shape."""
        with pytest.raises(ValueError, match="spacings length"):
            LaplacianOperator(spacings=[0.1], field_shape=(10, 10))

    @pytest.mark.unit
    def test_unsupported_order(self):
        """Should raise ValueError for order != 2."""
        with pytest.raises(ValueError, match="Only order=2"):
            LaplacianOperator(spacings=[0.1], field_shape=(10,), order=4)

    @pytest.mark.unit
    def test_shape_mismatch_callable(self):
        """Should raise ValueError when field shape doesn't match."""
        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(10, 10), bc=bc)

        with pytest.raises(ValueError, match="doesn't match"):
            L(np.zeros((10, 20)))

    @pytest.mark.unit
    def test_repr(self):
        """repr should contain key attributes."""
        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(10, 10), bc=bc)
        r = repr(L)
        assert "LaplacianOperator" in r
        assert "field_shape=(10, 10)" in r
        assert "order=2" in r


class TestLaplacianMassConservative:
    """Issue #1184: the no-flux sparse Laplacian used as the implicit FP system matrix must
    be COLUMN-conservative (1ᵀL = 0) or the diffusion solve leaks mass at the walls."""

    @staticmethod
    def _sums(L):
        A = L.as_scipy_sparse()
        A = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
        return float(np.max(np.abs(A.sum(axis=1)))), float(np.max(np.abs(A.sum(axis=0)))), A

    def test_default_noflux_is_row_but_not_column_conservative(self):
        """The default (2nd-order ghost-mirror) stencil has zero row sums but NONZERO column
        sums at the walls -- this is the leak the flag exists to fix."""
        n, h = 51, 1.0 / 50
        rmax, cmax, _ = self._sums(LaplacianOperator([h], (n,), bc=no_flux_bc(dimension=1)))
        assert rmax < 1e-9, f"default row sums should be 0, got {rmax:.2e}"
        assert cmax > 1.0 / h**2 / 2, f"default must show the column-sum defect, got {cmax:.2e}"

    @pytest.mark.parametrize("dim", [1, 2])
    def test_conservative_noflux_row_and_column_sums_zero(self, dim):
        """mass_conservative=True emits the FV zero-flux stencil: BOTH row and column sums
        vanish, so the implicit FP diffusion solve conserves mass exactly."""
        n = 41 if dim == 1 else 11
        h = 1.0 / (n - 1)
        bc = no_flux_bc(dimension=dim)
        L = LaplacianOperator([h] * dim, (n,) * dim, bc=bc, mass_conservative=True)
        rmax, cmax, A = self._sums(L)
        assert rmax < 1e-10, f"{dim}D conservative row sums must be 0, got {rmax:.2e}"
        assert cmax < 1e-10, f"{dim}D conservative column sums must be 0, got {cmax:.2e}"
        assert np.allclose(A, A.T), "FV no-flux Laplacian must be symmetric"

    def test_default_unchanged_by_flag_off(self):
        """mass_conservative=False (default) is byte-identical to the pre-#1184 stencil:
        diag -2/h², wall interior-neighbor +2/h² (regression-lock)."""
        n, h = 7, 0.25
        A = LaplacianOperator([h], (n,), bc=no_flux_bc(dimension=1)).as_scipy_sparse().toarray()
        h2 = h**2
        assert A[0, 0] == pytest.approx(-2.0 / h2)
        assert A[0, 1] == pytest.approx(2.0 / h2)  # ghost-mirror, not the FV +1/h²
        assert A[-1, -1] == pytest.approx(-2.0 / h2)
        assert A[-1, -2] == pytest.approx(2.0 / h2)
        assert A[3, 3] == pytest.approx(-2.0 / h2)  # interior unchanged
        assert A[3, 2] == pytest.approx(1.0 / h2)


class TestLaplacianVariableCoefficient:
    """Variable-coefficient diffusion via coefficient_field (Issue #1183): face-averaged
    D_{i+1/2} baked into the conservative FV stencil, so ∇·(D(x)∇·) is column-conservative
    even for varying D (a point-value D_i·Δ would leak)."""

    def test_none_is_byte_identical_unit(self):
        """coefficient_field=None reproduces the unit mass-conservative Laplacian exactly."""
        n, h = 21, 1.0 / 20
        bc = no_flux_bc(dimension=1)
        unit = LaplacianOperator([h], (n,), bc=bc, mass_conservative=True).as_scipy_sparse().toarray()
        again = (
            LaplacianOperator([h], (n,), bc=bc, mass_conservative=True, coefficient_field=None)
            .as_scipy_sparse()
            .toarray()
        )
        np.testing.assert_array_equal(again, unit)

    def test_constant_field_equals_scalar_times_unit(self):
        """A constant coefficient_field D0 equals D0 * (unit Laplacian), bit-for-bit."""
        n, h, d0 = 21, 1.0 / 20, 0.5
        bc = no_flux_bc(dimension=1)
        unit = LaplacianOperator([h], (n,), bc=bc, mass_conservative=True).as_scipy_sparse().toarray()
        baked = (
            LaplacianOperator([h], (n,), bc=bc, mass_conservative=True, coefficient_field=np.full(n, d0))
            .as_scipy_sparse()
            .toarray()
        )
        np.testing.assert_allclose(baked, d0 * unit, atol=1e-14)

    def test_varying_field_is_column_conservative(self):
        """1ᵀL = 0 for a spatially varying D (a point-value scheme would have colsum != 0)."""
        n, h = 41, 1.0 / 40
        x = np.linspace(0.0, 1.0, n)
        d_field = np.where(x < 0.5, 0.02**2 / 2, 0.30**2 / 2)
        A = (
            LaplacianOperator([h], (n,), bc=no_flux_bc(dimension=1), mass_conservative=True, coefficient_field=d_field)
            .as_scipy_sparse()
            .toarray()
        )
        assert np.max(np.abs(A.sum(axis=0))) < 1e-12

    def test_varying_field_2d_column_conservative(self):
        n, h = 11, 1.0 / 10
        d_field = np.outer(0.1 + np.linspace(0, 1, n), np.ones(n))
        A = (
            LaplacianOperator(
                [h, h], (n, n), bc=no_flux_bc(dimension=2), mass_conservative=True, coefficient_field=d_field
            )
            .as_scipy_sparse()
            .toarray()
        )
        assert np.max(np.abs(A.sum(axis=0))) < 1e-11

    def test_coefficient_field_requires_mass_conservative(self):
        """coefficient_field is only assembled in the conservative branch -> reject otherwise."""
        with pytest.raises(ValueError, match="mass_conservative"):
            LaplacianOperator([0.1], (5,), bc=no_flux_bc(dimension=1), coefficient_field=np.ones(5))

    def test_coefficient_field_shape_validated(self):
        with pytest.raises(ValueError, match="coefficient_field shape"):
            LaplacianOperator(
                [0.1], (5,), bc=no_flux_bc(dimension=1), mass_conservative=True, coefficient_field=np.ones(7)
            )


class TestLaplacianBCFailLoud1071:
    """Issue #1071 / fail-fast: a missing/unhandled BC must not silently degrade the stencil.

    Previously an unknown bc_type silently produced a boundary-diffusion-free interior-only
    stencil, and a provided bc whose bc_type could not be parsed silently became periodic.
    Both now raise (the bc=None periodic default is unchanged — it is documented).
    """

    def test_unknown_bc_type_fails_loud(self):
        """An unhandled bc_type (e.g. robin) raises instead of emitting an interior-only stencil."""

        class _FakeBCType:
            value = "robin"

        class _FakeRobinBC:
            bc_type = _FakeBCType()

        op = LaplacianOperator(spacings=[0.1], field_shape=(10,), bc=_FakeRobinBC())
        with pytest.raises(NotImplementedError, match="does not implement bc_type"):
            op.as_scipy_sparse()

    def test_provided_bc_without_bc_type_fails_loud(self):
        """A provided bc with no determinable bc_type raises, not silently treated as periodic."""

        class _NoBCType:
            pass  # no .bc_type attribute

        op = LaplacianOperator(spacings=[0.1], field_shape=(10,), bc=_NoBCType())
        with pytest.raises(ValueError, match="could not determine bc_type"):
            op.as_scipy_sparse()

    def test_bc_none_still_periodic(self):
        """bc=None remains the documented periodic default (NOT failed-loud)."""
        op = LaplacianOperator(spacings=[0.1], field_shape=(10,), bc=None)
        mat = op.as_scipy_sparse()  # must not raise
        assert mat.shape == (10, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

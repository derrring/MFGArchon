"""Pinning test for Issue #1189: sigma->D migration in FP matrix-assembly files.

Verifies byte-identity (np.testing.assert_array_equal) between the original
inline sigma**2 arithmetic and the canonical diffusion_from_volatility(sigma)
at each migrated site.

Run BEFORE the refactor: assertions against the ORIGINAL patterns pass.
Run AFTER the refactor: assertions against the MIGRATED patterns pass.

The structural invariant being tested: for every sigma and dx pair,
    sigma**2 / (2 * dx**2)   ==  diffusion_from_volatility(sigma) / dx**2
    sigma**2 / dx**2          ==  2 * diffusion_from_volatility(sigma) / dx**2
    sigma**2 / dx**2          ==  2 * diffusion_from_volatility(sigma) / dx**2

These are exact IEEE-754 equalities (0.5*x**2 is the same bit pattern as
x**2/2 due to IEEE multiply-by-0.5 == right-shift-exponent for normal floats).
"""

from __future__ import annotations

import pytest

import numpy as np
import numpy.testing as npt

from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

# ---------------------------------------------------------------------------
# Equivalence proofs (dtype-independent, representative sigma/dx values)
# ---------------------------------------------------------------------------

SIGMA_VALUES = [0.1, 0.2, 0.5, 1.0, 2.0]
DX_VALUES = [0.01, 0.05, 0.1, 0.5]


@pytest.mark.parametrize("sigma", SIGMA_VALUES)
@pytest.mark.parametrize("dx", DX_VALUES)
def test_half_sigma_sq_over_dxsq_equals_D_over_dxsq(sigma, dx):
    """sigma**2/(2*dx**2) == diffusion_from_volatility(sigma)/dx**2.

    This is the core identity for off-diagonal Laplacian coefficients:
        coeff = -(sigma**2) / (2 * dx**2)   [original]
              = -D / dx**2                   [refactored]
    """
    dx_sq = dx * dx
    original = sigma**2 / (2 * dx_sq)
    D = diffusion_from_volatility(sigma)
    refactored = D / dx_sq
    npt.assert_array_equal(
        original,
        refactored,
        err_msg=f"sigma={sigma}, dx={dx}: sigma**2/(2*dx**2) != D/dx**2",
    )


@pytest.mark.parametrize("sigma", SIGMA_VALUES)
@pytest.mark.parametrize("dx", DX_VALUES)
def test_sigma_sq_over_dxsq_equals_2D_over_dxsq(sigma, dx):
    """sigma**2/dx**2 == 2*diffusion_from_volatility(sigma)/dx**2.

    This is the identity for diagonal Laplacian coefficients (sum of off-diagonals
    with both neighbors present, or the ghost-point-reflected boundary term):
        diagonal += sigma**2 / dx**2   [original]
                  = 2 * D / dx**2      [refactored]
    """
    dx_sq = dx * dx
    original = sigma**2 / dx_sq
    D = diffusion_from_volatility(sigma)
    refactored = 2 * D / dx_sq
    npt.assert_array_equal(
        original,
        refactored,
        err_msg=f"sigma={sigma}, dx={dx}: sigma**2/dx**2 != 2*D/dx**2",
    )


@pytest.mark.parametrize("sigma", SIGMA_VALUES)
@pytest.mark.parametrize("dx", DX_VALUES)
def test_sigma_sq_over_dxsq_equals_negative_2D_boundary(sigma, dx):
    """-(sigma**2)/dx**2 == -2*diffusion_from_volatility(sigma)/dx**2.

    Ghost-point boundary off-diagonal coefficient (Issue #668 fix pattern):
        coeff = -(sigma**2) / dx**2   [original]
              = -2 * D / dx**2        [refactored]
    """
    dx_sq = dx * dx
    original = -(sigma**2) / dx_sq
    D = diffusion_from_volatility(sigma)
    refactored = -2 * D / dx_sq
    npt.assert_array_equal(
        original,
        refactored,
        err_msg=f"sigma={sigma}, dx={dx}: -(sigma**2)/dx**2 != -2*D/dx**2",
    )


# ---------------------------------------------------------------------------
# End-to-end matrix coefficient tests (validate actual matrix entries)
# ---------------------------------------------------------------------------


class _FakeGrid:
    """Minimal grid stub for testing matrix-assembly functions."""

    def __init__(self, shape):
        self.shape = shape
        self._ndim = len(shape)

    def get_index(self, multi_idx):
        """Flatten multi-index to flat index (C order)."""
        flat = 0
        stride = 1
        for d in reversed(range(self._ndim)):
            flat += multi_idx[d] * stride
            stride *= self.shape[d]
        return flat


class _FakeBC:
    """Minimal BC stub (non-periodic, non-uniform)."""

    is_uniform = False
    type = "neumann"


def _run_gradient_centered_interior(sigma, dx, dt=0.01, coupling=1.0):
    """Call add_interior_entries_gradient_centered and return matrix triplets."""
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm_alg_gradient_centered import (
        add_interior_entries_gradient_centered,
    )

    N = 5
    shape = (N,)
    grid = _FakeGrid(shape)
    # Interior point (not at boundary)
    flat_idx = 2
    multi_idx = (2,)
    u_flat = np.zeros(N)
    spacing = (dx,)

    row_indices, col_indices, data_values = [], [], []
    add_interior_entries_gradient_centered(
        row_indices=row_indices,
        col_indices=col_indices,
        data_values=data_values,
        flat_idx=flat_idx,
        multi_idx=multi_idx,
        shape=shape,
        ndim=1,
        dt=dt,
        sigma=sigma,
        coupling_coefficient=coupling,
        spacing=spacing,
        u_flat=u_flat,
        grid=grid,
        boundary_conditions=_FakeBC(),
    )
    return dict(zip(col_indices, data_values, strict=True))


def _run_gradient_upwind_interior(sigma, dx, dt=0.01, coupling=1.0):
    """Call add_interior_entries_gradient_upwind and return matrix triplets."""
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm_alg_gradient_upwind import (
        add_interior_entries_gradient_upwind,
    )

    N = 5
    shape = (N,)
    grid = _FakeGrid(shape)
    flat_idx = 2
    multi_idx = (2,)
    u_flat = np.zeros(N)
    spacing = (dx,)

    row_indices, col_indices, data_values = [], [], []
    add_interior_entries_gradient_upwind(
        row_indices=row_indices,
        col_indices=col_indices,
        data_values=data_values,
        flat_idx=flat_idx,
        multi_idx=multi_idx,
        shape=shape,
        ndim=1,
        dt=dt,
        sigma=sigma,
        coupling_coefficient=coupling,
        spacing=spacing,
        u_flat=u_flat,
        grid=grid,
        boundary_conditions=_FakeBC(),
    )
    return dict(zip(col_indices, data_values, strict=True))


def _run_bc_no_flux_interior(sigma, dx, dt=0.01, coupling=1.0):
    """Call add_boundary_no_flux_entries with a point at corner(boundary in d=0,interior in d=1 if 2D).
    For 1D, test the interior-in-d branch by passing a 1D point at index (1,) — boundary only in ndim=1D
    means all points are potentially boundary, so we use a 2D grid and test a corner point
    that is interior in the y-dimension.
    """
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm_bc import add_boundary_no_flux_entries

    Nx, Ny = 5, 5
    shape = (Nx, Ny)
    grid = _FakeGrid(shape)
    # Point at (0, 2): at_lower_boundary in d=0, interior in d=1
    flat_idx = grid.get_index((0, 2))
    multi_idx = (0, 2)
    u_flat = np.zeros(Nx * Ny)
    spacing = (dx, dx)

    row_indices, col_indices, data_values = [], [], []
    add_boundary_no_flux_entries(
        row_indices=row_indices,
        col_indices=col_indices,
        data_values=data_values,
        flat_idx=flat_idx,
        multi_idx=multi_idx,
        shape=shape,
        ndim=2,
        dt=dt,
        sigma=sigma,
        coupling_coefficient=coupling,
        spacing=spacing,
        u_flat=u_flat,
        grid=grid,
    )
    return dict(zip(col_indices, data_values, strict=True))


@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("dx", [0.05, 0.1])
def test_gradient_centered_interior_diagonal_coefficient(sigma, dx):
    """Diagonal of the interior stencil = 1/dt + 2*D/dx^2 (Issue #1189)."""
    dt = 0.01
    entries = _run_gradient_centered_interior(sigma, dx, dt=dt)
    flat_idx = 2
    diagonal = entries[flat_idx]
    D = diffusion_from_volatility(sigma)
    expected_diag = 1.0 / dt + 2 * D / dx**2
    npt.assert_allclose(
        diagonal,
        expected_diag,
        rtol=1e-12,
        err_msg=f"sigma={sigma}, dx={dx}: centered interior diagonal mismatch",
    )


@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("dx", [0.05, 0.1])
def test_gradient_centered_interior_offdiagonal_coefficient(sigma, dx):
    """Off-diagonal of the interior stencil (zero drift) = -D/dx^2 (Issue #1189)."""
    dt = 0.01
    entries = _run_gradient_centered_interior(sigma, dx, dt=dt)
    D = diffusion_from_volatility(sigma)
    # With zero U field, off-diagonals are exactly -D/dx^2
    for idx in [1, 3]:  # neighbors of flat_idx=2
        npt.assert_allclose(
            entries[idx],
            -D / dx**2,
            rtol=1e-12,
            err_msg=f"sigma={sigma}, dx={dx}: centered off-diagonal mismatch at col {idx}",
        )


@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("dx", [0.05, 0.1])
def test_gradient_upwind_interior_diagonal_coefficient(sigma, dx):
    """Diagonal of the upwind interior stencil = 1/dt + 2*D/dx^2 (Issue #1189)."""
    dt = 0.01
    entries = _run_gradient_upwind_interior(sigma, dx, dt=dt)
    flat_idx = 2
    diagonal = entries[flat_idx]
    D = diffusion_from_volatility(sigma)
    expected_diag = 1.0 / dt + 2 * D / dx**2
    npt.assert_allclose(
        diagonal,
        expected_diag,
        rtol=1e-12,
        err_msg=f"sigma={sigma}, dx={dx}: upwind interior diagonal mismatch",
    )


@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("dx", [0.05, 0.1])
def test_gradient_upwind_interior_offdiagonal_coefficient(sigma, dx):
    """Off-diagonal of the upwind interior stencil (zero drift) = -D/dx^2 (Issue #1189)."""
    dt = 0.01
    entries = _run_gradient_upwind_interior(sigma, dx, dt=dt)
    D = diffusion_from_volatility(sigma)
    for idx in [1, 3]:
        npt.assert_allclose(
            entries[idx],
            -D / dx**2,
            rtol=1e-12,
            err_msg=f"sigma={sigma}, dx={dx}: upwind off-diagonal mismatch at col {idx}",
        )


@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("dx", [0.05, 0.1])
def test_bc_no_flux_interior_in_d_coefficients(sigma, dx):
    """Interior-in-d branch of add_boundary_no_flux_entries uses D correctly (Issue #1189)."""
    dt = 0.01
    entries = _run_bc_no_flux_interior(sigma, dx, dt=dt)
    D = diffusion_from_volatility(sigma)
    # In d=1, the interior-in-d branch should contribute -D/dx^2 to the y-neighbors
    # Point (0, 2), Ny=5: y-neighbors are (0,1)=flat_idx_minus and (0,3)=flat_idx_plus
    grid = _FakeGrid((5, 5))
    y_plus_flat = grid.get_index((0, 3))
    y_minus_flat = grid.get_index((0, 1))
    # Off-diagonal in y: -D/dx^2 base (zero drift, so no advection term)
    npt.assert_allclose(
        entries.get(y_plus_flat, 0.0),
        -D / dx**2,
        rtol=1e-12,
        err_msg=f"sigma={sigma}, dx={dx}: bc no-flux interior y+ coefficient mismatch",
    )
    npt.assert_allclose(
        entries.get(y_minus_flat, 0.0),
        -D / dx**2,
        rtol=1e-12,
        err_msg=f"sigma={sigma}, dx={dx}: bc no-flux interior y- coefficient mismatch",
    )


# ---------------------------------------------------------------------------
# Torch diffusion_from_volatility_torch byte-identity tests
# ---------------------------------------------------------------------------


def test_diffusion_from_volatility_torch_numpy_parity():
    """diffusion_from_volatility_torch(sigma) == diffusion_from_volatility(sigma) for scalars."""
    from mfgarchon.utils.pde_coefficients import diffusion_from_volatility_torch

    for sigma in SIGMA_VALUES:
        result_torch = diffusion_from_volatility_torch(sigma)
        result_numpy = diffusion_from_volatility(sigma)
        npt.assert_array_equal(
            float(result_torch),
            float(result_numpy),
            err_msg=f"sigma={sigma}: diffusion_from_volatility_torch != diffusion_from_volatility",
        )


def test_diffusion_from_volatility_torch_preserves_autograd():
    """diffusion_from_volatility_torch preserves torch autograd graph."""
    pytest.importorskip("torch")
    import torch

    from mfgarchon.utils.pde_coefficients import diffusion_from_volatility_torch

    sigma = torch.tensor(0.3, requires_grad=True, dtype=torch.float64)
    D = diffusion_from_volatility_torch(sigma)
    # d(D)/d(sigma) = d(0.5*sigma^2)/d(sigma) = sigma
    grad = torch.autograd.grad(D, sigma)[0]
    npt.assert_allclose(
        float(grad),
        float(sigma),
        rtol=1e-10,
        err_msg="autograd of diffusion_from_volatility_torch should give sigma",
    )


def test_diffusion_from_volatility_torch_byte_identity_scalar():
    """diffusion_from_volatility_torch(sigma) == 0.5*sigma**2 for float scalars (byte-equal)."""
    from mfgarchon.utils.pde_coefficients import diffusion_from_volatility_torch

    for sigma in SIGMA_VALUES:
        npt.assert_array_equal(
            diffusion_from_volatility_torch(sigma),
            0.5 * sigma**2,
            err_msg=f"sigma={sigma}: diffusion_from_volatility_torch not byte-identical to 0.5*sigma**2",
        )

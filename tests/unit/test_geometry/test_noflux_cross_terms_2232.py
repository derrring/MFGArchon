"""A no-flux wall must pass zero flux, including the part built from transverse gradients. #2232.

`div(D grad m)` under no-flux integrates to zero over the domain -- that is what no-flux means, and
the discretisation is already in flux form, so the domain sum telescopes to exactly the boundary
face fluxes. Edge-padded ghosts zero `dm/dx` at an x wall, which kills `D_xx dm/dx` and leaves
`D_xy dm/dy`. With a diagonal tensor `D_xy = 0` and the wall face carried nothing; with an
off-diagonal tensor it carried the leak.

The checks below are on PROPERTIES -- the domain integral, and self-adjointness -- not on boundary
values. #2231 declined to assert over the boundary ring precisely because asserting values there
pins the current treatment as a contract, and this file keeps that discipline.

Self-adjointness is the sharper of the two and was not designed for: `div(D grad .)` with symmetric
`D` under no-flux is self-adjoint in the continuum. The discretisation was 9.8e-02 away from it
before and 5.5e-17 after, which is independent evidence that the boundary treatment is now right
rather than merely conservative -- a scheme can be conservative and still wrong at the wall.
"""

import pytest

import numpy as np

from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.numerical.tensor_calculus import _tensor_diffusion_2d, _tensor_diffusion_nd

DX, DY = 0.025, 0.035
DIAGONAL = np.array([[0.6, 0.0], [0.0, 0.25]])
CROSS = np.array([[0.6, 0.2], [0.2, 0.25]])


def _operator_matrix(nx: int, ny: int, tensor: np.ndarray) -> np.ndarray:
    """The matrix the operator applies, from its action on the standard basis.

    A single application to one smooth field is not a conservation test: a centred Gaussian is
    nearly symmetric, so its leftover wall fluxes nearly cancel and the sum comes out at 1e-15
    even when the operator does not conserve. The issue's own "a single application sums to 0.0"
    was that artefact. Acting on the basis has no symmetry to hide behind.
    """
    bc = no_flux_bc(dimension=2)
    return np.column_stack(
        [
            _tensor_diffusion_2d(np.eye(nx * ny)[k].reshape(nx, ny), tensor, DX, DY, bc, None, 0.0).ravel()
            for k in range(nx * ny)
        ]
    )


@pytest.mark.parametrize(("name", "tensor"), [("diagonal", DIAGONAL), ("cross", CROSS)])
def test_the_operator_conserves_mass_under_no_flux(name, tensor):
    """Column sums are the mass the operator adds per unit of density at each node.

    `diagonal` is the control in the same parametrisation: it held before the fix and must still
    hold, so a change that bought `cross` at its expense fails here.
    """
    a = _operator_matrix(6, 7, tensor)
    assert np.abs(a.sum(axis=0)).max() < 1e-12, f"{name} tensor leaks: {np.abs(a.sum(axis=0)).max():.3e}"


@pytest.mark.parametrize(("name", "tensor"), [("diagonal", DIAGONAL), ("cross", CROSS)])
def test_a_constant_field_is_unchanged(name, tensor):
    """Row sums: the other half, and the half that already held.

    Stated separately because conservation and consistency-on-constants are different properties
    and a wall treatment can buy one with the other. Both must hold.
    """
    a = _operator_matrix(6, 7, tensor)
    assert np.abs(a.sum(axis=1)).max() < 1e-10, f"{name} moves a constant: {np.abs(a.sum(axis=1)).max():.3e}"


@pytest.mark.parametrize(("name", "tensor"), [("diagonal", DIAGONAL), ("cross", CROSS)])
def test_the_operator_is_self_adjoint_for_a_symmetric_tensor(name, tensor):
    """The property conservation alone does not imply, on a uniform grid where the weight is scalar.

    Measured relative asymmetry with the cross tensor: 9.818e-02 before, 5.450e-17 after.
    """
    a = _operator_matrix(6, 7, tensor)
    asymmetry = np.abs(a - a.T).max() / np.abs(a).max()
    assert asymmetry < 1e-12, f"{name} tensor is not self-adjoint: {asymmetry:.3e}"


def test_the_nd_path_conserves_too():
    """`_tensor_diffusion_nd` carried its own copy of the same wall, so it needs its own check.

    Taken for d >= 3, where the 2-D specialisation never runs. Measured on the operator integral
    of a random field: -9.749e-03 before, -2.046e-17 after.
    """
    shape, spacings = (5, 6, 7), (0.02, 0.03, 0.025)
    cross_3d = np.array([[0.6, 0.15, 0.05], [0.15, 0.25, 0.1], [0.05, 0.1, 0.4]])
    field = np.random.default_rng(3).random(shape)
    result = _tensor_diffusion_nd(field, cross_3d, spacings, no_flux_bc(dimension=3))
    integral = result.sum() * float(np.prod(spacings))
    assert abs(integral) < 1e-14, f"3-D no-flux leaks: {integral:.3e}"


def test_an_asymmetric_field_is_what_makes_the_leak_visible():
    """Why the checks above act on the basis, kept as an executable statement rather than a comment.

    A centred Gaussian's wall fluxes nearly cancel by symmetry. This asserts only that the fixed
    operator conserves on BOTH fields -- it does not pin the Gaussian's pre-fix near-cancellation,
    which was a property of that field and not of the scheme.
    """
    nx, ny = 21, 21
    x = np.arange(nx) * DX
    y = np.arange(ny) * DY
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    bc = no_flux_bc(dimension=2)
    centred = np.exp(-(((grid_x - x.mean()) / 0.08) ** 2 + ((grid_y - y.mean()) / 0.10) ** 2))
    skewed = np.exp(-(((grid_x - x[3]) / 0.08) ** 2 + ((grid_y - y[-4]) / 0.10) ** 2))
    for label, field in (("centred", centred), ("skewed", skewed)):
        integral = _tensor_diffusion_2d(field, CROSS, DX, DY, bc, None, 0.0).sum() * DX * DY
        assert abs(integral) < 1e-13, f"{label} field leaks: {integral:.3e}"

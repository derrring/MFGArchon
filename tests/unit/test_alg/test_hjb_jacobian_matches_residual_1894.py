"""The assembled Jacobian must be the Jacobian of the residual it linearises.

An external oracle: a directional finite difference of `compute_hjb_residual` is a law the Jacobian
must reproduce, computed independently of it. Nothing checked this, and the Jacobian's diffusion
block restated the interior three-point stencil while the residual applies a BC-aware Laplacian --
so both boundary rows were wrong by an amount exactly proportional to sigma^2 (#1894).

Why it hid: a wrong Jacobian makes Newton converge slowly or not at all rather than return a wrong
answer -- the residual decides the root -- so it surfaced as inner-solver stalls (#1878) and as the
outer iteration consuming non-roots (#1873), neither of which points at the boundary. And it is
identically absent at sigma = 0, which is what the `fdm_upwind` capability fixture runs.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import compute_hjb_jacobian, compute_hjb_residual
from mfgarchon.backends import create_backend
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc


def _problem(sigma: float, nx: int) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(
        geometry=grid,
        Nt=10,
        T=1.0,
        sigma=sigma,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


def _directional_error(sigma: float, nx: int = 21, eps: float = 1e-6):
    """max |J v - dF/dv| per component, for a fixed direction, plus the split by location."""
    problem = _problem(sigma, nx)
    bc = problem.geometry.boundary_conditions
    bounds = np.array([[0.0, 1.0]])
    backend = create_backend("numpy")

    x = np.linspace(0.0, 1.0, nx)
    u = -3.0 * np.exp(-8 * (x - 0.5) ** 2)
    u_next = np.zeros(nx)
    m = np.exp(-10 * (x - 0.5) ** 2)
    m /= m.sum() * (1.0 / (nx - 1))

    jac = compute_hjb_jacobian(u, u, m, problem, 5, backend, None, True, bc=bc, domain_bounds=bounds).toarray()

    def residual(state):
        return np.asarray(
            compute_hjb_residual(state, u_next, m, problem, 5, backend, None, True, bc=bc, domain_bounds=bounds),
            dtype=float,
        )

    rng = np.random.default_rng(1894)
    v = rng.standard_normal(nx)
    v /= np.linalg.norm(v)
    fd = (residual(u + eps * v) - residual(u - eps * v)) / (2.0 * eps)
    err = np.abs(jac @ v - fd)
    return err[0], err[-1], float(err[1:-1].max())


@pytest.mark.parametrize("sigma", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_the_jacobian_linearises_the_residual_at_the_boundary(sigma: float):
    """The boundary rows are the ones that were wrong; sigma = 0 is the control that was always right.

    Before the fix these read 3.1e-09, 6.5e-02, 4.0e-01, 1.6e+00, 6.5e+00 at row 0 -- the ratio to
    sigma^2 constant at 6.4568 to five figures.
    """
    first, last, _ = _directional_error(sigma)
    assert first < 1e-6, f"row 0: |Jv - dF/dv| = {first:.3e}"
    assert last < 1e-6, f"row -1: |Jv - dF/dv| = {last:.3e}"


def test_the_interior_was_already_right_and_still_is():
    """Guards the fix from over-reaching: the interior is a finite-difference truncation floor, not
    zero, and it must not move. It was 3.32e-06 before and after."""
    _, _, interior = _directional_error(0.5)
    assert 1e-7 < interior < 1e-4, f"interior max = {interior:.3e}"


def test_the_extraction_refuses_an_operator_that_has_no_jacobian():
    """A nonlinear operator has no constant Jacobian, and that must fail rather than be approximated.

    The first version of this test used a pentadiagonal stub, on the assumption that the extraction
    could only represent a narrow band. It cannot stay: the fallback tier probes one column at a
    time and represents any *linear* operator exactly, so a wider band is no longer the thing that
    cannot be carried. Nonlinearity is.
    """
    from mfgarchon.alg.numerical.hjb_solvers import base_hjb

    def nonlinear(u, dx, bc=None, domain_bounds=None, time=0.0):
        return np.asarray(u, dtype=float) ** 2

    original = base_hjb._compute_laplacian_1d
    base_hjb._compute_laplacian_1d = nonlinear
    try:
        with pytest.raises(ValueError, match="not linear in U"):
            base_hjb._bc_laplacian_bands(9, 0.125, None, 0.0)
    finally:
        base_hjb._compute_laplacian_1d = original


def test_a_wide_operator_is_carried_rather_than_refused():
    """The fallback tier's reason to exist, stated as a test rather than left to the comment.

    Obstacle masks, source terms and nonlocal operators all produce operators the comb probes cannot
    attribute; before the fallback existed they raised, and five otherwise-passing tests went red.
    """
    from mfgarchon.alg.numerical.hjb_solvers import base_hjb

    def pentadiagonal(u, dx, bc=None, domain_bounds=None, time=0.0):
        a = np.asarray(u, dtype=float)
        return a + np.roll(a, 2)

    original = base_hjb._compute_laplacian_1d
    base_hjb._compute_laplacian_1d = pentadiagonal
    try:
        sub, diag, sup, extras = base_hjb._bc_laplacian_bands(9, 0.125, None, 0.0)
        probe = np.linspace(0.3, 2.1, 9)
        got = diag * probe
        got[1:] += sub[1:] * probe[:-1]
        got[:-1] += sup[:-1] * probe[1:]
        for i, j, value in extras:
            got[i] += value * probe[j]
        assert np.allclose(got, pentadiagonal(probe, 0.125)), "the wide operator was not represented"
    finally:
        base_hjb._compute_laplacian_1d = original


def test_periodic_wrap_with_a_volatility_field_does_not_collapse_the_jacobian():
    """The combination this file could not see: periodic BC (extras non-empty) AND array sigma.

    Every other case in this module builds `no_flux_bc`, for which `_bc_laplacian_bands` returns
    `extras == []`, so the off-band assembly path was entered by no test here. With a volatility
    field the off-band value was formed as `-_diffusion * v` with `_diffusion` an (Nx,) array;
    `Jac[i, j] += <array>` raised ValueError, and a pre-existing handler meant for the band shapes
    swapped the whole Jacobian for `(1/dt) * I`. Measured at Nx=21, T=0.2, Nt=10, sigma(x)=0.3+0.4x:
    nnz fell 63 -> 21, `np.allclose(J, I/dt)` was True, and the backward solve returned a non-root
    while 5998 tests stayed green.

    Asserted here as structure rather than as a residual comparison, because the collapse is total:
    an identity Jacobian has no wrap entries and a diagonal of exactly 1/dt.
    """
    nx, dt = 21, 0.02
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=periodic_bc(dimension=1))
    problem = MFGProblem(
        geometry=grid,
        Nt=10,
        T=0.2,
        sigma=0.5,
        components=MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: -(np.asarray(m) ** 2)
            ),
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
        ),
    )
    x = np.linspace(0.0, 1.0, nx)
    field = 0.3 + 0.4 * x
    u = np.zeros(nx)
    m = np.ones(nx) / nx

    jac = compute_hjb_jacobian(u, u, m, problem, t_idx_n=9, sigma_at_n=field, bc=grid.boundary_conditions)
    dense = np.asarray(jac.todense())

    assert not np.allclose(dense, np.eye(nx) / dt), "the Jacobian collapsed to (1/dt) * I"
    assert int((np.abs(dense) > 1e-14).sum()) == 63, "the two periodic wrap entries are missing"

    # The wrap entries must carry the ROW's diffusion, not the column's: dRes_i/dU_j = -D_i L[i, j].
    wrap = [(i, j) for i in range(nx) for j in range(nx) if abs(i - j) > 1 and abs(dense[i, j]) > 1e-14]
    assert len(wrap) == 2, f"expected exactly two off-band entries, got {wrap}"
    dx = 1.0 / (nx - 1)
    for i, j in wrap:
        # dRes_i/dU_j = -D_i * L[i, j], and the 3-point periodic Laplacian's off-band value is 1/dx^2.
        expected = -(0.5 * field[i] ** 2) / dx**2
        assert dense[i, j] == pytest.approx(expected, rel=1e-9), (
            f"wrap entry ({i},{j}) = {dense[i, j]} != {expected}; it does not carry row {i}'s diffusion"
        )

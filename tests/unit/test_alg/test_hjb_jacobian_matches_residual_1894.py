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
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc, periodic_bc


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


@pytest.mark.parametrize("nx", [9, 21, 201, 801])
def test_the_comb_tier_actually_fires_for_a_banded_operator(nx):
    """The O(Nx) tier must be reached, not merely present.

    It was not. `pack` attributed a comb response to every tooth, and a comb is nonzero near EVERY
    tooth, so each tooth received one spurious off-band entry per other tooth's row -- 240 of them
    at Nx=21 under no-flux, where the true operator has none. The control vector then failed every
    time and the O(Nx^2) fallback always ran: probe count was `Nx + 9` for every Nx and every BC,
    and the whole two-tier design was dead code. The full suite could not see it, because nothing
    observed which tier ran; it cost 1311 ms at Nx=1601 against 3.5 ms once fixed.

    Counting probes is the only observable that distinguishes the tiers, so it is what this pins.
    """
    from mfgarchon.alg.numerical.hjb_solvers import base_hjb

    original = base_hjb._compute_laplacian_1d
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    base_hjb._compute_laplacian_1d = counting
    try:
        for bc in (no_flux_bc(dimension=1), periodic_bc(dimension=1)):
            calls.clear()
            base_hjb._bc_laplacian_bands(nx, 1.0 / (nx - 1), bc, 0.0)
            assert len(calls) <= 9, (
                f"nx={nx}, {type(bc).__name__}: {len(calls)} probes -- the comb tier did not fire, "
                f"so the O(Nx^2) fallback ran on a banded operator"
            )
    finally:
        base_hjb._compute_laplacian_1d = original


def test_nx_6_degenerates_the_comb_and_that_is_recorded_rather_than_assumed_away():
    """Nx=6 is the one size where the comb cannot fire, and it is an in-tree fixture.

    `edges` takes {0, 1, 4, 5}, leaving `interior = [2, 3]`; a length below 3 collapses `stride` to
    1, so the single comb probes two ADJACENT columns, `pack` cannot attribute them, the control
    vector fails and tier 2 runs. Not a BC effect -- it is arithmetic on the index sets, identical
    under every BC.

    Pinned because the test above parametrises nx over [9, 21, 201, 801] and a docstring in
    `_extract_bands` generalised that population to "a constant 9 probes for every Nx", which is
    false twice over: the count on that set is 8, and Nx=6 costs 12. `tests/conftest.py`'s
    `tiny_problem` is Nx=6, so the fallback fires in-tree on every run. Found by review (#1899).

    The bands are still exact at Nx=6 -- tier 2 is exact for any structure -- so this records a
    cost, not a defect.
    """
    from mfgarchon.alg.numerical.hjb_solvers import base_hjb

    original = base_hjb._compute_laplacian_1d

    def count_for(nx, bc):
        calls = []

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        base_hjb._compute_laplacian_1d = counting
        try:
            bands = base_hjb._bc_laplacian_bands(nx, 1.0 / (nx - 1), bc, 0.0)
        finally:
            base_hjb._compute_laplacian_1d = original
        return len(calls), bands

    for bc in (no_flux_bc(dimension=1), dirichlet_bc(dimension=1, value=0.0), periodic_bc(dimension=1)):
        n6, bands6 = count_for(6, bc)
        n7, _ = count_for(7, bc)
        assert n6 > 9, f"Nx=6 no longer degenerates ({n6} probes); the recorded cost is stale"
        assert n7 <= 9, f"Nx=7 should still comb ({n7} probes)"

        # ...and the answer is still right, which is why this is a cost and not a defect.
        nx, dx = 6, 1.0 / 5
        zero = base_hjb._compute_laplacian_1d(np.zeros(nx), dx, bc=bc, time=0.0)
        reference = np.zeros((nx, nx))
        for j in range(nx):
            e = np.zeros(nx)
            e[j] = 1.0
            reference[:, j] = base_hjb._compute_laplacian_1d(e, dx, bc=bc, time=0.0) - zero
        sub, diag, sup, extras = bands6
        rebuilt = np.diag(diag) + np.diag(sub[1:], -1) + np.diag(sup[:-1], 1)
        for i, j, v in extras:
            rebuilt[i, j] += v
        assert np.abs(rebuilt - reference).max() < 1e-12, "tier 2 did not extract Nx=6 exactly"


@pytest.mark.parametrize("bc_name", ["no_flux", "periodic", "periodic_endpoint_inclusive"])
def test_extracted_bands_equal_a_column_by_column_reference(bc_name):
    """External oracle for the extraction: apply the operator to every basis vector.

    The comb tier is a shortcut; this is the thing it is a shortcut for, computed without any comb.
    Both periodic conventions are covered because the wrap position differs between them
    (exclusive at (0, Nx-1), ENDPOINT_INCLUSIVE at (0, Nx-2)) and that difference caused three
    successive wrong structural assumptions while #1894 was being written.
    """
    from mfgarchon.alg.numerical.hjb_solvers import base_hjb

    nx = 21
    dx = 1.0 / (nx - 1)
    if bc_name == "no_flux":
        bc = no_flux_bc(dimension=1)
    elif bc_name == "periodic":
        bc = periodic_bc(dimension=1)
    else:
        bc = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=periodic_bc(dimension=1)
        ).boundary_conditions

    zero = base_hjb._compute_laplacian_1d(np.zeros(nx), dx, bc=bc, time=0.0)
    reference = np.column_stack([base_hjb._compute_laplacian_1d(e, dx, bc=bc, time=0.0) - zero for e in np.eye(nx)])

    sub, diag, sup, extras = base_hjb._bc_laplacian_bands(nx, dx, bc, 0.0)
    got = np.diag(diag)
    for i in range(1, nx):
        got[i, i - 1] = sub[i]
    for i in range(nx - 1):
        got[i, i + 1] = sup[i]
    for i, j, value in extras:
        got[i, j] += value

    assert np.abs(got - reference).max() == pytest.approx(0.0, abs=1e-12), (
        f"{bc_name}: extracted bands differ from a column-by-column probe by {np.abs(got - reference).max():.3e}"
    )

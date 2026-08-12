"""The Jacobian's advection block must be the derivative of the gradient the residual applied.

Same external oracle as #1894 (a finite difference of `compute_hjb_residual`, computed without
reference to the Jacobian), pointed at the other block. Two defects, both measured on this fixture:

  item 4 -- the wall rows were the interior stencil. Under no-flux the true central row 0 is
    {0: -4, 1: +4} and the Jacobian wrote {1: +4}; under Dirichlet the true row is {0: +4, 1: +4},
    so the missing entry has the OPPOSITE SIGN between two BCs of the same scheme.
  item 3 -- the branch was selected by a second rule. The residual selects on sign(central), the
    Jacobian selected on sign(grad_upwind); they part at 8 of 41 nodes on a random field and at
    none on a monotone one, which is what every test used.

Why it hid, beyond that: `use_upwind=True` is the default, and under no-flux upwind the true wall
row is EMPTY while the BC forces p=0 there, so `dH/dp` multiplies the spurious diagonal away. The
one configuration the capability fixture runs is the one configuration that masks the defect.

The analytic block is reached only when `backend is None` (`HJBFDMSolver(analytic_jacobian=True)`,
#1607). Passing a NumPy backend routes to the per-point FD fallback instead -- a different code
path, whose agreement says nothing about this one. Every test here passes `backend=None`.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import base_hjb
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import (
    _advection_bands,
    _bc_laplacian_bands,
    _compute_gradient_array_1d,
    _compute_laplacian_1d,
    compute_hjb_jacobian,
    compute_hjb_residual,
)
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, neumann_bc, no_flux_bc, periodic_bc, robin_bc

NX = 21
DX = 1.0 / (NX - 1)
X = np.linspace(0.0, 1.0, NX)
BOUNDS = np.array([[0.0, 1.0]])

BC_FACTORIES = {
    "no_flux": lambda: no_flux_bc(dimension=1),
    "dirichlet": lambda: dirichlet_bc(dimension=1, value=0.0),
    "periodic": lambda: periodic_bc(dimension=1),
}


def _fixture(bc_name: str, sigma: float = 0.3):
    bc = BC_FACTORIES[bc_name]()
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=bc)
    problem = MFGProblem(
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
    m = np.exp(-10 * (X - 0.5) ** 2)
    m /= m.sum() * DX
    return problem, bc, m


def _residual(problem, bc, m, upwind: bool):
    u_next = np.zeros(NX)

    def call(state):
        return np.asarray(
            compute_hjb_residual(state, u_next, m, problem, 5, None, None, upwind, bc=bc, domain_bounds=BOUNDS),
            dtype=float,
        )

    return call


def _jacobian(problem, bc, m, u, upwind: bool):
    return compute_hjb_jacobian(u, u, m, problem, 5, None, None, upwind, bc=bc, domain_bounds=BOUNDS).toarray()


def _fd_columns(residual, u, eps: float = 1e-6):
    """dF/dU by a two-sided difference, one column at a time. Independent of the Jacobian code."""
    out = np.zeros((NX, NX))
    for j in range(NX):
        e = np.zeros(NX)
        e[j] = 1.0
        out[:, j] = (residual(u + eps * e) - residual(u - eps * e)) / (2.0 * eps)
    return out


def _switching_nodes(u, bc, upwind: bool):
    """Rows where the upwind branch is switching -- the residual is not differentiable there.

    A two-sided FD across such a row averages two different one-sided operators and equals neither,
    so it is not a valid oracle at those rows. Central differencing has no branch, hence no
    switching nodes. Returned so every test can say out loud which rows it is not asserting on.
    """
    if not upwind:
        return np.zeros(NX, dtype=bool)
    g_c = _compute_gradient_array_1d(np.asarray(u, dtype=float), DX, bc=bc, upwind=False, time=0.0)
    return np.abs(g_c) < 1e-9


# States with no switching node under any of the three BCs; `_no_switching_states_are_actually_smooth`
# is the control that keeps that claim honest as the fixture changes.
SMOOTH_STATES = {
    "monotone": 2.0 * X**2,
    "bump_off_node": -3.0 * np.exp(-8 * (X - 0.53) ** 2),
    "rough": 0.4 * np.sin(4 * np.pi * X) + 0.1 * np.random.default_rng(1896).standard_normal(NX),
    "piecewise_linear": np.abs(X - 0.53),
}


@pytest.mark.parametrize("bc_name", list(BC_FACTORIES))
@pytest.mark.parametrize("upwind", [False, True])
@pytest.mark.parametrize("state", list(SMOOTH_STATES))
def test_no_switching_states_are_actually_smooth(bc_name: str, upwind: bool, state: str):
    """Positive control for the exclusion below: these states must have nothing to exclude.

    Without this, `_switching_nodes` growing to cover every row would make every assertion in this
    file vacuous while the suite stayed green -- the exclusion would be laundering the measurement.
    """
    _, bc, _ = _fixture(bc_name)
    switching = _switching_nodes(SMOOTH_STATES[state], bc, upwind)
    assert not switching.any(), f"{state} under {bc_name}: switching at rows {np.nonzero(switching)[0].tolist()}"


@pytest.mark.parametrize("bc_name", list(BC_FACTORIES))
@pytest.mark.parametrize("upwind", [False, True])
def test_the_wall_rows_are_the_residuals_own_stencil(bc_name: str, upwind: bool):
    """#1896 item 4, all six cells. These are the rows the hardcoded interior stencil got wrong.

    Before the fix, at Nx=9: central row 0 read {1: +4} against a true {0: -4, 1: +4} (no-flux) and
    {0: +4, 1: +4} (Dirichlet), and the periodic wrap entry was absent entirely.
    """
    problem, bc, m = _fixture(bc_name)
    u = SMOOTH_STATES["bump_off_node"]
    err = np.abs(_jacobian(problem, bc, m, u, upwind) - _fd_columns(_residual(problem, bc, m, upwind), u))
    assert err[0].max() < 1e-5, f"row 0: {err[0].max():.3e}"
    assert err[-1].max() < 1e-5, f"row -1: {err[-1].max():.3e}"


@pytest.mark.parametrize("bc_name", list(BC_FACTORIES))
@pytest.mark.parametrize("upwind", [False, True])
@pytest.mark.parametrize("state", list(SMOOTH_STATES))
def test_every_row_linearises_the_residual(bc_name: str, upwind: bool, state: str):
    """The whole matrix, not just the ends -- item 3's divergence is interior.

    `piecewise_linear` is the discriminating case for how the branch is recovered: forward and
    backward agree in VALUE on every linear stretch, so reading the branch off values alone is
    blind there while the two ROWS still differ. That failure measured 4.000e+01, eps-independent.
    """
    problem, bc, m = _fixture(bc_name)
    u = SMOOTH_STATES[state]
    err = np.abs(_jacobian(problem, bc, m, u, upwind) - _fd_columns(_residual(problem, bc, m, upwind), u))
    worst = int(np.argmax(err.max(axis=1)))
    assert err.max() < 1e-5, f"worst row {worst}: {err.max():.3e}"


def test_the_two_branch_selection_rules_actually_disagree_on_this_fixture():
    """Positive control for item 3: without a state where they part, the test above proves nothing.

    The residual selects on sign(central); the superseded Jacobian selected on sign(grad_upwind).
    On a monotone state they agree at every node, which is why this was invisible.
    """
    _, bc, _ = _fixture("no_flux")
    rough = SMOOTH_STATES["rough"]
    central = _compute_gradient_array_1d(rough, DX, bc=bc, upwind=False, time=0.0)
    upwind_grad = _compute_gradient_array_1d(rough, DX, bc=bc, upwind=True, time=0.0)
    parted = (central >= 0) != (upwind_grad >= 0)
    assert parted.sum() >= 2, f"the rules agree everywhere on `rough` ({parted.sum()} nodes differ)"

    monotone = SMOOTH_STATES["monotone"]
    c_mono = _compute_gradient_array_1d(monotone, DX, bc=bc, upwind=False, time=0.0)
    u_mono = _compute_gradient_array_1d(monotone, DX, bc=bc, upwind=True, time=0.0)
    assert not ((c_mono >= 0) != (u_mono >= 0)).any(), "monotone was supposed to be the blind case"


def test_a_switching_node_gets_a_clarke_element_not_an_average():
    """Where the residual is not differentiable, the row must still be one of the admissible ones.

    The two-sided FD is not the oracle here -- it averages the two one-sided operators and equals
    neither. So each branch is isolated instead: tilt the two neighbours antisymmetrically to pin
    the selection to one side without moving the node's own value, and take the FD there. The
    Jacobian row must converge to ONE of the two as the tilt shrinks, and the two must stay far
    apart (otherwise "matches a branch" is satisfied by anything).
    """
    problem, bc, m = _fixture("no_flux")
    row = NX // 2
    u = np.abs(X - 0.5)  # kink exactly on `row`: central gradient there is 0
    assert _switching_nodes(u, bc, True)[row], "fixture no longer switches at this row"

    jac = _jacobian(problem, bc, m, u, True)
    residual = _residual(problem, bc, m, True)

    def branch_row(direction: float, tilt: float) -> np.ndarray:
        shifted = u.copy()
        shifted[row - 1] -= direction * tilt
        shifted[row + 1] += direction * tilt
        return _fd_columns(residual, shifted)[row]

    forward, backward = branch_row(-1.0, 1e-5), branch_row(+1.0, 1e-5)
    separation = float(np.abs(forward - backward).max())
    nearest = min(float(np.abs(jac[row] - forward).max()), float(np.abs(jac[row] - backward).max()))
    assert separation > 1.0, f"the two branches coincide ({separation:.3e}); nothing is being tested"
    assert nearest < 0.05 * separation, f"row {row} matches neither branch: {nearest:.3e} vs {separation:.3e}"

    # The tilt is a perturbation of the state, so `nearest` is O(tilt), not zero. Shrinking it by
    # 10x must shrink the gap by ~10x -- that is what says the row IS a branch rather than near one.
    coarse = min(
        float(np.abs(jac[row] - branch_row(-1.0, 1e-4)).max()),
        float(np.abs(jac[row] - branch_row(+1.0, 1e-4)).max()),
    )
    assert nearest < 0.2 * coarse, f"gap does not shrink with the tilt: {coarse:.3e} -> {nearest:.3e}"


def test_the_flat_state_the_repo_defaults_to_is_a_total_branch_degeneracy():
    """`u_terminal = 0` is this repo's own default, and it makes every node a tie.

    Forward and backward agree in value everywhere (both zero), so the value comparison decides
    nothing at any row. The residual's advection term is quadratic in p, so the true derivative is
    zero and the two-sided FD leaves an O(eps) tail rather than a defect -- pinned by its scaling,
    since a fixed tolerance here would pass over a genuinely wrong row of the same size.
    """
    problem, bc, m = _fixture("no_flux")
    u = np.zeros(NX)
    jac = _jacobian(problem, bc, m, u, True)
    residual = _residual(problem, bc, m, True)
    coarse = float(np.abs(jac - _fd_columns(residual, u, eps=1e-4)).max())
    fine = float(np.abs(jac - _fd_columns(residual, u, eps=1e-6)).max())
    assert fine < 0.02 * coarse, f"eps-independent error at the flat state: {coarse:.3e} -> {fine:.3e}"


@pytest.mark.parametrize("bc_name", list(BC_FACTORIES))
def test_a_volatility_field_does_not_disturb_the_advection_block(bc_name: str):
    """Row-indexed sigma is #1894's B1; the advection block row-indexes dH/dp for the same reason.

    An (Nx,) coefficient written whole into an off-band entry raises ValueError, and the assembly's
    pre-existing handler answers that by replacing the entire Jacobian with (1/dt)*I -- a plausible
    wrong answer rather than a crash. Periodic is the case that has off-band entries at all.
    """
    problem, bc, m = _fixture(bc_name)
    sigma_field = 0.2 + 0.3 * np.sin(np.pi * X) ** 2
    u = SMOOTH_STATES["bump_off_node"]
    jac = compute_hjb_jacobian(u, u, m, problem, 5, None, sigma_field, True, bc=bc, domain_bounds=BOUNDS).toarray()
    assert not np.allclose(jac, np.eye(NX) / problem.dt), "the Jacobian collapsed to (1/dt)*I"

    u_next = np.zeros(NX)

    def residual(state):
        return np.asarray(
            compute_hjb_residual(state, u_next, m, problem, 5, None, sigma_field, True, bc=bc, domain_bounds=BOUNDS),
            dtype=float,
        )

    assert float(np.abs(jac - _fd_columns(residual, u)).max()) < 1e-5


# The premise the whole upwind construction rests on. Every BC this repo can build must satisfy it,
# because the bands are assembled from `central` and `laplacian` and never from a one-sided stencil.
IDENTITY_BCS = {
    "no_flux": lambda: no_flux_bc(dimension=1),
    "dirichlet": lambda: dirichlet_bc(dimension=1, value=0.7),
    "neumann": lambda: neumann_bc(dimension=1, value=0.3),
    "periodic": lambda: periodic_bc(dimension=1),
    "robin_mixed": lambda: robin_bc(dimension=1, alpha=1.0, beta=1.0, value=0.5),
    "robin_neumann_like": lambda: robin_bc(dimension=1, alpha=0.0, beta=1.0, value=0.0),
    "robin_sign_flipped": lambda: robin_bc(dimension=1, alpha=2.0, beta=-1.0, value=1.0),
}


@pytest.mark.parametrize("bc_name", list(IDENTITY_BCS))
@pytest.mark.parametrize(
    "state",
    ["smooth", "random", "linear"],
)
def test_the_one_sided_stencils_are_algebra_on_the_two_operators_that_have_owners(bc_name: str, state: str):
    """forward = central + (dx/2)*laplacian, backward = central - (dx/2)*laplacian, walls included.

    This is why the upwind Jacobian needs no third stencil of its own: both one-sided operators are
    combinations of two the residual already owns. Robin is in the list because it is the BC most
    likely to pad the Laplacian differently from the gradient, which is the one way this breaks.
    """
    bc = IDENTITY_BCS[bc_name]()
    u = {
        "smooth": -3.0 * np.exp(-8 * (X - 0.53) ** 2),
        "random": np.random.default_rng(1896).standard_normal(NX),
        "linear": 2.0 * X + 0.3,
    }[state]
    central = _compute_gradient_array_1d(u, DX, bc=bc, upwind=False, time=0.0)
    lap = _compute_laplacian_1d(u, DX, bc=bc, time=0.0)
    upwind = _compute_gradient_array_1d(u, DX, bc=bc, upwind=True, time=0.0)
    forward, backward = central + DX / 2 * lap, central - DX / 2 * lap
    residual = float(np.minimum(np.abs(upwind - forward), np.abs(upwind - backward)).max())
    assert residual < 1e-12, f"{bc_name}/{state}: grad_upwind is neither one-sided form ({residual:.3e})"
    assert float(np.abs(forward - backward).max()) > 1e-9, "the two forms coincide; nothing is discriminated"


def test_a_broken_identity_raises_instead_of_picking_the_closer_wrong_answer(monkeypatch):
    """The failure mode this guards is silent, so the guard is part of the contract.

    If some future BC padded the Laplacian differently from the gradient, the branch would be
    recovered by taking the nearer of two wrong reconstructions -- a plausible Jacobian, no error,
    and Newton merely converging badly. No BC in the repo breaks the identity today (all seven
    above hold to 1e-14), so it is broken here deliberately.
    """
    bc = no_flux_bc(dimension=1)
    u = -3.0 * np.exp(-8 * (X - 0.53) ** 2)
    bands = _bc_laplacian_bands(NX, DX, bc, 0.0)
    real = base_hjb._compute_laplacian_1d
    monkeypatch.setattr(base_hjb, "_compute_laplacian_1d", lambda *a, **k: real(*a, **k) * 3.0 + 1.0)

    with pytest.raises(ValueError, match="#1896"):
        _advection_bands(u, DX, bc, 0.0, True, bands)


@pytest.mark.parametrize("upwind", [False, True])
def test_the_legacy_no_bc_path_linearises_its_own_residual_too(upwind: bool):
    """`bc=None` is a separate dispatch branch: both operators fall back to the periodic %Nx roll.

    The parametrised cases above all pass a real BC object, so none of them reaches it. It matters
    because the residual takes the same branch -- whatever operator `bc=None` selects, the Jacobian
    has to be the derivative of that one and not of the BC-aware one.
    """
    problem, _, m = _fixture("no_flux")
    u = SMOOTH_STATES["bump_off_node"]
    u_next = np.zeros(NX)

    def residual(state):
        return np.asarray(compute_hjb_residual(state, u_next, m, problem, 5, None, None, upwind, bc=None), dtype=float)

    jac = compute_hjb_jacobian(u, u, m, problem, 5, None, None, upwind, bc=None).toarray()
    assert float(np.abs(jac - _fd_columns(residual, u)).max()) < 1e-5

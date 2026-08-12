"""The Jacobian's advection block must be the derivative of the gradient the residual applied.

Same external oracle as #1894 (a finite difference of `compute_hjb_residual`, computed without
reference to the Jacobian), pointed at the other block. Two defects, both measured on this fixture:

  item 4 -- the wall rows were the interior stencil. Under no-flux the true central row 0 is
    {0: -4, 1: +4} and the Jacobian wrote {1: +4}; under Dirichlet the true row is {0: +4, 1: +4},
    so the missing entry has the OPPOSITE SIGN between two BCs of the same scheme.
  item 3 -- the branch was selected by a second rule. The residual selects on sign(central), the
    Jacobian selected on sign(grad_upwind); they part at a median 10 of 41 nodes on a noisy field
    (200 seeds, range 6 to 16) and at none at all on a monotone one, which is what every test
    used.

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
from mfgarchon.core.hamiltonian import HEvalState, QuadraticControlCost, SeparableHamiltonian
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
    # Robin is the BC most likely to break the forward/backward identity, by padding the Laplacian
    # differently from the gradient. alpha == beta specifically: the ghost is u_ghost = a*u[0] + c
    # with a = (2 + alpha/beta)/(2 - alpha/beta), so a == 3 there, and that is the wall where review
    # found the wrong branch being chosen -- 2.0000e+01 at row 0.
    "robin_alpha_eq_beta": lambda: robin_bc(dimension=1, alpha=1.0, beta=1.0, value=0.0),
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


# States with no switching node under any BC above; `test_no_switching_states_are_actually_smooth`
# is the control that keeps that claim honest as the fixture changes.
SMOOTH_STATES = {
    "monotone": 2.0 * X**2,
    "bump_off_node": -3.0 * np.exp(-8 * (X - 0.53) ** 2),
    "rough": 0.4 * np.sin(4 * np.pi * X) + 0.1 * np.random.default_rng(1896).standard_normal(NX),
    "piecewise_linear": np.abs(X - 0.53),
}


@pytest.mark.parametrize("bc_name", list(BC_FACTORIES))
@pytest.mark.parametrize("state", list(SMOOTH_STATES))
def test_no_switching_states_are_actually_smooth(bc_name: str, state: str):
    """Positive control for the exclusion below: these states must have nothing to exclude.

    Without this, `_switching_nodes` growing to cover every row would make every assertion in this
    file vacuous while the suite stayed green -- the exclusion would be laundering the measurement.
    """
    _, bc, _ = _fixture(bc_name)
    switching = _switching_nodes(SMOOTH_STATES[state], bc, True)
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


def test_the_flat_state_leaves_an_eps_tail_not_a_defect():
    """`u_terminal = 0` is this repo's own default, and it makes every node a tie.

    Read the name carefully: this pins the FINITE-DIFFERENCE tail, not the branch recovery. At this
    state `dH/dp` is identically zero, so the advection block is multiplied away entirely -- review
    (#1899) showed that replacing the whole output of `_advection_bands` with zero bands changes the
    assembled Jacobian by exactly 0.0 here. No mutation confined to the branch recovery can move
    this assertion, and the earlier name ("a total branch degeneracy") implied otherwise.

    The tie case IS covered, by `piecewise_linear` in `test_every_row_linearises_the_residual` and
    by `test_a_tied_wall_row_...`, both of which are tied in value with `dH/dp != 0`.

    What this does test: the residual's advection term is quadratic in p, so the true derivative is
    zero and the two-sided FD leaves an O(eps) tail rather than a defect -- pinned by its scaling,
    since a fixed tolerance would pass over a genuinely wrong row of the same size.
    """
    problem, bc, m = _fixture("no_flux")
    u = np.zeros(NX)
    # The premise, measured rather than implied by the name.
    x_grid = problem.geometry.get_spatial_grid()
    grad = _compute_gradient_array_1d(u, DX, bc=bc, upwind=True, time=0.0)
    dh_dp = problem.hamiltonian_class.evaluate_dp(
        HEvalState(x=x_grid, p=grad.reshape(-1, 1), m=np.asarray(m, dtype=float), t=0.0)
    ).ravel()
    assert np.abs(dh_dp).max() == 0.0, (
        f"dH/dp is no longer identically zero here ({np.abs(dh_dp).max():.3e}); this test would then "
        f"be reachable by the advection block and its name should say so"
    )
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


def test_a_tied_wall_row_that_is_not_a_switching_node_still_gets_the_right_branch():
    """The fallback's hardest customer: the branch is well defined, and unmeasurable.

    Forward and backward agree in VALUE wherever the Laplacian vanishes. At a WALL that depends on
    the ghost rule, and the cleanest witness is a BC CONSISTENT WITH THE STATE: a linear state of
    slope -2 under `neumann(du/dn = 2)` has its ghost continue the line exactly at the LOW wall, so
    `lap[0] = 0` while `central[0] = -2` -- nowhere near a switching node. The residual takes one
    specific branch, the two ROWS differ, and choosing by the tie alone is worth `4.0000e+01` here.

    The SIGN of `central` is the whole point, and it is asserted below rather than assumed. On a
    tied row `|g_up - backward| <= |g_up - forward|` holds by construction, so the tie-agnostic
    rule always answers "backward"; `g_c >= 0` can only disagree with it where `central < 0`. A
    witness with `central > 0` passes under both rules and measures nothing.

    ~~Robin with alpha == beta~~ was the original witness (found by review of #1899). It stopped
    tying once #1904 threaded the real grid spacing into the ghost buffer: that wall's tie was an
    artefact of the `dx = 1.0` fallback, not a property of the BC. The precondition below is what
    reported that, which is the whole reason it is asserted rather than assumed.

    ~~a slope +2 state read at the HIGH wall~~ replaced it and was itself non-discriminating
    [CORRECTED 2026-08-13, found by independent review of #1906]: `central[-1] = +2 > 0`, the half
    on which the two rules agree, so deleting the tie-break left this test green. What reddened
    instead were four `test_every_row_linearises_the_residual[piecewise_linear-True-*]` cases,
    interior rows that pre-date #1896 -- a real kill count attached to the wrong claim. Slope -2 at
    the low wall is the same construction moved onto the half where the rules part.

    Asserted as a measurement against the residual, not by inspecting which part of the recovery
    fired, so it survives a reimplementation of the branch recovery.
    """
    bc = neumann_bc(dimension=1, value=2.0)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=bc)
    problem = MFGProblem(
        geometry=grid,
        Nt=10,
        T=1.0,
        sigma=0.3,
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
    u = -2.0 * X + 0.3

    assert abs(DX * _compute_laplacian_1d(u, DX, bc=bc, time=0.0)[0]) < 1e-12, "the wall row no longer ties"
    central = _compute_gradient_array_1d(u, DX, bc=bc, upwind=False, time=0.0)
    assert abs(central[0]) > 0.1, f"row 0 is a switching node ({central[0]:.3e}); the test proves nothing"
    assert central[0] < 0, f"central[0] = {central[0]:.3e} > 0; both branch rules agree there"

    err = np.abs(_jacobian(problem, bc, m, u, True) - _fd_columns(_residual(problem, bc, m, True), u))
    assert err[0].max() < 1e-5, f"row 0: {err[0].max():.3e}"


def test_a_cancelled_wrap_entry_is_dropped_rather_than_kept_at_rounding_scale():
    """Pins the sparsity, which the agreement tests cannot see: they compare dense arrays.

    Under periodic upwind, row 0's forward difference reads `U[1]`, not the wrap column, so the
    central and Laplacian contributions to that entry cancel exactly. Keeping the residue would
    leave a `~1e-16` entry in the matrix — numerically invisible, but it changes `nnz` and so
    changes what every downstream sparse solve is handed.
    """
    _, bc, _ = _fixture("periodic")
    lap_bands = _bc_laplacian_bands(NX, DX, bc, 0.0)
    # A strictly monotone state cannot be periodic, so the branch is pinned at row 0 only: the wrap
    # ghost makes `central[0]` follow the sign of `u[1] - u[-1]`. These two differ there and agree
    # nowhere else that matters, which is exactly the discriminating pair.
    takes_forward, takes_backward = -np.sin(2 * np.pi * X), np.sin(2 * np.pi * X)
    assert _compute_gradient_array_1d(takes_forward, DX, bc=bc, upwind=False, time=0.0)[0] < 0
    assert _compute_gradient_array_1d(takes_backward, DX, bc=bc, upwind=False, time=0.0)[0] > 0

    _, _, _, fwd_extras = _advection_bands(takes_forward, DX, bc, 0.0, True, lap_bands)
    _, _, _, back_extras = _advection_bands(takes_backward, DX, bc, 0.0, True, lap_bands)
    scale = 1.0 / DX
    assert all(abs(v) > 1e-6 * scale for _, _, v in fwd_extras), f"a cancelled entry survived: {fwd_extras}"
    assert all(abs(v) > 1e-6 * scale for _, _, v in back_extras), f"a cancelled entry survived: {back_extras}"

    fwd_cells = {(i, j) for i, j, _ in fwd_extras}
    back_cells = {(i, j) for i, j, _ in back_extras}
    assert fwd_cells != back_cells, "both branches reach the same wrap columns; the drop is invisible here"
    assert 0 not in {i for i, _ in fwd_cells}, f"row 0 took forward, so it must not reach a wrap column: {fwd_extras}"


def test_the_jacobian_follows_a_changed_selection_rule_without_being_told(monkeypatch):
    """The reason the branch is measured rather than restated, made testable.

    On every configuration this repo can build, measuring which one-sided form `grad_upwind`
    returned and restating `sign(central) >= 0` give the same answer — so no ordinary test can tell
    a measurement from a second copy of the rule, and the mutation "let the fallback decide
    everything" kills nothing. The difference only appears when the rule CHANGES, which is exactly
    the failure #1896 item 3 is: a second copy that silently stopped agreeing.

    So change it. `gradient_upwind` is inverted here — forward where it took backward — and both the
    residual and the Jacobian go through it. A Jacobian that measures follows; one that restates
    `sign(central)` is now wrong on every non-tied row and cannot pass.
    """
    from mfgarchon.operators.stencils.finite_difference import gradient_backward, gradient_forward

    def inverted(u, axis, h, xp=np):
        fwd, bwd = gradient_forward(u, axis, h, xp), gradient_backward(u, axis, h, xp)
        return xp.where((fwd + bwd) / 2.0 >= 0, fwd, bwd)  # the OPPOSITE of Godunov

    monkeypatch.setattr(base_hjb, "gradient_upwind", inverted)

    problem, bc, m = _fixture("no_flux")
    u = SMOOTH_STATES["rough"]
    # Control: the inverted rule must actually differ from the real one on this state, or the
    # monkeypatch proves nothing.
    central = _compute_gradient_array_1d(u, DX, bc=bc, upwind=False, time=0.0)
    assert (central > 0).any(), "state does not exercise the backward branch"
    assert (central < 0).any(), "state does not exercise the forward branch"

    err = np.abs(_jacobian(problem, bc, m, u, True) - _fd_columns(_residual(problem, bc, m, True), u))
    assert err.max() < 1e-5, f"the Jacobian did not follow the rule change: {err.max():.3e}"


@pytest.mark.parametrize("upwind", [False, True])
def test_a_time_dependent_boundary_reaches_the_advection_block(upwind: bool):
    """`time` is threaded into all three operator calls, and nothing varied it on this path.

    Review (#1899) found that mutating `current_time` to 0.0 inside the `_advection_bands` call
    survives the whole suite: every fixture in this file uses a time-independent BC, so the
    threading was carried by no assertion. A BC whose value moves with `t` makes the wall rows
    depend on it, and the Jacobian must linearise the residual *at that time*, not at zero.
    """
    # The SIGN matters, and the control below is what caught it. With a large POSITIVE wall value
    # the ghost is far above the interior, so `central < 0` at both walls and upwinding selects the
    # branch that never reads the ghost -- the upwind gradient is then genuinely independent of the
    # BC value, and a test built on it would assert nothing while looking fine.
    bc = dirichlet_bc(dimension=1, value=lambda t: -(2.0 + 5.0 * t))
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=bc)
    problem = MFGProblem(
        geometry=grid,
        Nt=10,
        T=1.0,
        sigma=0.3,
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
    u = SMOOTH_STATES["bump_off_node"]
    u_next = np.zeros(NX)
    t = 0.6

    # Control: the BC must actually differ between t=0 and t, or the test cannot see the threading.
    at_zero = _compute_gradient_array_1d(u, DX, bc=bc, upwind=upwind, time=0.0)
    at_t = _compute_gradient_array_1d(u, DX, bc=bc, upwind=upwind, time=t)
    assert np.abs(at_zero - at_t).max() > 1.0, "the BC is not time-dependent on this fixture"

    def residual(state):
        return np.asarray(
            compute_hjb_residual(
                state, u_next, m, problem, 5, None, None, upwind, bc=bc, domain_bounds=BOUNDS, current_time=t
            ),
            dtype=float,
        )

    jac = compute_hjb_jacobian(
        u, u, m, problem, 5, None, None, upwind, bc=bc, domain_bounds=BOUNDS, current_time=t
    ).toarray()
    assert float(np.abs(jac - _fd_columns(residual, u)).max()) < 1e-5


@pytest.mark.parametrize("nx", [21, 51, 201])
def test_a_large_wall_value_does_not_trip_the_identity_guard(nx: int):
    """The guard must not raise on bands it is about to build exactly right.

    `forward`/`backward` recover a small number by cancelling `g_c` against `(dx/2)*lap`, so the
    reconstruction's rounding floor is set by those CANCELLED terms. Scaling the tolerance by
    `g_up` alone made the guard fire whenever a wall value was large enough that the cancelled
    magnitude exceeded it by ~1/eps: at Nx=51 with a Dirichlet value of 1e6, mismatch 4.172e-09
    against an atol of 1.98e-09 while the cancelled terms were 5e+07. Found by review (#1899).

    It escapes to the user: `compute_hjb_jacobian` is called OUTSIDE the `try` in
    `newton_hjb_step`, so the ValueError propagates out of the solve.
    """
    dx = 1.0 / (nx - 1)
    x = np.linspace(0.0, 1.0, nx)
    u = x**2
    for value in (7.0, 1.0e4, 1.0e6):
        bc = dirichlet_bc(dimension=1, value=value)
        lap_bands = _bc_laplacian_bands(nx, dx, bc, 0.0)
        cancelled = float(np.abs(dx / 2 * _compute_laplacian_1d(u, dx, bc=bc, time=0.0)).max())
        survivor = float(np.abs(_compute_gradient_array_1d(u, dx, bc=bc, upwind=True, time=0.0)).max())
        # Control: this fixture must actually exercise the regime, or it asserts nothing.
        if value == 1.0e6:
            assert cancelled > 1e6 * max(survivor, 1.0), (
                f"nx={nx}: cancelled {cancelled:.3e} vs survivor {survivor:.3e} -- no longer the trip regime"
            )
        _advection_bands(u, dx, bc, 0.0, True, lap_bands)  # must not raise

"""Pilot gates for the Hamiltonian single-source contract (Issue #1071, Phase 0+1).

These pin the #1071 PILOT: the Phase-0 contract (``HEvalState`` /
``HamiltonianValues`` / granular ``evaluate_H`` / ``evaluate_dp`` / convenience
``evaluate``) and the Phase-1 migration of ``base_hjb`` residual + analytic
Jacobian onto those primitives.

Gates (per the LOCKED RFC "Post-panel revision"):

* (a) BYTE-IDENTITY -- ``compute_hjb_residual`` and ``compute_hjb_jacobian`` on the
  matched-λ=1 LQ golden equal a reference assembled from the *inline* forms the
  pre-#1071 code used (``np.asarray(H(...))`` / ``np.asarray(H.dp(...))``), pinned
  with ``np.testing.assert_array_equal`` (atol=0).
* (b) INVOCATION-COUNT PIN -- on a NON-LQ FD-``dp`` Hamiltonian (analytic ``dp``
  would make the LQ golden blind to over-compute): the residual path triggers ZERO
  ``evaluate_dp`` calls and the Jacobian path ZERO ``evaluate_H`` calls, over one
  full solve and in isolation.
* (c) INDEPENDENCE -- residual never computes ``∂H/∂p``; Jacobian never recomputes
  ``H`` (the isolation half of (b)).
* (e) CONVENIENCE CONSISTENCY -- ``evaluate()`` returns the same ``H`` / ``dH_dp``
  as the granular primitives.

Plus the Phase-0 contract surface (``HEvalState`` alias, ``HamiltonianValues``
shape, h_eval delegation byte-identity).
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import base_hjb
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import (
    _compute_gradient_array_1d,
    _compute_laplacian_1d,
    compute_hjb_jacobian,
    compute_hjb_residual,
)
from mfgarchon.alg.numerical.hjb_solvers.h_eval import eval_dH_dp_batch, eval_H_batch
from mfgarchon.core.hamiltonian import (
    HamiltonianBase,
    HamiltonianValues,
    HEvalState,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

# ---------------------------------------------------------------------------
# Phase 0 -- contract surface
# ---------------------------------------------------------------------------


def _batch(n=9, d=1):
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 1.0, n).reshape(-1, 1) if d == 1 else rng.uniform(size=(n, d))
    m = rng.uniform(0.1, 1.0, size=n)
    p = rng.uniform(-1.0, 1.0, size=(n, d))
    return x, m, p


def test_heval_state_grad_u_alias_and_fields():
    x, m, p = _batch()
    st = HEvalState(x=x, p=p, m=m, t=0.4)
    assert np.array_equal(st.grad_u, p)  # grad_u is the momentum alias
    assert st.t == 0.4
    # frozen: assignment must fail
    import dataclasses

    try:
        st.t = 0.5  # type: ignore[misc]
        raise AssertionError("HEvalState must be frozen")
    except dataclasses.FrozenInstanceError:
        pass


def test_granular_primitives_byte_identical_to_inline():
    """evaluate_H / evaluate_dp ARE the pre-#1071 inline batch forms, atol=0."""
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    x, m, p = _batch()
    st = HEvalState(x=x, p=p, m=m, t=0.3)
    np.testing.assert_array_equal(H.evaluate_H(st), np.asarray(H(x, m, p, t=0.3), dtype=float))
    np.testing.assert_array_equal(H.evaluate_dp(st), np.asarray(H.dp(x, m, p, t=0.3), dtype=float))
    assert H.evaluate_H(st).dtype == np.float64
    assert H.evaluate_dp(st).dtype == np.float64


def test_h_eval_helpers_delegate_byte_identical():
    """The kept eval_*_batch shims delegate to the primitives (single source, no third layer)."""
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=2.0), coupling=lambda m: m**3)
    x, m, p = _batch()
    st = HEvalState(x=x, p=p, m=m, t=0.1)
    np.testing.assert_array_equal(eval_H_batch(H, x, m, p, 0.1), H.evaluate_H(st))
    np.testing.assert_array_equal(eval_dH_dp_batch(H, x, m, p, 0.1), H.evaluate_dp(st))


def test_evaluate_convenience_consistent_with_primitives():
    """Gate (e): the convenience evaluate() returns the same H / dH_dp as the primitives."""
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.5), coupling=lambda m: m**2)
    x, m, p = _batch()
    st = HEvalState(x=x, p=p, m=m, t=0.2)
    H.physical_sigma = 0.7  # required by the convenience (Phase-0 scaffold)
    hv = H.evaluate(st)
    assert isinstance(hv, HamiltonianValues)
    np.testing.assert_array_equal(hv.H, H.evaluate_H(st))
    np.testing.assert_array_equal(hv.dH_dp, H.evaluate_dp(st))
    # sigma ALWAYS an array (physical volatility), broadcast to (N,)
    assert isinstance(hv.sigma, np.ndarray)
    assert hv.sigma.shape == (x.shape[0],)
    np.testing.assert_array_equal(hv.sigma, np.full(x.shape[0], 0.7))


def test_evaluate_requires_physical_sigma_fail_fast():
    """No silent σ fallback: evaluate() without physical_sigma raises (fail-fast)."""
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    x, m, p = _batch()
    st = HEvalState(x=x, p=p, m=m, t=0.0)
    assert H.physical_sigma is None
    try:
        H.evaluate(st)
        raise AssertionError("evaluate() must fail fast when physical_sigma is unset")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Phase 1 -- base_hjb migration, byte-identity golden (matched-λ=1 LQ)
# ---------------------------------------------------------------------------


def _lq_problem(nx=21):
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),  # matched λ=1
    )
    return MFGProblem(geometry=geometry, T=1.0, Nt=10, components=components)


def _golden_state(problem):
    nx = problem.geometry.get_grid_shape()[0]
    xs = np.linspace(0.0, 1.0, nx)
    u_cur = np.sin(2 * np.pi * xs) + 0.3 * xs**2
    u_np1 = 0.5 * np.cos(np.pi * xs)
    m = 1.0 + 0.5 * np.exp(-8 * (xs - 0.4) ** 2)
    return u_cur, u_np1, m


def test_residual_byte_identical_to_inline_assembly():
    """Gate (a): compute_hjb_residual == reference assembled from the inline H form."""
    problem = _lq_problem()
    H = problem.hamiltonian_class
    nx = problem.geometry.get_grid_shape()[0]
    dx = problem.geometry.get_grid_spacing()[0]
    dt = problem.dt
    u_cur, u_np1, m = _golden_state(problem)
    bc = no_flux_bc(dimension=1)
    sigma, t = 0.3, 0.1

    out = compute_hjb_residual(
        u_cur,
        u_np1,
        m,
        problem,
        t_idx_n=0,
        backend=None,
        sigma_at_n=sigma,
        use_upwind=True,
        bc=bc,
        domain_bounds=None,
        current_time=t,
    )

    # Reference: reproduce the assembly with the literal pre-#1071 inline H form.
    lap = _compute_laplacian_1d(u_cur, dx, bc=bc, domain_bounds=None, time=t)
    grad = _compute_gradient_array_1d(u_cur, dx, bc=bc, upwind=True, time=t)
    x_grid = problem.geometry.get_spatial_grid()
    h_inline = np.asarray(H(x_grid, np.asarray(m, float), grad.reshape(-1, 1), t=t), dtype=float).ravel()
    ref = np.zeros(nx)
    ref += (u_cur - u_np1) / dt
    ref += -diffusion_from_volatility(sigma, kind="field") * lap
    ref += h_inline

    np.testing.assert_array_equal(out, ref)


def test_jacobian_byte_identical_to_inline_assembly():
    """Gate (a): compute_hjb_jacobian == reference assembled from the inline dp form."""
    import scipy.sparse as sparse

    problem = _lq_problem()
    H = problem.hamiltonian_class
    nx = problem.geometry.get_grid_shape()[0]
    dx = problem.geometry.get_grid_spacing()[0]
    dt = problem.dt
    u_cur, _u_np1, m = _golden_state(problem)
    bc = no_flux_bc(dimension=1)
    sigma, t = 0.3, 0.1

    out = compute_hjb_jacobian(
        u_cur,
        u_cur,
        m,
        problem,
        t_idx_n=0,
        backend=None,
        sigma_at_n=sigma,
        use_upwind=True,
        bc=bc,
        domain_bounds=None,
        current_time=t,
    )

    # Reference: reproduce the diagonal-Jacobian assembly with the inline dp form.
    grad = _compute_gradient_array_1d(u_cur, dx, bc=bc, upwind=True, time=t)
    x_grid = problem.geometry.get_spatial_grid()
    dH_dp = np.asarray(H.dp(x_grid, np.asarray(m, float), grad.reshape(-1, 1), t=t), dtype=float).ravel()

    J_D = np.zeros(nx)
    J_L = np.zeros(nx)
    J_U = np.zeros(nx)
    J_D += 1.0 / dt
    J_D += sigma**2 / dx**2
    val_off = -(sigma**2) / (2 * dx**2)
    J_L += val_off
    J_U += val_off
    inv_dx = 1.0 / dx
    backward = grad >= 0
    J_D += dH_dp * np.where(backward, inv_dx, -inv_dx)
    J_L += dH_dp * np.where(backward, -inv_dx, 0.0)
    J_U += dH_dp * np.where(backward, 0.0, inv_dx)
    j_l_roll = np.roll(J_L, -1)
    j_u_roll = np.roll(J_U, 1)
    ref = sparse.spdiags([j_l_roll, J_D, j_u_roll], [-1, 0, 1], nx, nx, format="csr").tocsr()

    np.testing.assert_array_equal(out.toarray(), ref.toarray())


# ---------------------------------------------------------------------------
# Phase 1 -- invocation-count pin + independence (non-LQ FD-dp Hamiltonian)
# ---------------------------------------------------------------------------


class _NonLQFDHamiltonian(HamiltonianBase):
    """Non-LQ Hamiltonian with NO analytic ``dp`` -> ``dp`` uses the base finite
    differences. ``H = (1/3)|p|^3 + sin(2πx) + m^2``: a quadratic control cost
    would make ``dp`` analytic and free, hiding any residual-path over-compute, so
    the cubic + FD-``dp`` form is what makes the invocation gate meaningful.
    """

    def __call__(self, x, m, p, t=0.0):
        p = np.asarray(p, dtype=float)
        x = np.asarray(x, dtype=float)
        m = np.asarray(m, dtype=float)
        if p.ndim == 2:
            kin = np.sum(np.abs(p) ** 3, axis=1) / 3.0
            pot = np.sin(2 * np.pi * x[:, 0])
            return kin + pot + m**2
        return float(np.sum(np.abs(p) ** 3) / 3.0 + np.sin(2 * np.pi * np.atleast_1d(x)[0]) + float(np.mean(m)) ** 2)


class _CountingHamiltonian(_NonLQFDHamiltonian):
    """Spy: counts method-level evaluate_H / evaluate_dp invocations.

    Counts the PRIMITIVES (not raw ``__call__``): the FD ``dp`` legitimately calls
    ``__call__`` to probe ``H``, but that is internal to ``evaluate_dp`` and must NOT
    register as an ``evaluate_H`` (residual-path) call.
    """

    def __init__(self):
        super().__init__()
        self.n_eval_H = 0
        self.n_eval_dp = 0

    def evaluate_H(self, state):
        self.n_eval_H += 1
        return super().evaluate_H(state)

    def evaluate_dp(self, state):
        self.n_eval_dp += 1
        return super().evaluate_dp(state)


def _nonlq_problem(nx=21):
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    H = _CountingHamiltonian()
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=H,
    )
    return MFGProblem(geometry=geometry, T=0.5, Nt=5, components=components), H


def test_residual_path_triggers_zero_dp_evals():
    """Gates (b)+(c): the residual computes H only -- ZERO evaluate_dp calls."""
    problem, H = _nonlq_problem()
    nx = problem.geometry.get_grid_shape()[0]
    xs = np.linspace(0.0, 1.0, nx)
    u_cur = np.sin(2 * np.pi * xs)
    u_np1 = 0.2 * xs
    m = 1.0 + 0.3 * np.cos(np.pi * xs)
    bc = no_flux_bc(dimension=1)

    H.n_eval_H = 0
    H.n_eval_dp = 0
    compute_hjb_residual(
        u_cur,
        u_np1,
        m,
        problem,
        t_idx_n=0,
        backend=None,
        sigma_at_n=0.3,
        use_upwind=True,
        bc=bc,
        domain_bounds=None,
        current_time=0.1,
    )
    assert H.n_eval_dp == 0, "residual path must not compute ∂H/∂p"
    assert H.n_eval_H >= 1, "residual path must compute H"


def test_jacobian_path_triggers_zero_extra_H_evals():
    """Gates (b)+(c): the analytic Jacobian computes ∂H/∂p only -- ZERO evaluate_H calls."""
    problem, H = _nonlq_problem()
    nx = problem.geometry.get_grid_shape()[0]
    xs = np.linspace(0.0, 1.0, nx)
    u_cur = np.sin(2 * np.pi * xs)
    m = 1.0 + 0.3 * np.cos(np.pi * xs)
    bc = no_flux_bc(dimension=1)

    H.n_eval_H = 0
    H.n_eval_dp = 0
    compute_hjb_jacobian(
        u_cur,
        u_cur,
        m,
        problem,
        t_idx_n=0,
        backend=None,
        sigma_at_n=0.3,
        use_upwind=True,
        bc=bc,
        domain_bounds=None,
        current_time=0.1,
    )
    assert H.n_eval_H == 0, "Jacobian path must not recompute H"
    assert H.n_eval_dp >= 1, "Jacobian path must compute ∂H/∂p"


def test_full_solve_invocation_split():
    """Gate (b) over one full solve: both primitives exercised, never cross-contaminated.

    The migrated batch residual/Jacobian path runs with ``backend=None`` (the pure
    BC-aware NumPy path the multi-pop solves use), so the full backward Newton solve
    is driven through ``solve_hjb_system_backward(backend=None)``. The two base_hjb
    assembly functions are wrapped so every primitive call is attributed to its path:
    over the full solve (Newton + line search) the residual must contribute ZERO
    ``evaluate_dp`` calls and the Jacobian ZERO ``evaluate_H`` calls.
    """
    problem, H = _nonlq_problem()
    nt = problem.Nt + 1
    nx = problem.geometry.get_grid_shape()[0]
    bc = no_flux_bc(dimension=1)

    residual_dp_calls = 0
    jacobian_H_calls = 0
    orig_residual = base_hjb.compute_hjb_residual
    orig_jacobian = base_hjb.compute_hjb_jacobian

    def spy_residual(*a, **k):
        nonlocal residual_dp_calls
        before = H.n_eval_dp
        out = orig_residual(*a, **k)
        residual_dp_calls += H.n_eval_dp - before
        return out

    def spy_jacobian(*a, **k):
        nonlocal jacobian_H_calls
        before = H.n_eval_H
        out = orig_jacobian(*a, **k)
        jacobian_H_calls += H.n_eval_H - before
        return out

    base_hjb.compute_hjb_residual = spy_residual
    base_hjb.compute_hjb_jacobian = spy_jacobian
    try:
        H.n_eval_H = 0
        H.n_eval_dp = 0
        M_density = np.ones((nt, nx))
        base_hjb.solve_hjb_system_backward(
            M_density,
            np.zeros(nx),
            np.zeros((nt, nx)),
            problem,
            backend=None,
            use_upwind=True,
            bc=bc,
            domain_bounds=None,
        )
    finally:
        base_hjb.compute_hjb_residual = orig_residual
        base_hjb.compute_hjb_jacobian = orig_jacobian

    assert H.n_eval_H > 0, "full solve must compute H (residual path)"
    assert H.n_eval_dp > 0, "full solve must compute ∂H/∂p (Jacobian path)"
    assert residual_dp_calls == 0, "residual path leaked ∂H/∂p evals over the full solve"
    assert jacobian_H_calls == 0, "Jacobian path leaked H evals over the full solve"

#!/usr/bin/env python3
"""HJBGFDM must preserve one spatial volatility field through every path (Issue #1725).

The original defect appeared in the batch residual and Jacobian, which resolved sigma through
``_get_sigma_value(None)`` and collapsed a field to its mean. The same resolution fork survived
in LLF, DMP, and Howard. These tests pin one solve-level collocation-space coefficient across all
of those consumers.

Every pre-existing array test used a CONSTANT array (``np.full(Nx, 0.5)``), which is its own
mean and therefore passes under both behaviours. These tests use a non-constant field, which is
the only kind that can tell them apart.
"""

import warnings

import pytest

import numpy as np
from scipy.sparse import csr_matrix

from mfgarchon.alg.numerical.coupling.base_mfg import BaseCouplingIterator
from mfgarchon.alg.numerical.gfdm_components import monotonicity_enforcer
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import Hyperrectangle, TensorProductGrid
from mfgarchon.geometry.boundary.conditions import no_flux_bc

_N = 21
_SIGMA_LO, _SIGMA_HI = 0.1, 0.5  # mean 0.3


class _HJBKwargBuilder(BaseCouplingIterator):
    """Minimal stand-in for the coupling seam.

    Subclasses rather than borrowing the single method: `_build_hjb_kwargs` now delegates to
    sibling helpers on the class (`_matches_problem_sigma`, `_require_kwarg`, #1783), and a double
    that copies one method silently loses them. Inheriting keeps the double honest about what the
    seam actually depends on.
    """

    def __init__(self):
        self._hjb_sig_params = {"volatility_field"}
        self._hjb_solver_name = "HJBGFDMSolver"
        self.problem = None  # no problem: a scalar can never be shown equivalent to sigma

    def solve(self, *args, **kwargs):  # pragma: no cover - abstract stub, never called
        raise NotImplementedError

    def get_results(self, *args, **kwargs):  # pragma: no cover - abstract stub, never called
        raise NotImplementedError


def _problem(sigma=0.3):
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[_N],
            boundary_conditions=no_flux_bc(dimension=1),
        ),
        components=MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.5 * np.asarray(x) ** 2,
        ),
        T=0.1,
        Nt=5,
        sigma=sigma,
    )


def _problem_2d(nx, ny, sigma):
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[nx, ny],
            boundary_conditions=no_flux_bc(dimension=2),
        ),
        components=MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
            m_initial=lambda point: np.exp(-np.sum((np.asarray(point) - 0.5) ** 2)),
            u_terminal=lambda point: 0.5 * np.sum(np.asarray(point) ** 2),
        ),
        T=0.1,
        Nt=2,
        sigma=sigma,
    )


def _solver(
    problem,
    points,
    *,
    llf_augmentation=False,
    llf_l_H=None,
    max_newton_iterations=None,
    boundary_conditions=None,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(
            problem,
            collocation_points=np.asarray(points).reshape(-1, 1),
            delta=0.2,
            monotonicity_scheme="none",
            llf_augmentation=llf_augmentation,
            llf_l_H=llf_l_H,
            max_newton_iterations=max_newton_iterations,
            boundary_conditions=boundary_conditions,
        )


def _solve(
    volatility_field,
    *,
    points=None,
    llf_augmentation=False,
    llf_l_H=None,
    return_solver=False,
):
    problem = _problem()
    x = np.linspace(0.0, 1.0, _N) if points is None else np.asarray(points)
    solver = _solver(
        problem,
        x,
        llf_augmentation=llf_augmentation,
        llf_l_H=llf_l_H,
    )
    M = np.tile(np.exp(-10 * (x - 0.5) ** 2), (problem.Nt + 1, 1))
    U = solver.solve_hjb_system(
        M,
        0.5 * x**2,
        np.zeros((problem.Nt + 1, len(x))),
        volatility_field=volatility_field,
    )
    return (U, solver) if return_solver else U


@pytest.mark.unit
def test_nonconstant_array_field_is_not_collapsed_to_its_mean():
    """The defect, stated as the test that was missing.

    linspace(0.1, 0.5) has mean 0.3. Before the fix these two solves were byte-identical.
    """
    field = np.linspace(_SIGMA_LO, _SIGMA_HI, _N)
    u_field = _solve(field)
    u_mean = _solve(float(field.mean()))

    assert not np.array_equal(u_field, u_mean), (
        "the spatial volatility field was collapsed to its mean: the solve is byte-identical "
        f"to passing the scalar {field.mean()}"
    )
    # And the difference must be of the size the field's spread implies, not float noise.
    assert np.max(np.abs(u_field - u_mean)) > 1e-6


@pytest.mark.unit
def test_nonconstant_callable_field_is_not_collapsed_to_the_domain_center():
    """Same defect via the callable branch, which evaluated once at the domain center."""

    def sigma_of_x(pt):
        return _SIGMA_LO + (_SIGMA_HI - _SIGMA_LO) * np.asarray(pt).reshape(-1)[0]

    u_callable = _solve(sigma_of_x)
    u_center = _solve(float(sigma_of_x(np.array([0.5]))))

    assert not np.array_equal(u_callable, u_center), (
        "the callable volatility was evaluated once at the domain center rather than per node"
    )


@pytest.mark.unit
def test_coupling_builder_forwards_the_nonconstant_field_unchanged():
    """The public coupling seam must not replace the array before GFDM sees it."""
    field = np.linspace(_SIGMA_LO, _SIGMA_HI, _N)

    forwarded = _HJBKwargBuilder()._build_hjb_kwargs(volatility_field=field)["volatility_field"]

    assert forwarded is field
    assert not np.array_equal(_solve(forwarded), _solve(float(field.mean())))


@pytest.mark.unit
def test_scalar_field_is_byte_identical_to_the_problem_sigma():
    """The fix must not perturb the scalar path.

    Byte-identity, not a tolerance: consolidating coefficient retrieval is a refactor on this
    path and any float difference would mean it is not.
    """
    u_override = _solve(0.3)
    u_problem = _solve(None)
    np.testing.assert_array_equal(
        u_override,
        u_problem,
        err_msg="scalar volatility_field=0.3 must reproduce problem.sigma=0.3 bit for bit",
    )


@pytest.mark.unit
def test_residual_jacobian_directional_derivative_agrees_under_field():
    """The actual batch residual and Jacobian must consume the same per-node field."""
    problem = _problem()
    x = np.linspace(0.0, 1.0, _N)
    solver = _solver(problem, x)
    field = np.linspace(_SIGMA_LO, _SIGMA_HI, _N)
    M = np.tile(np.exp(-10 * (x - 0.5) ** 2), (problem.Nt + 1, 1))
    solver.solve_hjb_system(M, 0.5 * x**2, volatility_field=field)

    u = 0.2 * np.sin(np.pi * x) + 0.1 * x
    u_next = 0.9 * u
    m = M[0]
    H = problem.hamiltonian_class
    direction = np.cos(2.0 * np.pi * x) + 0.2

    def residual(candidate):
        grad, lap = solver._compute_derivatives_vectorized(candidate)
        return solver._compute_hjb_residual_hamiltonian(
            candidate,
            u_next,
            m,
            grad,
            lap,
            H,
            0.0,
        )

    grad, _ = solver._compute_derivatives_vectorized(u)
    jacobian = solver._compute_hjb_jacobian_hamiltonian(grad, m, H, 0.0)
    epsilon = 1e-6
    finite_difference = (residual(u + epsilon * direction) - residual(u - epsilon * direction)) / (2.0 * epsilon)
    jacobian_action = jacobian @ direction
    relative_error = np.linalg.norm(finite_difference - jacobian_action) / np.linalg.norm(jacobian_action)

    assert relative_error < 1e-7


@pytest.mark.unit
def test_llf_path_consumes_nonconstant_field():
    """LLF must augment each node's base sigma, not the field's scalar mean."""
    field = np.linspace(_SIGMA_LO, _SIGMA_HI, _N)
    u_field, solver = _solve(
        field,
        llf_augmentation=True,
        llf_l_H=0.0,
        return_solver=True,
    )
    u_mean = _solve(
        float(field.mean()),
        llf_augmentation=True,
        llf_l_H=0.0,
    )

    np.testing.assert_array_equal(solver._llf_sigma_eff, field)
    assert not np.array_equal(u_field, u_mean)


@pytest.mark.unit
def test_callable_field_is_evaluated_once_per_collocation_node():
    """A time-invariant spatial callable is normalized once, not inside every Newton probe."""
    calls = 0

    def sigma_of_x(point):
        nonlocal calls
        calls += 1
        x = float(np.asarray(point).reshape(-1)[0])
        return _SIGMA_LO + (_SIGMA_HI - _SIGMA_LO) * x

    _solve(sigma_of_x)

    assert calls == _N


@pytest.mark.unit
@pytest.mark.parametrize("has_defaults", [False, True], ids=["required-full-signature", "defaulted-full-signature"])
def test_non_spatial_callable_fails_before_evaluation(has_defaults):
    """GFDM must not freeze the public ``sigma(t, x, m)`` contract into one static field."""
    calls = 0

    if has_defaults:

        def sigma(t=0.0, x=None, m=None):
            nonlocal calls
            calls += 1
            return 0.2

    else:

        def sigma(t, x, m):
            nonlocal calls
            calls += 1
            return 0.2

    problem = _problem(sigma)
    x = np.linspace(0.0, 1.0, _N)
    solver = _solver(problem, x)
    M = np.ones((problem.Nt + 1, _N))

    with pytest.raises(NotImplementedError, match=r"space-only.*sigma\(x\)"):
        solver.solve_hjb_system(M, np.zeros(_N))

    assert calls == 0


@pytest.mark.unit
def test_hybrid_grid_field_is_mapped_to_collocation_points():
    """A grid-indexed field and the equivalent spatial callable must produce the same solve."""
    grid = np.linspace(0.0, 1.0, _N)
    collocation = np.linspace(0.0, 1.0, 17) ** 1.2
    field_on_grid = _SIGMA_LO + (_SIGMA_HI - _SIGMA_LO) * grid

    def sigma_of_x(point):
        x = float(np.asarray(point).reshape(-1)[0])
        return _SIGMA_LO + (_SIGMA_HI - _SIGMA_LO) * x

    problem = _problem()
    M_grid = np.tile(np.exp(-10 * (grid - 0.5) ** 2), (problem.Nt + 1, 1))
    U_terminal_grid = 0.5 * grid**2

    array_solver = _solver(problem, collocation)
    callable_solver = _solver(_problem(), collocation)
    U_array = array_solver.solve_hjb_system(
        M_density=M_grid,
        U_terminal=U_terminal_grid,
        volatility_field=field_on_grid,
    )
    U_callable = callable_solver.solve_hjb_system(
        M_density=M_grid,
        U_terminal=U_terminal_grid,
        volatility_field=sigma_of_x,
    )

    np.testing.assert_allclose(U_array, U_callable, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_native_2d_grid_field_is_mapped_to_collocation_points():
    """A multidimensional scalar field keeps the problem's native spatial shape."""
    nx, ny = 5, 4
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    field_on_grid = 0.1 + 0.2 * xx + 0.1 * yy
    problem = _problem_2d(nx, ny, field_on_grid)
    collocation = np.column_stack([xx.ravel(), yy.ravel()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = HJBGFDMSolver(
            problem,
            collocation_points=collocation,
            delta=1.5,
            monotonicity_scheme="none",
            max_newton_iterations=0,
        )

    solver.solve_hjb_system(
        M_density=np.ones((problem.Nt + 1, nx, ny)),
        U_terminal=np.zeros((nx, ny)),
    )

    expected = 0.1 + 0.2 * collocation[:, 0] + 0.1 * collocation[:, 1]
    np.testing.assert_allclose(solver._solve_sigma, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.unit
def test_ambiguous_2d_field_requires_explicit_solve_override():
    """A ``(d,d)`` grid field is accepted only where the API explicitly declares field semantics."""
    field_on_grid = np.array([[0.1, 0.2], [0.3, 0.4]])
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 4), indexing="ij")
    collocation = np.column_stack([xx.ravel(), yy.ravel()])
    problem = _problem_2d(2, 2, 0.3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = HJBGFDMSolver(
            problem,
            collocation_points=collocation,
            delta=1.5,
            monotonicity_scheme="none",
            max_newton_iterations=0,
        )

    solver.solve_hjb_system(
        M_density=np.ones((problem.Nt + 1, 2, 2)),
        U_terminal=np.zeros((2, 2)),
        volatility_field=field_on_grid,
    )

    expected = 0.1 + 0.2 * collocation[:, 0] + 0.1 * collocation[:, 1]
    np.testing.assert_allclose(solver._solve_sigma, expected, rtol=1e-14, atol=1e-14)

    problem.volatility_field = field_on_grid
    with pytest.raises(NotImplementedError, match=r"ambiguous.*solve_hjb_system"):
        HJBGFDMSolver(problem, collocation)


@pytest.mark.unit
def test_problem_owned_implicit_field_stays_collocation_indexed():
    """An implicit-domain field must not be reinterpreted on an invented uniform grid."""
    points = np.array([0.0, 0.06, 0.19, 0.43, 0.71, 0.92, 1.0])
    field = np.array([0.11, 0.18, 0.14, 0.33, 0.29, 0.47, 0.41])
    geometry = Hyperrectangle(np.array([[0.0, 1.0]]))
    geometry._num_spatial_points_cached = len(points)
    geometry.get_spatial_grid = lambda: points.reshape(-1, 1)
    problem = MFGProblem(
        geometry=geometry,
        components=MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
            m_initial=lambda point: np.exp(-10 * (np.asarray(point) - 0.5) ** 2),
            u_terminal=lambda point: 0.5 * np.asarray(point) ** 2,
        ),
        T=0.1,
        Nt=2,
        sigma=field,
    )
    solver = _solver(
        problem,
        points,
        max_newton_iterations=0,
        boundary_conditions=no_flux_bc(dimension=1),
    )

    solver.solve_hjb_system(
        M_density=np.ones((problem.Nt + 1, len(points))),
        U_terminal=np.zeros(len(points)),
    )

    np.testing.assert_array_equal(solver._solve_sigma, field)


@pytest.mark.unit
def test_column_field_fails_before_sparse_assembly():
    """An ``(N, 1)`` field is not a one-dimensional scalar volatility field."""
    column = np.linspace(_SIGMA_LO, _SIGMA_HI, _N).reshape(-1, 1)

    with pytest.raises(ValueError, match=r"volatility_field.*one-dimensional"):
        _solve(column)


@pytest.mark.unit
def test_spatiotemporal_field_fails_before_sparse_assembly():
    """GFDM's space-only coefficient contract must reject an unimplemented time axis."""
    field = np.tile(np.linspace(_SIGMA_LO, _SIGMA_HI, _N), (_problem().Nt + 1, 1))

    with pytest.raises(ValueError, match=r"volatility_field.*one-dimensional"):
        _solve(field)


@pytest.mark.unit
def test_mismatched_field_length_fails_in_coefficient_resolution():
    """A field that matches neither grid nor collocation space must not fall back to its mean."""
    field = np.linspace(_SIGMA_LO, _SIGMA_HI, _N - 1)

    with pytest.raises(ValueError, match=r"volatility_field.*shape"):
        _solve(field)


@pytest.mark.unit
def test_dmp_guard_uses_each_nodes_resolved_diffusion(monkeypatch):
    """The DMP diagnostic must inspect each solve's row-wise diffusion, without stale cache."""
    field = np.linspace(_SIGMA_LO, _SIGMA_HI, _N)
    _, solver = _solve(field, return_solver=True)
    captured = []

    def capture_diffusion(d_lap, d_grad, diffusion_coeff, interior_indices=None, atol=1e-12):
        captured.append(np.asarray(diffusion_coeff).copy())
        return float("inf")

    monkeypatch.setattr(
        monotonicity_enforcer,
        "critical_drift_for_dmp",
        capture_diffusion,
    )
    solver.check_dmp = True
    solver.monotonicity_scheme = "joint_socp"
    solver._maybe_warn_dmp(np.zeros(_N))

    next_field = field[::-1].copy()
    x = np.linspace(0.0, 1.0, _N)
    M = np.tile(np.exp(-10 * (x - 0.5) ** 2), (solver.problem.Nt + 1, 1))
    solver.check_dmp = False
    solver.monotonicity_scheme = "none"
    solver.solve_hjb_system(M, 0.5 * x**2, volatility_field=next_field)
    solver.check_dmp = True
    solver.monotonicity_scheme = "joint_socp"
    solver._maybe_warn_dmp(np.zeros(_N))

    assert len(captured) == 2
    np.testing.assert_array_equal(captured[0], 0.5 * field**2)
    np.testing.assert_array_equal(
        captured[1],
        0.5 * next_field**2,
    )


@pytest.mark.unit
def test_dmp_threshold_consumes_rowwise_diffusion():
    """The DMP helper must use the diffusion belonging to each Laplacian row."""
    d_lap = csr_matrix(
        [
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    d_grad = [
        csr_matrix(
            [
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    ]

    threshold = monotonicity_enforcer.critical_drift_for_dmp(
        d_lap,
        d_grad,
        diffusion_coeff=np.array([0.1, 0.4, 0.9]),
        interior_indices=np.array([0, 1]),
    )

    assert threshold == pytest.approx(0.1)
    with pytest.raises(ValueError, match=r"diffusion_coeff.*shape"):
        monotonicity_enforcer.critical_drift_for_dmp(
            d_lap,
            d_grad,
            diffusion_coeff=np.array([0.1, 0.4]),
        )

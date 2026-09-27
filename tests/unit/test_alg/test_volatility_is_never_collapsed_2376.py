"""#2376: a non-scalar volatility is never collapsed to a representative scalar.

``MFGProblem`` once reduced an array volatility to its mean and a callable one to the literal
``1.0`` (a factor of 400 in ``D`` for ``sigma = 0.05``), and every reader of that scalar solved a
different problem without a word. There is no such scalar any more: ``problem.volatility`` is held
as supplied, and each consumer either reads it or refuses it by name.

ADMISSION (#2227). Class 3, a pin of the fix: each test below reddens when the refusal or the read
it names is taken out, which is the measurement its assertion is. The fixture is 1-D where the
property is a scalar contract, and 2-D only for the tensor rows, which have no 1-D reader.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.model import Conditions, Model
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

N = 11
RAMP = np.linspace(0.2, 0.6, N)  # mean 0.4: a consumer averaging it would look plausible


def _hamiltonian():
    return SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))


def _components(dim=1):
    """The MultiPopulationMFGProblem inputs, which that class takes as components."""
    return MFGComponents(
        hamiltonian=_hamiltonian(),
        m_initial=lambda x: np.exp(-10 * np.sum((np.atleast_1d(np.asarray(x)) - 0.5) ** 2, axis=-1)),
        u_terminal=lambda x: np.zeros(np.asarray(x).shape[:-1]) if dim > 1 else 0.0 * np.asarray(x),
    )


def _v1_problem(bounds, nx_points, **volatility):
    """The v1.0 constructor: the volatility and its kind sit on the Model (#2375 ruling 6)."""
    dim = len(bounds)
    if dim == 1:
        conditions = Conditions(
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.0 * np.asarray(x),
            T=0.2,
        )
    else:
        conditions = Conditions(
            m_initial=lambda x: np.exp(-10 * np.sum((np.asarray(x) - 0.5) ** 2, axis=-1)),
            u_terminal=lambda x: np.zeros(np.asarray(x).shape[:-1]),
            T=0.2,
        )
    grid = TensorProductGrid(bounds=bounds, Nx_points=nx_points, boundary_conditions=no_flux_bc(dimension=dim))
    with warnings.catch_warnings():
        # The fixture's density is not normalised, and no pin here turns on its mass.
        warnings.filterwarnings("ignore", message="initial density mass", category=UserWarning)
        return MFGProblem(
            model=Model(hamiltonian=_hamiltonian(), **volatility), domain=grid, conditions=conditions, Nt=4
        )


def _problem(**volatility):
    return _v1_problem([(0.0, 1.0)], [N], **volatility)


def _problem_2d(**volatility):
    return _v1_problem([(0.0, 1.0), (0.0, 1.0)], [6, 5], **volatility)


FIELD = {"volatility": RAMP, "volatility_kind": "field"}
CALLABLE = {"volatility": lambda t, x, m: 0.05}


@pytest.mark.parametrize("volatility", [FIELD, CALLABLE], ids=["field", "callable"])
def test_scalar_only_hjb_solvers_refuse_by_name(volatility):
    """WENO and SL each read one constant volatility; a field or callable is refused, naming the solver."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver, HJBWENOSolver

    problem = _problem(**volatility)
    with pytest.raises(NotImplementedError, match="HJBWENOSolver uses one scalar volatility"):
        HJBWENOSolver(problem)
    with pytest.raises(NotImplementedError, match="HJBSemiLagrangianSolver reads one constant volatility"):
        HJBSemiLagrangianSolver(problem)


def test_the_adjoint_consistent_provider_refuses_a_field():
    """The provider squares one scalar; the BC state now carries the problem's volatility as supplied."""
    from mfgarchon.geometry.boundary import AdjointConsistentProvider

    problem = _problem(**FIELD)
    state = {"m_current": np.ones(N), "geometry": problem.geometry, "sigma": problem.volatility}
    with pytest.raises(NotImplementedError, match="AdjointConsistentProvider uses one scalar volatility"):
        AdjointConsistentProvider(side="left").compute(state)
    # Control: the same state with a scalar computes.
    state["sigma"] = 0.4
    assert np.isfinite(AdjointConsistentProvider(side="left").compute(state))


def test_the_hjb_residual_refuses_a_callable_it_cannot_evaluate():
    """The 1-D residual has no (t, m) to evaluate a callable at; the per-timestep driver does."""
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _volatility_at_n

    with pytest.raises(NotImplementedError, match="evaluated at t_n"):
        _volatility_at_n(_problem(**CALLABLE), None)
    np.testing.assert_array_equal(_volatility_at_n(_problem(**FIELD), None), RAMP)


def test_the_strict_adjoint_fp_step_refuses_a_callable():
    """Before #2376 the step took problem.sigma, the literal 1.0 for any callable."""
    from scipy import sparse

    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    problem = _problem(**CALLABLE)
    solver = FPFDMSolver(problem)
    with pytest.raises(NotImplementedError, match="adjoint_mode='off'"):
        solver.solve_fp_step_adjoint_mode(np.ones(N) / N, sparse.csr_matrix((N, N)))


def test_gfdm_falls_back_to_the_field_not_to_one():
    """HJBGFDMSolver's pre-solve volatility is the field at the collocation points, never 1.0.

    Before #2376 ``_get_sigma_value(None)`` returned ``float(getattr(problem, "sigma", 1.0))``: the
    mean for an array, and 1.0 for a callable problem -- reached by a standalone Howard solve and by
    the construction-time LLF base.
    """
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

    points = np.linspace(0.0, 1.0, N).reshape(-1, 1)
    solver = HJBGFDMSolver(_problem(**FIELD), collocation_points=points, monotonicity_scheme="none")
    np.testing.assert_allclose(solver._get_sigma_value(None), RAMP, rtol=0, atol=1e-15)
    assert solver._get_sigma_value(3) == pytest.approx(RAMP[3], abs=1e-15)
    with pytest.raises(NotImplementedError, match="space-only volatility callable"):
        HJBGFDMSolver(_problem(**CALLABLE), collocation_points=points, monotonicity_scheme="none")._get_sigma_value(
            None
        )


def test_fvm_refuses_the_problems_tensor_even_when_its_entries_are_equal():
    """An all-equal matrix passes FVM's constancy check and would be read as the scalar s."""
    from mfgarchon.alg.numerical.fp_solvers import FPFVMSolver

    problem = _problem_2d(volatility=np.full((2, 2), 0.3), volatility_kind="tensor")
    with pytest.raises(NotImplementedError, match="the problem's is a tensor"):
        FPFVMSolver(problem)._scalar_diffusion(None)


def test_the_pairing_guard_compares_arrays_by_content():
    """Two problems whose array volatilities share a mean are different problems.

    Before #2376 the guard compared each problem's collapsed mean, so this pair passed.
    """
    from mfgarchon.alg.numerical.coupling.base_mfg import assert_paired_solver_sigma

    class _Solver:
        def __init__(self, problem):
            self.problem = problem

    a = _problem(**FIELD)
    b = _problem(volatility=RAMP[::-1].copy(), volatility_kind="field")  # same mean, other field
    assert float(np.mean(a.volatility)) == pytest.approx(float(np.mean(b.volatility)))
    with pytest.raises(ValueError, match="different volatility"):
        assert_paired_solver_sigma(_Solver(a), _Solver(b), "test")
    assert_paired_solver_sigma(_Solver(a), _Solver(a), "test")  # control: the same problem passes


def test_distinct_per_population_volatilities_are_refused():
    """No multi-population solver reads a population's own volatility; all were solved at population 0's."""
    from mfgarchon.extensions.multi_population import MultiPopulationMFGProblem

    kwargs = {
        "num_populations": 2,
        "spatial_bounds": [(0.0, 1.0)],
        "spatial_discretization": [N - 1],
        "T": 0.2,
        "Nt": 4,
        "components": _components(),
    }
    with pytest.raises(NotImplementedError, match="population 0's"):
        MultiPopulationMFGProblem(volatility=[0.1, 0.3], **kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="initial density mass", category=UserWarning)
        assert MultiPopulationMFGProblem(volatility=[0.2, 0.2], **kwargs).volatility == 0.2


# --- The #2378 review round: the None paths, a declared kind, and refusals by name. -------------


def _hjb_fdm_2d(problem, **override):
    """A standalone nD HJB-FDM solve, the route FixedPointIterator and BlockIterator take with None.

    The terminal cost is asymmetric in the two axes, so a transposed or swapped diffusion shows.
    """
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    x, y = np.meshgrid(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 5), indexing="ij")
    U_terminal = (x - 0.5) ** 2 + 0.5 * (y - 0.3) ** 2
    M = np.ones((problem.Nt + 1, 6, 5))
    return HJBFDMSolver(problem).solve_hjb_system(M, U_terminal, np.zeros_like(M), **override)


# The problem's own volatility reaches a solver two ways: as the None default (a standalone solve,
# BlockIterator), and as the object itself, which MFGProblem.solve() forwards as volatility_field.
# Both must be read by the declared kind, so every kind-dispatch pin below runs on both.
ROUTES = pytest.mark.parametrize("route", ["none", "same-object"])


def _route(problem, route):
    return {} if route == "none" else {"volatility_field": problem.volatility}


def _hjb_fdm_reads_the_diagonal_tensor(route):
    """The problem's own volatility is dispatched by its declared kind, on either route.

    Before, the (d, d) tensor dispatch looked only at an explicit override: with None the problem's
    tensor went to the field reader and raised a grid-shape error, so FixedPointIterator,
    BlockIterator and a standalone solve failed where MFGProblem.solve, which forwards the object,
    ran. The reference is the per-axis override, which reaches the tensor path by its own route.
    """
    problem = _problem_2d(volatility=np.diag([0.3, 0.2]), volatility_kind="tensor")

    np.testing.assert_array_equal(
        _hjb_fdm_2d(problem, **_route(problem, route)), _hjb_fdm_2d(problem, volatility_field=np.array([0.3, 0.2]))
    )


def test_hjb_fdm_reads_the_problems_diagonal_tensor_on_the_none_path():
    _hjb_fdm_reads_the_diagonal_tensor("none")


def test_hjb_fdm_reads_the_problems_diagonal_tensor_on_the_same_object_route():
    """MFGProblem.solve() forwards the problem's own volatility; a None-only pin left it open."""
    _hjb_fdm_reads_the_diagonal_tensor("same-object")


UNASSEMBLED_TENSORS = pytest.mark.parametrize(
    ("volatility", "what"),
    [
        (np.array([[0.3, 0.1], [0.1, 0.2]]), "non-diagonal"),
        (np.broadcast_to(np.diag([0.3, 0.2]), (6, 5, 2, 2)).copy(), "per point"),
        (lambda t, x, m: np.diag([0.3, 0.2]), "a callable"),
    ],
    ids=["non-diagonal", "per-point", "callable"],
)


@UNASSEMBLED_TENSORS
def test_hjb_fdm_refuses_a_problem_tensor_it_would_solve_at_the_wrong_diffusion(volatility, what):
    """HJB-FDM's tensor path is a constant diagonal Laplacian with weights sigma_ii^2/2.

    A non-diagonal Sigma loses the cross term and gets A_ii = 1/2 sum_k Sigma_ik^2 wrong as well
    (0.045 against 0.05 here); a per-point or callable one would be averaged over the grid. Before,
    MFGProblem.solve reached the first with a warning that named only the cross term.
    """
    problem = _problem_2d(volatility=volatility, volatility_kind="tensor")
    with pytest.raises(NotImplementedError, match=rf"only as a constant diagonal.*is {what}.*HJBSemiLagrangianSolver"):
        _hjb_fdm_2d(problem)


@UNASSEMBLED_TENSORS
def test_hjb_fdm_refuses_a_problem_tensor_on_the_same_object_route(volatility, what):
    """The route MFGProblem.solve() takes: it forwards the problem's own volatility as the override.

    A pin on the None route alone left this one open -- deleting the identity arm of the dispatch
    brought back the warned-about wrong solve on MFGProblem.solve() with the whole gate green
    (#2378 re-review).
    """
    problem = _problem_2d(volatility=volatility, volatility_kind="tensor")
    with pytest.raises(NotImplementedError, match=rf"only as a constant diagonal.*is {what}.*HJBSemiLagrangianSolver"):
        _hjb_fdm_2d(problem, volatility_field=problem.volatility)


def test_hjb_fdm_solves_a_2d_field_and_a_constant_one_is_the_scalar():
    """The nD field path multiplied a grid-shaped D by the flattened Laplacian and crashed.

    A constant field solves to the scalar's answer. For a varying one the reference is independent
    of that path: a field and a terminal cost that vary along x only
    make the 2-D solution y-independent and equal to the 1-D HJB-FDM solve on the same x grid, which
    is a different code path (base_hjb's 1-D Newton). A D laid onto the grid in the wrong order --
    raveled Fortran-style, say -- passes a constant-field check and a ramp-vs-mean check alike, and
    fails this one (#2378 re-review).
    """
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    constant = _problem_2d(volatility=np.full((6, 5), 0.3), volatility_kind="field")
    np.testing.assert_allclose(_hjb_fdm_2d(constant), _hjb_fdm_2d(_problem_2d(volatility=0.3)), rtol=0, atol=1e-14)

    ramp_x = np.linspace(0.1, 0.5, 6)
    problem_2d = _problem_2d(volatility=np.repeat(ramp_x[:, None], 5, axis=1), volatility_kind="field")
    x2 = np.linspace(0.0, 1.0, 6)[:, None] * np.ones((6, 5))
    M2 = np.ones((problem_2d.Nt + 1, 6, 5))
    U2 = HJBFDMSolver(problem_2d).solve_hjb_system(M2, (x2 - 0.3) ** 2, np.zeros_like(M2))

    problem_1d = _v1_problem([(0.0, 1.0)], [6], volatility=ramp_x, volatility_kind="field")
    x1 = np.linspace(0.0, 1.0, 6)
    M1 = np.ones((problem_1d.Nt + 1, 6))
    U1 = HJBFDMSolver(problem_1d).solve_hjb_system(M1, (x1 - 0.3) ** 2, np.zeros_like(M1))

    assert np.abs(U2 - U2[..., :1]).max() < 1e-12, "an x-only problem solved to a y-dependent U"
    np.testing.assert_allclose(U2[..., 0], U1, rtol=0, atol=1e-7)  # measured 2.8e-9: the two Newton stops


def _problem_2x2(**volatility):
    """A 2 x 2 grid, where a (d, d) field and a (d, d) tensor have the same shape."""
    return _v1_problem([(0.0, 1.0), (0.0, 1.0)], [2, 2], **volatility)


def _fp_fdm_reads_a_declared_field_on_a_d_by_d_grid(route):
    """FP-FDM dispatched the problem's array by shape, so a declared 2 x 2 field was solved as Sigma.

    A constant field is the scalar exactly; the same array declared a tensor is a different
    diffusion, A = 1/2 Sigma Sigma^T, which is the control that the fixture can tell them apart.
    """
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    m0 = np.array([[2.0, 0.5], [1.0, 0.5]])  # non-uniform: a uniform density is invariant under any diffusion
    scalar = FPFDMSolver(_problem_2x2(volatility=0.3)).solve_fp_system(m0)
    declared_field = _problem_2x2(volatility=np.full((2, 2), 0.3), volatility_kind="field")
    field = FPFDMSolver(declared_field).solve_fp_system(m0, **_route(declared_field, route))
    declared_tensor = _problem_2x2(volatility=np.full((2, 2), 0.3), volatility_kind="tensor")
    tensor = FPFDMSolver(declared_tensor).solve_fp_system(m0, **_route(declared_tensor, route))

    np.testing.assert_array_equal(field, scalar)
    assert not np.allclose(tensor, scalar)


def test_fp_fdm_reads_a_declared_field_as_a_field_on_a_d_by_d_grid():
    _fp_fdm_reads_a_declared_field_on_a_d_by_d_grid("none")


def test_fp_fdm_reads_a_declared_field_on_a_d_by_d_grid_on_the_same_object_route():
    _fp_fdm_reads_a_declared_field_on_a_d_by_d_grid("same-object")


def _fp_fdm_callable_drift_reads_the_problems_tensor(route):
    """The callable-drift route passed None through as a field, so the problem's tensor was misread.

    With a zero drift it is the pure-diffusion route, which reads the tensor by kind.
    """
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    problem = _problem_2d(volatility=np.diag([0.3, 0.2]), volatility_kind="tensor")
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 5), indexing="ij")
    m0 = np.exp(-10 * ((x - 0.5) ** 2 + (y - 0.3) ** 2))

    np.testing.assert_array_equal(
        FPFDMSolver(problem).solve_fp_system(m0, drift_field=lambda t, x, m: 0.0, **_route(problem, route)),
        FPFDMSolver(problem).solve_fp_system(m0),
    )


def test_fp_fdm_callable_drift_reads_the_problems_tensor():
    _fp_fdm_callable_drift_reads_the_problems_tensor("none")


def test_fp_fdm_callable_drift_reads_the_problems_tensor_on_the_same_object_route():
    _fp_fdm_callable_drift_reads_the_problems_tensor("same-object")


@ROUTES
def test_hjb_fdm_reads_a_declared_field_as_a_field_on_a_d_by_d_grid(route):
    """HJB-FDM dispatched the problem's array by shape, so a declared 2 x 2 field was solved as Sigma.

    A constant field cannot show it -- HJB-FDM's tensor path keeps sigma_ii^2/2, which for a constant
    field is its own D -- so the field varies along x. The reference is the same x-only field on a
    2 x 3 grid, whose shape no dispatch can mistake for a tensor: with an x-only terminal cost both
    solutions are y-independent and must agree column for column.
    """
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    def solve(shape, **kw):
        problem = _v1_problem(
            [(0.0, 1.0), (0.0, 1.0)],
            list(shape),
            volatility=np.repeat([[0.3], [0.6]], shape[1], axis=1),
            volatility_kind="field",
        )
        x = np.linspace(0.0, 1.0, shape[0])[:, None] * np.ones(shape)
        M = np.ones((problem.Nt + 1, *shape))
        return HJBFDMSolver(problem).solve_hjb_system(M, (x - 0.3) ** 2, np.zeros_like(M), **_route(problem, route))

    np.testing.assert_allclose(solve((2, 2))[..., 0], solve((2, 3))[..., 0], rtol=0, atol=1e-12)


@ROUTES
def test_the_particle_callable_drift_path_reads_a_declared_field_on_a_d_by_d_grid(route):
    """Read by shape, a declared 2 x 2 field full of sigma became the noise matrix Sigma = sigma 1 1^T.

    That matrix drives both axes with one Brownian increment, so every particle's dx and dy are equal;
    a field draws them independently. Started at the centre, where one step reaches no wall, the
    increments' correlation separates the two readings (1.0 against measured 0.013), and their
    spread is the field's sigma sqrt(dt).
    """
    from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver

    problem = _problem_2x2(volatility=np.full((2, 2), 0.3), volatility_kind="field")
    solver = FPParticleSolver(problem, num_particles=2000, density_mode="hybrid", seed=3)
    solver.solve_fp_system(
        np.ones((2, 2)),
        drift_field=lambda t, x, m: np.zeros_like(x),
        initial_particles=np.full((2000, 2), 0.5),
        **_route(problem, route),
    )
    step = np.asarray(solver._particle_history[1]) - np.asarray(solver._particle_history[0])

    assert abs(np.corrcoef(step[:, 0], step[:, 1])[0, 1]) < 0.2
    np.testing.assert_allclose(step.std(axis=0), 0.3 * np.sqrt(problem.dt), rtol=0.1)


def test_the_strict_adjoint_fp_step_refuses_the_problems_tensor():
    """The step assembles sigma^2/2 per point; a (d, d) tensor would be read as a 2 x 2 field."""
    from scipy import sparse

    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    problem = _problem_2d(volatility=np.diag([0.3, 0.2]), volatility_kind="tensor")
    with pytest.raises(NotImplementedError, match=r"assembles an isotropic diffusion.*volatility_kind='tensor'"):
        FPFDMSolver(problem).solve_fp_step_adjoint_mode(np.ones((6, 5)), sparse.csr_matrix((30, 30)))


def test_the_particle_grid_drift_path_refuses_the_problems_field():
    """With an ndarray drift, the path consumes one scalar; it averaged the problem's field before.

    The None default reaches it from the coupled solve; the override route is pinned in #1248's file.
    """
    from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver

    U = np.tile(0.3 * (np.linspace(0.0, 1.0, N) - 0.5) ** 2, (5, 1))
    with pytest.raises(NotImplementedError, match="FPParticleSolver's grid-drift path"):
        FPParticleSolver(_problem(**FIELD), num_particles=100).solve_fp_system(np.ones(N), drift_field=U)


def test_the_particle_callable_drift_path_refuses_a_callable_tensor():
    """It evaluates a callable as one sigma per particle, and a (d, d) output would be raveled into it."""
    from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver

    problem = _problem_2d(volatility=lambda t, x, m: np.diag([0.3, 0.2]), volatility_kind="tensor")
    with pytest.raises(NotImplementedError, match=r"one sigma per particle.*volatility_kind='tensor'"):
        FPParticleSolver(problem, num_particles=100).solve_fp_system(np.ones((6, 5)), drift_field=lambda t, x, m: 0.0)


@ROUTES
def test_the_particle_callable_drift_path_reads_a_declared_tensor_on_a_d_by_d_grid(route):
    """The other half of the same grid: a declared full-rank 2 x 2 tensor is the noise matrix.

    Sigma = [[0.3, 0.15], [0.15, 0.3]] gives Sigma Sigma^T = [[0.1125, 0.09], [0.09, 0.1125]], so the
    increments correlate at 0.8 (measured 0.797) with spread sqrt(0.1125 dt). Read as a field on
    this grid -- which a shape-keyed field branch ahead of the matrix branches did, had it not been
    shadowed -- they would be independent (#2378 re-review, D1).
    """
    from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver

    problem = _problem_2x2(volatility=np.array([[0.3, 0.15], [0.15, 0.3]]), volatility_kind="tensor")
    solver = FPParticleSolver(problem, num_particles=4000, density_mode="hybrid", seed=5)
    solver.solve_fp_system(
        np.ones((2, 2)),
        drift_field=lambda t, x, m: np.zeros_like(x),
        initial_particles=np.full((4000, 2), 0.5),
        **_route(problem, route),
    )
    step = np.asarray(solver._particle_history[1]) - np.asarray(solver._particle_history[0])

    assert np.corrcoef(step[:, 0], step[:, 1])[0, 1] == pytest.approx(0.8, abs=0.05)
    np.testing.assert_allclose(step.std(axis=0), np.sqrt(0.1125 * problem.dt), rtol=0.05)


@pytest.mark.parametrize(
    ("hjb", "hjb_kind", "fp", "fp_kind", "differ"),
    [
        (0.3, None, lambda t, x, m: 0.3, None, True),
        (RAMP, "field", lambda t, x, m: 0.3, None, True),
        (lambda t, x, m: 0.3, None, lambda t, x, m: 0.3, None, False),
        (0.3, None, np.full(N, 0.3), "field", False),
        (0.3, None, np.full(N, 0.4), "field", True),
        (0.3, None, np.full((2, 2), 0.3), "tensor", True),
        (np.full((2, 2), 0.3), "field", np.full((2, 2), 0.3), "tensor", True),
    ],
    ids=[
        "scalar-vs-callable",
        "field-vs-callable",
        "two-callables",
        "scalar-vs-equal-constant",
        "scalar-vs-other-constant",
        "scalar-vs-all-equal-tensor",
        "field-vs-tensor-same-entries",
    ],
)
def test_the_pairing_guard_on_mixed_kinds(hjb, hjb_kind, fp, fp_kind, differ):
    """Exactly one callable against a value differs: no evaluation can show them equal (#1316).

    At 0f937601 these pairs were refused only because the callable collapsed to 1.0; 23bbb9e5 let
    them through. Two callables stay identity-only, which is what keeps Mock doubles passing
    (#1489). A scalar and a constant FIELD equal to it are one problem; an all-equal tensor is not,
    since its A = 1/2 Sigma Sigma^T is not sigma^2/2 I, and 928bfa8d let that pair through too.
    """
    from types import SimpleNamespace

    from mfgarchon.alg.numerical.coupling.base_mfg import assert_paired_solver_sigma

    pair = (
        SimpleNamespace(problem=SimpleNamespace(volatility=hjb, volatility_kind=hjb_kind)),
        SimpleNamespace(problem=SimpleNamespace(volatility=fp, volatility_kind=fp_kind)),
    )
    if differ:
        with pytest.raises(ValueError, match="different volatility"):
            assert_paired_solver_sigma(*pair, "test")
    else:
        assert_paired_solver_sigma(*pair, "test")


def test_save_experiment_data_stores_a_callable_volatility_by_tag(tmp_path):
    """np.savez pickles the parameter dict, and a callable would fail the whole save (#2378 review)."""
    from mfgarchon.utils.experiment_manager import load_experiment_data, save_experiment_data

    problem = _problem(**CALLABLE)
    zeros = np.zeros((problem.Nt + 1, N))
    path = save_experiment_data(
        problem, zeros, zeros, "test", 1, np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), 0.0, str(tmp_path)
    )

    assert path, "the save failed"
    params = load_experiment_data(path)["problem_params"]
    assert params["sigma"] == "callable"
    assert params["volatility_kind"] is None

    # An array is tagged in the file name by its kind: a tensor is not a field.
    tensor = _problem_2d(volatility=np.diag([0.3, 0.2]), volatility_kind="tensor")
    zeros = np.zeros((tensor.Nt + 1, 6, 5))
    path = save_experiment_data(
        tensor, zeros, zeros, "test", 1, np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), 0.0, str(tmp_path)
    )
    assert "_sigtensor_" in path

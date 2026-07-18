"""Semi-Lagrangian per-node failures must propagate, not substitute a stale value (Issue #1635).

Two per-point loops caught `Exception`, logged a warning and assigned
``U_star[i] = U_next[i]`` -- the value at t^{n+1}, i.e. no update at all for that
node. The substituted value is finite by construction, so the NaN/Inf guard in
`solve_hjb_system` could not see it: the solver returned a plausible, silently wrong
value function, with no machine-readable record of which nodes were contaminated.

Reached at defaults: `enable_adaptive_substepping` is True and CFL > 1 forces
`n_substeps > 1`, which routes through the substepping loop with no opt-in flag.

The two loops were byte-identical copies; they now share one owner
(`HJBSemiLagrangianSolver._advect_pointwise`) so the fix cannot re-fork.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

NX = 41
NT = 10


class _PoisonedPotential:
    """A user-supplied potential that raises once, at a single interior grid node.

    Ordinary user code -- nothing patched inside the library. This is exactly how a
    real per-node numerics failure reaches the solver.
    """

    def __init__(self, x_bad: float, tol: float = 1e-9):
        self.x_bad = x_bad
        self.tol = tol
        self.n_raised = 0

    def __call__(self, x, t=0.0):
        arr = np.asarray(x, dtype=float)
        if np.any(np.abs(arr - self.x_bad) < self.tol):
            self.n_raised += 1
            raise FloatingPointError(f"injected failure at x={self.x_bad}")
        # Scalar: the per-point path passes one node at a time and the caller
        # reduces the result to a Python float.
        return 0.0


def _problem(potential=None, control_cost: float = 1.0) -> MFGProblem:
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=control_cost),
        potential=potential,
    )
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, sigma=0.1),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1)),
        # A steep terminal cost drives max|grad u| up so CFL > 1 and substepping engages.
        conditions=Conditions(u_terminal=lambda x: 20.0 * (x - 0.5) ** 2, m_initial=lambda x: 1.0, T=0.5),
        Nt=NT,
    )


def _solve(problem, **kwargs):
    solver = HJBSemiLagrangianSolver(problem, **kwargs)
    U_terminal = np.array([20.0 * (x - 0.5) ** 2 for x in problem.geometry.coordinates[0]])
    M = np.ones((NT + 1, NX)) / NX
    return solver.solve_hjb_system(M, U_terminal, U_coupling_prev=np.zeros((NT + 1, NX)))


def test_substepping_path_is_live_at_defaults():
    """Guards the premise: if substepping stopped engaging, the other tests would pass vacuously."""
    solver = HJBSemiLagrangianSolver(_problem())
    assert solver.enable_adaptive_substepping is True

    _, n_substeps, _ = solver._compute_cfl_and_substeps(np.linspace(0.0, 20.0, NX), solver.dt)
    assert n_substeps > 1, "CFL did not force substepping; this fixture no longer exercises the loop"


def test_per_node_failure_raises_instead_of_returning_a_wrong_answer():
    """The whole defect: this used to return a finite, plausible, silently wrong U."""
    potential = _PoisonedPotential(x_bad=0.5)

    with pytest.raises(RuntimeError, match="Semi-Lagrangian update failed at grid point"):
        _solve(_problem(potential))

    assert potential.n_raised > 0, "the injected failure never fired; the test proves nothing"


def test_raise_names_the_node_and_the_time():
    """A diagnostic the caller can act on -- the old handler left no trace at all."""
    with pytest.raises(RuntimeError) as excinfo:
        _solve(_problem(_PoisonedPotential(x_bad=0.5)))

    message = str(excinfo.value)
    assert "grid point" in message
    assert "x=" in message
    assert "t=" in message
    assert "dt=" in message
    assert isinstance(excinfo.value.__cause__, FloatingPointError), "the original error must be chained"


def test_rk4_fallback_path_also_fails_loud():
    """The second copy of the loop: live under characteristic_solver='rk4'."""
    with pytest.raises(RuntimeError, match="Semi-Lagrangian update failed at grid point"):
        _solve(_problem(_PoisonedPotential(x_bad=0.5)), characteristic_solver="rk4")


def test_clean_solve_is_unaffected():
    """No poisoned node: the solver still returns a finite value function."""
    result = _solve(_problem())

    assert result.shape == (NT + 1, NX)
    assert np.isfinite(result).all()


def _gentle_problem(potential=None, control_cost: float = 1.0) -> MFGProblem:
    """A flat terminal cost keeps CFL well below 1, so substepping does NOT engage.

    That matters: while substepping is active it handles every timestep, so site A
    (the rk4 fallback of the fixed-dt path) is never reached -- measured, `rk4` and
    the default both enter the shared helper the same 94 times, all via site B. Only
    at CFL <= 1 does rk4 route through site A (40 calls; the default, 0).
    """
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=control_cost),
        potential=potential,
    )
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, sigma=0.1),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(u_terminal=lambda x: 0.05 * (x - 0.5) ** 2, m_initial=lambda x: 1.0, T=0.5),
        Nt=40,
    )


def _solve_gentle(problem, **kwargs):
    solver = HJBSemiLagrangianSolver(problem, **kwargs)
    coords = problem.geometry.coordinates[0]
    U_terminal = np.array([0.05 * (x - 0.5) ** 2 for x in coords])
    M = np.ones((41, len(coords))) / len(coords)
    return solver.solve_hjb_system(M, U_terminal, U_coupling_prev=np.zeros((41, len(coords))))


def test_site_a_is_reached_only_by_rk4_below_the_cfl_limit():
    """Guards the premise of the site-A pin below; without it that test is vacuous."""
    calls = {"n": 0}
    original = HJBSemiLagrangianSolver._advect_pointwise

    def spy(self, *args):
        calls["n"] += 1
        return original(self, *args)

    HJBSemiLagrangianSolver._advect_pointwise = spy
    try:
        calls["n"] = 0
        _solve_gentle(_gentle_problem())
        default_calls = calls["n"]
        calls["n"] = 0
        _solve_gentle(_gentle_problem(), characteristic_solver="rk4")
        rk4_calls = calls["n"]
    finally:
        HJBSemiLagrangianSolver._advect_pointwise = original

    assert default_calls == 0, "the default path should take the vectorized batch route here"
    assert rk4_calls > 0, "rk4 below the CFL limit must route through the shared helper (site A)"


def test_site_a_preserves_t_val_and_lambda():
    """Site A's arguments, pinned where site A is the only consumer of the helper."""
    time_dependent = lambda x, t=0.0: 5.0 * float(t) * float(np.asarray(x).reshape(-1)[0])  # noqa: E731

    u_time_dep = _solve_gentle(_gentle_problem(time_dependent), characteristic_solver="rk4")
    u_time_indep = _solve_gentle(_gentle_problem(lambda x, t=0.0: 0.0), characteristic_solver="rk4")
    assert not np.allclose(u_time_dep, u_time_indep), "site A is not transporting t_val"


@pytest.mark.parametrize("control_cost", [1.0, 3.0, 30.0])
def test_characteristic_foot_is_scaled_by_one_over_lambda(control_cost):
    """The foot velocity must be `grad(u)/lambda`, asserted on the owner directly.

    An end-to-end comparison across control_cost cannot pin this: lambda also enters
    the Lax-Oleinik value update, so dropping the `/lam` in the foot still leaves
    cheap-vs-costly solutions different and the test green. Spy on the trace call and
    check the velocity itself.
    """
    problem = _gentle_problem(control_cost=control_cost)
    solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4")

    grad_u = np.linspace(-1.0, 1.0, len(solver.x_grid))
    seen: list[float] = []
    original = solver._trace_characteristic_backward

    def spy(x, velocity, dt):
        seen.append(float(velocity))
        return original(x, velocity, dt)

    solver._trace_characteristic_backward = spy
    solver._advect_pointwise(np.zeros_like(grad_u), np.ones_like(grad_u) / len(grad_u), grad_u, t_val=0.1, dt=solver.dt)

    expected = grad_u / control_cost
    assert np.allclose(seen, expected), (
        f"foot velocity should be grad(u)/lambda with lambda={control_cost}; "
        f"got {seen[:3]}... expected {expected[:3].tolist()}..."
    )


@pytest.mark.parametrize("solver_kwargs", [{}, {"characteristic_solver": "rk4"}], ids=["substepping", "rk4"])
def test_consolidation_preserves_t_val_and_lambda_at_both_sites(solver_kwargs):
    """Pinning test the consolidation owes (repo rule: one owner + a pinning test).

    `_advect_pointwise` is the single owner of `t_val`, the substep `dt` and the
    control-cost `lambda` for BOTH call sites. Nothing else notices if a call site
    drifts: passing `t_val=0.0` or dropping the `1/lambda` foot scaling leaves all
    388 pre-existing SL tests green.

    Parametrized over both entry points because they reach different sites: the
    default (`explicit_euler`) takes the vectorized batch path and reaches the shared
    helper only via CFL substepping, while `rk4` routes every node through it. A test
    that exercised only one would leave the other's arguments unpinned -- and a
    lambda difference observed on the default path leaks in through the batch path,
    so it must be measured where the helper is the only consumer.
    """
    time_dependent = lambda x, t=0.0: 5.0 * float(t) * float(np.asarray(x).reshape(-1)[0])  # noqa: E731

    u_time_dep = _solve(_problem(time_dependent), **solver_kwargs)
    u_time_indep = _solve(_problem(lambda x, t=0.0: 0.0), **solver_kwargs)
    assert not np.allclose(u_time_dep, u_time_indep), (
        "t_val is not reaching the Hamiltonian: a time-dependent potential changed nothing"
    )

    u_cheap = _solve(_problem(control_cost=1.0), **solver_kwargs)
    u_costly = _solve(_problem(control_cost=3.0), **solver_kwargs)
    assert not np.allclose(u_cheap, u_costly), "the 1/lambda foot scaling is not reaching the characteristic trace"


def test_clean_solve_actually_advances_the_value_function():
    """`solve_hjb_system` must transport information, not merely return finite arrays.

    A no-op advection -- compute the update, discard it, return U_next unchanged --
    passes a shape+finite assertion. Pin that the backward solve moves u away from
    the terminal condition.
    """
    result = _solve(_problem())

    assert not np.allclose(result[0], result[-1]), "the backward solve left u at its terminal condition"


def test_both_loops_share_one_owner():
    """Consolidation pin: a re-fork would reintroduce the divergence this issue is about."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(HJBSemiLagrangianSolver))
    owners = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_advect_pointwise"]
    assert len(owners) == 1, "the pointwise advection sweep must have exactly one owner"

    # Search executable code only -- docstrings legitimately quote the old pattern.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        stale = (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "U_star"
            and isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and value.value.id == "U_next"
        )
        assert not stale, f"stale-value substitution U_star[i] = U_next[i] reappeared at line {node.lineno}"

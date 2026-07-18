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


def _problem(potential=None) -> MFGProblem:
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
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

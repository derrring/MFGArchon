"""The SL HJB's sub-steps carry a Neumann value since #2141 -- the half the applicator test cannot see.

`test_neumann_wall_sign_2141.py` hands `InterpolationApplicator` its spacing itself. The solver's call
site, `HJBSemiLagrangianSolver._enforce_boundary_conditions`, passed none, so a fix to the applicator
alone left this path where it was. Only a solve adjudicates that half.

Oracle: u = A (x - 1/2)^2 with V = 2 A^2 (x - 1/2)^2 is exact at sigma = 0 for the quadratic control
cost, and its outward normal derivative is A on both walls, so `neumann_bc(value=A)` is its data.
A = -2 sends the characteristics out of the domain at a foot CFL up to 2.8 (dt = h), so the default
path cuts each step into sub-steps, and their post-step enforcement goes through
`InterpolationApplicator`. Before #2141 that dropped g, and the error with g matched `no_flux_bc()`'s:
1.88 / 1.90 / 1.92 against 1.87 / 1.90 / 1.92 at 21 / 41 / 81 points. Now it converges at first order,
0.755 / 0.389 / 0.197.

The solver refuses an inhomogeneous Neumann value at construction (#1936, #2461). This test lifts that
refusal for itself, as `test_hjb_sl_boundary_time_2453.py` does, to measure the path an implementation
of #1936 will use.
"""

from __future__ import annotations

import numpy as np

import mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian as hsl
from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, mixed_bc, neumann_bc

A = -2.0


def _relative_error(n: int, u, du, bc) -> float:
    """Stationary exact u at sigma = 0: V = u'(x)^2 / 2 makes H(x, u') = |u'|^2 / 2 - V vanish."""

    def potential(t, x):
        v = 0.5 * du(np.asarray(x, dtype=float)) ** 2
        return float(np.ravel(v)[0]) if np.size(v) == 1 else v

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=bc)
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0), potential=potential),
            volatility=0.0,
        ),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: u(np.asarray(x, dtype=float)), T=1.0),
        Nt=n - 1,
    )
    exact = u(grid.coordinates[0])
    U = HJBSemiLagrangianSolver(problem).solve_hjb_system(np.ones((n, n)), exact, np.zeros((n, n)))
    return float(np.max(np.abs(U[0] - exact)) / np.abs(exact).max())


def _sub_step_counter(monkeypatch) -> list[int]:
    monkeypatch.setattr(HJBSemiLagrangianSolver, "honors_inhomogeneous_neumann", True)
    counts: list[int] = []
    original = hsl.cfl_substeps
    monkeypatch.setattr(hsl, "cfl_substeps", lambda *a, **k: counts.append(original(*a, **k)) or counts[-1])
    return counts


def test_the_default_path_sub_steps_carry_the_neumann_value(monkeypatch):
    counts = _sub_step_counter(monkeypatch)
    errors = [
        _relative_error(n, lambda x: A * (x - 0.5) ** 2, lambda x: 2 * A * (x - 0.5), neumann_bc(value=A, dimension=1))
        for n in (21, 41, 81)
    ]

    assert counts, "the solve never asked for a sub-step schedule"
    assert max(counts) > 1, "no step sub-stepped, so this did not reach the sub-step enforcement"
    assert errors[-1] < 0.25, (
        f"relative error {errors} at 21/41/81 points; with the Neumann value dropped it is about 1.9, "
        f"the no-flux error (#2141)"
    )
    assert all(errors[i] / errors[i + 1] > 1.8 for i in range(2)), f"not first order: {errors}"


def test_each_wall_carries_its_own_value(monkeypatch):
    """The data above is the same on both walls, so it cannot tell a wall's value from the other's. Here
    u = 1.5 x - 2 x^2 has du/dn = -1.5 at the left wall and -2.5 at the right, and the characteristics
    leave at both. Measured: 0.953 / 0.488 / 0.247 at 21 / 41 / 81 points; with the two walls' values
    swapped it does not converge (4.6 -> 4.1), and before #2141 it matched no-flux (2.9 -> 3.0)."""
    counts = _sub_step_counter(monkeypatch)
    a, b = 1.5, -2.0
    bc = mixed_bc(
        [
            BCSegment(name="left", bc_type=BCType.NEUMANN, value=-a, boundary="x_min"),
            BCSegment(name="right", bc_type=BCType.NEUMANN, value=a + 2 * b, boundary="x_max"),
        ],
        dimension=1,
        domain_bounds=np.array([[0.0, 1.0]]),
    )
    errors = [_relative_error(n, lambda x: a * x + b * x**2, lambda x: a + 2 * b * x, bc) for n in (21, 41, 81)]

    assert max(counts) > 1, "no step sub-stepped, so this did not reach the sub-step enforcement"
    assert errors[-1] < 0.3, f"relative error {errors} at 21/41/81 points"
    assert all(errors[i] / errors[i + 1] > 1.8 for i in range(2)), f"not first order: {errors}"

"""A solve that reports convergence must return a root of the residual it certified.

`solve_hjb_timestep_newton` ran Newton to convergence and then overwrote the boundary values with a
different discretisation of the same boundary condition, so the array it returned was not a root and
`converged` was True -- which means the non-convergence warning, the only existing instrument, was
structurally blind to it (#1900).

Measured on `scripts/capability_matrix.py::_smoke_problem` before the fix: the enforcement changed
the returned array on 50 of 50 inner solves and destroyed convergence on 11 of 11 that had achieved
it. At t_idx=9 Newton reached 4.424e-07 against a 1e-6 tolerance and returned an array whose
residual was 4.338e-02.

Three implementations of no-flux were in play, which is why this is #1894/#1896's class one layer up:

    pad_array_with_ghosts          mirror ghost              O(h^2)   <- the residual; defines the problem
    enforce_neumann_value_nd       u[0] = (4u[1] - u[2])/3   O(h^2)   <- exists, called by no solver
    base_hjb (hand-rolled, 1-D)    u[0] = u[1] - g*dx        O(h)     <- deleted here

Which one owns the condition was measured rather than argued: over Nx in {41, 81, 161, 321, 641} at
fixed t_idx, enforced and un-enforced boundary values converge to the SAME limit with the gap falling
8.9e-03 -> 4.0e-05, i.e. O(h^2). Dirichlet and Robin are the opposite case -- the residual only
approaches the boundary value at O(h) there -- so their branches stay.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import base_hjb
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _get_bc_info_1d
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    robin_bc,
)

BOUNDS = np.array([[0.0, 1.0]])

BCS = {
    "no_flux": lambda: no_flux_bc(dimension=1),
    "neumann_zero": lambda: neumann_bc(dimension=1, value=0.0),
    "dirichlet": lambda: dirichlet_bc(dimension=1, value=0.7),
    "robin": lambda: robin_bc(dimension=1, alpha=1.0, beta=1.0, value=0.5),
}


def _problem(bc, nx: int = 21, sigma: float = 0.3):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=bc)
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


def _solve_and_measure(bc, nx: int = 21, tol: float = 1e-9):
    """Run one backward Newton solve; return (reported_converged, residual_of_returned_array)."""
    problem = _problem(bc, nx)
    dx = float(problem.geometry.get_grid_spacing()[0])
    x = np.linspace(0.0, 1.0, nx)
    u_next = 0.5 * np.cos(np.pi * x)
    m = np.exp(-10 * (x - 0.5) ** 2)
    m /= m.sum() * dx

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = base_hjb.solve_hjb_timestep_newton(
            u_next,
            u_next,
            m,
            problem,
            max_newton_iterations=60,
            newton_tolerance=tol,
            t_idx_n=5,
            backend=None,
            sigma_at_n=0.3,
            use_upwind=True,
            bc=bc,
            domain_bounds=BOUNDS,
            current_time=0.5,
        )
    warned = any("inner Newton did not converge" in str(w.message) for w in caught)
    residual = base_hjb.compute_hjb_residual(
        np.asarray(returned, dtype=float),
        u_next,
        m,
        problem,
        5,
        None,
        0.3,
        True,
        bc=bc,
        domain_bounds=BOUNDS,
        current_time=0.5,
    )
    return (
        (not warned),
        base_hjb.hjb_residual_norm(np.asarray(residual, dtype=float), dx),
        np.asarray(returned, dtype=float),
    )


@pytest.mark.parametrize(
    "bc_name",
    [
        "no_flux",
        "neumann_zero",
        pytest.param(
            "dirichlet",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Dirichlet enforcement is still load-bearing and still overwrites the root: "
                    "residual 5.442e+01 while reporting converged. The residual's ghost padding only "
                    "APPROACHES the boundary value at O(h) (0.589 -> 0.696 against an exact 0.7 over "
                    "Nx 41..641), so the branch cannot simply be deleted the way no-flux's was. The "
                    "fix is row replacement in the residual AND the Jacobian, which #542's own "
                    "discussion point 2 named and which is its own change. strict=True so this "
                    "reddens the day it starts passing."
                ),
            ),
        ),
    ],
)
def test_a_solve_that_reports_converged_returns_a_root(bc_name: str):
    """The law: `converged` is a claim about the array handed back, not about an array discarded.

    This is an external oracle -- `F` is defined by the residual and the claim is made by the
    solver, so it cannot go tautological. Before the fix it was violated by 4e+04 on every no-flux
    solve that converged.
    """
    tol = 1e-9
    converged, residual, _ = _solve_and_measure(BCS[bc_name](), tol=tol)
    if not converged:
        pytest.skip(f"{bc_name}: this solve does not converge, so the law has nothing to bind")
    assert residual < tol, (
        f"{bc_name}: reported converged, but the returned array has residual {residual:.3e} "
        f"against the tolerance {tol:.3e} it certified"
    )


def test_the_no_flux_case_actually_converges_here_so_the_law_is_not_vacuous():
    """Positive control for the skip above.

    If every configuration failed to converge, the parametrised test would skip its way to green
    and assert nothing. No-flux is the case the defect was measured on and it must reach the
    tolerance, or this file is not testing what it claims.
    """
    converged, residual, _ = _solve_and_measure(no_flux_bc(dimension=1), tol=1e-9)
    assert converged, "no-flux no longer converges on this fixture; the law above would be vacuous"
    assert residual < 1e-9


@pytest.mark.parametrize("nx", [21, 41, 81])
def test_removing_the_enforcement_still_leaves_the_no_flux_condition_satisfied(nx: int):
    """Guards the opposite failure: a small residual bought by solving a different problem.

    The residual owns du/dn = 0 through its ghost padding, so the converged solution must satisfy a
    discrete no-flux condition without being forced to. Asserted against the scheme's own ghost
    treatment: with the mirror ghost u[-1] = u[0], the discrete backward difference at the wall is
    identically zero by construction, so what is checkable is the ONE-SIDED difference staying
    bounded and the solution not developing a boundary layer as h shrinks.
    """
    bc = no_flux_bc(dimension=1)
    converged, _, u = _solve_and_measure(bc, nx=nx, tol=1e-9)
    assert converged, f"nx={nx}: did not converge, cannot judge the boundary"

    dx = 1.0 / (nx - 1)
    slope_left = abs(u[1] - u[0]) / dx
    slope_right = abs(u[-1] - u[-2]) / dx
    interior = np.abs(np.diff(u)).max() / dx
    assert slope_left <= interior + 1e-9, f"nx={nx}: left wall slope {slope_left:.3e} exceeds interior {interior:.3e}"
    assert slope_right <= interior + 1e-9, (
        f"nx={nx}: right wall slope {slope_right:.3e} exceeds interior {interior:.3e}"
    )


def test_the_surviving_enforcement_uses_the_grids_own_spacing():
    """The block computed `span / Nx` where the spacing is `span / (Nx - 1)`.

    Inert under no-flux, because it multiplied `g = 0` -- and that branch is gone now anyway. It
    survives for Robin, where the spacing enters the denominator `alpha + beta/dx`: at Nx=21 the
    wrong value is 1/21 against a true 1/20. Third known site of the node-count/interval-count
    confusion (#1889, #1896 item 8).

    A FACED segment is required. A uniform `robin_bc()` carries no face, so `_get_bc_info_1d` falls
    through to hardcoded `alpha=1, beta=0` and the whole formula collapses to `u[0] = g` with the
    spacing never read -- a separate defect, filed on its own. Using the uniform constructor here
    would make this test pass while measuring nothing.
    """
    nx = 21
    alpha, beta, g = 1.0, 1.0, 0.5
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="L", bc_type=BCType.ROBIN, alpha=alpha, beta=beta, value=g, boundary="x_min"),
            BCSegment(name="R", bc_type=BCType.ROBIN, alpha=alpha, beta=beta, value=g, boundary="x_max"),
        ],
        dimension=1,
    )
    # Control: the coefficients must actually arrive, or the assertion below is about nothing.
    _, _, got_alpha, got_beta = _get_bc_info_1d(bc, "left", 0.5)
    assert (got_alpha, got_beta) == (alpha, beta), (
        f"the faced segment's coefficients did not arrive: alpha={got_alpha}, beta={got_beta}"
    )

    _, _, u = _solve_and_measure(bc, nx=nx, tol=1e-9)
    true_dx = 1.0 / (nx - 1)
    wrong_dx = 1.0 / nx
    expected_true = (g + beta * u[1] / true_dx) / (alpha + beta / true_dx)
    expected_wrong = (g + beta * u[1] / wrong_dx) / (alpha + beta / wrong_dx)

    assert abs(expected_true - expected_wrong) > 1e-6, (
        f"the two spacings agree to {abs(expected_true - expected_wrong):.3e}; fixture cannot discriminate"
    )
    assert abs(u[0] - expected_true) < 1e-9, (
        f"u[0]={u[0]:.12f} matches neither: true-dx {expected_true:.12f}, wrong-dx {expected_wrong:.12f}"
    )


def test_a_uniform_robin_bc_loses_its_alpha_and_beta_before_reaching_the_enforcement():
    """`robin_bc()` is silently enforced as Dirichlet, because its coefficients never arrive.

    `robin_bc(dimension=1, alpha=1, beta=1, value=g)` builds ONE segment named "uniform" whose
    `face` is None. `_get_bc_info_1d`'s priority-1 loop requires `seg.face is not None and
    seg.face == target_face`, so it never matches, and priority 2 returns the hardcoded
    `default_alpha, default_beta = 1.0, 0.0` -- discarding the segment's real coefficients. With
    beta = 0 the enforcement's `denom = alpha + beta/dx` is 1 and `u[0] = (g + 0)/1 = g`, i.e.
    Dirichlet. Measured: the segment carries (1.0, 1.0) and the accessor returns (1.0, 0.0); with
    alpha=2, beta=-1 it still returns (1.0, 0.0).

    A FACED segment works correctly, which is why this was invisible -- the CLAUDE.md example for
    adjoint-consistent BCs uses `boundary="x_min"`.

    xfail(strict=True): this asserts the CORRECT behaviour, so it reddens the day the accessor is
    fixed and the marker has to come off. Filed separately from #1900.
    """
    bc = robin_bc(dimension=1, alpha=2.0, beta=-1.0, value=1.0)
    assert [(s.alpha, s.beta) for s in bc.segments] == [(2.0, -1.0)], "constructor no longer stores them"
    _, _, alpha, beta = _get_bc_info_1d(bc, "left", 0.0)
    assert (alpha, beta) == (2.0, -1.0), (
        f"a uniform robin_bc reaches the enforcement as alpha={alpha}, beta={beta} -- "
        f"beta=0 collapses the Robin formula to Dirichlet"
    )


test_a_uniform_robin_bc_loses_its_alpha_and_beta_before_reaching_the_enforcement = pytest.mark.xfail(
    strict=True,
    reason="verified defect: uniform robin_bc segments carry no face, so alpha/beta are replaced by defaults",
)(test_a_uniform_robin_bc_loses_its_alpha_and_beta_before_reaching_the_enforcement)

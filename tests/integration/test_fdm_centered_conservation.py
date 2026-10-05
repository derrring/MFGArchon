"""Issue #1149: the `FDM_CENTERED` scheme must conserve mass.

`FDM_CENTERED` used to route its FP to the non-conservative `gradient_centered`
(`v.grad(m)`) advection, which leaks probability mass through no-flux walls (lost
~58% on a 1D Neumann congestion MFG). It now routes to `divergence_centered`
(`div(v m)`, telescoping flux, zero boundary flux) -- 2nd-order, central, and
mass-conservative. The non-conservative gradient form was later removed altogether (#2007).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.factory.scheme_factory import create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types.schemes import NumericalScheme


def _problem(n=41, nt=40):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(
        hamiltonian=H, m_initial=lambda x: np.exp(-40 * (x - 0.35) ** 2), u_terminal=lambda x: 0.5 * (x - 0.5) ** 2
    )
    return MFGProblem(geometry=grid, components=comp, T=0.5, Nt=nt, volatility=0.3, coupling_coefficient=0.5)


def _mass(grid, field):
    """Mass on the grid's own measure (#2145), for one field or a whole (time, x) history.

    Every `sum(m) * dx` in this file was the cell-centred integral, and `TensorProductGrid` is
    node-centred: the wall lies ON x_0, so the two end nodes own half a cell each and the measure is
    the trapezoid. Those weights ARE the control volumes the divergence-form flux telescopes
    against. Measuring the rectangle while the scheme conserves the trapezoid is not a small
    mismatch -- on the #1975 census fixture the divergence family lost a quarter of the true mass
    while the rectangle read machine zero.
    """
    return np.asarray(grid.integrate(np.asarray(field, dtype=float)), dtype=float)


def test_fdm_centered_routes_to_conservative_divergence_centered():
    """The centered scheme's FP must be the conservative divergence form, not the
    non-conservative gradient form."""
    _, fp = create_paired_solvers(_problem(), NumericalScheme.FDM_CENTERED)
    assert fp.advection_scheme == "divergence_centered"


def test_fdm_centered_conserves_mass_under_no_flux():
    """An FP solve conserves mass to machine precision on a no-flux domain (Issue #1149).

    The density is placed AT the wall (columns 0-2) and the drift pushes toward it, so the
    boundary-face flux is exercised. The boundary handler previously evaluated that face
    velocity one-sided while the interior used a central stencil -> double-valued face flux
    -> leak; a mid-domain density (as an earlier version of this test used) is blind to it."""
    n, nt = 41, 40
    prob = _problem(n=n, nt=nt)
    _, fp = create_paired_solvers(prob, NumericalScheme.FDM_CENTERED)
    x = np.linspace(0.0, 1.0, n)
    # Issue #1632: `drift_field=<ndarray>` is the VELOCITY channel, and `divergence_centered`
    # does not read it -- so the drift this test believed it was applying was discarded and the
    # solve ran at zero drift. The boundary face flux the docstring describes was never
    # exercised. Route the drift through `potential_field=U` instead, which this scheme does
    # consume: for the smooth separable H above the solver forms alpha = -c*grad(U) internally.
    # U increasing in x gives a leftward velocity, pushing the wall-adjacent bump into the wall.
    # The potential must be NON-LINEAR. The #1149 fix makes the boundary handler evaluate the
    # shared face velocity with the interior's central stencil, (U[2]-U[0])/(2*dx), instead of the
    # one-sided (U[1]-U[0])/dx. Those are algebraically identical for a linear U, so a linear
    # potential leaves this test blind to the very defect it pins: reverting both walls to the
    # one-sided form keeps the mass drift at 3.9e-15 under `U = x`, and moves it to 3.0e-02 here.
    potential = np.tile(0.3 * np.sin(np.pi * x), (nt + 1, 1))
    # A bump at EACH wall: `U = x + x^2/2` gave a leftward drift everywhere, so the right wall
    # carried ~5e-15 of density and every right-wall mutation was invisible. `0.3*sin(pi*x)` has
    # an interior maximum, so the drift runs into both walls, and it is curved, so the central
    # and one-sided face stencils differ there. Measured: reverting the LEFT wall to one-sided
    # gives 3.390e-03 and the RIGHT wall 3.390e-03 -- symmetric, where before it was 3.001e-02
    # and 3.664e-15 (blind).
    m0 = np.exp(-200 * (x - 0.05) ** 2) + np.exp(-200 * (x - 0.95) ** 2)
    m0 /= float(_mass(prob.geometry, m0))

    traj = fp.solve_fp_system(m0, potential_field=potential)
    mass = _mass(prob.geometry, traj)
    assert np.all(np.isfinite(traj))
    assert np.max(np.abs(mass - mass[0])) < 1e-12, (
        f"mass drift {np.max(np.abs(mass - mass[0])):.2e} (no-flux must conserve to machine precision)"
    )


@pytest.mark.integration
def test_fdm_centered_coupled_solve_conserves_mass():
    """End-to-end coupled MFG with FDM_CENTERED keeps mass ~1 (no wall leak)."""
    prob = _problem(n=31, nt=20)
    res = prob.solve(scheme=NumericalScheme.FDM_CENTERED, max_iterations=120, tolerance=1e-6, verbose=False)
    M = np.asarray(res.M)
    assert np.all(np.isfinite(M))
    # The #1149 bug leaked ~57% (terminal mass ~0.43). With the conservative scheme AND the
    # boundary-flux fix the coupled solve conserves mass to ~machine precision.
    #
    # The target is the INITIAL mass, not 1. `_problem` hands over an unnormalised Gaussian and
    # #1887 removed the rescale that used to turn it into a probability density, so `== 1` would now
    # pin a property of nothing. It also never belonged here: mass 1 is a property of the initial
    # condition, and what the scheme owes is that it transports whatever it was given.
    mass = _mass(prob.geometry, M)
    assert abs(mass[-1] - mass[0]) < 1e-9 * abs(mass[0]), (
        f"terminal mass {mass[-1]:.8f} against an initial {mass[0]:.8f} (centered must conserve)"
    )


@pytest.mark.integration
def test_callable_drift_explicit_path_respects_no_flux_no_periodic_wrap():
    """Issue #1181: the callable-drift explicit FP path must use the domain's no-flux BC,
    not the periodic default. With a constant leftward drift on a no-flux domain, mass piles
    at the LEFT wall and must NOT appear at the RIGHT wall. Pre-fix the advection omitted the
    BC argument -> periodic default -> mass exiting the left wall re-entered at the right wall
    (right-edge mass ~0.19); the fix passes boundary_conditions so the right half stays empty.
    """
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver

    n, nt, T = 81, 50, 0.5
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: np.asarray(m) * 0.0,
        coupling_dm=lambda m: np.asarray(m) * 0.0,
    )
    comps = MFGComponents(
        m_initial=lambda x: np.exp(-((np.asarray(x) - 0.5) ** 2) / 0.01),
        u_terminal=lambda x: np.asarray(x) * 0.0,
        hamiltonian=H,
    )
    prob = MFGProblem(geometry=grid, T=T, Nt=nt, volatility=0.05, components=comps)
    x = np.linspace(0.0, 1.0, n)
    dx = x[1] - x[0]  # the partial-region screen below is still a plain rectangle sum
    m0 = np.exp(-((x - 0.5) ** 2) / 0.01)
    m0 /= float(_mass(grid, m0))
    # callable drift signature is (t, grid, density); constant leftward velocity toward x=0
    M = FPFDMSolver(prob).solve_fp_system(m0.copy(), drift_field=lambda t, g, m: np.full(n, -0.3))
    assert np.all(np.isfinite(M))
    # No periodic wrap: a periodic default re-enters mass exiting the LEFT wall AT the RIGHT
    # wall, giving O(0.1) there. The direct wrap signal is the right-wall value itself.
    assert M[-1, -1] < 1e-6, f"mass wrapped to the right wall (no-flux violated): M[-1,-1]={M[-1, -1]:.3e}"
    # The far-right region stays a smooth, negligible diffusion tail. The conservative FV advection
    # (#1184) retains the mass the old scheme leaked and is slightly more diffusive, so the tail
    # (~4e-5) is fatter than the pre-#1184 leaking scheme (<1e-5) but still ~1e4x below the O(0.1)
    # periodic-wrap level this test guards against.
    # A partial-region sum, not a full-domain integral, so it keeps the rectangle: only one of its
    # two ends is a wall node, and the threshold is an order-of-magnitude screen against O(0.1),
    # four orders above the ~4e-5 it measures. #2145 changes this by less than the last printed digit.
    far_right_mass = M[-1, int(0.7 * n) :].sum() * dx
    assert far_right_mass < 1e-3, (
        f"mass wrapped through the no-flux wall: far-right (x>=0.7) mass {far_right_mass:.3e} "
        f"(periodic default gives O(0.1); a diffusion tail is ~4e-5)"
    )
    # Conservative FV advection (#1184) conserves mass exactly even under strong wall-directed drift.
    # The defect pin that stood here is retired, by its own retirement condition. It recorded that
    # this path -- `FPFDMSolver` with a CALLABLE drift -- routed its advection through the FV kernel
    # `fp_fvm_flux.axis_flux_divergence`, which divided every cell by `dx` while this solver's
    # diffusion used node control volumes, leaving a trapezoid drift of 2.388e-05. The kernel now
    # takes its control volumes from `quadrature_weights_1d`, the pin tripped at 1.110e-15, and its
    # message said to replace it with the bound the rest of this file uses.
    mass = _mass(grid, M)
    assert abs(mass[-1] - mass[0]) < 1e-9 * abs(mass[0]), (
        f"mass not conserved: {mass[-1]:.8f} against an initial {mass[0]:.8f}"
    )
    # Mass should pile toward the LEFT wall (where the leftward drift transports it).
    assert M[-1, 0] > M[-1, -1], "leftward drift did not pile mass at the left wall"

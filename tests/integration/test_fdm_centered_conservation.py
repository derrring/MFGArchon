"""Issue #1149: the `FDM_CENTERED` scheme must conserve mass.

`FDM_CENTERED` used to route its FP to the non-conservative `gradient_centered`
(`v.grad(m)`) advection, which leaks probability mass through no-flux walls (lost
~58% on a 1D Neumann congestion MFG). It now routes to `divergence_centered`
(`div(v m)`, telescoping flux, zero boundary flux) -- 2nd-order, central, and
mass-conservative. The non-conservative form stays available as an explicit
`advection_scheme` but is no longer the centered default.
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
    return MFGProblem(geometry=grid, components=comp, T=0.5, Nt=nt, sigma=0.3, coupling_coefficient=0.5)


def test_fdm_centered_routes_to_conservative_divergence_centered():
    """The centered scheme's FP must be the conservative divergence form, not the
    non-conservative gradient form."""
    _, fp = create_paired_solvers(_problem(), NumericalScheme.FDM_CENTERED)
    assert fp.advection_scheme == "divergence_centered"


def test_fdm_centered_conserves_mass_under_no_flux():
    """An FP solve with a non-trivial drift on a no-flux domain conserves mass to machine
    precision (Issue #1149: gradient_centered leaked through the walls)."""
    n, nt = 41, 40
    prob = _problem(n=n, nt=nt)
    _, fp = create_paired_solvers(prob, NumericalScheme.FDM_CENTERED)
    x = np.linspace(0.0, 1.0, n)
    dx = x[1] - x[0]
    drift = np.tile(0.5 * (x - 0.5) ** 2, (nt + 1, 1))  # steady confining drift toward 0.5
    m0 = np.exp(-40 * (x - 0.35) ** 2)
    m0 /= m0.sum() * dx

    traj = fp.solve_fp_system(m0, drift_field=drift)
    mass = np.array([traj[k].sum() * dx for k in range(nt + 1)])
    assert np.all(np.isfinite(traj))
    assert np.max(np.abs(mass - mass[0])) < 1e-9, (
        f"mass drift {np.max(np.abs(mass - mass[0])):.2e} (no-flux must conserve)"
    )


def test_gradient_centered_still_available_and_leaks():
    """The non-conservative form is still selectable explicitly, and demonstrably does NOT
    conserve mass -- documenting why it is no longer the centered default."""
    n, nt = 41, 40
    prob = _problem(n=n, nt=nt)
    _, fp = create_paired_solvers(
        prob, NumericalScheme.FDM_CENTERED, fp_config={"advection_scheme": "gradient_centered"}
    )
    assert fp.advection_scheme == "gradient_centered"
    x = np.linspace(0.0, 1.0, n)
    dx = x[1] - x[0]
    drift = np.tile(0.5 * (x - 0.5) ** 2, (nt + 1, 1))
    m0 = np.exp(-40 * (x - 0.35) ** 2)
    m0 /= m0.sum() * dx
    traj = fp.solve_fp_system(m0, drift_field=drift)
    mass = np.array([traj[k].sum() * dx for k in range(nt + 1)])
    assert np.max(np.abs(mass - mass[0])) > 1e-3, "gradient_centered is expected to violate conservation"


@pytest.mark.integration
def test_fdm_centered_coupled_solve_conserves_mass():
    """End-to-end coupled MFG with FDM_CENTERED keeps mass ~1 (no wall leak)."""
    prob = _problem(n=31, nt=20)
    res = prob.solve(scheme=NumericalScheme.FDM_CENTERED, max_iterations=120, tolerance=1e-6, verbose=False)
    M = np.asarray(res.M)
    x = np.linspace(0.0, 1.0, 31)
    dx = x[1] - x[0]
    assert np.all(np.isfinite(M))
    # The #1149 bug leaked ~57% (terminal mass ~0.43); the conservative form keeps it
    # near 1 (residual is the centered scheme's mild boundary inexactness under the strong
    # coupled drift, ~0.5%, not a wall leak). A 2% bound catches a regression to the leak.
    assert abs(M[-1].sum() * dx - 1.0) < 2e-2, (
        f"terminal mass {M[-1].sum() * dx:.4f} (centered must not leak; bug was 0.43)"
    )

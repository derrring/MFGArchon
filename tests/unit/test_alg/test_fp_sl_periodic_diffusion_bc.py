"""Pinning test for Issue #1257: FPSLSolver periodic domain diffusion seam fix.

Bug: _adjoint_sl_step_1d built the CN matrix with zero-flux (Neumann) boundary
stencils unconditionally; _adjoint_sl_step_nd called adi_diffusion_step without
bc_type (defaults 'neumann').  On a periodic domain this paired periodic advection
with Neumann diffusion, producing an O(1) seam-flux error every step.

Fix: _get_diffusion_bc_type() maps 'periodic' -> 'periodic', else 'neumann'.
_adjoint_sl_step_1d branches to solve_crank_nicolson_diffusion_1d(..., bc_type)
for periodic; _adjoint_sl_step_nd passes bc_type=self._get_diffusion_bc_type().

Pinning test logic (1D):
  * Place a unit spike at x=0 (the periodic seam), zero drift.
  * After one diffusion sub-step the spike MUST spread symmetrically:
    m_new[1] (right neighbour) == m_new[-2] (left neighbour, wrap).
  * With the buggy zero-flux stencil at i=0, L[0] = (m[1]-m[0])/dx^2 only
    diffuses rightward, so m_new[1] >> m_new[-2].

The mirror is index -2, not index -1: `np.linspace(x_min, x_max, n)` is endpoint-inclusive,
so index -1 is the same physical point as index 0 and holds a copy of the spike. This file
said -1 until Issue #1820, which corrected the convention in the solver -- the assertion had
been comparing the spike's neighbour against the spike itself.
"""

from __future__ import annotations

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import periodic_bc


def _periodic_problem(n: int = 41, nt: int = 10, sigma: float = 0.3) -> MFGProblem:
    """Minimal MFGProblem with periodic BC for FPSLSolver construction."""
    bc = periodic_bc(dimension=1)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=bc)
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0,
        coupling_dm=lambda m: 0.0,
    )
    comp = MFGComponents(
        hamiltonian=H,
        m_initial=lambda x: np.ones_like(x),
        u_terminal=lambda x: x * 0,
    )
    return MFGProblem(geometry=grid, components=comp, T=1.0, Nt=nt, sigma=sigma)


def test_periodic_diffusion_seam_symmetry_1d():
    """After one diffusion sub-step with a spike at the seam, mass spreads symmetrically.

    Still the direct numerical witness of Issue #1257 -- a Neumann stencil gives
    m_new[1] >> its mirror -- but the mirror is index -2, not index -1.

    Issue #1820 corrected the grid convention: `np.linspace(x_min, x_max, n)` is
    endpoint-inclusive, so index 0 and index -1 are the SAME physical point and index
    -1 holds a copy of the spike rather than a neighbour of it. The symmetric partner
    of index 1 (x = dx) is index -2 (x = 1 - dx). Measured after the correction:
    m_new[1]/m_new[-2] = 1.000000 exactly, m_new[2] == m_new[-3], m_new[0] == m_new[-1],
    and total mass 1.000000. The previous assertion compared the spike's neighbour
    against the spike, which held only under the endpoint-exclusive reading.
    """
    n = 41
    prob = _periodic_problem(n=n, sigma=0.3)
    fp = FPSLSolver(prob)

    # Spike at x=0 (index 0), the periodic seam.  Normalised so sum*dx = 1.
    x = np.linspace(0.0, 1.0, n)
    dx = x[1] - x[0]
    m = np.zeros(n)
    m[0] = 1.0 / dx  # unit mass at left boundary / seam

    # Zero drift: only the diffusion sub-step acts.
    alpha = np.zeros(n)
    dt = prob.dt
    sigma = prob.sigma

    # Run ONE step.
    m_new = fp._adjoint_sl_step_1d(m, alpha, dt, sigma)

    # The right neighbour (index 1) and its mirror across the seam (index -2) must
    # receive equal mass. Tolerance: they must agree to within 1% relative error.
    ratio = m_new[1] / (m_new[-2] + 1e-300)
    assert abs(ratio - 1.0) < 0.01, (
        f"Periodic diffusion seam asymmetry: m_new[1]={m_new[1]:.6f}, "
        f"m_new[-2]={m_new[-2]:.6f}, ratio={ratio:.4f}.  "
        "Expected ratio ~1 (periodic CN); got >> 1 means Neumann zero-flux stencil "
        "is still in use (Issue #1257)."
    )
    # Mass, which this step must conserve exactly: zero drift means diffusion alone, and a
    # periodic domain has no boundary for mass to cross. NOT an endpoint-equality assertion --
    # that one cannot fail while the implementation ends in `np.append(interior, interior[0])`,
    # so it pins the last line rather than the scheme (Issue #1820).
    mass_before = float(np.trapezoid(m, x))
    mass_after = float(np.trapezoid(m_new, x))
    assert abs(mass_after / mass_before - 1.0) < 1e-12, (
        f"periodic diffusion changed total mass by {abs(mass_after / mass_before - 1.0):.3e} "
        f"({mass_before:.6f} -> {mass_after:.6f}) with zero drift"
    )


def test_periodic_diffusion_uses_diffusion_bc_type():
    """_get_diffusion_bc_type returns 'periodic' for a periodic-domain solver."""
    prob = _periodic_problem()
    fp = FPSLSolver(prob)
    assert fp._get_diffusion_bc_type() == "periodic", (
        "_get_diffusion_bc_type() must return 'periodic' for a periodic domain (Issue #1257)"
    )


def test_neumann_domain_preserves_zero_flux_stencil():
    """Neumann (no-flux) domain still uses the FV zero-flux path, not periodic CN.

    Regression guard: ensure the fix does not accidentally route Neumann domains
    through the periodic solver (which would be wrong for mass conservation).
    """
    from mfgarchon.geometry.boundary import no_flux_bc

    bc = no_flux_bc(dimension=1)
    n = 41
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=bc)
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0,
        coupling_dm=lambda m: 0.0,
    )
    comp = MFGComponents(
        hamiltonian=H,
        m_initial=lambda x: np.ones_like(x),
        u_terminal=lambda x: x * 0,
    )
    prob = MFGProblem(geometry=grid, components=comp, T=1.0, Nt=10, sigma=0.3)
    fp = FPSLSolver(prob)

    assert fp._get_diffusion_bc_type() == "neumann", (
        "_get_diffusion_bc_type() must return 'neumann' for a no-flux domain (regression guard for Issue #1257 fix)"
    )

    # Spike at the left boundary — with no-flux/zero-flux the spike is reflected,
    # so m_new[1] > m_new[-1] (boundary acts as a mirror, not a seam).
    #
    # A narrow Gaussian rather than a delta (Issue #1683). A single-cell delta of height
    # 1/dx drives the CN diffusion to -7.5 in one step: the stencil question this test
    # asks is answerable without an unphysical density, and the old input only passed
    # because the positivity clip repaired it. Measured, w = 2dx clips **nothing** and
    # the assertion below is unchanged.
    x = np.linspace(0.0, 1.0, n)
    dx = x[1] - x[0]
    m = np.exp(-(((x - 0.0) / (2 * dx)) ** 2))
    m = m / (m.sum() * dx)
    alpha = np.zeros(n)
    m_new = fp._adjoint_sl_step_1d(m, alpha, prob.dt, prob.sigma)

    # No-flux left boundary: diffusion only goes rightward, so m_new[1] > m_new[-1].
    assert m_new[1] > m_new[-1], (
        "Neumann domain: spike at left boundary should diffuse rightward only, "
        "so m_new[1] should exceed m_new[-1] (zero-flux stencil)."
    )

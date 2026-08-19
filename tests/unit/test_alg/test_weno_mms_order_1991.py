"""MMS order verification for HJBWENOSolver (Issue #1991).

`source_term` is the slot MMS forcing enters through, and most HJB solvers do not accept it
-- so no manufactured solution could reach WENO at all, and its order was not merely
unverified but unmeasurable. This threads it through the 1D path and pins what the
resulting measurement shows.

(A census by parameter name gave 8 of 11 lacking it. At the time that over-counted:
`HJBGFDMSolver` carried the same arithmetic slot under the name `running_cost`, at the opposite
sign and with a different convention -- `f(n) -> (n_points,)` rather than `f(t, x)`. #1999 has
since removed that channel, so GFDM now reaches the slot through `source_term` alone and the
census is no longer an over-count on that solver.

The sign relationship the two slots had is a plain negation, `additive_source = -source_term`,
fixed by two documented primitives: `h_eval.py` returns the residual
`-u_t + H(+additive_source) - D*lap_u`, while the source contract at `base_hjb.py:435` is
`F(u) = (u-u_next)/dt + H - S = 0`; setting both to zero gives the sign. Note the historical
hazard: the line that once "stated it directly" stated it with the WRONG sign -- "Added to
H_total = H(x,p,m) + L(t,x)". Measured by running this same manufactured pair through GFDM at
sigma=1: `-r_u` gives 3.1621e-03 -> 7.9413e-04, EOC 2.0; `+r_u` sits flat at 1.4233 -> 1.4241
and never converges. So a port carrying an FDM source into GFDM unchanged would converge on the
wrong solution, and the unified channel #1991 wants can be built on a settled premise.)

The oracle is external in AGENTS.md's sense: an exact solution constructed independently of
the scheme, not a second code path. The manufactured pair is the 1D reduction of the coupled
MMS in the GFDM paper (`chapters/appendix.tex`, eq:mms_reference / eq:mms_system):

    ubar(t,x) = a1(t) cos(c x),   a1(t) = 1 + (T - t)/(2T),   c = 2 pi / L

made exact for  -u_t - (sigma^2/2) u_xx + (1/2) u_x^2 = r_u.

`ubar_x = -a1 c sin(c x)` vanishes at both walls, so the manufactured solution is exactly
compatible with the no-flux boundary the solver imposes -- the boundary is not a separate
error source here.

What the measurement shows:

    sigma = 1.0000   EOC 2.02, 1.98, 2.00           diffusion-dominated
    sigma = 0.0005   EOC 5.41, 5.04, 5.01           advection-dominated, clean
    sigma = 0.0500   EOC 5.70, 5.43, 0.86           advection-dominated, CONTAMINATED

The error has two terms of OPPOSITE sign: the HJ-WENO5 reconstruction at O(h^5) and the
central second-difference diffusion at O(h^2), the latter scaling with sigma^2. They cancel,
and an earlier version of this docstring asserted the same model with both terms positive --
which confines the EOC to [2, 5] and so forbids the 0.86 sitting in its own table. Signed
error at x = 0, sigma = 0.05, Nt = 40 (the EOC ladders above are Nt = 20; the sign change
reproduces at either, but these magnitudes are the Nt = 40 ones):

    Nx= 21  -2.982166e-04
    Nx= 41  -5.751384e-06
    Nx= 81  +1.248895e-07     <- sign change: the two terms cross here
    Nx=161  +7.366815e-08
    Nx=321  +1.964453e-08

So at sigma = 0.05 the 5.43 is inflated by the approaching cancellation and the 0.86 is its
far side -- neither is a clean rate, and neither is "the order crossing back to 2". Pushing
the diffusion out of range at sigma = 0.0005 removes the cancellation and gives 5.41, 5.04,
5.01: HJ-WENO5 delivers its advertised fifth order, and the scheme's overall order is capped
by the second-order diffusion discretisation, which is a design property, not a defect.

WHAT THIS STUDY CANNOT SEE. `a1` is linear in `t`, so the exact trajectory is a straight line
and SSP-RK3 integrates it exactly: the L-infinity error is 5.7514e-06 at Nx=41, sigma=0.05,
for EVERY Nt from 10 to 320, EOC 0.00 throughout. That is deliberate -- it isolates the spatial operator --
but it means no number here establishes third-order-in-time. It cuts the useful way for the
stage-time tests: because the correct assignment drives the temporal error to zero rather than
merely small, a wrong stage time is not buried under it but stands out as a flat O(dt) floor.

The Nt-independence at Nx=161 is 7.3668145e-08 vs 7.3668131e-08 across Nt = 40 to 320 --
identical to seven significant figures, not bit-identical as previously claimed; the sub-step
sequence does change with Nt. The conclusion that the floor is spatial survives; the word did
not.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBWENOSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

L = 20.0
T = 4.0
C = 2.0 * np.pi / L


def _a1(t):
    return 1.0 + (T - t) / (2.0 * T)


def _u_exact(t, x):
    return _a1(t) * np.cos(C * x)


def _source(t, x, sigma):
    x = np.asarray(x, dtype=float)
    u_t = (-1.0 / (2.0 * T)) * np.cos(C * x)
    u_xx = -_a1(t) * C**2 * np.cos(C * x)
    u_x = -_a1(t) * C * np.sin(C * x)
    return -u_t - 0.5 * sigma**2 * u_xx + 0.5 * u_x**2


def _problem(nx, nt, sigma, dimension=1):
    bounds = [(0.0, L)] * dimension
    grid = TensorProductGrid(
        bounds=bounds, Nx_points=[nx] * dimension, boundary_conditions=no_flux_bc(dimension=dimension)
    )
    if dimension == 1:
        m_initial = lambda x: np.ones_like(np.asarray(x, dtype=float)) / L  # noqa: E731
        u_terminal = lambda x: _u_exact(T, x)  # noqa: E731
    else:
        m_initial = lambda x, y: np.ones_like(np.asarray(x, dtype=float)) / (L * L)  # noqa: E731
        u_terminal = lambda x, y: np.zeros_like(np.asarray(x, dtype=float))  # noqa: E731
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=m_initial,
        u_terminal=u_terminal,
    )
    return MFGProblem(geometry=grid, components=comps, T=T, Nt=nt, sigma=sigma)


def _linf_error(nx, nt, sigma):
    problem = _problem(nx, nt, sigma)
    solver = HJBWENOSolver(problem)
    x = np.linspace(0.0, L, nx)
    m = np.tile(np.ones(nx) / L, (nt + 1, 1))
    u_T = _u_exact(T, x)
    U = solver.solve_hjb_system(
        M_density=m,
        U_terminal=u_T,
        U_coupling_prev=np.tile(u_T, (nt + 1, 1)),
        source_term=lambda t, xx: _source(t, xx, sigma),
    )
    return np.abs(U[0] - _u_exact(0.0, x)).max()


def _eoc(nx_coarse, nx_fine, nt, sigma):
    e_c = _linf_error(nx_coarse, nt, sigma)
    e_f = _linf_error(nx_fine, nt, sigma)
    h_c = L / (nx_coarse - 1)
    h_f = L / (nx_fine - 1)
    return np.log(e_c / e_f) / np.log(h_c / h_f)


def test_weno_accepts_a_source_term_at_all():
    """The precondition: without this slot no manufactured solution can reach WENO."""
    nx, nt, sigma = 41, 20, 1.0
    problem = _problem(nx, nt, sigma)
    solver = HJBWENOSolver(problem)
    x = np.linspace(0.0, L, nx)
    m = np.tile(np.ones(nx) / L, (nt + 1, 1))
    u_T = _u_exact(T, x)

    forced = solver.solve_hjb_system(
        M_density=m,
        U_terminal=u_T,
        U_coupling_prev=np.tile(u_T, (nt + 1, 1)),
        source_term=lambda t, xx: _source(t, xx, sigma),
    )
    unforced = solver.solve_hjb_system(M_density=m, U_terminal=u_T, U_coupling_prev=np.tile(u_T, (nt + 1, 1)))
    assert np.isfinite(forced).all()
    # Discrimination: the source must actually change the answer, or every order figure
    # below would be measuring an unforced solve that happens to sit near the exact solution.
    assert np.abs(forced - unforced).max() > 1e-6, "source_term reached the solver but changed nothing"


def test_diffusion_dominated_order_is_two():
    """sigma = 1: the central second-difference diffusion sets the rate, not WENO5."""
    order = _eoc(41, 81, nt=20, sigma=1.0)
    assert 1.7 < order < 2.3, f"expected ~2 (diffusion-limited), measured {order:.2f}"


def test_advection_dominated_order_reaches_weno5():
    """sigma tiny: HJ-WENO5 sets the rate, uncontaminated, and must clear fourth order.

    sigma = 0.0005 rather than 0.05, and Nx 41->81 rather than 21->41, both deliberately. At
    sigma = 0.05 the O(h^2) diffusion term and the O(h^5) reconstruction have opposite signs and
    cancel near h ~ 0.3, which inflates the 41->81 slope to 5.43 and collapses the next one to
    0.86; and the 21->41 pair is pre-asymptotic, its EOC of 5.70 sitting ABOVE the design order,
    which is the diagnostic that it is not yet asymptotic. Neither is a sound certificate even
    though both clear the threshold. Here the measured ladder is 5.41, 5.04, 5.01.

    Bounded below rather than pinned two-sided: the assertion should fail when the reconstruction
    degrades, not when the crossover moves.
    """
    order = _eoc(41, 81, nt=20, sigma=0.0005)
    assert order > 4.0, f"HJ-WENO5 should clear 4th order in the advection-dominated regime, got {order:.2f}"


def test_source_term_is_refused_rather_than_dropped_in_multi_d():
    """The multi-D path is a dimensional split, so a source added per axis sweep would be
    applied `dimension` times per step. Refusing beats silently solving the wrong problem."""
    problem = _problem(nx=11, nt=5, sigma=1.0, dimension=2)
    solver = HJBWENOSolver(problem)
    m = np.ones((6, 11, 11)) / (L * L)
    u_T = np.zeros((11, 11))
    with pytest.raises(NotImplementedError, match="1991"):
        solver.solve_hjb_system(
            M_density=m,
            U_terminal=u_T,
            U_coupling_prev=np.tile(u_T, (6, 1, 1)),
            # Shape-correct even though it is never called: a scalar return is not a valid
            # source under the base contract, and the test should not model an unsupported call.
            source_term=lambda t, x: np.zeros(np.asarray(x).shape[0]),
        )

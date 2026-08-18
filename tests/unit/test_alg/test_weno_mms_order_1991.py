"""MMS order verification for HJBWENOSolver (Issue #1991).

`source_term` is the slot MMS forcing enters through, and 8 of 11 HJB solvers did not
accept it -- so no manufactured solution could reach WENO at all, and its order was not
merely unverified but unmeasurable. This threads it through the 1D path and pins what the
resulting measurement shows.

The oracle is external in AGENTS.md's sense: an exact solution constructed independently of
the scheme, not a second code path. The manufactured pair is the 1D reduction of the coupled
MMS in the GFDM paper (`chapters/appendix.tex`, eq:mms_reference / eq:mms_system):

    ubar(t,x) = a1(t) cos(c x),   a1(t) = 1 + (T - t)/(2T),   c = 2 pi / L

made exact for  -u_t - (sigma^2/2) u_xx + (1/2) u_x^2 = r_u.

`ubar_x = -a1 c sin(c x)` vanishes at both walls, so the manufactured solution is exactly
compatible with the no-flux boundary the solver imposes -- the boundary is not a separate
error source here.

What the measurement showed, and why both rates are correct:

    sigma = 1.00   EOC 2.02, 1.98, 2.00      diffusion-dominated
    sigma = 0.05   EOC 5.70, 5.43            advection-dominated

The total error is C5*h^5 from the HJ-WENO5 reconstruction plus C2*h^2 from the central
second-difference diffusion, with C2 proportional to sigma^2. At sigma = 1 the second-order
term dominates everywhere. At sigma = 0.05 the fifth-order term dominates until h^5 falls
below it, after which the rate crosses back to 2 -- measured at Nx = 161 -> 321, EOC 1.91,
and confirmed spatial rather than temporal because the error is bit-identical across
Nt = 40, 80, 160, 320.

So WENO5 does deliver its advertised order; the scheme's overall order is capped by the
diffusion discretisation, which is a design property and not a defect.
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
    """sigma small: HJ-WENO5 sets the rate and must clear fourth order.

    Bounded below at 4 rather than pinned near 5 because the C2*h^2 diffusion term is present
    at every level and pulls the fitted slope down as h shrinks; the crossover is real and
    documented in the module docstring, so a tight two-sided band here would be pinning the
    crossover location rather than the reconstruction order.
    """
    order = _eoc(21, 41, nt=20, sigma=0.05)
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
            M_density=m, U_terminal=u_T, U_coupling_prev=np.tile(u_T, (6, 1, 1)), source_term=lambda t, x: 0.0
        )

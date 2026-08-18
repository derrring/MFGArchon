"""GFDM accepts the package-wide MMS source, and the order it then measures (Issue #1991).

MMS is the only *external* oracle in this library -- every other check compares one code path
against another -- and its forcing enters through `source_term`. `HJBGFDMSolver` did not accept
that argument, so the capability gate at `coupling/base_mfg.py:215` rejected it with "Use an FDM
HJB solver", while GFDM in fact had the channel all along under the name `running_cost`. The gate
keys on a parameter NAME, so it was measuring a proxy for the capability it was asked about.

`source_term` and `running_cost` are NOT the same quantity and are deliberately not unified:

- `running_cost` is model data. `HJBHowardSolver` documents this slot as "the non-quadratic-in-alpha
  part of the Lagrangian (potential V(x), congestion g(x, m), etc.)", so it MAY depend on `m`; the
  alpha-dependent half of the Lagrangian is `control_lagrangian`, which sits inside the Legendre
  transform rather than beside it.
- `source_term` is artificial forcing for verification and must not depend on `m`.

They share one arithmetic slot with opposite signs -- `h_eval.assemble_hjb_residual` returns
`-u_t + H(+running_cost) - D*lap_u` while the source contract in `base_hjb` is
`F(u) = (u-u_next)/dt + H - S = 0`, hence `running_cost = -source_term` -- so the solver keeps them
as separate attributes and adds them only at the call site.

Manufactured pair is the 1D reduction of the coupled MMS in the GFDM paper
(`chapters/appendix.tex`, eq:mms_reference / eq:mms_system):

    ubar(t,x) = a1(t) cos(c x),   a1(t) = 1 + (T - t)/(2T),   c = 2 pi / L

exact for `-u_t - (sigma^2/2) u_xx + (1/2) u_x^2 = r_u`.
"""

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
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
    a = np.cos(C * np.asarray(x, dtype=float))
    return float(_a1(t) * a.reshape(-1)[0]) if a.size == 1 else _a1(t) * a.reshape(-1)


def _source(t, x, sigma):
    x = np.asarray(x, dtype=float).reshape(-1)
    u_t = (-1.0 / (2.0 * T)) * np.cos(C * x)
    u_xx = -_a1(t) * C**2 * np.cos(C * x)
    u_x = -_a1(t) * C * np.sin(C * x)
    return -u_t - 0.5 * sigma**2 * u_xx + 0.5 * u_x**2


def _linf(nx, nt=20, sigma=1.0, sign=-1.0):
    """L-inf error at t=0. `sign=+1` flips the source, as a discrimination control."""
    x = np.linspace(0.0, L, nx)
    grid = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda xx: np.ones_like(np.asarray(xx, dtype=float)) / L,
        u_terminal=lambda xx: _u_exact(T, xx),
    )
    problem = MFGProblem(geometry=grid, components=comps, T=T, Nt=nt, sigma=sigma)
    solver = HJBGFDMSolver(problem, collocation_points=x.reshape(-1, 1), delta=3.0 * L / (nx - 1))
    m = np.tile(np.ones(nx) / L, (nt + 1, 1))
    u_T = _u_exact(T, x)
    U = solver.solve_hjb_system(
        M_density=m,
        U_terminal=u_T,
        U_coupling_prev=np.tile(u_T, (nt + 1, 1)),
        source_term=lambda t, xx: -sign * _source(t, xx, sigma),
    )
    return float(np.abs(np.asarray(U)[0].reshape(-1) - _u_exact(0.0, x)).max())


def test_gfdm_accepts_the_package_wide_source_argument():
    """The gate keys on the name, so the name is the capability as far as the coupler is concerned."""
    import inspect

    params = set(inspect.signature(HJBGFDMSolver.solve_hjb_system).parameters)
    assert "source_term" in params, "the capability gate at base_mfg.py:215 tests for this name"
    assert "running_cost" in params, "the modelling concept must survive; it is not the same quantity"


def test_mms_reaches_gfdm_and_it_converges():
    """External oracle: an exact solution built independently of the scheme.

    GFDM's second-order Taylor reconstruction makes the exact Laplacian moments on a uniform
    cloud, so second order is the expected rate here, not a shortfall.
    """
    e_c, e_f = _linf(21), _linf(41)
    order = np.log(e_c / e_f) / np.log(2.0)
    assert 1.7 < order < 2.3, f"expected ~2, measured {order:.2f} (errors {e_c:.3e} -> {e_f:.3e})"


def test_the_source_sign_is_not_free():
    """Discrimination: flipping the source must destroy convergence, or the test proves nothing.

    `running_cost = -source_term` is a sign convention bridging two residual framings, and a
    convergence assertion that passes under either sign would be measuring nothing. Measured:
    the correct sign gives EOC 2.00/1.99, the flipped one sits flat at 1.42.
    """
    flipped_c, flipped_f = _linf(21, sign=+1.0), _linf(41, sign=+1.0)
    assert flipped_f > 1.0, f"a flipped source should not converge, got {flipped_f:.3e}"
    order = np.log(flipped_c / flipped_f) / np.log(2.0)
    assert order < 0.5, f"flipped source converged at order {order:.2f}; the sign is not being honoured"

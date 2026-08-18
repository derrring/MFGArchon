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

WHAT THIS MEASURES, AND WHAT IT CANNOT. `a1(t)` is LINEAR in `t`, so for implicit backward
Euler `(u(t_{n+1}) - u(t_n))/dt` equals `u_t` exactly and the temporal truncation error is
identically zero. Measured at nx=161 refining nt only: 1.9624e-04 at nt=10 against 1.9307e-04
at nt=80 -- 1% over an 8x refinement, EOC_t 0.01. With a quadratic or exponential `a1` the same
refinement gives EOC_t 1.00. So the EOC 2 below is the order of the SPATIAL operator, with `nt`
held fixed, and no number here establishes anything about the time discretisation.

This matters for whoever strengthens the MMS next: making `a1` nonlinear collapses the space
study to EOC 0.27/0.03/0.01 at fixed nt, because the temporal error then dominates. That is the
manufactured solution changing, not a GFDM regression.
"""

import pytest

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


def _linf(nx, nt=20, sigma=1.0, sign=-1.0, **solver_kw):
    """L-inf error at t=0. `sign=+1` flips the source, as a discrimination control."""
    x = np.linspace(0.0, L, nx)
    grid = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda xx: np.ones_like(np.asarray(xx, dtype=float)) / L,
        u_terminal=lambda xx: _u_exact(T, xx),
    )
    problem = MFGProblem(geometry=grid, components=comps, T=T, Nt=nt, sigma=sigma)
    solver = HJBGFDMSolver(problem, collocation_points=x.reshape(-1, 1), delta=3.0 * L / (nx - 1), **solver_kw)
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
    # Issue #1999: and there is no second additive channel beside it. The alpha-independent part
    # of the Lagrangian -- V(x,t) + f(m) -- is the Hamiltonian's, and a `running_cost=` parameter
    # could only carry the same quantity a second time: supplied alongside a Hamiltonian that
    # already held a potential, it double-counted silently (#2001).
    assert "running_cost" not in params, (
        "a caller must not be able to inject an alpha-free cost that bypasses the Hamiltonian"
    )


def test_mms_reaches_gfdm_and_it_converges():
    """External oracle: an exact solution built independently of the scheme.

    Second order is the expected rate here, not a shortfall. The mechanism is deliberately not
    asserted: the obvious explanation -- that GFDM's second-order Taylor reconstruction makes the
    Laplacian moments exact on a uniform cloud -- is under-determined, since the rate survives
    jittering the interior points by 40% of h, so uniformity is not what carries it.
    """
    e_c, e_f = _linf(21), _linf(41)
    order = np.log(e_c / e_f) / np.log(2.0)
    assert 1.7 < order < 2.3, f"expected ~2, measured {order:.2f} (errors {e_c:.3e} -> {e_f:.3e})"


_HOWARD = {
    "inner_solver": "howard",
    "monotonicity_scheme": "joint_socp",
    "monotonicity_application": "precompute",
}


def test_the_howard_inner_solver_also_honours_the_source():
    """The source must reach BOTH inner solvers, or the capability gate lies.

    `_mms_source_fn` was read only in the Newton branch, so this configuration accepted
    `source_term` and discarded it bitwise -- measured, |U(source) - U(no source)| = 0.000e+00
    at two resolutions. Since the gate at `coupling/base_mfg.py:215` keys on the parameter
    NAME, accepting the name while dropping the argument turns that gate's false negative into
    a false positive: it would certify GFDM as source-capable in a configuration that silently
    solves the wrong problem, which is precisely what #1424 exists to prevent.
    """
    with_src = _linf(21, **_HOWARD)
    order = np.log(with_src / _linf(41, **_HOWARD)) / np.log(2.0)
    assert 1.7 < order < 2.3, f"Howard path expected ~2, measured {order:.2f}"

    flipped = _linf(41, sign=+1.0, **_HOWARD)
    assert flipped > 1.0, f"a flipped source should not converge on the Howard path, got {flipped:.3e}"


def test_the_source_reaches_gfdm_in_2d():
    """1D is not enough for a meshfree nD method, and the paper's manufactured pair is 2D.

    Every MMS pin added today is 1D. GFDM exists to work on scattered clouds in n dimensions,
    the source is flattened to the collocation ordering, and the shape guard above is about a
    2D array's point order -- so a 1D-only pin leaves the dimension where the guard matters
    entirely unexercised.

    2D reduction of the same pair: `ubar = a1(t)(cos(c x1) + cos(c x2))`, whose normal
    derivative vanishes on all four walls, so it stays no-flux compatible.
    """
    n1 = 9
    xs = np.linspace(0.0, L, n1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])

    def u_ex(t, p):
        return _a1(t) * (np.cos(C * p[:, 0]) + np.cos(C * p[:, 1]))

    def src(t, p, sigma=1.0):
        a, da = _a1(t), -1.0 / (2.0 * T)
        cos_sum = np.cos(C * p[:, 0]) + np.cos(C * p[:, 1])
        u_t = da * cos_sum
        lap = -a * C**2 * cos_sum
        grad2 = (a * C) ** 2 * (np.sin(C * p[:, 0]) ** 2 + np.sin(C * p[:, 1]) ** 2)
        return -u_t - 0.5 * sigma**2 * lap + 0.5 * grad2

    grid = TensorProductGrid(
        bounds=[(0.0, L), (0.0, L)], Nx_points=[n1, n1], boundary_conditions=no_flux_bc(dimension=2)
    )
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x, y: np.ones_like(np.asarray(x, dtype=float)) / (L * L),
        u_terminal=lambda x, y: _a1(T) * (np.cos(C * np.asarray(x)) + np.cos(C * np.asarray(y))),
    )
    nt = 8
    problem = MFGProblem(geometry=grid, components=comps, T=T, Nt=nt, sigma=1.0)
    solver = HJBGFDMSolver(problem, collocation_points=pts, delta=3.0 * L / (n1 - 1))
    m = np.ones((nt + 1, pts.shape[0])) / (L * L)
    u_T = u_ex(T, pts)

    def solve(sign):
        U = solver.solve_hjb_system(
            M_density=m,
            U_terminal=u_T,
            U_coupling_prev=np.tile(u_T, (nt + 1, 1)),
            source_term=lambda t, p: -sign * src(t, np.asarray(p).reshape(-1, 2)),
        )
        return float(np.abs(np.asarray(U)[0].reshape(-1) - u_ex(0.0, pts)).max())

    forced, flipped = solve(-1.0), solve(+1.0)
    assert flipped > 3.0 * forced, f"2D source is not discriminating: forced={forced:.3e} flipped={flipped:.3e}"


@pytest.mark.slow
def test_the_per_point_residual_path_also_honours_the_source():
    """A THIRD arithmetic site, with its own sign, and nothing else reaches it.

    `qp_optimization_level != "none"` switches off the batch residual and uses the per-point
    loop, where the source enters at `hjb_gfdm.py:2332` as `H = H + running_cost[i]` -- separate
    from `h_eval.assemble_hjb_residual` (Newton batch) and from `howard_running_cost`'s `-rc`.
    Three sites, three conventions to keep straight. Flipping the sign at that one line leaves
    every other test in this module green while the path sits at the flat-1.42 non-convergence
    signature, so this is the failure nothing else discriminates.

    One flipped-sign control, not an order study. Marked slow: the QP-per-point assembly costs
    ~24s for the pair even at nx=11/nt=4, so the gate skips it and nightly runs it. Measured
    separation at the shipped size: correct 1.275e-02 against flipped 1.429e+00, 112x.
    """
    # v0.25.0 (#1070) removed `qp_optimization_level=`; the axes are passed directly.
    kw = {"monotonicity_scheme": "qp_m_matrix", "monotonicity_application": "always"}
    correct = _linf(21, nt=4, **kw)
    flipped = _linf(21, nt=4, sign=+1.0, **kw)
    assert flipped > 1.0, f"flipped source should not converge on the per-point path, got {flipped:.3e}"
    assert flipped / correct > 10.0, (
        f"per-point path barely distinguishes the sign: correct={correct:.3e} flipped={flipped:.3e}"
    )


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

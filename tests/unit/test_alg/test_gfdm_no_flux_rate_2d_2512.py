"""#2512 (a) item 1: HJB-GFDM's reflecting cell, the 2-D box with every face no-flux, converges to an exact reflected
solution at the order of each distinct discrete equation the class ships (user ruling 2026-10-22, #2512 comment
6333790265).

The solution is exp17's, as `test_positive_qp_fan_reflecting_1689` builds it and imported from there:
``u* = phi + C (T - t)``, ``phi = a (q(x) + q(y))``, ``q(s) = s^3 - 1.5 s^2``, so ``d_n phi = 0`` on every edge; ``m*``
is frozen and ``V`` absorbs the residual, so the equation holds exactly, and implicit Euler is exact in time, so the
error is the spatial one. The clouds are exp17's Sobol clouds, refined as N_int 900 -> 1600 -> 2500 with the
neighbourhood radius scaled with the spacing, ``delta = 4.5 h``, ``h = 1/sqrt(N_int)`` (0.15 at N_int 900). A fixed
delta is not a refinement: there the error saturates near 6.7e-4 from N_int 900.

The equations, and what was measured at 29cdb9d6 over these two refinements on scrambles 42, 3 and 7 (#2512 comment
6333097202):
- ``monotonicity_scheme="none"`` with Newton: EOC 1.88 to 2.08, band [1.7, 2.3].
- ``"none"`` with Howard: EOC 0.97 to 1.01. In d >= 2 Howard is handed ``discretisation="upwind_projection"``, a
  one-sided advection stencil, so first order is the equation's own (`HJBGFDMSolver`'s ``inner_solver`` docstring).
  The band [0.7, 1.3] excludes second order, so a switch to the central stencil is red here.
- ``"joint_socp"`` with Newton: EOC 1.90 to 2.03, band [1.7, 2.3], slow.
- ``"joint_socp"`` with Howard: EOC 0.82 to 0.91, still rising (0.71 to 0.72 from N_int 400); band [0.7, 1.3], slow.
  Measured at 29cdb9d6 for the change that added this file.
- ``"positive_qp_fan"`` with Howard has its own boundary rows and no rate on these clouds; its reflecting cell is
  1689's tolerance pin.

With the no-flux row read along the face's tangent instead of its normal (`_bc_row_for_point`), every row here is red:
@MUT@
"""

from __future__ import annotations

import warnings

import pytest
from test_positive_qp_fan_reflecting_1689 import NT, T_END, _bc, _m_star, _problem, _sobol, _u_star

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

_LADDER = (900, 1600, 2500)
_SECOND, _FIRST = (1.7, 2.3), (0.7, 1.3)


def _error(n_int: int, scramble: int, scheme: str, inner: str) -> float:
    """The L2 error at t = 0 over the interior nodes of exp17's cloud with ``n_int`` interior nodes."""
    points, boundary = _sobol(n_int, seed=scramble)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = HJBGFDMSolver(
            _problem(_bc("box")),
            collocation_points=points,
            boundary_indices=boundary,
            delta=4.5 / np.sqrt(n_int),
            monotonicity_scheme=scheme,
            inner_solver=inner,
        )
        U = solver.solve_hjb_system(M_density=np.tile(_m_star(points), (NT + 1, 1)), U_terminal=_u_star(T_END, points))
    interior = np.setdiff1d(np.arange(len(points)), boundary)
    return float(np.sqrt(np.mean((np.asarray(U)[0] - _u_star(0.0, points))[interior] ** 2)))


def _orders(scramble: int, scheme: str, inner: str) -> tuple[list[float], list[float]]:
    errors = [_error(n_int, scramble, scheme, inner) for n_int in _LADDER]
    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(np.sqrt(_LADDER[i + 1] / _LADDER[i])) for i in range(len(errors) - 1)
    ]
    return errors, orders


_SCRAMBLES = [42, 3, 7]
_INNER = [pytest.param("newton", _SECOND, id="newton"), pytest.param("howard", _FIRST, id="howard")]


@pytest.mark.parametrize("scramble", _SCRAMBLES)
@pytest.mark.parametrize(("inner", "band"), _INNER)
def test_the_default_stencils_converge_at_their_equations_order_on_reflecting_walls(inner, band, scramble):
    """``monotonicity_scheme="none"``: second order with Newton, first with Howard's upwind advection."""
    errors, orders = _orders(scramble, "none", inner)
    assert all(band[0] <= order <= band[1] for order in orders), (
        f"none/{inner}, scramble {scramble}: {errors}, {orders}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("scramble", _SCRAMBLES)
@pytest.mark.parametrize(("inner", "band"), _INNER)
def test_joint_socp_converges_at_its_equations_order_on_reflecting_walls(inner, band, scramble):
    """``monotonicity_scheme="joint_socp"``, whose stencils need cvxpy (the ``numerical`` extra)."""
    pytest.importorskip("cvxpy", reason="joint_socp stencils require cvxpy (the `numerical` extra)")
    errors, orders = _orders(scramble, "joint_socp", inner)
    assert all(band[0] <= order <= band[1] for order in orders), (
        f"joint_socp/{inner}, scramble {scramble}: {errors}, {orders}"
    )

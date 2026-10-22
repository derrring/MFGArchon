"""#2512 (a) item 1: HJB-GFDM's default scheme on the 2-D box with every face no-flux converges at second order to an
exact reflected solution.

The solution is exp17's, as `test_positive_qp_fan_reflecting_1689` builds it and imported from there:
``u* = phi + C (T - t)``, ``phi = a (q(x) + q(y))``, ``q(s) = s^3 - 1.5 s^2``, so ``d_n phi = 0`` on every edge; ``m*``
is frozen and ``V`` absorbs the residual, so the equation holds exactly, and implicit Euler is exact in time, so the
error is the spatial one. The clouds are exp17's Sobol clouds, refined as N_int 900 -> 1600 -> 2500 with the
neighbourhood radius scaled with the spacing, ``delta = 4.5 h``, ``h = 1/sqrt(N_int)`` (0.15 at N_int 900). A fixed
delta is not a refinement: there the error saturates near 6.7e-4 from N_int 900.

Measured at 29cdb9d6 on scrambles 42, 3 and 7 (#2512 comment 6333097202): EOC 1.88 to 2.08 over these two refinements.
With the no-flux row read along the face's tangent instead of its normal, the errors are 0.11 to 1.3 at every N of
the ladder, on all three scrambles, and every row is red.

Only ``monotonicity_scheme="none"`` with Newton is pinned. Which (scheme, inner solver) pairs the family owes is the
user's to rule (same comment); with Howard in d >= 2 the advection is a first-order upwind stencil, so that pair's
order is one, by construction.
"""

from __future__ import annotations

import warnings

import pytest
from test_positive_qp_fan_reflecting_1689 import NT, T_END, _bc, _m_star, _problem, _sobol, _u_star

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

_LADDER = (900, 1600, 2500)


def _error(n_int: int, scramble: int) -> float:
    """The L2 error at t = 0 over the interior nodes of exp17's cloud with ``n_int`` interior nodes."""
    points, boundary = _sobol(n_int, seed=scramble)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = HJBGFDMSolver(
            _problem(_bc("box")),
            collocation_points=points,
            boundary_indices=boundary,
            delta=4.5 / np.sqrt(n_int),
            monotonicity_scheme="none",
            inner_solver="newton",
        )
        U = solver.solve_hjb_system(M_density=np.tile(_m_star(points), (NT + 1, 1)), U_terminal=_u_star(T_END, points))
    interior = np.setdiff1d(np.arange(len(points)), boundary)
    return float(np.sqrt(np.mean((np.asarray(U)[0] - _u_star(0.0, points))[interior] ** 2)))


@pytest.mark.parametrize("scramble", [42, 3, 7])
def test_the_default_scheme_converges_at_second_order_on_reflecting_walls(scramble):
    """EOC in h over each refinement of the ladder lies in [1.7, 2.3], the band of 1991's 1-D rate."""
    errors = [_error(n_int, scramble) for n_int in _LADDER]
    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(np.sqrt(_LADDER[i + 1] / _LADDER[i])) for i in range(len(errors) - 1)
    ]
    assert all(1.7 <= order <= 2.3 for order in orders), f"scramble {scramble}: errors {errors}, EOC {orders}"

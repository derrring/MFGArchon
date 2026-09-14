"""The 1-D FDM HJB against a manufactured solution where the upwind rule's minimum case decides (#2308).

``gradient_upwind`` selected the one-sided difference on ``sign((backward + forward) / 2)``. At a
discrete local minimum, ``backward < 0 < forward``, that returns the difference of smaller
magnitude, where the Godunov momentum for ``|p|^2/2`` is 0. The numerical Hamiltonian is then
non-monotone on exactly that quadrant, and the implicit step still has a unique root, so Newton
converges to it and nothing warns.

The oracle is external: ``u(t, x) = cos(kx) exp(-D k^2 (T - t))`` with ``k = 2 pi`` and ``D = sigma^2/2``
solves the backward heat equation, so the source ``S = u_x^2 / 2`` makes it exact for
``-u_t - D u_xx + |u_x|^2/2 = S``. The cells are the ones where ``r = (kh / sigma)^2 >= 9.87``, where
the old rule's error exceeded Rouy-Tourin's by 5.5x or more in #2308's table; the implicit row stops
being monotone already at r > 1 (derived there). Measured at Nt = 100 on this branch, with the rule
swapped in process for the second row:

    (Nx, sigma)        (11, 0.2)   (21, 0.1)   (41, 0.05)
    Rouy-Tourin        0.2422      0.1257      0.0599
    sign of central    6.4504      2.3446      0.5682

The bounds are twice the first row, so the second row sits 12.9x, 9.4x and 4.7x above them; a
regression smaller than 2x passes. For r <= 3.86 the ratio of the two rules' errors is 0.44 to 1.66 in
#2308's table, which is why the MMS convergence test in ``tests/integration/test_mms_validation.py``,
at r <= 0.154, cannot see this.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid, periodic_bc

_T = 0.3
_K = 2.0 * np.pi
_NT = 100


def _error(nx: int, sigma: float) -> float:
    diffusion = sigma**2 / 2

    def exact(t, x):
        return np.cos(_K * x) * np.exp(-diffusion * _K**2 * (_T - t))

    def source(t, x):
        return 0.5 * (_K * np.sin(_K * np.ravel(x)) * np.exp(-diffusion * _K**2 * (_T - t))) ** 2

    components = MFGComponents(
        m_initial=lambda x: np.ones_like(x),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=periodic_bc(dimension=1))
    problem = MFGProblem(geometry=grid, T=_T, Nt=_NT, sigma=sigma, components=components)
    x = grid.coordinates[0]
    U = HJBFDMSolver(problem).solve_hjb_system(
        M_density=np.ones((_NT + 1, nx)),
        U_terminal=exact(_T, x),
        U_coupling_prev=np.zeros((_NT + 1, nx)),
        source_term=source,
    )
    return float(np.sqrt(np.mean((np.asarray(U)[0] - exact(0.0, x)) ** 2)))


@pytest.mark.parametrize(("nx", "sigma", "bound"), [(11, 0.2, 0.5), (21, 0.1, 0.25), (41, 0.05, 0.12)])
def test_a_coarse_grid_solve_is_not_wrecked_at_the_minimum(nx: int, sigma: float, bound: float):
    r = (_K / (nx - 1) / sigma) ** 2
    assert r > 9.8, f"r = {r:.2f}: this cell is outside the regime where the minimum case decides"
    error = _error(nx, sigma)
    assert error < bound, (
        f"Nx={nx}, sigma={sigma}: RMS error at t=0 is {error:.4f} against a bound of {bound}. The "
        f"sign-of-central upwind rule gives 6.45 / 2.34 / 0.568 on these cells (#2308)."
    )

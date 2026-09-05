"""Which wall the SL/CN family's call sites actually take, measured. Issue #2243.

#2237 gave the six implementations of the implicit-diffusion Neumann wall one owner and made each
call site name its treatment; five named ``half_wall``. #2243 switched them to ``mirror`` -- the
stencil #2145 had already established as the one that conserves the mass an endpoint-inclusive grid
carries, and which is second order at the wall instead of first.

`tests/unit/test_utils/test_neumann_cn_wall_2237.py` pins what each TREATMENT does. It cannot pin
which one a call site takes, and that is the whole of this issue. So this file measures the shipped
routines and never reads ``treatment=`` from the source -- a test asserting on that keyword would
pass on a call site that named ``mirror`` and reached a different operator, which is precisely the
failure mode #2237 found three times (three files named a treatment their code did not implement).

Both oracles are independent of the scheme: an exact heat solution, and exact quadrature
invariance. Each assertion carries the value the OTHER wall produces, so the numbers say what the
check would have caught rather than only that it passed.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.adjoint.operators import build_diffusion_matrix, build_diffusion_matrix_2d
from mfgarchon.alg.numerical.fp_solvers import FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import adi_diffusion_step, solve_crank_nicolson_diffusion_1d
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.numerical.quadrature import quadrature_weights_1d

SIGMA = 0.4
T_END = 0.05
NSTEP = 200
D = SIGMA**2 / 2.0


def _heat_eoc(stepper, levels=(21, 41, 81, 161), dims: int = 1) -> tuple[list[float], list[float]]:
    """L-inf error and EOC against ``exp(-D k^2 pi^2 t) prod cos(pi x_k)``, computed analytically.

    That mode has zero normal derivative on every wall, so it is admissible under no-flux and the
    scheme is measured against a solution it should reproduce exactly in the limit -- no reference
    run, no other solver, nothing that moves when the SL family is reorganised (#2238).
    """
    errors = []
    for n in levels:
        x = np.linspace(0.0, 1.0, n)
        mesh = np.meshgrid(*([x] * dims), indexing="ij")
        u = np.ones_like(mesh[0])
        for g in mesh:
            u = u * np.cos(np.pi * g)
        exact = np.exp(-D * dims * np.pi**2 * T_END) * u.copy()
        for _ in range(NSTEP):
            u = stepper(u, T_END / NSTEP, x)
        errors.append(float(np.max(np.abs(u - exact))))
    eoc = [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]
    return errors, eoc


def test_the_1d_crank_nicolson_call_site_is_second_order_at_the_wall():
    """The acceptance criterion #2243 states: EOC 0.73 / 0.87 / 0.94 must become 2.00 / 2.00 / 2.00.

    Nothing else in this routine changed, so the wall is the only thing this can be reading. The
    half wall's numbers are in the message: at nx=161 it errs by 2.05e-03 against 1.22e-06 here,
    a factor of 1.7e3.
    """
    errors, eoc = _heat_eoc(lambda u, dt, x: solve_crank_nicolson_diffusion_1d(u.copy(), dt, SIGMA, x, "neumann"))
    assert all(o == pytest.approx(2.0, abs=0.15) for o in eoc), (
        f"solve_crank_nicolson_diffusion_1d is not second order at the wall: EOC {eoc} from "
        f"{errors}. The pre-#2243 half wall gave 0.73 / 0.87 / 0.94 from 2.05e-03 at nx=161."
    )


@pytest.mark.parametrize("dims", [1, 2])
def test_the_adi_call_site_is_second_order_at_the_wall(dims):
    """The ADI sweep is a SEPARATE call site (`solve_1d_diffusion_along_axis`), not the 1-D routine.

    d = 2 is here because the sweep is per-axis and a wall on one axis coexists with an interior on
    the other; d = 1 is here because `_adi_diffusion_step` dispatches to the 1-D routine at d = 1,
    so a wall that changed with the dimension would be invisible in either alone. This is a
    code-path argument, not "prefer 2D".
    """
    levels = (21, 41, 81, 161) if dims == 1 else (11, 21, 41, 81)

    def step(u, dt, x):
        dx = x[1] - x[0]
        return adi_diffusion_step(u.copy(), dt, SIGMA, np.array([dx] * dims), u.shape, "neumann")

    errors, eoc = _heat_eoc(step, levels=levels, dims=dims)
    assert all(o == pytest.approx(2.0, abs=0.15) for o in eoc), (
        f"adi_diffusion_step is not second order at the wall in {dims}D: EOC {eoc} from {errors}."
    )


def test_the_fp_solver_conserves_the_grid_measure_and_not_the_rectangle_rule():
    """#2145's decision, asserted through a real `FPSLSolver` solve rather than on the operator.

    Both halves are load-bearing. Asserting only that the grid measure is held would pass on a
    solver that held BOTH -- and the rectangle drift is what says the wall changed, since
    `half_wall` has the two the other way round: 3.3e-14 rectangle against 1.1e-02 trapezoid on
    the nx=81 version of this fixture.
    """
    nx, nt, t_end, sigma = 41, 200, 0.6, 0.5
    x = np.linspace(0.0, 1.0, nx)
    m0 = np.exp(-60.0 * (x - 0.5) ** 2)

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    components = MFGComponents(
        m_initial=lambda xx: np.exp(-60.0 * (np.asarray(xx) - 0.5) ** 2),
        u_terminal=lambda xx: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: 0.0 * np.asarray(m),
            coupling_dm=lambda m: 0.0 * np.asarray(m),
        ),
    )
    problem = MFGProblem(geometry=grid, T=t_end, Nt=nt, sigma=sigma, components=components)
    M = np.asarray(FPSLSolver(problem).solve_fp_system(M_initial=m0.copy(), potential_field=np.zeros((nt + 1, nx))))

    # CONTROL. The two measures differ by `dx*(m[0] + m[-1])/2`, so a fixture whose wall values do
    # not move cannot separate them however long it runs -- #2237's own MMS is such a fixture
    # (its exact solution holds `m[0] + m[-1] == 2` for all t) and reports both conserved.
    wall_sum_0 = float(m0[0] + m0[-1])
    wall_sum_t = float(M[-1][0] + M[-1][-1])
    assert wall_sum_t > 1e4 * max(wall_sum_0, 1e-12), (
        f"the wall values did not move ({wall_sum_0:.3e} -> {wall_sum_t:.3e}); this fixture cannot "
        f"tell the two measures apart and the assertions below would pass on either wall"
    )

    w = quadrature_weights_1d(x)
    dx = x[1] - x[0]
    grid_measure = abs(w @ M[-1] - w @ m0) / (w @ m0)
    rectangle = abs(M[-1].sum() * dx - m0.sum() * dx) / (m0.sum() * dx)
    assert grid_measure < 1e-12, f"the grid measure drifted by {grid_measure:.3e}; `mirror` holds it"
    assert rectangle > 1e-4, (
        f"the rectangle rule was also conserved ({rectangle:.3e}). Neither wall conserves both, so "
        f"this means the wall is not `mirror` -- `half_wall` has exactly these two swapped."
    )


def test_build_diffusion_matrix_answers_the_same_at_one_and_two_dimensions():
    """The seventh site. `build_diffusion_matrix` routes 1-D to `build_diffusion_matrix_1d` and
    nD to a Kronecker path over `_build_1d_laplacian`, which held its own copy of the wall -- and
    #2237's census could not see it, because it looks for `dt/dx^2` beside a tridiagonal assembly
    and this one applies `alpha` at the caller.

    Comparing the two nD assemblies is what caught it: they agreed to 0.000e+00 while both were on
    the half wall, and switching only the named call sites made them disagree by 6.4e-03 -- a 2-D
    wall row with a doubled diagonal against a single-weight neighbour, which is no stencil at all.
    """
    args = ((5, 5), 0.25, 0.4, 0.01, 0.5, "neumann")
    kron = build_diffusion_matrix(*args).toarray()
    direct = build_diffusion_matrix_2d(*args).toarray()
    assert np.abs(kron - direct).max() == 0.0, (
        f"the two nD diffusion assemblies disagree by {np.abs(kron - direct).max():.3e}; one of "
        f"them is on a different wall, and `build_diffusion_matrix` then answers by dimension"
    )


def test_the_diffusion_matrix_is_self_adjoint_in_the_grid_measure():
    """Self-adjointness MOVED inner product with the wall; it was not lost.

    `div(D grad .)` under no-flux is self-adjoint in the continuum, and the discrete statement has
    to be taken in the measure the grid carries. Bare symmetry is that statement under UNIFORM
    weights, which this grid does not have -- the same substitution #2145 found behind `1^T L = 0`.
    """
    n = 7
    a = build_diffusion_matrix(n, 1.0 / (n - 1), SIGMA, 0.01, 0.5, "neumann").toarray()
    weighted = np.diag(quadrature_weights_1d(np.linspace(0.0, 1.0, n))) @ a
    assert np.abs(weighted - weighted.T).max() < 1e-14, (
        f"not self-adjoint in the grid measure: {np.abs(weighted - weighted.T).max():.3e}"
    )
    # The other half: bare symmetry must now FAIL, or the wall did not move. `half_wall` has these
    # two the other way round -- exactly symmetric, and 7.2e-03 asymmetric in the grid measure.
    assert np.abs(a - a.T).max() > 1e-6, (
        "the matrix is symmetric in the UNIFORM inner product, which is the half wall's signature"
    )

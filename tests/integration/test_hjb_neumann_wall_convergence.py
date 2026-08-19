"""The HJB solvers converge to the Neumann wall, pinned by a symmetry oracle.

WHY A SYMMETRY ORACLE RATHER THAN AN MMS
----------------------------------------
`d_n u = 0` at a reflecting wall is a *property* of the exact solution, not a manufactured field, so
it needs no source term and no exact-solution transcription. Take `u_T(x) = cos(2*pi*x)` on [0, 1]:
it is even about both walls. Under zero coupling the HJB flow
``-d_t u - (sigma^2/2) u_xx + (1/2)|u_x|^2 = 0`` preserves evenness -- ``u_xx`` does, and ``|u_x|^2``
is the square of an odd function -- so ``d_n u == 0`` at both walls for every t, exactly. Any
non-zero one-sided difference at the wall is discretization error, and it must vanish under
refinement.

The control is a CONSTANT terminal condition: the solution is then constant in x, so the same
one-sided difference is zero to machine precision. Measured below at 4.4e-15 or exact zero. Without
it a harness bug that reported zero for everything would read as a perfect pass.

WHAT THIS PINS, AND WHAT IT IS NOT
----------------------------------
It is a REGRESSION PIN, not a bug hunt. A 2026-08-17 measurement recorded `HJBSemiLagrangianSolver`
and `HJBWENOSolver` violating this at O(1) -- ~1.0 still at Nx=161. That is no longer reproducible:
measured 2026-08-19 at sigma=0.5, T=0.5, Nt=20, all three converge at roughly first order. The
configurations are not identical so no causation is claimed here, but the earlier record's own open
question was whether `HJBFDMSolver`'s then-exact zero came from `_apply_neumann_enforcement`
overwriting the wall row, which #1902 deleted -- and FDM now shows a converging 6.2e-02 rather than
an exact zero, which is consistent with the scheme genuinely converging instead of the value being
painted on.

So this test exists to keep that true, not to expose anything.

WHAT IT CANNOT SEE
------------------
- **1D only, and deliberately.** `d_n u = 0` is a scalar condition per wall with no tangential
  component to get wrong, so it is fully expressed in 1D (AGENTS.md, "the dimension must be able to
  express the property"). A 2D version would cost runtime and separate no additional mutant.
- **Zero coupling.** With `m` uniform and no coupling term the evenness argument is exact; with a
  coupling the manufactured symmetry would have to be argued for `f(m)` too.
- It says nothing about the FP side's wall, which is `J.n = 0` and a different condition
  (#1728/#2006).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

L, T, NT, SIGMA = 1.0, 0.5, 20, 0.5
LEVELS = (21, 41, 81)


def _wall_gradient(solver_cls, nx: int, *, constant_terminal: bool = False) -> float:
    """max |d_n u| over the two walls, as a one-sided difference of the solved u(0, .)."""
    x = np.linspace(0.0, L, nx)
    grid = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda _p: 1.0 / L,
        u_terminal=lambda _p: 0.0,
    )
    problem = MFGProblem(geometry=grid, components=components, T=T, Nt=NT, sigma=SIGMA)
    u_terminal = np.ones(nx) if constant_terminal else np.cos(2.0 * np.pi * x)
    u = np.asarray(
        solver_cls(problem).solve_hjb_system(
            M_density=np.tile(np.ones(nx) / L, (NT + 1, 1)),
            U_terminal=u_terminal,
            U_coupling_prev=np.tile(u_terminal, (NT + 1, 1)),
        )
    )[0]
    dx = L / (nx - 1)
    return float(max(abs(u[1] - u[0]) / dx, abs(u[-1] - u[-2]) / dx))


@pytest.mark.integration
@pytest.mark.parametrize("solver_cls", [HJBFDMSolver, HJBWENOSolver, HJBSemiLagrangianSolver], ids=lambda c: c.__name__)
def test_the_control_is_exact(solver_cls):
    """A constant terminal condition gives a constant solution: the metric must read ~0.

    This is the positive control on the harness. It runs first because a harness that returned zero
    for everything would make every assertion below pass.
    """
    assert _wall_gradient(solver_cls, 41, constant_terminal=True) < 1e-12


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("solver_cls", [HJBFDMSolver, HJBWENOSolver], ids=lambda c: c.__name__)
def test_the_wall_gradient_converges_at_first_order(solver_cls):
    """FDM and WENO are clean per level: measured orders 1.03/1.02 and 1.00/1.00.

    Baseline levels at Nx = 21/41/81 — FDM 1.29e-01, 6.23e-02, 3.05e-02 (orders 1.05, 1.03);
    WENO 4.23e-02, 2.11e-02, 1.06e-02 (1.00, 0.99). Nx=161 is measured and consistent (FDM 1.50e-02,
    WENO 5.28e-03) but costs 162 s for WENO alone, which would make this the slowest test in its
    shard for one more order value.

    SENSITIVITY, measured rather than assumed. Injecting a wall-gradient perturbation that decays as
    ``nx**-p`` and running this same assertion:

        p = 0.0  (wall never flattens)   caught
        p = 0.5  (half order)            caught
        p = 0.8  (the band's own edge)   passes
        p = 1.0  (first order)           passes

    So it catches anything decaying slower than the band's lower bound, and nothing above it. A
    first draft of that check used ``0.3*cos(2*pi*x)/sqrt(nx)``, which passed -- because ``cos'`` is
    zero at the wall, so the perturbation never touched the quantity being measured. The mutation
    has to move the metric, not merely the field.
    """
    errors = [_wall_gradient(solver_cls, nx) for nx in LEVELS]
    orders = [float(np.log(errors[i] / errors[i + 1]) / np.log(2.0)) for i in range(len(errors) - 1)]
    assert all(0.8 <= o <= 1.3 for o in orders), f"wall gradient is not first order: {orders} ({errors})"
    assert errors[-1] < 5e-02, f"level at the finest grid is {errors[-1]:.3e}"


@pytest.mark.integration
@pytest.mark.slow
def test_the_semi_lagrangian_wall_gradient_decreases():
    """SL is asserted on the NET order only, because its per-level ratios are genuinely irregular.

    Measured 8.650e-02, 4.114e-02, 2.683e-02 — per-level orders +1.07, +0.62, net 0.85 over this
    ladder (extending to Nx=161 gives 6.935e-03, a +1.95 step and net 1.21, so the band below is
    wide on purpose). The irregularity is not noise in the metric: the solver reports adaptive
    substepping using 33 / 112 / 89 substeps at different resolutions, so the effective time
    discretization changes between levels. Asserting a per-level band here would be a threshold
    fitted to one run; asserting monotone decrease plus a net order is what the measurement supports.
    """
    errors = [_wall_gradient(HJBSemiLagrangianSolver, nx) for nx in LEVELS]
    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1)), (
        f"wall gradient did not decrease monotonically under refinement: {errors}"
    )
    net = float(np.log(errors[0] / errors[-1]) / np.log(LEVELS[-1] / LEVELS[0]))
    assert 0.7 <= net <= 1.6, f"net order over the ladder is {net:.2f} (errors {errors})"

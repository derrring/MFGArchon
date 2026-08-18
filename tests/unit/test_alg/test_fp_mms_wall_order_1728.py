"""MMS order study for the FP-FDM advection schemes at a no-flux wall carrying drift.

Closes the gap named in #1728: the MMS test previously cited as covering `divergence_upwind`'s
spatial order runs at zero drift, so substituting any other advection scheme leaves its output
bit-identical and the stated order has never been under test.

The construction
----------------
An impermeable wall is J·n = 0 with J = αm − D∇m, D = σ²/2. The general zero-flux (Gibbs) pair is

    phi any smooth potential,   b* = −∇phi,   m*(x) = Z⁻¹ exp(−phi(x)/D),

for which ∇m* = −(∇phi/D) m* gives J = b* m* − D(−∇phi/D)m* ≡ 0 **everywhere**, not merely on the
wall. Hence the source term is identically zero: the exact solution is stationary and source-free,
and no `source_term` channel is needed -- which matters, because only `FPFDMSolver` and
`FPFVMSolver` accept one.

Two instances of that family are measured, and they are independent of each other:

- **linear** `phi = −A·x`, so b* = A is a constant vector and m* = Z⁻¹exp(A·x/D). Dimension-agnostic
  for any d and any A. Its normal drift b*·n is non-zero *at the wall node itself*.
- **Gibbs** `phi = A(cos Kx₁ + cos Kx₂)` with K = 4π, D = 1/8, A = D ln5/4, T = 1 on (0,1)²: the
  published source-free exact reflected MFG of the GFDM paper (app:source_free_benchmark,
  def:source_free_benchmark), whose m* has contrast max/min = 5. Here ∂_ν phi = 0 on ∂Ω, so b*·n
  vanishes **at the wall node** -- but not on the boundary cell's interior face, where
  |∂₁phi| = A K sin(K dx) = 0.195 at dx = 1/40. A published oracle reaching the same verdicts as
  the linear one is worth more than either alone.

Why the study runs over dimension and over sign
-----------------------------------------------
On the wall x_k = 0 the outward normal is −e_k, so A has normal component −A_k and tangential
components A_j (j ≠ k). In d = 1 there is no tangential direction at all: all three tangential
mutants tried against this file (drop the tangential advection at wall rows, flip its sign, read the
potential along the wrong axis) are **exact no-ops in 1D**, max error difference 0.000e+00, and two
of the three separate in 2D.

The sign is parametrized because A > 0 alone never exercises the upwind *selection*: every face
velocity has one sign, so only one branch of each `if alpha >= 0 / else` is taken, and a mutant that
always upwinds from the left reproduces the library's output to bitwise equality. The library's own
numbers are unchanged by the flip -- the problem is mirror-symmetric under x -> L − x -- while such
a mutant is not, so the coverage is free.

WHAT THIS STUDY CANNOT SEE
--------------------------
- **It cannot attribute the order to the wall alone.** The measured order is a *min* over the wall
  closure and the interior stencil: substituting a centered interior for `divergence_upwind` still
  gives EOC 0.89/0.94, and substituting a centered *wall* still gives 0.89/0.95 in 1D. That is why
  the first-order tests assert an error **level** as well as an order -- the level separates them
  (library 1.63e-2, centered-interior mutant 2.98e-2 at Nx = 21, d = 1).
- **The linear instance alone cannot distinguish the two interior advection forms** -- but the
  pair can, which is why both are here. With constant A, ∇·(αm) = α·∇m + m∇·α reduces to α·∇m, so
  `gradient_upwind`'s interior collapses to `divergence_upwind`'s flux difference, and repointing
  the former's *wall* at the conservative routine makes the two agree to 2.5e-14 there. The Gibbs
  instance is the complement: its potential is non-linear, so m∇·α ≠ 0, and the SAME wall
  repointing leaves `gradient_upwind` non-convergent (5.81e-1 -> 8.02e-1, EOC −0.007 -> 0.108)
  while it fully repairs the linear instance (6.69e-1 -> 2.23e-2, EOC 0.937). So the wall defect
  and the interior-form defect are separated here without any source term: a non-constant
  SOURCE-FREE potential suffices, and a sourced MMS is not needed for this question.
- **It cannot see a dt error floor**, because the exact solution is stationary. `NT` is held fixed
  while Nx refines for that reason -- not because the schemes are implicit, which buys unconditional
  stability and nothing about accuracy. Measured: errors move < 2% from NT = 10 to NT = 640, and the
  orders are unchanged under NT ~ Nx and NT ~ Nx². For a *non-stationary* manufactured solution the
  same fixed-NT refinement would be confounded, and the explicit FP solvers are confounded by it
  here (`fvm:upwind` reads EOC −1.34 at fixed NT and +0.81 once dt refines with dx).
"""

from __future__ import annotations

from itertools import pairwise

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

L = 1.0
NT = 40

# Linear instance. Distinct components, one negative: distinctness breaks a transposed reshape,
# the negative exercises the other branch of the upwind selection.
LIN_T, LIN_SIGMA = 0.5, 1.0
LIN_D = 0.5 * LIN_SIGMA**2
A_FULL = np.array([0.7, -0.4, 0.2])

# Gibbs instance -- the GFDM paper's published constants, reproduced exactly.
GIBBS_T, GIBBS_D = 1.0, 1.0 / 8.0
GIBBS_SIGMA = float(np.sqrt(2.0 * GIBBS_D))
GIBBS_K = 4.0 * np.pi
GIBBS_A = GIBBS_D * np.log(5.0) / 4.0

LEVELS = {1: (21, 41, 81), 2: (21, 41)}

# Error level at the coarsest level, above which the scheme is not the library's upwind one.
# Set from measurement, midway between the library and the centered-interior mutant:
#   d = 1  library 1.6305e-2  mutant 2.9791e-2
#   d = 2  library 2.2303e-2  mutant 3.6212e-2
UPWIND_LEVEL_BOUND = {1: 2.3e-2, 2: 2.9e-2}
# Same construction for the Gibbs instance: library 5.5032e-2, centered-interior mutant 1.6857e-1.
GIBBS_LEVEL_BOUND = 1.1e-1


def _build(nx: int, d: int, sigma: float, horizon: float) -> tuple[TensorProductGrid, MFGProblem]:
    grid = TensorProductGrid(bounds=[(0.0, L)] * d, Nx_points=[nx] * d, boundary_conditions=no_flux_bc(dimension=d))
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: 1.0 / L**d,
        u_terminal=lambda x: 0.0,
    )
    return grid, MFGProblem(geometry=grid, components=components, T=horizon, Nt=NT, sigma=sigma)


def _zero_flux_pair(grid: TensorProductGrid, phi: np.ndarray, diffusion: float):
    """Return (m*, U) for the zero-flux pair of a potential phi given on the flat point list.

    m* = Z^-1 exp(-phi/D) and U = phi, since the solver forms alpha = -c grad(U) with c = 1 for
    QuadraticControlCost(1.0) and the pair needs b* = -grad(phi). The potential channel is used
    rather than the velocity channel because only the interface-velocity schemes read the latter
    (#1632).
    """
    shape = tuple(grid.Nx_points)
    dx = L / (shape[0] - 1)
    m = np.exp(-phi / diffusion).reshape(shape)
    m /= m.sum() * dx ** len(shape)
    return m, phi.reshape(shape)


def _linear_phi(grid: TensorProductGrid, A: np.ndarray) -> np.ndarray:
    return -(grid.get_spatial_grid() @ A)


def _gibbs_phi(grid: TensorProductGrid) -> np.ndarray:
    points = grid.get_spatial_grid()
    return GIBBS_A * np.cos(GIBBS_K * points).sum(axis=1)


def _solve_error(scheme: str, nx: int, instance: str, sign: int = 1, d: int = 2) -> float:
    """Relative L-inf distance between m(T) and the stationary exact solution."""
    if instance == "linear":
        sigma, horizon, diffusion = LIN_SIGMA, LIN_T, LIN_D
    else:
        sigma, horizon, diffusion, d = GIBBS_SIGMA, GIBBS_T, GIBBS_D, 2
    grid, problem = _build(nx, d, sigma, horizon)
    phi = _linear_phi(grid, sign * A_FULL[:d]) if instance == "linear" else _gibbs_phi(grid)
    m0, u = _zero_flux_pair(grid, phi, diffusion)
    U = np.broadcast_to(u, (NT + 1, *u.shape)).copy()
    solver = FPFDMSolver(problem, advection_scheme=scheme)
    M = np.asarray(solver.solve_fp_system(M_initial=m0, potential_field=U))
    return float(np.abs(M[-1] - m0).max() / m0.max())


def _errors(scheme: str, instance: str, sign: int, d: int) -> list[float]:
    return [_solve_error(scheme, nx, instance, sign, d) for nx in LEVELS[d]]


def _orders(errors: list[float]) -> list[float]:
    return [float(np.log(a / b) / np.log(2.0)) for a, b in pairwise(errors)]


@pytest.mark.parametrize("d", [1, 2])
def test_manufactured_pair_carries_zero_flux(d: int) -> None:
    """Positive control on the oracle itself, before any solver is measured against it.

    The construction claims J ≡ 0. If it is wrong -- or if the flat (N, d) point list is reshaped in
    the wrong order -- every order below is measured against a wrong reference. The finite difference
    used here is second order, so the residual must fall by ~4 per refinement; a scrambled field does
    not converge at all (measured on an F-order reshape: order 0.001).
    """
    A = A_FULL[:d]
    residuals = []
    for nx in (21, 41):
        grid, _ = _build(nx, d, LIN_SIGMA, LIN_T)
        m, _ = _zero_flux_pair(grid, _linear_phi(grid, A), LIN_D)
        dx = L / (nx - 1)
        residuals.append(
            max(
                float(np.abs(A[k] * m - LIN_D * np.gradient(m, dx, axis=k, edge_order=2)).max() / m.max())
                for k in range(d)
            )
        )
    order = np.log(residuals[0] / residuals[1]) / np.log(2.0)
    assert order > 1.8, f"flux control is not converging at 2nd order: residuals={residuals}"


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("d", [1, 2])
def test_divergence_upwind_is_first_order_at_a_drifting_wall(d: int, sign: int) -> None:
    """#1728: the scheme's advertised first order, measured with the advection term active.

    Two assertions, and the second is not decoration. The ORDER is a min over the wall closure and
    the interior stencil, so it alone cannot say which produced the O(h) -- a centered interior still
    reads 0.89/0.94. The LEVEL separates them. Together they are mutation-red against a centered
    interior, a centered wall, and a deleted upwind selection; the order bound alone catches none of
    the three, and the upper bound is what rejects `divergence_centered` (~1.97).
    """
    errors = _errors("divergence_upwind", "linear", sign, d)
    orders = _orders(errors)
    assert all(0.8 <= o <= 1.4 for o in orders), f"expected first order, observed {orders}"
    assert errors[0] < UPWIND_LEVEL_BOUND[d], (
        f"error level {errors[0]:.4e} exceeds {UPWIND_LEVEL_BOUND[d]:.1e}: the order is still ~1 but "
        f"the constant moved, which is what a centered interior or a centered wall looks like here."
    )


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("d", [1, 2])
def test_divergence_centered_is_second_order_at_a_drifting_wall(d: int, sign: int) -> None:
    orders = _orders(_errors("divergence_centered", "linear", sign, d))
    assert all(o > 1.8 for o in orders), f"expected second order, observed {orders}"


@pytest.mark.parametrize("d", [1, 2])
def test_gradient_upwind_does_not_converge_at_a_drifting_wall(d: int) -> None:
    """RECORDED DEFECT, not a contract. Fixing it trips this test, and that is intended.

    `gradient_upwind` imposes ∂_n m = 0 (the comment "No-flux: dm/dx = 0" in
    add_boundary_no_flux_entries_gradient_upwind) rather than J·n = 0, and drops the wall-normal
    advective flux. Each wall row's steady state is therefore m_0 = m_1, and A·∇m = D∆m under
    ∂_n m = 0 admits only constants -- so the scheme converges to a UNIFORM limit while the exact
    solution spans exp(A·L/D) = 4.06. At T = 0.5 the computed span is 1.10, a partially relaxed
    field; extended to T = 4.0 it reaches 1.0000, the limit itself.

    #1075 names the wrong condition in its title and is closed; the constructor warns about it; and
    `test_solver_bc_support_census_1975.py::test_the_gradient_schemes_impose_a_zero_gradient_wall_and_leak`
    already pins the mass-drift half on the same drive. What is new here is the accuracy form -- the
    scheme does not converge at all -- and the magnitude: the warning's "leaks O(1e-2)" is the
    zero-drift figure, an order of magnitude below the 1.4e-1 measured at A = 0.7.

    When the wall is rewritten to J·n = 0, this test fails and its message says what to do with it.
    """
    orders = _orders(_errors("gradient_upwind", "linear", 1, d))
    assert all(abs(o) < 0.2 for o in orders), (
        f"gradient_upwind now converges (orders={orders}) -- the ∂_n m = 0 wall appears to be fixed. "
        f"Delete this defect pin and give the scheme a real order assertion."
    )


def test_divergence_upwind_is_first_order_on_the_published_gibbs_instance() -> None:
    """The GFDM paper's source-free exact reflected MFG, as an independent published oracle.

    Its drift is tangential AT the wall node (∂_ν phi = 0), which the linear instance's is not, and
    its potential is non-linear, which exercises a part of the drift reconstruction the linear one
    cannot. The order verdict must nevertheless agree, since it does not depend on the choice of phi.
    """
    errors = [_solve_error("divergence_upwind", nx, "gibbs") for nx in LEVELS[2]]
    orders = _orders(errors)
    assert all(0.8 <= o <= 1.4 for o in orders), f"expected first order, observed {orders}"
    assert errors[0] < GIBBS_LEVEL_BOUND, (
        f"error level {errors[0]:.4e} exceeds {GIBBS_LEVEL_BOUND:.1e} on the published instance"
    )


def test_gradient_upwind_is_wrong_on_a_non_linear_potential_for_a_second_reason() -> None:
    """RECORDED DEFECT, not a contract -- and a DIFFERENT defect from the wall pinned above.

    The linear instance has constant α, so ∇·(αm) = α·∇m there and the two interior forms coincide;
    fixing the wall repairs `gradient_upwind` completely on it (6.69e-1 -> 2.23e-2, EOC 0.937,
    matching `divergence_upwind`). On this instance α is not constant, m∇·α ≠ 0, and the gradient
    form is therefore discretizing a different operator. The same wall repointing does NOT repair it
    (5.81e-1 -> 8.02e-1, EOC −0.007 -> 0.108) -- which is the evidence that `gradient_upwind` is not
    the FP operator with a bad wall, but a different operator.

    Retirement condition, and it is NOT the wall fix: this pin retires when the interior form is
    corrected to carry m∇·α, or when the scheme is removed. A wall-only change leaves it passing,
    correctly.
    """
    errors = [_solve_error("gradient_upwind", nx, "gibbs") for nx in LEVELS[2]]
    assert all(e > 0.5 for e in errors), (
        f"gradient_upwind is no longer grossly wrong on a non-linear potential (errors={errors}). "
        f"If the interior form now carries m*div(alpha), delete this pin and assert a real order."
    )

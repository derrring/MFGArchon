"""MMS order study for the FP-FDM advection schemes at a non-tangential no-flux wall.

Closes the gap named in #1728: the MMS test previously cited as covering `divergence_upwind`'s
spatial order runs at zero drift, so substituting any other advection scheme leaves its output
bit-identical and the stated order has never been under test.

Manufactured solution
---------------------
An impermeable wall is J·n = 0 with J = αm − D∇m, D = σ²/2. For a **constant** drift vector A the
density that makes the flux vanish everywhere — not merely on the wall — is

    m̄(x) = C exp(A·x / D),

since ∇m̄ = (A/D) m̄ gives J = A m̄ − D (A/D) m̄ ≡ 0. Hence the source term is identically zero:
the manufactured solution is stationary, and the measurement is whether a scheme preserves it.
No `source_term` channel is needed, so this oracle is available to every FP solver.

Why constant A: in the interior ∇·(αm) = α·∇m + m ∇·α reduces to α·∇m when α is constant, so the
divergence and gradient forms coincide away from the wall. The probe therefore isolates the
**wall treatment**, which is what distinguishes the two boundary architectures (#2005).

Why the study runs over dimension: on the wall x_k = 0 the outward normal is −e_k, so A has normal
component −A_k and tangential components A_j (j ≠ k). In d = 1 there is no tangential direction at
all, so a scheme mishandling the tangential part cannot be discriminated there.

Drive: α = −c∇U with U(x) = −A·x reproduces A exactly for QuadraticControlCost(1.0). The potential
channel is used rather than the velocity channel because only the interface-velocity schemes read
the latter (#1632).

Cell Péclet is |A| dx / D = 0.07 at the coarsest level, so `divergence_centered` is nowhere near
its stability limit: the tests below discriminate on order, not on stability (#1728's fourth
acceptance condition).
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
T = 0.5
SIGMA = 1.0
D = 0.5 * SIGMA**2
NT = 40

# Distinct components, so a reshape that transposed the field would break the flux control below.
A_FULL = np.array([0.7, 0.4, 0.2])


def _build(nx: int, d: int) -> tuple[TensorProductGrid, MFGProblem]:
    grid = TensorProductGrid(bounds=[(0.0, L)] * d, Nx_points=[nx] * d, boundary_conditions=no_flux_bc(dimension=d))
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: 1.0 / L**d,
        u_terminal=lambda x: 0.0,
    )
    return grid, MFGProblem(geometry=grid, components=components, T=T, Nt=NT, sigma=SIGMA)


def _manufactured(grid: TensorProductGrid, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (m̄, U) on the grid's own layout. Both exact, both time-independent."""
    points = grid.get_spatial_grid()  # (N, d)
    shape = tuple(grid.Nx_points)
    w = points @ A  # A·x
    m = np.exp(w / D).reshape(shape)
    dx = L / (shape[0] - 1)
    m /= m.sum() * dx ** len(shape)
    return m, (-w).reshape(shape)


def _solve_error(scheme: str, nx: int, d: int) -> float:
    """Relative L-inf distance between m(T) and the stationary exact solution."""
    grid, problem = _build(nx, d)
    m0, u = _manufactured(grid, A_FULL[:d])
    U = np.broadcast_to(u, (NT + 1, *u.shape)).copy()
    solver = FPFDMSolver(problem, advection_scheme=scheme)
    M = np.asarray(solver.solve_fp_system(M_initial=m0, potential_field=U))
    return float(np.abs(M[-1] - m0).max() / m0.max())


def _observed_orders(scheme: str, d: int, levels: tuple[int, ...]) -> list[float]:
    errors = [_solve_error(scheme, nx, d) for nx in levels]
    return [float(np.log(a / b) / np.log(2.0)) for a, b in pairwise(errors)]


LEVELS = {1: (21, 41, 81), 2: (21, 41)}


@pytest.mark.parametrize("d", [1, 2])
def test_manufactured_pair_carries_zero_flux(d: int) -> None:
    """Positive control on the oracle itself, before any solver is measured against it.

    The construction claims J ≡ 0. If it is wrong -- or if the flat (N, d) point list is reshaped
    in the wrong order -- every order below is measured against a wrong reference. The finite
    difference used here is second order, so the residual must fall by ~4 per refinement; a
    scrambled field would not converge at all.
    """
    A = A_FULL[:d]
    residuals = []
    for nx in (21, 41):
        grid, _ = _build(nx, d)
        m, _ = _manufactured(grid, A)
        dx = L / (nx - 1)
        residuals.append(
            max(
                float(np.abs(A[k] * m - D * np.gradient(m, dx, axis=k, edge_order=2)).max() / m.max()) for k in range(d)
            )
        )
    order = np.log(residuals[0] / residuals[1]) / np.log(2.0)
    assert order > 1.8, f"flux control is not converging at 2nd order: residuals={residuals}"


@pytest.mark.parametrize("d", [1, 2])
def test_divergence_upwind_is_first_order_at_a_drifting_wall(d: int) -> None:
    """#1728: the scheme's advertised first order, measured with the advection term active.

    The upper bound is what makes this mutation-red. `divergence_centered` scores ~1.97 on the same
    manufactured solution, so substituting it fails here -- which is precisely the discrimination
    the zero-drift MMS test cited in #1728 does not have.
    """
    orders = _observed_orders("divergence_upwind", d, LEVELS[d])
    assert all(0.8 <= o <= 1.4 for o in orders), f"expected first order, observed {orders}"


@pytest.mark.parametrize("d", [1, 2])
def test_divergence_centered_is_second_order_at_a_drifting_wall(d: int) -> None:
    orders = _observed_orders("divergence_centered", d, LEVELS[d])
    assert all(o > 1.8 for o in orders), f"expected second order, observed {orders}"


@pytest.mark.parametrize("d", [1, 2])
def test_gradient_upwind_does_not_converge_at_a_drifting_wall(d: int) -> None:
    """RECORDED DEFECT, not a contract. Fixing it trips this test, and that is intended.

    `gradient_upwind` imposes ∂_n m = 0 at the wall (see the comment "No-flux: dm/dx = 0" in
    add_boundary_no_flux_entries_gradient_upwind) rather than J·n = 0. With A·n ≠ 0 those differ,
    and the gradient-form equation under ∂_n m = 0 has a UNIFORM steady state, so the scheme
    converges to its own wrong limit: the exact profile spans a factor exp(A·L/D) = 4.06 across
    the domain and the computed one spans 1.10.

    #1075 records the mass-conservation half of this and is closed; the constructor emits a
    UserWarning citing it. Neither states that the solution is also wrong in shape -- the warning
    is about mass only, and its "leaks O(1e-2)" is the zero-drift figure, an order of magnitude
    below the 1.4e-1 measured here at A = 0.7.

    When the wall is rewritten to J·n = 0, this test fails and its message says what to do with it.
    """
    orders = _observed_orders("gradient_upwind", d, LEVELS[d])
    assert all(abs(o) < 0.2 for o in orders), (
        f"gradient_upwind now converges (orders={orders}) -- the ∂_n m = 0 wall appears to be "
        f"fixed. Delete this defect pin and give the scheme a real order assertion."
    )

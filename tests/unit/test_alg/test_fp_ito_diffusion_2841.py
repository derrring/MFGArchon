"""FP-FDM's diffusion of a volatility that varies in space is Ito's, d_i d_j(A_ij m) (#2841).

For dX = alpha dt + Sigma dW the FP equation is d_t m + div(alpha m) = d_i d_j(A_ij m), A = Sigma Sigma^T / 2, the
adjoint of the HJB's tr(A D^2 u) (CONVENTIONS § 5; user ruling 2026-10-30): Delta(D m) for a field. The implicit step
had assembled D Delta m, which gains mass under NO_FLUX (7.96e-2 by T = 2 on the issue's row); the explicit step, the
adjoint mode and the per-point tensor step had div(A grad m), which conserves mass but tends to a constant density.
Each oracle below is one of the two: mass, or the Ito law's fixed point m = c / D.
"""

from __future__ import annotations

import pytest

import numpy as np
import scipy.sparse as sparse

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid, no_flux_bc
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions


def _grid(dim: int, n: int) -> TensorProductGrid:
    return TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )


def _problem(grid: TensorProductGrid, nt: int, t_end: float, m_initial=lambda p: 1.0, **volatility) -> MFGProblem:
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)),
        **(volatility or {"volatility": 0.4}),
    )
    conditions = Conditions(m_initial=m_initial, u_terminal=lambda p: 0.0, T=t_end)
    return MFGProblem(model=model, domain=grid, conditions=conditions, Nt=nt)


def _nodes(grid: TensorProductGrid) -> list[np.ndarray]:
    return np.meshgrid(*[np.asarray(a) for a in grid.coordinates], indexing="ij")


def _mass(grid: TensorProductGrid, m: np.ndarray) -> np.ndarray:
    w = np.asarray(grid.measure_weights())
    return np.array([float(np.sum(level * w)) for level in np.asarray(m)])


def _tensor(sigma_xx: np.ndarray, sigma_yy: np.ndarray, sigma_xy: float) -> np.ndarray:
    tensor = np.zeros((*sigma_xx.shape, 2, 2))
    tensor[..., 0, 0], tensor[..., 1, 1] = sigma_xx, sigma_yy
    tensor[..., 0, 1] = tensor[..., 1, 0] = sigma_xy
    return tensor


@pytest.mark.parametrize(
    "path", ["implicit_1d", "implicit_2d", "implicit_drifted", "explicit_callable_drift", "tensor"]
)
def test_a_varying_volatility_conserves_mass_under_no_flux(path: str):
    """The issue's 1-D row on the default pair, a 2-D row whose m0 varies along x, where sigma does (a cos(pi x)cos(pi y)
    start stays flat in x on average and cannot see the defect), a drifted row, the callable-drift step, and a per-point
    anisotropic tensor, whose mixed operators carry the wall closure: the mass stays where it started."""
    dim = 2 if path in ("implicit_2d", "tensor") else 1
    n = 9 if path == "tensor" else (11 if dim == 2 else 21)
    nt, t_end = (20, 0.1) if path == "tensor" else (40, 2.0)
    grid = _grid(dim, n)
    x = _nodes(grid)
    m0 = 1 + 0.5 * np.cos(np.pi * x[0]) + (0.3 * np.cos(np.pi * x[1]) if dim == 2 else 0.0)
    sigma = 0.3 + 0.2 * x[0]
    if path == "tensor":
        volatility = {"volatility": _tensor(sigma, 0.25 + 0.1 * x[1], 0.05), "volatility_kind": "tensor"}
    else:
        volatility = {"volatility": sigma, "volatility_kind": "field"}
    solver = FPFDMSolver(_problem(grid, nt, t_end), boundary_conditions=no_flux_bc(dimension=dim))
    if path == "explicit_callable_drift":
        m = solver.solve_fp_system(
            m0, drift_field=lambda t, xx, mm: 0.5 * np.ones_like(mm), show_progress=False, **volatility
        )
    else:
        u = np.broadcast_to(x[0] if path == "implicit_drifted" else 0.0 * x[0], (nt + 1, *m0.shape)).copy()
        m = solver.solve_fp_system(m0, potential_field=u, show_progress=False, **volatility)
    mass = _mass(grid, m)
    assert np.max(np.abs(mass / mass[0] - 1)) < 1e-12, mass / mass[0] - 1


def test_the_coupled_solve_conserves_mass_with_a_field_volatility():
    """The issue's coupled row: problem.solve() on the default FDM pair returned a density 6.96e-2 heavier, with no
    warning. Its m0 varies where sigma does; from a constant one the row-scaled step gained nothing."""
    grid = _grid(1, 21)
    x = _nodes(grid)[0]
    problem = _problem(
        grid,
        40,
        2.0,
        m_initial=lambda p: 1 + 0.5 * np.cos(np.pi * np.asarray(p, dtype=float)),
        volatility=0.3 + 0.2 * x,
        volatility_kind="field",
    )
    mass = _mass(grid, problem.solve(max_iterations=3).M)
    assert abs(mass[-1] / mass[0] - 1) < 1e-12, mass[-1] / mass[0] - 1


@pytest.mark.parametrize("path", ["implicit", "explicit_callable_drift", "adjoint_mode", "tensor"])
def test_one_over_d_is_a_fixed_point_of_every_fdm_path(path: str):
    """With no drift, Ito's Delta(D m) takes m = c / D to 0 exactly (the unit stencil takes a constant to 0), so one step
    returns it; D Delta m and div(D grad m) move it. The tensor is sigma(x) I, a field written as a per-point tensor."""
    dim, n, dt = (2, 9, 0.01) if path == "tensor" else (1, 21, 0.1)
    grid = _grid(dim, n)
    x = _nodes(grid)
    sigma = 0.3 + 0.2 * x[0]
    m0 = 1.0 / (sigma**2 / 2)
    solver = FPFDMSolver(_problem(grid, 1, dt), boundary_conditions=no_flux_bc(dimension=dim))
    if path == "adjoint_mode":
        m1 = solver.solve_fp_step_adjoint_mode(m0, sparse.csr_matrix((n, n)), volatility=sigma, volatility_kind="field")
    elif path == "explicit_callable_drift":
        m1 = solver.solve_fp_system(
            m0,
            drift_field=lambda t, xx, mm: np.zeros_like(mm),
            volatility=sigma,
            volatility_kind="field",
            show_progress=False,
        )[-1]
    else:
        volatility = (
            {"volatility": _tensor(sigma, sigma, 0.0), "volatility_kind": "tensor"}
            if path == "tensor"
            else {"volatility": sigma, "volatility_kind": "field"}
        )
        m1 = solver.solve_fp_system(m0, potential_field=np.zeros((2, *m0.shape)), show_progress=False, **volatility)[-1]
    assert np.max(np.abs(m1 - m0)) < 1e-12 * np.max(m0), np.max(np.abs(m1 - m0))


def test_a_flux_under_a_field_volatility_moves_the_mass_by_its_q():
    """FLUX (#2581) reaches the implicit step under a field, and each step changes the mass by -dt q (1-D, a point
    face): the step conserves mass under NO_FLUX, the identity's hypothesis, which the row-scaled step broke."""
    n, nt, dt, q = 21, 5, 0.1, 0.05
    grid = _grid(1, n)
    x = _nodes(grid)[0]
    bc = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(
                name=f,
                bc_type=BCType.FLUX if f == "x_max" else BCType.NO_FLUX,
                value=q if f == "x_max" else 0.0,
                boundary=f,
            )
            for f in ("x_min", "x_max")
        ],
    )
    m = FPFDMSolver(_problem(grid, nt, nt * dt), boundary_conditions=bc).solve_fp_system(
        np.ones(n),
        potential_field=np.zeros((nt + 1, n)),
        volatility=0.3 + 0.2 * x,
        volatility_kind="field",
        show_progress=False,
    )
    assert np.allclose(np.diff(_mass(grid, m)), -q * dt, rtol=0, atol=1e-12), np.diff(_mass(grid, m))


def test_a_uniform_per_point_tensor_with_an_off_diagonal_is_the_constant_tensor():
    """A per-point tensor whose entries all agree goes through the owner, component by component; it equals the
    constant tensor's step only if the off-diagonal component counts both A_xy and A_yx."""
    n, dt = 9, 0.01
    grid = _grid(2, n)
    x = _nodes(grid)
    m0 = 1 + 0.5 * np.cos(np.pi * x[0]) + 0.3 * np.cos(np.pi * x[1]) + 0.2 * x[0] * x[1]
    constant = np.array([[0.3, 0.1], [0.1, 0.25]])
    steps = []
    for volatility in (constant, np.broadcast_to(constant, (n, n, 2, 2)).copy()):
        solver = FPFDMSolver(_problem(grid, 1, dt), boundary_conditions=no_flux_bc(dimension=2))
        steps.append(
            solver.solve_fp_system(
                m0,
                potential_field=np.zeros((2, n, n)),
                volatility=volatility,
                volatility_kind="tensor",
                show_progress=False,
            )[-1]
        )
    assert np.max(np.abs(steps[1] - steps[0])) < 1e-13 * np.max(m0), np.max(np.abs(steps[1] - steps[0]))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_the_boundary_flux_of_a_varying_diffusion_is_its_ito_flux(dim: int):
    """compute_boundary_flux's diffusive flux of a varying D is -grad(D m): at m = c / D, D m is constant and every
    face carries no diffusive flux, where D grad m (on D's mean, as it was in 2-D and 3-D) would not."""
    from mfgarchon.utils.numerical.flux_diagnostics import FluxDiagnostics

    n = 9
    axes = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * dim, indexing="ij")
    d = (0.3 + 0.2 * axes[0]) ** 2 / 2
    result = FluxDiagnostics(dimension=dim, spacing=(1.0 / (n - 1),) * dim).compute_boundary_flux(
        1.0 / d, velocity_field=np.zeros_like(d), diffusion=d
    )
    assert max(abs(f) for f in result.boundary_fluxes.values()) < 1e-12, result.boundary_fluxes

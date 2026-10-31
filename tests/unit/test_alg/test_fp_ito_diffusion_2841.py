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
from mfgarchon.geometry import TensorProductGrid, no_flux_bc, periodic_bc
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
    warning. Its m0 varies where sigma does; from a constant one the row-scaled step gained nothing. U is 0 on this row
    (no coupling cost, a zero terminal cost), so it checks that the coupled route reaches the fixed step, not a drifted
    coupled step; the drifted rows are in the mass test."""
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


@pytest.mark.parametrize("path", ["implicit", "implicit_2d", "explicit_callable_drift", "adjoint_mode", "tensor"])
def test_one_over_d_is_a_fixed_point_of_every_fdm_path(path: str):
    """With no drift, Ito's Delta(D m) takes m = c / D to 0 exactly (the unit stencil takes a constant to 0), so one step
    returns it; D Delta m and div(D grad m) move it. The tensor is sigma(x) I, a field written as a per-point tensor; in
    2-D, D varies along the first axis only, so a coefficient laid out in the other order is not a fixed point."""
    dim, n, dt = (2, 9, 0.01) if path in ("tensor", "implicit_2d") else (1, 21, 0.1)
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
    # and its size: a uniform array is the scalar, on a density and a drift that carry flux through every face
    m = 1 + 0.5 * np.cos(np.pi * axes[0]) + 0.2 * axes[-1]
    diag = FluxDiagnostics(dimension=dim, spacing=(1.0 / (n - 1),) * dim)
    alpha = 0.3 * np.ones_like(m) if dim == 1 else np.stack([0.3 * np.ones_like(m)] * dim, axis=-1)
    scalar = diag.compute_boundary_flux(m, velocity_field=alpha, diffusion=0.08).boundary_fluxes
    uniform = diag.compute_boundary_flux(m, velocity_field=alpha, diffusion=np.full(m.shape, 0.08)).boundary_fluxes
    assert all(abs(uniform[k] - scalar[k]) < 1e-12 for k in scalar), (uniform, scalar)


def _one_solve(grid, path: str, m0: np.ndarray, nt: int, dt: float, bc, **volatility) -> np.ndarray:
    solver = FPFDMSolver(_problem(grid, nt, nt * dt), boundary_conditions=bc)
    if path == "adjoint_mode":
        n = m0.size
        steps = [m0]
        for _ in range(nt):
            steps.append(solver.solve_fp_step_adjoint_mode(steps[-1], sparse.csr_matrix((n, n)), **volatility))
        return np.asarray(steps)
    if path == "explicit_callable_drift":

        def drift(t, xx, mm):
            return 0.2 * (np.ones_like(mm) if m0.ndim == 1 else np.ones((m0.ndim, *np.shape(mm))))

        return np.asarray(solver.solve_fp_system(m0, drift_field=drift, show_progress=False, **volatility))
    u = np.broadcast_to(0.1 * np.cos(2 * np.pi * _nodes(grid)[0]), (nt + 1, *m0.shape)).copy()
    return np.asarray(solver.solve_fp_system(m0, potential_field=u, show_progress=False, **volatility))


@pytest.mark.parametrize(
    ("path", "dim", "wall"),
    [
        ("implicit", 1, "no_flux"),
        ("implicit", 2, "no_flux"),
        ("explicit_callable_drift", 1, "no_flux"),
        ("adjoint_mode", 1, "no_flux"),
        ("implicit", 1, "periodic"),
        ("explicit_callable_drift", 1, "periodic"),
        ("adjoint_mode", 1, "periodic"),
    ],
)
def test_a_nearly_uniform_field_is_the_scalar_step(path: str, dim: int, wall: str):
    """sigma = 0.4 + 1e-11 x goes through the owner and 0.4 through the scalar path, so the two must agree to the
    field's own variation: this pins D's size on each field path, which mass and the fixed point c / D cannot see (both hold
    for any multiple of D, zero included). On a periodic grid, main's callable-drift step and adjoint mode had read no
    coefficient there and diffused with D = 1 whatever sigma was."""
    n = 9 if dim == 2 else 21
    bc = (no_flux_bc if wall == "no_flux" else periodic_bc)(dimension=dim)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=bc)
    x = _nodes(grid)
    m0 = 1 + 0.5 * np.cos(2 * np.pi * x[0]) + (0.3 * np.cos(np.pi * x[1]) if dim == 2 else 0.0)
    field = _one_solve(grid, path, m0, 3, 0.05, bc, volatility=0.4 + 1e-11 * x[0], volatility_kind="field")
    scalar = _one_solve(grid, path, m0, 3, 0.05, bc, volatility=0.4)
    assert np.max(np.abs(field - scalar)) < 1e-9 * np.max(m0), np.max(np.abs(field - scalar))


def test_a_dirichlet_wall_holds_its_value_under_a_field_volatility():
    """The implicit step's Dirichlet rows are constraints, m = g, and carry no diffusion of their own (its assembly
    leaves them out of the unit stencil); under a field the wall still reads g at every step."""
    n, nt, dt, g = 21, 5, 0.05, 0.3
    grid = _grid(1, n)
    x = _nodes(grid)[0]
    bc = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="x_min", bc_type=BCType.DIRICHLET, value=g, boundary="x_min"),
            BCSegment(name="x_max", bc_type=BCType.NO_FLUX, value=0.0, boundary="x_max"),
        ],
    )
    m = FPFDMSolver(_problem(grid, nt, nt * dt), boundary_conditions=bc).solve_fp_system(
        1 + 0.5 * np.cos(np.pi * x),
        potential_field=np.zeros((nt + 1, n)),
        volatility=0.3 + 0.2 * x,
        volatility_kind="field",
        show_progress=False,
    )
    assert np.allclose(np.asarray(m)[1:, 0], g, rtol=0, atol=1e-13), np.asarray(m)[1:, 0]


@pytest.mark.parametrize(
    ("path", "dim"),
    [
        ("implicit", 1),
        ("explicit_callable_drift", 1),
        ("explicit_callable_drift", 2),
        ("adjoint_mode", 1),
        ("tensor", 2),
    ],
)
def test_a_periodic_grid_reads_its_seam_at_the_first_copy(path: str, dim: int):
    """On an endpoint-inclusive periodic grid the last node is the first, so it has one coefficient, every component of
    A included. With a sigma that differs at its two copies (0.3 + 0.2x), and for the tensor an off-diagonal that does
    too, each step equals bit for bit the step on sigma with its first copy written into the last, as the implicit
    step's constraint rows read it (#1822); and it conserves mass and keeps the copies equal."""
    n, nt, dt = (9, 10, 0.002) if path == "tensor" else ((9, 10, 0.05) if dim == 2 else (21, 10, 0.05))
    bc = periodic_bc(dimension=dim)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=bc)
    x = _nodes(grid)
    m0 = 1 + 0.5 * np.cos(2 * np.pi * x[0]) + (0.3 * np.cos(2 * np.pi * x[1]) if dim == 2 else 0.0)
    sigma = 0.3 + 0.2 * x[0]
    if path == "tensor":
        volatility, kind = _tensor(sigma, 0.25 + 0.1 * x[1], 0.05 * (1 + x[0] + x[1])), "tensor"
    else:
        volatility, kind = sigma, "field"

    def first_copy(a: np.ndarray) -> np.ndarray:
        a = np.array(a, copy=True)
        for axis in range(dim):
            last, first = [slice(None)] * a.ndim, [slice(None)] * a.ndim
            last[axis], first[axis] = -1, 0
            a[tuple(last)] = a[tuple(first)]
        return a

    def run(v: np.ndarray) -> np.ndarray:
        return _one_solve(
            grid, "implicit" if path == "tensor" else path, m0, nt, dt, bc, volatility=v, volatility_kind=kind
        )

    m = run(volatility)
    assert np.array_equal(m, run(first_copy(volatility)))
    mass = _mass(grid, m)
    assert np.max(np.abs(mass / mass[0] - 1)) < 1e-12, mass / mass[0] - 1
    assert np.max(np.abs(np.take(m[-1], -1, axis=0) - np.take(m[-1], 0, axis=0))) < 1e-12


@pytest.mark.parametrize("off_diagonal", ["zero_on_a_line", "negative"])
def test_the_tensor_step_is_the_ito_sum_component_by_component(off_diagonal: str):
    """One tensor step with no drift is m0 + dt sum_{i<=j} U_ij(A_ij m0), U_ij the diffusion kernel at the unit tensor
    of the component (e_i e_i^T, or e_i e_j^T + e_j e_i^T for i < j). An off-diagonal that is zero on a line, or
    negative, is still a component of A and is summed."""
    from mfgarchon.operators.differential.diffusion import apply_diffusion
    from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

    n, dt = 9, 0.002
    grid = _grid(2, n)
    x = _nodes(grid)
    s_xy = 0.05 * (x[0] - 0.5) if off_diagonal == "zero_on_a_line" else np.full(x[0].shape, -0.05)
    sigma = _tensor(0.3 + 0.2 * x[0], 0.25 + 0.1 * x[1], s_xy)
    m0 = 1 + 0.5 * np.cos(np.pi * x[0]) + 0.3 * np.cos(np.pi * x[1])
    m1 = FPFDMSolver(_problem(grid, 1, dt), boundary_conditions=no_flux_bc(dimension=2)).solve_fp_system(
        m0, potential_field=np.zeros((2, n, n)), volatility=sigma, volatility_kind="tensor", show_progress=False
    )[-1]
    a = np.asarray(diffusion_from_volatility(sigma, kind="tensor"))
    expected = np.zeros_like(m0)
    for i, j in ((0, 0), (0, 1), (1, 1)):
        e = np.zeros((2, 2))
        e[i, j] = e[j, i] = 1.0
        unit = np.broadcast_to(e, (n, n, 2, 2))
        expected = expected + apply_diffusion(a[..., i, j] * m0, unit, [1.0 / (n - 1)] * 2, bc=no_flux_bc(dimension=2))
    expected = m0 + dt * expected
    assert np.max(np.abs(m1 - expected)) < 1e-13 * np.max(m0), np.max(np.abs(m1 - expected))


def test_a_diagonal_per_point_tensor_is_refused_for_its_wall_not_for_an_off_diagonal():
    """A partly no-flux BC is refused on the tensor step by its advection (a Dirichlet wall exchanges mass); a diagonal
    per-point Sigma has no off-diagonal component, so that is the refusal, not the kernel's off-diagonal one."""
    n = 9
    grid = _grid(2, n)
    x = _nodes(grid)
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(
                name=f,
                bc_type=BCType.DIRICHLET if f == "x_min" else BCType.NO_FLUX,
                value=0.2 if f == "x_min" else 0.0,
                boundary=f,
            )
            for f in ("x_min", "x_max", "y_min", "y_max")
        ],
    )
    with pytest.raises(NotImplementedError, match="mass_conservative divergence supports only"):
        FPFDMSolver(_problem(grid, 1, 0.002), boundary_conditions=bc).solve_fp_system(
            np.ones((n, n)),
            potential_field=np.zeros((2, n, n)),
            volatility=_tensor(0.3 + 0.2 * x[0], 0.25 + 0.1 * x[1], 0.0),
            volatility_kind="tensor",
            show_progress=False,
        )

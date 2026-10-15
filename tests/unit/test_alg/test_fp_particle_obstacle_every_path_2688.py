"""Every forward-step path of `FPParticleSolver` records its particles inside an obstacle domain, or refuses (#2688).

Three CPU paths step particles -- the 1-D grid drift, the n-D grid drift, and the callable drift -- and each crossed
with the outer boundary condition it can take, uniform or segment-aware (absorbing), and with `preserve_indices` and
the caller's `initial_particles` where those exist. Each ends its step in `_finish_step`: the outer boundary condition
on the raw step, then the obstacle placement for the survivors with their positions before the step. Before, the 1-D
grid path never placed (1990 of 2000 particles inside the obstacle at the peak, i2688/prod/product_base.log) and the
n-D paths placed BEFORE the wall reflection, which put particles back inside an obstacle that meets the wall (13438
particle-times). The GPU path, a periodic outer boundary and initial mass inside an obstacle are refused by name.

What reddens this file: any path whose step stops calling the placement, or calls it before the boundary condition
(its cell); the placement indexing ``previous`` without the absorbed mask (the mask test); the initial sampler
dropping ``navigable=`` (the cells' t = 0, whose jitter at the box's edge nodes leaves the domain).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, periodic_bc
from mfgarchon.geometry.implicit import DifferenceDomain, Hyperrectangle

_NT, _N = 15, 600

# 2-D: the obstacle meets the wall x = 1, so a wall reflection can land in it. 1-D: an interval cut from the middle.
_DOMAIN = {
    2: DifferenceDomain(
        Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), Hyperrectangle(np.array([[0.9, 1.05], [0.2, 0.8]]))
    ),
    1: DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0]])), Hyperrectangle(np.array([[0.4, 0.6]]))),
}


def _absorbing_at_x_max(d):
    boundaries = ["x_min", "x_max", "y_min", "y_max"][: 2 * d]
    return BoundaryConditions(
        segments=[
            BCSegment(name=b, bc_type=BCType.DIRICHLET if b == "x_max" else BCType.NO_FLUX, value=0.0, boundary=b)
            for b in boundaries
        ],
        dimension=d,
    )


def _problem(d, bc):
    n = 21 if d == 2 else 101
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[n] * d, boundary_conditions=bc)
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(geometry=grid, T=0.3, Nt=_NT, volatility=0.1 if d == 2 else 0.05, components=components), n


def _m_initial(d, n, centres):
    """Gaussian bumps, zero at the nodes inside the obstacle unless a bump is put there on purpose."""
    xs = np.linspace(0.0, 1.0, n)
    pts = xs.reshape(-1, 1) if d == 1 else np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1).reshape(-1, 2)
    m = sum(np.exp(-np.sum((pts - np.asarray(c)) ** 2, axis=1) / 0.002) for c in centres)
    if all(_DOMAIN[d].contains(np.atleast_2d(c))[0] for c in centres):
        m = m * _DOMAIN[d].contains(pts)
    m = m.reshape((n,) * d)
    return m / (m.sum() * (xs[1] - xs[0]) ** d)


def _drift(d, n, path):
    """Toward the obstacle and, in 2-D, toward the wall it meets: (1, -0.6); +2 in 1-D."""
    if path == "callable":
        if d == 1:
            return lambda t, x, m: 2.0 + 0.0 * np.asarray(x, dtype=float)
        return lambda t, x, m: np.column_stack([np.full(len(x), 1.0), np.full(len(x), -0.6)])
    xs = np.linspace(0.0, 1.0, n)
    if d == 1:
        return np.tile(-2.0 * xs, (_NT + 1, 1))  # the 1-D grid path differentiates U: U = -2x is drift +2
    drift = np.zeros((_NT + 1, n, n, 2))
    drift[..., 0], drift[..., 1] = 1.0, -0.6
    return drift


def _solve(d, bc, path, centres=None, initial_particles=None, **solver_kwargs):
    problem, n = _problem(d, bc)
    if centres is None:
        # An absorbing BC gets a second bump that the drift carries to the exit, so absorption and placement both act.
        absorbing = isinstance(bc, BoundaryConditions) and any(s.bc_type == BCType.DIRICHLET for s in bc.segments)
        centres = (
            ([(0.93, 0.88)] + ([(0.5, 0.5)] if absorbing else [])) if d == 2 else [0.3] + ([0.7] if absorbing else [])
        )
    solver = FPParticleSolver(
        problem, num_particles=_N, implicit_domain=_DOMAIN[d], seed=2688, boundary_conditions=bc, **solver_kwargs
    )
    kwargs = {"drift_field": _drift(d, n, path), "show_progress": False}
    if path == "grid" and d == 2:
        kwargs["drift_is_precomputed"] = True
    if path == "callable":
        kwargs["drift_needs_density"] = False
    if initial_particles is not None:
        kwargs["initial_particles"] = initial_particles
    else:
        kwargs["M_initial"] = _m_initial(d, n, centres)
    solver.solve_fp_system(**kwargs)
    return solver


def _navigable_start():
    return np.random.default_rng(2688).uniform([0.6, 0.6], [0.85, 0.95], (_N, 2))


_CELLS = {
    "1-D grid, reflecting": (1, "reflecting", "grid", {}),
    "1-D grid, absorbing": (1, "absorbing", "grid", {}),
    "n-D grid, reflecting": (2, "reflecting", "grid", {}),
    "n-D grid, absorbing": (2, "absorbing", "grid", {}),
    "callable 1-D, reflecting": (1, "reflecting", "callable", {}),
    "callable 1-D, absorbing": (1, "absorbing", "callable", {}),
    "callable 2-D, reflecting": (2, "reflecting", "callable", {}),
    "callable 2-D, absorbing": (2, "absorbing", "callable", {}),
    "callable 2-D, absorbing, preserve_indices": (2, "absorbing", "callable", {"preserve_indices": True}),
    "callable 2-D, reflecting, initial_particles": (2, "reflecting", "callable", {"initial": True}),
}


@pytest.mark.parametrize("cell", list(_CELLS))
def test_no_step_path_records_a_particle_outside_the_domain(cell):
    """Every recorded particle, at every time, is navigable; and the cell exercises what it names -- the placement
    acted, and an absorbing cell absorbed."""
    d, wall, path, options = _CELLS[cell]
    bc = _absorbing_at_x_max(d) if wall == "absorbing" else no_flux_bc(dimension=d)
    kwargs = {}
    if options.get("preserve_indices"):
        kwargs = {"preserve_indices": True, "density_mode": "query_only"}
    solver = _solve(
        d, bc, path, initial_particles=_navigable_start() if options.get("initial") or kwargs else None, **kwargs
    )

    outside = []
    for particles in solver.M_particles_trajectory:
        p = np.asarray(particles, dtype=float).reshape(-1, d)
        p = p[np.all(np.isfinite(p), axis=1)]  # preserve_indices marks the absorbed with NaN
        outside.append(int((~np.atleast_1d(_DOMAIN[d].contains(p))).sum()))
    assert sum(outside) == 0, f"{cell}: particles recorded outside the domain, per time: {outside}"
    placed = sum(sum(h.values()) for h in solver.obstacle_placement_history)
    assert placed > 0, f"{cell}: the placement never acted, so this cell does not test it"
    if wall == "absorbing":
        assert solver.total_absorbed > 0, (
            f"{cell}: nothing was absorbed, so this cell does not test the absorbing branch"
        )


def test_the_placement_reads_each_survivors_own_previous_position():
    """The boundary condition absorbs first, so the survivors are a subset; the placement must read each survivor's
    own position before the step, through the absorbed mask. Particle 0 leaves at the absorbing wall; particle 1 ends
    inside the obstacle having entered through its bottom face. Read without the mask, particle 1 would take particle
    0's start -- left of the obstacle -- and land on the face x = 0.9 instead of y = 0.2."""
    problem, _ = _problem(2, _absorbing_at_x_max(2))
    solver = FPParticleSolver(
        problem, num_particles=2, implicit_domain=_DOMAIN[2], seed=0, boundary_conditions=_absorbing_at_x_max(2)
    )
    previous = np.array([[0.85, 0.3], [0.95, 0.15]])
    raw = np.array([[1.2, 0.95], [0.98, 0.3]])
    recorded, absorbed, n_absorbed = solver._finish_step(raw, previous, [(0.0, 1.0), (0.0, 1.0)], True)
    assert n_absorbed == 1, n_absorbed
    assert absorbed.tolist() == [True, False], absorbed
    np.testing.assert_allclose(recorded, [[0.98, 0.2 - 1e-10]], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "case",
    [
        "periodic, n-D grid",
        "periodic, callable",
        "initial_particles inside the obstacle",
        "M_initial inside the obstacle, n-D",
        "M_initial inside the obstacle, 1-D nodes",
        "GPU path",
    ],
)
def test_what_the_placement_cannot_do_is_refused(case):
    """A periodic outer boundary (a wrapped step is not a segment, #2704), initial mass inside an obstacle (moving
    it changes the caller's measure), and the GPU path (its particles never meet the placement) are refused."""
    if case.startswith("periodic"):
        with pytest.raises(NotImplementedError, match=r"periodic outer boundary is not supported.*#2704"):
            _solve(2, periodic_bc(dimension=2), "grid" if "grid" in case else "callable")
    elif case.startswith("initial_particles"):
        start = _navigable_start()
        start[:3] = [0.95, 0.5]
        with pytest.raises(ValueError, match=r"initial_particles, 3 of 600 are inside an obstacle"):
            _solve(2, no_flux_bc(dimension=2), "callable", initial_particles=start)
    elif case == "M_initial inside the obstacle, n-D":
        with pytest.raises(
            ValueError, match=r"sampled cell\(s\) carrying .* no navigable point.*Set the initial density"
        ):
            _solve(2, no_flux_bc(dimension=2), "grid", centres=[(0.97, 0.5)])
    elif case == "M_initial inside the obstacle, 1-D nodes":
        with pytest.raises(
            ValueError, match=r"grid nodes it samples from m_initial, \d+ of 600 are inside an obstacle"
        ):
            _solve(1, no_flux_bc(dimension=1), "grid", centres=[0.5])
    else:
        pytest.importorskip("torch")
        problem, n = _problem(1, no_flux_bc(dimension=1))
        solver = FPParticleSolver(problem, num_particles=2000, implicit_domain=_DOMAIN[1], seed=2688, backend="torch")
        with pytest.raises(NotImplementedError, match=r"GPU path does not place particles"):
            solver.solve_fp_system(
                M_initial=_m_initial(1, n, [0.3]), drift_field=_drift(1, n, "grid"), show_progress=False
            )
        assert solver.current_strategy.name != "cpu", (
            "the selector sent this solve to the CPU path, so it tests nothing"
        )

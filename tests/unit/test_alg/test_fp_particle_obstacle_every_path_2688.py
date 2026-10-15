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

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, periodic_bc
from mfgarchon.geometry.implicit import DifferenceDomain, Hyperrectangle, UnionDomain

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
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        volatility=0.1 if d == 2 else 0.05,
    )
    conditions = Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.3)
    return MFGProblem(model=model, domain=grid, conditions=conditions, Nt=_NT), n


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
        "an implicit_domain box short of the grid's",
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
            ValueError, match=r"cell\(s\) carrying mass have no navigable point.*Leaving it out would drop"
        ):
            _solve(2, no_flux_bc(dimension=2), "grid", centres=[(0.97, 0.5)])
    elif case == "M_initial inside the obstacle, 1-D nodes":
        with pytest.raises(
            ValueError,
            match=r"grid node\(s\) carrying m_initial's mass are inside an obstacle.*Leaving it out would drop",
        ):
            _solve(1, no_flux_bc(dimension=1), "grid", centres=[0.5])
    elif case == "an implicit_domain box short of the grid's":
        # #2706 review 1, B1: the BC reflects at the grid's box, and the placement leaves to it whatever is outside
        # the domain's box, so a particle between the two would be recorded outside the domain.
        short = DifferenceDomain(
            Hyperrectangle(np.array([[0.0, 0.8], [0.0, 1.0]])), Hyperrectangle(np.array([[0.3, 0.4], [0.3, 0.4]]))
        )
        problem, _ = _problem(2, no_flux_bc(dimension=2))
        solver = FPParticleSolver(problem, num_particles=_N, implicit_domain=short, seed=0)
        with pytest.raises(ValueError, match=r"does not contain the grid's box"):
            solver.solve_fp_system(
                initial_particles=_navigable_start() * [0.9, 1.0],
                drift_field=_drift(2, 21, "callable"),
                drift_needs_density=False,
                show_progress=False,
            )
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


def test_a_step_that_absorbs_every_particle_records_none():
    """Every particle of the step leaves at the absorbing wall, so the placement gets no survivors: shape (0, 2)."""
    problem, _ = _problem(2, _absorbing_at_x_max(2))
    solver = FPParticleSolver(
        problem, num_particles=2, implicit_domain=_DOMAIN[2], seed=0, boundary_conditions=_absorbing_at_x_max(2)
    )
    recorded, _absorbed, n_absorbed = solver._finish_step(
        np.array([[1.2, 0.95], [1.3, 0.1]]), np.array([[0.85, 0.95], [0.85, 0.1]]), [(0.0, 1.0)] * 2, True
    )
    assert recorded.shape == (0, 2), recorded.shape
    assert n_absorbed == 2, n_absorbed


def test_a_reflected_step_is_placed_by_the_face_its_path_met():
    """#2706 review 1, M3: a reflected step's path is broken at the wall, and the face it met is on that path.

    From (0.985, 0.95) the raw step to (1.015, 0.7) hits the wall at (1.0, 0.825) and its reflected leg enters O
    through O's top face at (0.997, 0.8). The chord from the start to the reflected end, (0.985, 0.7), crosses the
    small obstacle Q's top face instead, a face the particle never approached; placed by the chord it landed at
    (0.985, 0.82). Placed by the path, it is projected onto O's top face: (0.985, 0.8 + 1e-10)."""
    obstacles = UnionDomain(
        [Hyperrectangle(np.array([[0.9, 1.05], [0.2, 0.8]])), Hyperrectangle(np.array([[0.984, 0.99], [0.81, 0.82]]))]
    )
    domain = DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), obstacles)
    problem, _ = _problem(2, no_flux_bc(dimension=2))
    solver = FPParticleSolver(problem, num_particles=1, implicit_domain=domain, seed=0)
    recorded, _, _ = solver._finish_step(np.array([[1.015, 0.7]]), np.array([[0.985, 0.95]]), [(0.0, 1.0)] * 2, False)
    np.testing.assert_allclose(recorded, [[0.985, 0.8 + 1e-10]], rtol=0, atol=1e-12)


def test_a_reflected_step_that_clips_a_flush_obstacles_corner_is_placed_on_its_top_face():
    """#2706: the obstacle [0.9, 1.0] x [0.2, 0.8] is flush with the wall x = 1. This step's first leg clips the
    obstacle's corner just before it hits the wall at (1.0, 0.79896), which lies ON the obstacle's face, phi = 0. That
    excursion is shorter than a scan step, so it was missed, the second leg began on a face that coincides with the
    wall, and the placement raised. Found by `i2688/m3/reflected_faces_code.py` on pop's flush rectangle. The face it
    met is the top face, so it is placed at (2 - raw_x, 0.8 + 1e-10)."""
    flush = DifferenceDomain(
        Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), Hyperrectangle(np.array([[0.9, 1.0], [0.2, 0.8]]))
    )
    problem, _ = _problem(2, no_flux_bc(dimension=2))
    solver = FPParticleSolver(problem, num_particles=1, implicit_domain=flush, seed=0)
    raw = np.array([[1.0275983976791534, 0.7639797123595877]])
    recorded, _, _ = solver._finish_step(
        raw, np.array([[0.9547534651728411, 0.8562972391037148]]), [(0.0, 1.0)] * 2, False
    )
    np.testing.assert_allclose(recorded, [[2.0 - raw[0, 0], 0.8 + 1e-10]], rtol=0, atol=1e-12)


@pytest.mark.parametrize("path", ["grid", "callable"])
def test_every_placement_in_a_solve_reads_its_own_particles_start(path):
    """#2706 review 1, S1: in a solve, each particle that ends a step in the obstacle is placed from its own start.

    The drift is linear, 8 * ((1, 0.5) - x), so each step's end is known from its start up to noise of order 1e-7,
    and particles enter the crossing rectangle through its left, top and bottom faces. Each recorded position is
    checked against the placement owner's answer from that particle's own start. A start read from another particle
    names another face and moves the placement by O(0.1)."""
    from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import project_to_navigable_or_defer

    nt, rate, target = 5, 8.0, np.array([1.0, 0.5])
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[21, 21], boundary_conditions=no_flux_bc(dimension=2))
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=1e-6
        ),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.5),
        Nt=nt,
    )
    solver = FPParticleSolver(
        problem, num_particles=_N, implicit_domain=_DOMAIN[2], seed=2706, boundary_conditions=no_flux_bc(dimension=2)
    )
    if path == "callable":
        rng = np.random.default_rng(2706)
        starts = np.vstack(
            [
                rng.uniform([0.86, 0.3], [0.89, 0.7], (_N // 3, 2)),
                rng.uniform([0.95, 0.85], [0.99, 0.95], (_N // 3, 2)),
                rng.uniform([0.95, 0.05], [0.99, 0.15], (_N - 2 * (_N // 3), 2)),
            ]
        )
        solver.solve_fp_system(
            initial_particles=starts,
            drift_field=lambda t, x, m: rate * (target - np.asarray(x)),
            drift_needs_density=False,
            show_progress=False,
        )
    else:
        xs = np.linspace(0.0, 1.0, 21)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        drift = np.zeros((nt + 1, 21, 21, 2))
        drift[..., 0], drift[..., 1] = rate * (1.0 - X), rate * (0.5 - Y)
        solver.solve_fp_system(
            M_initial=_m_initial(2, 21, [(0.85, 0.5), (0.95, 0.9), (0.95, 0.1)]),
            drift_field=drift,
            drift_is_precomputed=True,
            show_progress=False,
        )
    traj = [np.asarray(P, dtype=float).reshape(-1, 2) for P in solver.M_particles_trajectory]
    dt = problem.T / nt
    report: dict[str, int] = {}
    for t in range(len(traj) - 1):
        start = traj[t]
        expected = project_to_navigable_or_defer(
            start + rate * (target - start) * dt, _DOMAIN[2], previous=start, report=report
        )
        np.testing.assert_allclose(traj[t + 1], expected, rtol=0, atol=1e-4, err_msg=f"step {t}")
    assert report.get("face_of_entry", 0) > _N, f"the face map must act for many particles here: {report}"


def test_a_single_particle_steps_in_two_dimensions():
    """#2691: `reflect_positions`, `absorb_positions` and `wrap_positions` returned shape (d,) for a (1, d) row, and the
    step's placement then read a 2-D point as d points of dimension 1. A (1, d) row now comes back as (1, d)."""
    problem, _ = _problem(2, no_flux_bc(dimension=2))
    solver = FPParticleSolver(problem, num_particles=1, implicit_domain=_DOMAIN[2], seed=0, density_mode="query_only")
    solver.solve_fp_system(
        initial_particles=np.array([[0.85, 0.5]]),
        drift_field=_drift(2, 21, "callable"),
        drift_needs_density=False,
        show_progress=False,
    )
    for particles in solver.M_particles_trajectory:
        p = np.asarray(particles, dtype=float)
        assert p.reshape(-1, 2).shape == (1, 2), p.shape
        assert _DOMAIN[2].contains(p.reshape(-1, 2)).all(), p


def _interval_minus(lo, hi):
    return DifferenceDomain(Hyperrectangle(np.array([[0.0, 1.0]])), Hyperrectangle(np.array([[lo, hi]])))


_XS = np.linspace(0.0, 1.0, 11)  # spacing 0.1: the node 0.3 owns the cell [0.25, 0.35]


def test_a_jitter_in_an_obstacle_is_redrawn_within_its_cell():
    """t = 0 of #2688: the node 0.3 is navigable and its cell meets the obstacle at 0.32, so a jitter landing past it
    is redrawn -- within the same cell, so the cell keeps its mass: every sample lies in [0.25, 0.32)."""
    from mfgarchon.utils.numerical.particle import sample_from_density

    density = np.zeros(11)
    density[3] = 1.0
    x = sample_from_density(density, [_XS], 2000, seed=0, navigable=_interval_minus(0.32, 0.6).contains)[:, 0]
    assert np.all((x >= 0.25) & (x < 0.32)), (x.min(), x.max())
    assert np.ptp(x) > 0.05, "the redraw must spread over the cell's navigable part"


def test_a_cell_with_its_node_inside_but_a_navigable_part_is_sampled_there():
    """The node 0.3 is inside the obstacle [0.28, 0.6], but its cell's probe lattice holds 0.26, which is not: the
    cell has a navigable part, so its mass is sampled there rather than refused."""
    from mfgarchon.utils.numerical.particle import sample_from_density

    density = np.zeros(11)
    density[3] = 1.0
    x = sample_from_density(density, [_XS], 2000, seed=0, navigable=_interval_minus(0.28, 0.6).contains)[:, 0]
    assert np.all((x >= 0.25) & (x < 0.28)), (x.min(), x.max())


@pytest.mark.parametrize(("share", "refused"), [(1e-9, False), (1e-6, True)])
def test_mass_inside_an_obstacle_is_left_out_up_to_the_gates_threshold(share, refused):
    """#2706: the cell of the node 0.6 lies wholly inside the obstacle [0.45, 0.75]. Its share of the mass goes to
    `mass_fabrication_gate.drop_mass_or_raise`: above the gate's threshold (1e-8) the sample is refused, naming the
    share; at or below it the cell is left out and nothing is placed there."""
    from mfgarchon.utils.numerical.particle import sample_from_density

    domain = _interval_minus(0.45, 0.75)
    density = np.zeros(11)
    density[2], density[6] = 1.0 - share, share
    if refused:
        with pytest.raises(
            ValueError, match=r"Leaving it out would drop a fraction 1\.000e-06 of the total mass \(threshold 1e-08\)"
        ):
            sample_from_density(density, [_XS], 2000, seed=0, navigable=domain.contains)
        return
    x = sample_from_density(density, [_XS], 2000, seed=0, navigable=domain.contains)
    assert domain.contains(x).all()

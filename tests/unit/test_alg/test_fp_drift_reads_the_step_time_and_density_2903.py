"""The FP solvers that read `optimal_control` themselves hand it the step's time and the density they hold (#2903).

The drift that advances step n to n + 1 is read at t_n = n dt, with U^n and with the density the solver holds at step n
(docs/user/CONVENTIONS.md, "What the Fokker–Planck solver receives"). Before #2903 the weak-form family (FEM, meshless
Galerkin) read it at t = 0 with m = None, so a time-dependent control was solved there as a constant one; the particle
solver's three loops read the step's time with m = None. A control that used m directly raised; one that converted it
first gave NaN on FEM and meshless Galerkin, and raised on the particle solver. FVM and the network solver held the
convention already, and are the controls here.

- **Time** (a mutation-verified convention pin). alpha = (1 + 4 t) alpha_lib(p) at step n is the library's control
  applied to U'^n = (1 + 4 t_n) U^n, since the library's control is linear in p (on the network, positively
  homogeneous). So the solve with that control on U equals the library's solve on U', to rounding. It kills a time
  read at t_{n+1} or at 0.
- **Which density** (a mutation-verified convention pin). A recording control keeps each read's (t, x, m): the times
  are the t_n, and each m is the solver's own density at step n where the solver reads it. On the particle rows that
  is the returned density read at the recorded particles by `np.interp` or scipy's linear grid interpolator, not by the
  solver's own `_estimate_density_at_particles`. It kills m = None, the step-0 density, a wrong step, the density at
  FEM's nodes for its quadrature points, and an interpolator that returns the wrong values.
- **The density is the solver's copy** (a mutation-verified pin). A control that writes the density it is handed, in
  place, leaves the returned history as it was. It kills a hook handed the solver's own row `M[n]`: on meshless Galerkin,
  which hands its node values on, the write then reaches the returned `M[0]`. FEM interpolates them into an array of its
  own, so it is not a row here.
- **The default path** (kills a mutation no other test here runs). `problem.solve()` on a 1-D mesh picks FEM_P1, and
  every read there carries a density. A factory that routes a 1-D mesh away from FEM_P1 fails it alone. It does not
  check which density; the which-density test does.

Each fixture notes the layouts of p and m it hands, as CONVENTIONS' table states them; no layout is pinned. A problem
reads the control at its own probes when it is built (#2901); those reads are not the solver's, and are dropped.
"""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import Mesh1D, TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

T, NT, SIGMA = 0.5, 10, 0.3
DT = T / NT
#: The time oracle's tolerance. A solver that reads t_{n+1}, or a fixed time, is far outside it (the PR's mutants).
TIME_TOL = 1e-9
#: The particle rows' oracle is an independent linear interpolation of the returned density, equal to the solver's to
#: rounding; a wrong density is far outside it (the PR's mutants).
INTERP_TOL = 1e-12
#: What a "write" control floors the density it is handed at, in place: above _m0's minimum, so the write is seen.
WRITE_FLOOR = 0.6


class _Control:
    """``kind``: "record" (the library's control), "time" ((1 + 4 t) times it), "write" (the library's control, after
    flooring the density it was handed at WRITE_FLOOR in place). Every read's (t, x, m) is kept."""

    def _shaped(self, t, x, p, m, base):
        self.reads.append(
            (float(t), tuple(np.array(c, dtype=float) for c in x), None if m is None else np.array(m, dtype=float))
        )
        if self.kind == "write" and isinstance(m, np.ndarray) and m.ndim > 0:
            # A point read hands a scalar density; the row is about the array a solver hands (#2901's batch check
            # reads the control at one point too).
            np.maximum(m, WRITE_FLOOR, out=m)
        alpha = np.asarray(base(t=t, x=x, p=p, m=m), dtype=float)
        if self.kind == "time":
            return (1.0 + 4.0 * t) * alpha
        return alpha


class _Separable(_Control, SeparableHamiltonian):
    def __init__(self, kind):
        super().__init__(control_cost=QuadraticControlCost(control_cost=1.0))
        self.kind, self.reads = kind, []

    def optimal_control(self, t, x, p, m):
        return self._shaped(t, x, p, m, super().optimal_control)

    # A "time" control is the library's H scaled by 1 + 4t, so its H and dp scale with it: a problem refuses a control
    # that is not -dp at its probes, which run at t = 0, T/2 and T (#2901). The FP rows read only the control.
    def __call__(self, t, x, p, m):
        return self._time_factor(t) * super().__call__(t=t, x=x, p=p, m=m)

    def dp(self, t, x, p, m):
        return self._time_factor(t) * np.asarray(super().dp(t=t, x=x, p=p, m=m), dtype=float)

    def _time_factor(self, t):
        return 1.0 + 4.0 * t if self.kind == "time" else 1.0


class _NetworkControl(_Control):
    """The network solver reads its problem's own NetworkHamiltonian, so the control wraps it."""

    def __init__(self, inner, kind):
        self.inner, self.kind, self.reads = inner, kind, []

    def optimal_control(self, x, m, p, t):
        return self._shaped(t, x, p, m, self.inner.optimal_control)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _components(h, dim=1, level=1.0):
    """A constant density of mass 1 on the problem's measure (`MFGProblem.spatial_measure`), so the problem does not
    warn of its mass; the FP-only rows pass their own m0. The terminal cost has a gradient, so `problem.solve()` moves
    the density."""
    return MFGComponents(
        m_initial=lambda x: np.full_like(np.asarray(x[0], dtype=float), level),
        u_terminal=lambda x: 0.5 * (np.asarray(x[0], dtype=float) - 0.3) ** 2,
        hamiltonian=h,
    )


def _grid_problem(h, n, dim=1):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )
    problem = MFGProblem(geometry=grid, T=T, Nt=NT, volatility=SIGMA, components=_components(h, dim))
    h.reads.clear()  # construction's probe reads (#2901) are not the solver's
    return problem


def _mesh_problem(h):
    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=14)
    mesh.generate_mesh()
    mesh.boundary_conditions = no_flux_bc(dimension=1)
    # A 1-D mesh is measured on uniform cells, one per node: 15 cells of width 1/14, so mass 1 is the level 14/15.
    problem = MFGProblem(geometry=mesh, T=T, Nt=NT, volatility=SIGMA, components=_components(h, level=14 / 15))
    h.reads.clear()  # construction's probe reads (#2901) are not the solver's
    return problem


def _scale(time_scaled):
    """(1 + 4 t_n) per step when ``time_scaled``, else 1: the oracle's U' = (1 + 4 t_n) U^n."""
    return 1.0 + 4.0 * np.arange(NT + 1) * DT if time_scaled else np.ones(NT + 1)


def _u_1d(x, time_scaled):
    return np.stack([s * 0.5 * (x - 0.3) ** 2 * (1 + 0.1 * k) for k, s in enumerate(_scale(time_scaled))])


def _m0(x):
    return np.exp(-20 * (x - 0.5) ** 2)


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            return fn()
        finally:
            logging.disable(logging.NOTSET)


def _fem(kind, time_scaled=False):
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    h = _Separable(kind)  # p is (d, elements, points); m is (elements, points)
    solver = FPFEMSolver(_mesh_problem(h), order=1)
    x = solver._disc.dof_coordinates[:, 0]
    M = _quiet(lambda: solver.solve_fp_system(_m0(x), potential_field=_u_1d(x, time_scaled)))
    return SimpleNamespace(h=h, M=np.asarray(M), density=lambda n, read: (solver._basis.interpolate(M[n]).value, 0.0))


def _meshless(kind, time_scaled=False):
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver

    h = _Separable(kind)  # p is (nodes, d); m is (nodes,)
    x = np.linspace(0.0, 1.0, 15)
    solver = MeshlessGalerkinFPSolver(_grid_problem(h, 15), collocation_points=x[:, None], delta=3.5 / 14)
    M = _quiet(lambda: solver.solve_fp_system(_m0(x), potential_field=_u_1d(x, time_scaled)))
    return SimpleNamespace(h=h, M=np.asarray(M), density=lambda n, read: (M[n], 0.0))


def _particle_1d(kind, time_scaled=False):
    from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver

    h = _Separable(kind)  # p and m are (particles,)
    x = np.linspace(0.0, 1.0, 15)
    solver = FPParticleSolver(_grid_problem(h, 15), num_particles=2000, seed=2)
    M = np.asarray(
        _quiet(lambda: solver.solve_fp_system(_m0(x), potential_field=_u_1d(x, time_scaled), show_progress=False))
    )

    return SimpleNamespace(h=h, M=M, density=lambda n, read: (np.interp(read[1][0], x, M[n]), INTERP_TOL))


def _particle_2d(kind, time_scaled=False):
    from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver

    h = _Separable(kind)  # p is (particles, 2); m is (particles,)
    x = np.linspace(0.0, 1.0, 9)
    X, Y = np.meshgrid(x, x, indexing="ij")
    U = np.stack(
        [s * 0.5 * ((X - 0.3) ** 2 + (Y - 0.6) ** 2) * (1 + 0.1 * k) for k, s in enumerate(_scale(time_scaled))]
    )
    solver = FPParticleSolver(_grid_problem(h, 9, dim=2), num_particles=2000, seed=2)
    M = np.asarray(
        _quiet(
            lambda: solver.solve_fp_system(
                np.exp(-20 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)), potential_field=U, show_progress=False
            )
        )
    )

    def density(n, read):
        from scipy.interpolate import RegularGridInterpolator

        positions = np.stack(read[1], axis=-1)
        return RegularGridInterpolator((x, x), M[n], method="linear")(positions), INTERP_TOL

    return SimpleNamespace(h=h, M=M, density=density)


def _particle_gpu_loop(kind, time_scaled=False):
    """The 1-D GPU loop, on torch's CPU backend; the strategy is asserted first."""
    torch = pytest.importorskip("torch")
    from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver

    h = _Separable(kind)  # p and m are (particles,)
    x = np.linspace(0.0, 1.0, 41)
    np.random.seed(12345)
    torch.manual_seed(12345)  # this loop samples with torch's generator and steps with numpy's global stream
    solver = FPParticleSolver(_grid_problem(h, 41), num_particles=2000, kde_bandwidth=0.1, backend="torch")
    M = np.asarray(_quiet(lambda: solver.solve_fp_system(_m0(x), _u_1d(x, time_scaled))))
    assert solver.current_strategy.name != "cpu", (
        f"routed to the {solver.current_strategy.name!r} strategy, not the GPU loop, so this row tests the CPU loop"
    )

    return SimpleNamespace(h=h, M=M, density=lambda n, read: (np.interp(read[1][0], x, M[n]), INTERP_TOL))


def _fvm(kind, time_scaled=False):
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver

    h = _Separable(kind)  # p and m are one value per face
    x = np.linspace(0.0, 1.0, 15)
    problem = _grid_problem(h, 15)
    solver = FPFVMSolver(problem)
    M = np.asarray(
        _quiet(lambda: solver.solve_fp_system(_m0(x), potential_field=_u_1d(x, time_scaled), show_progress=False))
    )
    return SimpleNamespace(h=h, M=M, density=lambda n, read: (0.5 * (M[n][:-1] + M[n][1:]), 0.0))


def _network(kind, time_scaled=False):
    pytest.importorskip("igraph")
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
    from mfgarchon.extensions.topology import NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    net = GridNetwork(width=4, height=4)
    net.create_network()
    problem = NetworkMFGProblem(geometry=net, T=T, Nt=NT)
    h = _NetworkControl(problem.hamiltonian_class, kind)  # a node's control is one value per node
    problem.components._hamiltonian_class = h  # the backing field of MFGProblem.hamiltonian_class
    assert problem.hamiltonian_class is h
    n = problem.num_nodes
    m0 = np.random.default_rng(3).uniform(0.5, 1.5, n)
    U = np.stack([s * np.linspace(0.0, 2.0, n) * (1 + 0.1 * k) for k, s in enumerate(_scale(time_scaled))])
    M = np.asarray(
        _quiet(
            lambda: FPNetworkSolver(problem, scheme="explicit").solve_fp_system(m0 / m0.sum(), U, show_progress=False)
        )
    )
    return SimpleNamespace(h=h, M=M, density=lambda n_, read: (M[n_], 0.0))


ROWS = {
    "FEM P1": _fem,
    "meshless Galerkin": _meshless,
    "particle, CPU 1-D loop": _particle_1d,
    "particle, CPU nD loop": _particle_2d,
    "particle, GPU loop on torch's CPU backend": _particle_gpu_loop,
    "FVM (control)": _fvm,
    "network (control)": _network,
}


@pytest.mark.parametrize("row", list(ROWS))
def test_a_time_dependent_control_is_read_at_each_steps_time(row):
    scaled = ROWS[row]("time")
    oracle = ROWS[row]("record", time_scaled=True)
    gap = float(np.max(np.abs(scaled.M - oracle.M)))
    assert gap < TIME_TOL, (
        f"{row}: alpha = (1 + 4t) alpha_lib(p) on U differs from alpha_lib on U' = (1 + 4 t_n) U^n by {gap:.3e}; the "
        f"control is not read at t_n = n dt on the step it advances (#2903)"
    )


@pytest.mark.parametrize("row", list(ROWS))
def test_each_read_gets_the_steps_time_and_the_density_the_solver_holds(row):
    run = ROWS[row]("record")
    reads = run.h.reads
    times = sorted({t for t, _, _ in reads})
    assert np.allclose(times, DT * np.arange(NT), rtol=0, atol=1e-12), f"{row}: read at {times}, not at n dt (#2903)"
    for read in reads:
        t, _, m = read
        assert m is not None, f"{row}: read at t = {t} with m = None (#2903)"
        n = round(t / DT)
        expected, tol = run.density(n, read)
        expected = np.asarray(expected, dtype=float)
        assert m.shape == expected.shape, f"{row}: the m read at t = {t} has shape {m.shape}, not {expected.shape}"
        gap = float(np.max(np.abs(m - expected)))
        assert gap <= tol, (
            f"{row}: the m read at t = {t} is {gap:.3e} from the density the solver holds at step {n} (#2903)"
        )


def test_a_control_that_writes_the_density_it_is_handed_leaves_the_returned_history():
    run = _meshless("write")
    m0 = _m0(np.linspace(0.0, 1.0, 15))
    assert m0.min() < WRITE_FLOOR, "the initial density never falls below the floor, so a write cannot be seen"
    assert np.array_equal(run.M[0], m0), (
        f"meshless Galerkin: the returned M[0] is not the initial density (min {run.M[0].min():.4f}): a control that "
        f"writes the density it is handed reached the solver's history (#2903)"
    )


def test_the_default_solve_on_a_mesh_hands_the_control_a_density():
    """`problem.solve()` on a 1-D mesh picks FEM_P1 (the factory's scheme for an unstructured mesh). The reads come
    from FEM's quadrature points, two-dimensional per coordinate, and every one carries a density."""
    h = _Separable("record")
    result = _quiet(lambda: _mesh_problem(h).solve())
    assert h.reads, "the solve read optimal_control nowhere"
    assert all(read[1][0].ndim == 2 for read in h.reads), "a read did not come from FEM's quadrature points"
    assert all(read[2] is not None for read in h.reads), "a read was handed m = None (#2903)"
    assert np.all(np.isfinite(np.asarray(result.M)))

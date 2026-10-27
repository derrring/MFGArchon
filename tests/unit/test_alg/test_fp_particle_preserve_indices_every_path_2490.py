"""`preserve_indices=True` holds on every CPU particle path (Issue #2490, after #1119).

#1119 implemented the flag for the callable-drift path only. The 1-D path and the grid-drift n-D path raised under a
segment-aware BC, absorbing or not, which is the per-point route #2490's routing needs. Now the three share one owner
of the bookkeeping. The GPU path refuses a segment-aware BC (#1910), so it has no absorbed rows to keep.

The oracle is the flag's contract, read against the same solve with the flag off and the same seed:
- the density is bit-identical, since the flag changes only what is stored;
- each slice is full-size, its NaN rows are the particles absorbed so far, and its live rows are the compact slice's,
  in order;
- particle j keeps row j. With no noise and a constant drift of 1.5 onto an absorbing wall, each live row moves by
  exactly 1.5 dt per step, so a row that carries another particle after an absorption is off by the gap between the
  two particles' positions;
- with nothing absorbed, the trajectory is the compact one and has no NaN row: under a uniform BC, which keeps every
  row anyway, and under no-flux segments, which route through the segment-aware step and its bookkeeping.

Packing the survivors into the first rows, the defect of dropping the original-index update, passes the count and order
checks: the identity check is the one that fails, on each path.
"""

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc

# Seed 11: neither absorbing 2-D grid-drift solve here (sigma 0.2, and sigma 0 for the row-following test) may reach a
# step with exactly 2 survivors, which have no kernel (#2790). At #2556's C3-1 the density reader refuses such a step by
# name. Of seeds 5 to 20, those that reach one are 5-9, 12, 14, 16 and 20 at sigma 0.2 and 10 and 17 at sigma 0; the
# base raised on all of them but 8 and 12, which it passed through its ghost refit.
_N, _NT, _T, _SEED = 400, 20, 1.0, 11


def _m0(x):
    return float(np.exp(-20 * np.sum((np.asarray(x, dtype=float) - 0.6) ** 2)))


def _bc(dimension, wall):
    """``wall``: "uniform" no-flux, "no-flux segments" (segment-aware, absorbs nothing), or "absorbing" at x_max."""
    if wall == "uniform":
        return no_flux_bc(dimension=dimension)
    faces = ["x_min", "x_max", "y_min", "y_max"][: 2 * dimension]
    return BoundaryConditions(
        segments=[
            BCSegment(
                name=face,
                bc_type=BCType.DIRICHLET if wall == "absorbing" and face == "x_max" else BCType.NO_FLUX,
                value=0.0,
                boundary=face,
            )
            for face in faces
        ],
        dimension=dimension,
    )


def _solve(path, wall, preserve, sigma=0.2):
    """One solve on ``path``; returns (density, trajectory slices)."""
    dimension = 1 if path == "1-D grid drift" else 2
    bc = _bc(dimension, wall)
    shape = [41] if dimension == 1 else [21, 11]
    bounds = [(0.0, 1.0)] if dimension == 1 else [(0.0, 1.0), (0.0, 0.5)]
    grid = TensorProductGrid(bounds, Nx_points=shape, boundary_conditions=bc)
    with warnings.catch_warnings():
        # The bump's mass on the grid is not 1; the trajectory's bookkeeping does not depend on it.
        warnings.filterwarnings("ignore", message="initial density mass")
        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=sigma
            ),
            domain=grid,
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=_m0, T=_T),
            Nt=_NT,
        )
    density_mode = "query_only" if path == "callable drift" else "grid_only"
    solver = FPParticleSolver(
        problem,
        num_particles=_N,
        density_mode=density_mode,
        boundary_conditions=bc,
        preserve_indices=preserve,
        seed=_SEED,
    )
    if path == "callable drift":
        init = np.random.default_rng(_SEED).uniform([0.4, 0.1], [0.8, 0.4], (_N, 2))
        result = solver.solve_fp_system(
            initial_particles=init,
            drift_field=lambda t, x, m: np.column_stack([np.full(len(x), 1.5), np.zeros(len(x))]),
            volatility=sigma,
            drift_needs_density=False,
            show_progress=False,
        )
        density = result.M_grid
    else:
        # U = -1.5 x: the drift -grad U is 1.5 to the right, onto the x_max wall
        x = np.asarray(grid.get_spatial_grid(), dtype=float)[:, 0].reshape(shape)
        U = np.broadcast_to(-1.5 * x, (_NT + 1, *shape)).copy()
        m0 = np.asarray([_m0(p) for p in grid.get_spatial_grid()]).reshape(shape)
        density = solver.solve_fp_system(M_initial=m0, drift_field=U, show_progress=False)
    slices = [np.asarray(s, dtype=float) for s in solver.M_particles_trajectory]
    return np.asarray(density), slices


PATHS = ["1-D grid drift", "2-D grid drift", "callable drift"]


@pytest.mark.parametrize("path", PATHS)
def test_preserve_indices_keeps_each_particle_s_row_under_an_absorbing_wall(path):
    density_off, compact = _solve(path, "absorbing", preserve=False)
    density_on, preserved = _solve(path, "absorbing", preserve=True)

    assert np.array_equal(density_off, density_on), "the flag changes what is stored, not the solve"
    assert len(preserved) == len(compact)
    absorbed = [_N - len(c) for c in compact]
    assert absorbed[0] == 0
    assert absorbed[-1] > 0, "the fixture must absorb particles"
    for t, (full, live) in enumerate(zip(preserved, compact, strict=True)):
        rows = full.reshape(_N, -1)
        dead = np.isnan(rows).any(axis=1)
        assert rows.shape[0] == _N, f"slice {t} is not full-size"
        assert int(dead.sum()) == absorbed[t], f"slice {t}: {int(dead.sum())} NaN rows for {absorbed[t]} absorbed"
        assert np.array_equal(rows[~dead], np.asarray(live, dtype=float).reshape(-1, rows.shape[1])), (
            f"slice {t}: the live rows are not the compact slice in order"
        )
        if t:
            previous = np.isnan(preserved[t - 1].reshape(_N, -1)).any(axis=1)
            assert np.all(dead[previous]), f"slice {t}: an absorbed particle came back"


@pytest.mark.parametrize("wall", ["uniform", "no-flux segments"])
@pytest.mark.parametrize("path", PATHS)
def test_preserve_indices_with_nothing_absorbed_is_the_compact_trajectory(path, wall):
    density_off, compact = _solve(path, wall, preserve=False)
    density_on, preserved = _solve(path, wall, preserve=True)

    assert np.array_equal(density_off, density_on)
    assert len(preserved) == len(compact)
    for full, live in zip(preserved, compact, strict=True):
        assert not np.isnan(full).any()
        assert np.array_equal(full, live)


@pytest.mark.parametrize("path", PATHS)
def test_preserve_indices_rows_follow_their_own_particles(path):
    _, preserved = _solve(path, "absorbing", preserve=True, sigma=0.0)
    rows = [s.reshape(_N, -1) for s in preserved]
    dt = _T / _NT

    # From slice 1 on: the 2-D sampler can start a particle just outside the box, and the first step places it inside.
    start = rows[1][:, 0]
    assert int(np.isnan(rows[-1][:, 0]).sum()) == _N, "the fixture must absorb every particle by the end"
    for k in range(2, len(rows)):
        live = ~np.isnan(rows[k][:, 0])
        expected = start[live] + 1.5 * (k - 1) * dt
        assert np.max(np.abs(rows[k][live, 0] - expected), initial=0.0) < 1e-12, (
            f"slice {k}: a live row is not its own particle, moved by 1.5 dt per step"
        )

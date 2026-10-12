"""FP-FDM reads a Dirichlet value at the level it solves for, and at the wall node (#2535, #2512 row B2).

External oracle. With zero drift, m(t, x) = a + b (|x|^2 + 2 d D t), D = sigma^2 / 2, solves m_t = D lap m exactly. The
centred second difference is exact for a quadratic, and implicit Euler is exact for a solution linear in t. So
with the walls held at g(t, x) = m(t, x), FP-FDM reproduces m to roundoff. A wall read at t = 0 misses by 2 d D b t,
and the error spreads inward.

Measured at `76d8b1dc` (main):
- in 1-D with a per-face g(t): 9.000e-2 at T, which is exactly 2 d D b T;
- in 2-D with a uniform g(t, x): 2.18, since a callable value was read as 0.
At the head, both are below 2e-15.

The BC goes to the FP's own channel, `FPFDMSolver(..., boundary_conditions=bc)`. On the shared route,
`fp_view_of_shared_bc` makes a Dirichlet wall absorbing by design.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers import fp_fdm_time_stepping
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc

A, B, SIGMA, T, NT = 1.0, 0.5, 0.6, 0.5, 10
D = SIGMA**2 / 2


def _problem(d: int, n: int) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[n] * d, boundary_conditions=no_flux_bc(dimension=d))
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    return MFGProblem(geometry=grid, components=comps, T=T, Nt=NT, volatility=SIGMA)


@pytest.mark.parametrize("d", [1, 2])
def test_a_time_dependent_dirichlet_wall_is_read_at_each_level(d, monkeypatch):
    n = 21 if d == 1 else 9
    exact = lambda t, x: A + B * (np.sum(np.atleast_2d(x) ** 2, axis=-1) + 2 * d * D * t)  # noqa: E731
    if d == 1:  # per face, g(t): the mixed path through the accessor
        bc = BoundaryConditions(
            dimension=1,
            segments=[
                BCSegment(name="left", bc_type=BCType.DIRICHLET, value=lambda t: A + B * 2 * D * t, boundary="x_min"),
                BCSegment(
                    name="right", bc_type=BCType.DIRICHLET, value=lambda t: A + B * (1 + 2 * D * t), boundary="x_max"
                ),
            ],
        )
    else:  # uniform, g(t, x), read at the node
        bc = dirichlet_bc(dimension=2, value=lambda t, x: A + B * (x[0] ** 2 + x[1] ** 2 + 2 * d * D * t))
    # FP-FDM reads a Dirichlet node's value from the segment governing the node, through the value owner
    # `segment_value` (#2512 row B1); `get_bc_value_at_boundary`, which this recorded until then, is a wrapper of it.
    times = []
    owner = fp_fdm_time_stepping.segment_value

    def recording(segment, *, t=None, **kwargs):
        times.append(t)
        return owner(segment, t=t, **kwargs)

    monkeypatch.setattr(fp_fdm_time_stepping, "segment_value", recording)
    pts = np.stack(np.meshgrid(*[np.linspace(0.0, 1.0, n)] * d, indexing="ij"), axis=-1).reshape(-1, d)
    shape = (n,) * d
    M = np.asarray(
        FPFDMSolver(_problem(d, n), boundary_conditions=bc).solve_fp_system(
            exact(0.0, pts).reshape(shape), drift_field=np.zeros((NT + 1, *shape)), volatility=SIGMA
        )
    )
    levels = np.linspace(0.0, T, NT + 1)
    assert np.abs(M - np.stack([exact(t, pts).reshape(shape) for t in levels])).max() < 1e-12
    # Every level after the first is read at its own time, and only there; the first, at t = 0, only in 1-D.
    step_times = {round(k * T / NT, 12) for k in range(1, NT + 1)}
    read = {round(t, 12) for t in times}
    assert None not in times
    assert step_times <= read, sorted(step_times - read)
    assert read <= step_times | {0.0}, sorted(read - step_times)


def test_a_no_flux_wall_beside_a_dirichlet_one_keeps_its_density():
    """A BC whose only segment is Dirichlet, on x_min, leaves x_max to its no-flux default. The 1-D overwrite, at
    t = 0 and after every step, wrote both walls, since every segment was Dirichlet, and x_max read the BC's default
    value: the no-flux wall was held at 0.0 from level 0 on, at `76d8b1dc` (#2535). The post-step overwrite is gone,
    as the implicit step's rows hold g and the explicit steps refuse a Dirichlet wall. The level-0 write is by
    each wall's own type."""
    bc = BoundaryConditions(
        dimension=1,
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=2.0, boundary="x_min")],
        default_bc=BCType.NO_FLUX,
    )
    n = 21
    M = np.asarray(
        FPFDMSolver(_problem(1, n), boundary_conditions=bc).solve_fp_system(
            np.ones(n), drift_field=np.zeros((NT + 1, n)), volatility=SIGMA
        )
    )
    assert np.all(M[:, 0] == 2.0)
    # Mass flows in from the wall held at 2: the no-flux wall never drops below its initial density.
    assert np.all(M[:, -1] >= 1.0), M[:, -1]

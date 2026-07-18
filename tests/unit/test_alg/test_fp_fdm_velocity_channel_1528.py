"""The nD FDM velocity channel: reachability and honesty (Issue #1528 phase 2).

Two defects, both on `solve_fp_nd_full_system`:

1. The scalar drift coefficient was resolved unconditionally at function scope,
   before the drift channel was consulted. `fp_drift_coefficient` raises for any
   Hamiltonian whose optimal control is not ``-grad(U)/control_cost`` (MAXIMIZE,
   non-quadratic, regularized), so a MAXIMIZE problem could not run through the
   nD FDM solver *even when supplying* ``velocity_field`` -- the channel that
   exists precisely to carry a precomputed alpha* for those Hamiltonians.

Only ``divergence_upwind`` actually reads ``interface_velocity``, so only that
scheme leaves the coefficient unread; the others still fall back to ``-c*grad(U)``
and still resolve it. That they silently discard the caller's velocity is a
separate defect, tracked in #1632.

Each test fails if the fix is reverted.
"""

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_fp_nd_full_system
from mfgarchon.core.hamiltonian import (
    OptimizationSense,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

N = 8
NT = 4


def _problem(sense: OptimizationSense = OptimizationSense.MINIMIZE) -> MFGProblem:
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(lambda_=1.0, sense=sense),
        sense=sense,
    )
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, sigma=0.2),
        domain=TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=no_flux_bc(dimension=2),
        ),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=NT,
    )


def _uniform_density() -> np.ndarray:
    m0 = np.ones((N, N))
    return m0 / m0.sum()


def _velocity(vx: float = 0.0, vy: float = 0.0) -> np.ndarray:
    vel = np.zeros((NT + 1, 2, N, N))
    vel[:, 0, ...] = vx
    vel[:, 1, ...] = vy
    return vel


# --- defect 1: the coefficient must not be resolved where it is not consumed ---


@pytest.mark.parametrize("sense", [OptimizationSense.MINIMIZE, OptimizationSense.MAXIMIZE])
def test_velocity_channel_runs_for_both_senses(sense):
    """MAXIMIZE previously raised NotImplementedError from the eager coefficient read."""
    result = solve_fp_nd_full_system(_uniform_density(), None, _problem(sense), velocity_field=_velocity(vx=0.3))

    assert result.shape == (NT + 1, N, N)
    assert np.isfinite(result).all()
    assert result[-1].sum() == pytest.approx(1.0, abs=1e-9), "no-flux walls must conserve mass"


def test_maximize_still_rejected_on_the_u_channel():
    """The guard is not weakened: deriving the drift from U for MAXIMIZE is still wrong physics."""
    u_solution = np.zeros((NT + 1, N, N))
    with pytest.raises((NotImplementedError, ValueError)):
        solve_fp_nd_full_system(_uniform_density(), u_solution, _problem(OptimizationSense.MAXIMIZE))


def test_u_channel_unchanged_for_minimize():
    """Regression: the path that legitimately consumes the coefficient still resolves it."""
    u_solution = np.zeros((NT + 1, N, N))
    u_solution[:] = np.add.outer(np.linspace(0.0, 1.0, N) ** 2, np.zeros(N))

    result = solve_fp_nd_full_system(_uniform_density(), u_solution, _problem())

    assert np.isfinite(result).all()
    assert result[-1].sum() == pytest.approx(1.0, abs=1e-9)


# --- the accept-list must be truthful ---------------------------------------


def test_consuming_scheme_actually_honors_the_velocity():
    """Pins that the accept-list is truthful: a different velocity must change the answer.

    Without this, the guard could 'pass' by whitelisting a scheme that also ignores
    the parameter -- rejecting the honest schemes and silently dropping the velocity
    on the accepted one.
    """
    still = solve_fp_nd_full_system(
        _uniform_density(), None, _problem(), velocity_field=_velocity(), advection_scheme="divergence_upwind"
    )
    moving = solve_fp_nd_full_system(
        _uniform_density(),
        None,
        _problem(),
        velocity_field=_velocity(vx=2.5, vy=-1.7),
        advection_scheme="divergence_upwind",
    )

    assert not np.array_equal(still, moving), "velocity_field was discarded by the accepted scheme"
    assert np.abs(still - moving).max() > 1e-3

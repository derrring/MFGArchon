"""The network FP path stops instead of clipping-then-normalising to unit mass (#1683).

It clipped negatives to zero and then divided by the total, so the returned density was
non-negative **and** exactly unit-mass whatever the step produced -- the two properties a
caller would check to decide the solve was healthy, both supplied by the repair rather
than by the physics.

Measured on a 5x5 grid network before the change:

    steep value field (U scale 50)   clips 19/20 steps, max  1.66% of present mass
    dt past the CFL limit            clips  3/3  steps, max 60.09%
    9x9 grid, D=1.0                  clips  9/10 steps, max 59.81%

all returning unit mass.

**Why the renormalisation stays here, unlike GFDM (#1752).** Measured, not assumed: with
the renormalisation removed this scheme drifts 0.00% on two configurations and 0.01% on a
clipping one, because the graph inflow-outflow form conserves by construction. It is not
masking a non-conservative scheme the way GFDM's was. It is also a named user-facing option
(`enforce_mass_conservation=`), already gated off for mass-changing node BCs by #1478, not
a silent internal repair. The clip was the silent part, and that is what is gated.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
from mfgarchon.extensions.topology import NetworkMFGProblem
from mfgarchon.geometry.graph.network_geometry import GridNetwork

igraph = pytest.importorskip("igraph")

NT = 20
T = 1.0


def _solve(width=5, height=5, Nt=NT, diffusion=0.1, u_scale=1.0):
    network = GridNetwork(width=width, height=height)
    network.create_network()
    problem = NetworkMFGProblem(geometry=network, T=T, Nt=Nt)
    solver = FPNetworkSolver(problem, diffusion_coefficient=diffusion)

    n = network.num_nodes
    m0 = np.zeros(n)
    m0[n // 2] = 1.0
    nodes = np.arange(n, dtype=float)
    u = np.tile(u_scale * (nodes - n / 2.0) ** 2 / n, (Nt + 1, 1))
    return solver.solve_fp_system(m0, u)


def test_a_steep_value_field_stops():
    """Clipped 1.66% of the present mass on 19 of 20 steps and returned unit mass."""
    with pytest.raises(ValueError, match="would fabricate"):
        _solve(u_scale=50.0)


def test_a_timestep_past_the_cfl_limit_stops():
    """Clipped 60.09% on every step. The solve already warns that dt > dt_stable here --
    it warned, continued, and then normalised the consequence away."""
    with pytest.raises(ValueError, match="would fabricate"):
        _solve(Nt=3, diffusion=1.0, u_scale=5.0)


def test_the_message_names_the_step_the_scheme_and_a_remedy():
    """A diagnostic that reports a defect without naming a next step is read as noise."""
    with pytest.raises(ValueError) as exc:
        _solve(u_scale=50.0)
    message = str(exc.value)
    assert "Network FP solve: at step" in message
    assert "scheme='explicit'" in message
    assert "dt_stable" in message, "the CFL warning is the first thing to check; name it"
    assert "scheme='implicit'" in message


def test_a_healthy_solve_still_runs():
    """Negative control. A threshold that rejected round-off would make this path unusable
    -- the failure mode opposite to the one #1683 fixes."""
    result = _solve()
    assert np.all(np.isfinite(result))
    assert result.min() >= 0.0
    assert abs(float(result[-1].sum()) - 1.0) < 1e-9


def test_the_scheme_conserves_mass_without_the_renormalisation():
    """Justifies keeping `enforce_mass_conservation`, which GFDM's equivalent did not.

    The point of #1683 is that a repair which forces conservation makes a broken solve look
    healthy. That argument only applies if the underlying scheme is not conservative. Here
    it is: running the raw steps with the clip but WITHOUT the division by total mass leaves
    the final mass at 1.000000. Removing the option would therefore delete a user-facing
    knob without exposing anything -- so it stays, on evidence rather than by analogy.

    If this ever fails, the network scheme has stopped conserving and
    `enforce_mass_conservation` has silently become the same mask GFDM's was (#1752).
    """
    network = GridNetwork(width=5, height=5)
    network.create_network()
    problem = NetworkMFGProblem(geometry=network, T=T, Nt=NT)
    solver = FPNetworkSolver(problem, diffusion_coefficient=0.1)

    n = network.num_nodes
    m = np.zeros(n)
    m[n // 2] = 1.0
    nodes = np.arange(n, dtype=float)
    u = np.tile((nodes - n / 2.0) ** 2 / n, (NT + 1, 1))

    for k in range(NT):
        solver._current_rates = solver._precompute_transition_rates(m, u[k], k * (T / NT))
        m = np.maximum(solver._explicit_step(m, u[k], k * (T / NT)), 0.0)

    assert abs(float(m.sum()) - 1.0) < 1e-4, (
        f"unnormalised mass {float(m.sum()):.6f}: the graph inflow-outflow form no longer "
        f"conserves, so enforce_mass_conservation is now hiding a scheme defect (cf. #1752)"
    )

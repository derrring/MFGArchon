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

**`enforce_mass_conservation` went with it.** The first version of this change kept the
option, on the grounds that the scheme conserves anyway so the option was harmless. That
reasoning is backwards: measuring the option to be inert is an argument for deleting it, not
for keeping it. Conservation is a property of the discretisation, and a post-hoc rescale
does not make a scheme conservative -- it makes one look conservative. Neither branch of the
flag was useful: when the scheme conserves it does nothing, and when it does not it forces
the output to lie. There is no setting of a boolean that fixes a discretisation.

It had a second use that was worse than the per-step one: it divided the caller's **initial
condition** by its own total, so a density of mass 5.0 silently became the evolution of a
density of mass 1.0. The FP equation is linear in m, so that is not bookkeeping -- it
answers a different question by a factor the caller never sees.

Both are gone. The drift is reported instead.
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


def test_the_remedy_is_scheme_aware_and_names_only_what_was_measured_to_work():
    """The first version shipped one constant remedy for both schemes, and it was wrong twice.

    On the implicit path it told the reader to check a `dt > dt_stable` warning -- but
    `dt_stable` is only computed under `if self.scheme == "explicit"`, so the attribute does
    not exist and the warning structurally cannot have been emitted. It also suggested
    "use scheme='implicit'" to a user already on it.

    On the explicit path, review measured the suggestions: switching to implicit makes it worse
    (it raises earlier on the same configuration). "Lowering the diffusion coefficient does not
    work" was my own over-correction -- re-review measured that it does, at D <= 1e-4, so the
    message now says so and gives the real reason not to reach for it: D is physics, not a
    discretisation knob.

    The implicit text was wrong in the same way the original was. `_implicit_step` is IMEX --
    only the diffusion is implicit, the drift term is explicit -- so it does carry a timestep
    restriction, and increasing Nt does fix it (Nt=3 stops at 10.991%, Nt=20 completes). My
    version asserted "a negative density here is the spatial operator, not the timestep" while
    opening with "Increase Nt", contradicting both the code and itself.
    """
    with pytest.raises(ValueError) as exc:
        _solve(u_scale=50.0)
    explicit = str(exc.value)
    assert "Network FP solve: at step" in explicit
    assert "scheme='explicit'" in explicit
    assert "Increase Nt" in explicit
    assert "dt_stable" not in explicit, "no dt_stable advice on a path where it was measured not to help"
    assert "physics rather than a discretisation knob" in explicit, (
        "lowering D does clear the gate; the message must give the real reason not to, not deny it"
    )

    network = GridNetwork(width=5, height=5)
    network.create_network()
    problem = NetworkMFGProblem(geometry=network, T=T, Nt=3)
    solver = FPNetworkSolver(problem, diffusion_coefficient=1.0, scheme="implicit")
    assert not hasattr(solver, "dt_stable"), "if this gains a dt_stable the remedy below is stale"
    n = network.num_nodes
    m0 = np.zeros(n)
    m0[n // 2] = 1.0
    with pytest.raises(ValueError) as exc:
        solver.solve_fp_system(m0, np.tile(5.0 * (np.arange(n, dtype=float) - n / 2.0) ** 2 / n, (4, 1)))
    implicit = str(exc.value)
    assert "scheme='implicit'" in implicit
    assert "IMEX" in implicit, "the drift term is explicit; the message must not deny the timestep"
    assert "Increase Nt" in implicit
    assert "not the timestep" not in implicit, "measured: Nt=3 stops, Nt=20 completes"
    assert "use scheme='implicit'" not in implicit, "do not advise a scheme the caller is already on"


def test_a_healthy_solve_still_runs():
    """Negative control. A threshold that rejected round-off would make this path unusable
    -- the failure mode opposite to the one #1683 fixes."""
    result = _solve()
    assert np.all(np.isfinite(result))
    assert result.min() >= 0.0
    assert abs(float(result[-1].sum()) - 1.0) < 1e-9


def test_the_scheme_conserves_without_anything_enforcing_it():
    """Conservation is now a measured property of the scheme, not something imposed.

    The graph inflow-outflow form conserves by construction, which is exactly why the flag
    that used to divide by the total was pointless. This asserts the property directly, on
    the public path with nothing rescaling the output.

    If this ever fails, the scheme has stopped conserving -- and that is now visible, which
    is the whole point. It is not a reason to reinstate the division.
    """
    result = _solve()
    assert abs(float(result[-1].sum()) - 1.0) < 1e-4, (
        f"final mass {float(result[-1].sum()):.6f} from a unit initial mass: the graph "
        f"inflow-outflow form no longer conserves"
    )


def test_the_initial_condition_is_no_longer_silently_normalised():
    """It divided M[0] by its own total, so mass 5.0 became the evolution of mass 1.0.

    The FP equation is linear in m: rescaling the initial condition rescales the entire
    answer. A caller who passed an unnormalised density got a solve of a different problem,
    at a factor they were never told about.
    """
    network = GridNetwork(width=5, height=5)
    network.create_network()
    problem = NetworkMFGProblem(geometry=network, T=T, Nt=NT)
    solver = FPNetworkSolver(problem, diffusion_coefficient=0.1)

    n = network.num_nodes
    m0 = np.zeros(n)
    m0[n // 2] = 5.0
    nodes = np.arange(n, dtype=float)
    u = np.tile((nodes - n / 2.0) ** 2 / n, (NT + 1, 1))

    result = solver.solve_fp_system(m0, u)
    assert abs(float(result[0].sum()) - 5.0) < 1e-12, "the initial condition was rewritten"
    assert abs(float(result[-1].sum()) - 5.0) < 1e-3, (
        f"final mass {float(result[-1].sum()):.6f}: a linear equation must carry the caller's "
        f"normalisation through, not snap it to 1.0"
    )


def test_the_removed_flag_raises_on_both_values():
    """A user who passed False was asking for what is now the only behaviour.

    Accepting either value silently would hide that the semantics moved, which is the
    failure mode this whole issue is about.
    """
    network = GridNetwork(width=3, height=3)
    network.create_network()
    problem = NetworkMFGProblem(geometry=network, T=T, Nt=5)
    for value in (True, False):
        with pytest.raises(NotImplementedError, match="Issue #1683"):
            FPNetworkSolver(problem, enforce_mass_conservation=value)


def test_a_drift_that_does_happen_is_reported(caplog):
    """The other half of removing the division: if it does not conserve, say so.

    Reported through the logger, not `warnings.warn`, for two measured reasons found in
    review. `solve_fp_system` carries two `@deprecated_parameter` decorators, so
    `stacklevel=2` resolved to `utils/deprecation.py` instead of the caller. And the drift
    value sits in the message text, so `warnings`' dedup keyed on a string that changed every
    Picard iteration -- a 15-iteration coupled solve with an absorbing node emitted 15
    warnings for behaviour that is correct. The GFDM sibling reports its drift the same way.

    Driven by a perturbation rather than a broken scheme: unpatched, this configuration drifts
    exactly 0.0 and logs nothing, so the assertion below is about the report and not about the
    configuration happening to be noisy.
    """
    import logging

    network = GridNetwork(width=5, height=5)
    network.create_network()
    problem = NetworkMFGProblem(geometry=network, T=T, Nt=NT)
    solver = FPNetworkSolver(problem, diffusion_coefficient=0.1)

    n = network.num_nodes
    m0 = np.zeros(n)
    m0[n // 2] = 1.0
    nodes = np.arange(n, dtype=float)
    u = np.tile((nodes - n / 2.0) ** 2 / n, (NT + 1, 1))

    original = solver._explicit_step

    def leaky(m, u_cur, t):
        return 0.99 * original(m, u_cur, t)

    logger_name = "mfgarchon.alg.numerical.network_solvers.fp_network"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        solver.solve_fp_system(m0, u)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "unpatched, this configuration conserves exactly -- if it already logs, the test below "
        "proves nothing about the perturbation"
    )

    solver._explicit_step = leaky
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=logger_name):
        solver.solve_fp_system(m0, u)
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert records, "the drift was not reported"
    assert "total mass changed by" in records[0].getMessage()


def test_the_threshold_admits_this_scheme_s_own_discretisation_noise():
    """The shared default of 1e-8 rejected solves whose honest answer is a 0.006% drift.

    Review caught this PR citing exactly such a configuration as evidence that the scheme
    conserves -- while the shipped code refused to run it. The shared gate's premise ("round-off
    gives ~1e-15, a failed scheme gives O(1), there is no interesting regime between them") is
    false for this scheme: across 144 configurations the fabricated fraction forms a continuous
    ladder from 1e-9 to O(1).

    The measured gap this threshold sits in:

        true final drift < 1e-3  (honest)  -- fabricated at most 6.664e-06
        true final drift >= 1e-2 (broken)  -- fabricated at least 9.179e-04

    This test guards the lower end, which is the side a tightened threshold would break and
    the side no other test here covers. `test_a_steep_value_field_stops` guards the upper end.

    **No lower bound is asserted, deliberately.** An earlier version required
    `1e-6 < drift < 1e-3` and would have failed on *improvement*: the drift is a step function
    of how many steps happened to clip, not a smooth curve, so neighbouring configurations give
    1.1e-16 (u=8), 6.1e-08 (u=9), 5.8e-05 (u=10, pinned here) and 9.0e-04 (u=15). Two steps of
    Nt in the accurate direction drops it three orders. A test that goes red when the scheme
    gets better, saying "re-measure the threshold", is a false alarm dressed as diligence.

    What must hold is that this configuration **completes**: it is the one review found the
    shipped gate refusing while the PR body cited it as evidence.
    """
    result = _solve(u_scale=10.0)
    drift = abs(float(result[-1].sum()) - 1.0)
    assert drift < 1e-3, (
        f"drift {drift:.3e} is above the honest band this threshold was measured against; "
        f"the scheme changed and the threshold needs re-measuring, not this assertion widening"
    )


def test_the_drift_report_does_not_repeat_across_a_coupled_solve(caplog):
    """A coupled solve calls this once per Picard iteration, and neither channel dedups.

    `warnings.warn` keys its dedup on the message text, which carries the drift value and so
    differs every iteration; `logging` does not dedup at all. Both were measured emitting one
    record per call -- 15 iterations, 15 records, for behaviour that is correct on an absorbing
    node. My first fix diagnosed that correctly and then switched channels, which did not
    change the count and was strictly worse on identical repeats.

    A drift that grows is worth another line; one that repeats is not.
    """
    import logging

    network = GridNetwork(width=5, height=5)
    network.create_network()
    problem = NetworkMFGProblem(geometry=network, T=T, Nt=NT)
    solver = FPNetworkSolver(problem, diffusion_coefficient=0.1)

    n = network.num_nodes
    m0 = np.zeros(n)
    m0[n // 2] = 1.0
    u = np.tile((np.arange(n, dtype=float) - n / 2.0) ** 2 / n, (NT + 1, 1))

    original = solver._explicit_step
    solver._explicit_step = lambda m, u_cur, t: 0.99 * original(m, u_cur, t)

    logger_name = "mfgarchon.alg.numerical.network_solvers.fp_network"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        for _ in range(15):
            solver.solve_fp_system(m0, u)
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(records) == 1, (
        f"{len(records)} records from 15 identical solves; the drift does not grow, so after "
        f"the first there is nothing new to say"
    )


def test_a_growing_drift_is_still_reported(caplog):
    """The suppression must not swallow an escalation, which is the case worth seeing.

    Without this, the guard above could be satisfied by never reporting again -- the failure
    mode that made the deleted `_clip_warned` latch in the SL solver a defect rather than a fix.
    """
    import logging

    network = GridNetwork(width=5, height=5)
    network.create_network()
    problem = NetworkMFGProblem(geometry=network, T=T, Nt=NT)
    solver = FPNetworkSolver(problem, diffusion_coefficient=0.1)

    n = network.num_nodes
    m0 = np.zeros(n)
    m0[n // 2] = 1.0
    u = np.tile((np.arange(n, dtype=float) - n / 2.0) ** 2 / n, (NT + 1, 1))
    original = solver._explicit_step

    logger_name = "mfgarchon.alg.numerical.network_solvers.fp_network"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        for factor in (0.999, 0.99, 0.9):
            solver._explicit_step = (lambda f: lambda m, u_cur, t: f * original(m, u_cur, t))(factor)
            solver.solve_fp_system(m0, u)
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(records) == 3, f"{len(records)} records: each worsening drift must be reported"

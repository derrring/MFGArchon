"""The GPU particle path refuses an absorbing wall instead of reflecting it silently. #1910

`_solve_fp_system_gpu`'s docstring has carried "Segment-aware absorbing BC not yet implemented"
since #535 Phase 1, and the code matched it: `_needs_segment_aware_bc()` is consulted in
`_solve_fp_system_cpu`, `_solve_fp_system_cpu_nd` and `_solve_fp_system_callable_drift`, and
nowhere in the GPU method. The absorbing branch was unreachable there.

Measured before the refusal, 2000 particles, `dirichlet_bc(0.0)`, seed 12345:

    numpy  no_flux       absorbed 0
    numpy  dirichlet(0)  absorbed 4      <- the wall works
    torch  no_flux       absorbed 0
    torch  dirichlet(0)  absorbed 0      <- and returns a finite non-negative density of mass ~1

**Only the numpy pair is evidence, and the asymmetry is the point.** `total_absorbed` is
incremented in `_apply_boundary_conditions_segment_aware` and
`_apply_boundary_conditions_with_flux_limits`, both CPU-only, and `_solve_fp_system_gpu` neither
increments nor resets it. So the two torch rows read 0 whatever the wall does: they cannot
distinguish "reflected the particles" from "does not count". What established that the torch path
ignored the BC is the numpy contrast (0 against 4) plus the finite mass-~1 density torch returned
for a wall that should have removed particles. A caller asking for an absorbing wall got a
reflecting one and nothing said so.

This is not the implementation -- the GPU loop still cannot remove particles. It is the refusal,
which is what turns a known limitation from a wrong answer into an error.

ADMISSION (#2257). Class 3, a defect pin. The retirement condition is stated where it can fire:
when the GPU loop learns to remove particles the refusal stops being raised, this file goes red on
`test_the_gpu_path_refuses_an_absorbing_wall`, and the failure message says to delete it rather
than to restore the raise.

**A pin whose trigger is "nothing was raised" fires on more than the fix.** Whether
`_solve_fp_system_gpu` runs at all is decided by `strategy_selector.select_strategy(...)` and the
`current_strategy.name == "cpu"` dispatch, not by `backend="torch"`: a threshold change that routes
2000 particles to the CPU path -- which honours the absorbing wall -- also produces no raise, and
would print the retirement message for a guard that is still live. So each test here asserts that
the GPU strategy was actually selected -- in a `finally`, so the check runs on the raising path and
the non-raising path alike and precedes the retirement -- and the retirement text states what was
observed rather than declaring the capability landed.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc

_NX, _NT = 41, 15

_OBSERVED = "The GPU particle path did not refuse an absorbing wall."

_CAUSES = {
    "the GPU loop now absorbs": (
        "this pin has done its job. Do NOT restore the raise: delete this file and replace it with a "
        "test that the torch backend REMOVES particles at a Dirichlet wall, using "
        "`test_the_cpu_path_absorbs_at_the_same_wall` below as the oracle for how many. See #1910"
    ),
}


def _problem(bc):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_NX], boundary_conditions=bc)
    return MFGProblem(
        geometry=grid,
        T=0.5,
        Nt=_NT,
        sigma=0.15,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-50 * (x - 0.5) ** 2),
            u_terminal=lambda x: 0.0 * x,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        ),
    )


def _solver(backend: str, bc) -> FPParticleSolver:
    np.random.seed(12345)
    return FPParticleSolver(_problem(bc), num_particles=2000, kde_bandwidth=0.1, backend=backend)


def _solve(solver: FPParticleSolver) -> np.ndarray:
    x = np.linspace(0.0, 1.0, _NX)
    return solver.solve_fp_system(np.exp(-50 * (x - 0.5) ** 2), np.zeros((_NT + 1, _NX)))


def _assert_gpu_path_was_taken(solver: FPParticleSolver) -> None:
    """The pin's premise: this solve reached `_solve_fp_system_gpu`.

    `solve_fp_system` sets `current_strategy` before dispatching on `current_strategy.name`, so the
    attribute is populated even when the GPU branch then raises. Without this, a selector change
    that sends the solve to the CPU path is indistinguishable from the capability landing.
    """
    assert solver.current_strategy is not None, "no strategy recorded; solve_fp_system did not dispatch"
    assert solver.current_strategy.name != "cpu", (
        f"routed to the {solver.current_strategy.name!r} strategy, not the GPU path -- this pin's "
        f"premise does not hold and its result says nothing about the GPU absorbing branch"
    )


def test_the_gpu_path_refuses_an_absorbing_wall(still_refused):
    pytest.importorskip("torch")
    solver = _solver("torch", dirichlet_bc(0.0, dimension=1))
    # `premise=` rather than a line after the block. `still_refused` ends in `pytest.fail`, which
    # propagates out of the context manager, so a trailing assertion never runs on exactly the path
    # whose message would claim it had -- which is what this pin shipped with, once (#2288). The
    # fixture runs the premise in a `finally` inside the block, so it fires whether the solve raised
    # or not and before the absence is interpreted. The strategy is only recorded during the solve,
    # so it cannot be checked before. `causes` therefore has one entry: the premise excludes the
    # other, which is why the fixture accepts a single cause here and refuses one without it.
    with still_refused(
        "segment-aware absorbing",
        observed=_OBSERVED,
        causes=_CAUSES,
        premise=lambda: _assert_gpu_path_was_taken(solver),
        premise_establishes="the GPU strategy was selected, so this solve reached "
        "`_solve_fp_system_gpu` rather than the CPU path, which honours the wall",
    ):
        _solve(solver)


def test_the_gpu_path_still_runs_a_uniform_wall():
    """The refusal must be scoped to the BC it cannot honour, not to the backend.

    `total_absorbed` is deliberately not asserted here: it is a CPU-only counter (see the module
    docstring), so `== 0` on this path is the constructor's initial value and cannot fail.

    `_assert_gpu_path_was_taken` is what this test discriminates on -- it fails when the selector
    stops choosing the GPU strategy. The three assertions after it are admitted as happy-path
    checks and no mutant is offered for them: for a Gaussian KDE, non-negativity and a non-zero
    final row are close to structurally guaranteed, so only `isfinite` would plausibly catch
    anything, and only a blow-up. They say the solve returned something usable, not that it
    returned the right thing.
    """
    pytest.importorskip("torch")
    solver = _solver("torch", no_flux_bc(dimension=1))
    M = _solve(solver)
    _assert_gpu_path_was_taken(solver)

    assert np.all(np.isfinite(M)), "the GPU loop returned a non-finite density"
    assert np.all(M >= 0.0), "a KDE density went negative"
    assert M[-1].sum() > 0.0, "the final density is identically zero; no particles survived a reflecting wall"


def test_the_cpu_path_absorbs_at_the_same_wall():
    """The oracle for the refusal: the same wall on the backend that implements it.

    Without this, the refusal above could be pinning a wall that removes nothing on either backend,
    which is a different (and duller) fact. Both assertions can fail here, because the CPU path is
    the one that maintains the counter.

    The 4 particles quoted in the module docstring are deterministic, not a sample: `_solver` calls
    `np.random.seed(12345)` before construction and `FPParticleSolver` falls back to the global
    `np.random` when given no seed. The count is a fixture artifact, not a property of the wall --
    what is asserted is the sign.
    """
    solver = _solver("numpy", dirichlet_bc(0.0, dimension=1))
    _solve(solver)
    assert solver.total_absorbed > 0, "the absorbing wall removes nothing on CPU either"

    control = _solver("numpy", no_flux_bc(dimension=1))
    _solve(control)
    assert control.total_absorbed == 0, "a reflecting wall absorbed particles; the counter is wrong"

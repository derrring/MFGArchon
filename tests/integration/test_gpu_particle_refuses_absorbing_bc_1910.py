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

`no_flux` reading 0 on both backends is the positive control: the difference is the BC type, not
the backend. A caller asking for an absorbing wall got a reflecting one and nothing said so.

This is not the implementation -- the GPU loop still cannot remove particles. It is the refusal,
which is what turns a known limitation from a wrong answer into an error.

ADMISSION (#2257). Class 3, a defect pin, and the retirement condition is stated where it can fire:
when the GPU loop learns to remove particles the refusal stops being raised, this file goes red on
`test_the_gpu_path_refuses_an_absorbing_wall`, and the failure message says to delete it rather than
to restore the raise. That is the whole point of the class -- a pin whose retirement is invisible
becomes an obstacle to the fix it was written to demand.
"""

from __future__ import annotations

import contextlib

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc

pytestmark = pytest.mark.filterwarnings("ignore")

_NX, _NT = 41, 15

_RETIREMENT = """RETIREMENT CONDITION MET -- the GPU particle loop no longer refuses an absorbing
wall, which is what this pin existed to demand. Do NOT restore the raise. Delete this file and
replace it with a test of the behaviour that took its place: that the torch backend REMOVES
particles at a Dirichlet wall, with `test_the_cpu_path_absorbs_at_the_same_wall` below as the
oracle for how many. See #1910."""


@contextlib.contextmanager
def _still_refused(match: str):
    """Assert the refusal, and make its disappearance an instruction rather than a bare red.

    `pytest.raises` reports "DID NOT RAISE", which tells a future reader that something broke but
    not that the something is this pin's own success condition.
    """
    try:
        yield
    except NotImplementedError as exc:
        assert match in str(exc), f"refused, but not for the pinned reason: {exc}"
    else:
        pytest.fail(_RETIREMENT)


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


def _solve(backend: str, bc):
    np.random.seed(12345)
    solver = FPParticleSolver(_problem(bc), num_particles=2000, kde_bandwidth=0.1, backend=backend)
    x = np.linspace(0.0, 1.0, _NX)
    M = solver.solve_fp_system(np.exp(-50 * (x - 0.5) ** 2), np.zeros((_NT + 1, _NX)))
    return solver, M


def test_the_gpu_path_refuses_an_absorbing_wall():
    pytest.importorskip("torch")
    with _still_refused("segment-aware absorbing"):
        _solve("torch", dirichlet_bc(0.0, dimension=1))


def test_the_gpu_path_still_runs_a_uniform_wall():
    """The refusal must be scoped to the BC it cannot honour, not to the backend."""
    pytest.importorskip("torch")
    solver, M = _solve("torch", no_flux_bc(dimension=1))
    assert solver.total_absorbed == 0
    assert np.all(np.isfinite(M))


def test_the_cpu_path_absorbs_at_the_same_wall():
    """The external oracle. Without it, the refusal above could be pinning a wall that removes
    nothing on either backend, which is a different (and duller) fact."""
    solver, _ = _solve("numpy", dirichlet_bc(0.0, dimension=1))
    assert solver.total_absorbed > 0, "the absorbing wall removes nothing on CPU either"

    control, _ = _solve("numpy", no_flux_bc(dimension=1))
    assert control.total_absorbed == 0, "a reflecting wall absorbed particles; the counter is wrong"

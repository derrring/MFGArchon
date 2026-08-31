"""`CommonNoiseMFGSolver.solve()` completes a conditional solve, and the noise reaches it (#2191).

Before this, the class could not run at all. Two independent blockers, one behind the other:

1. `StochasticMFGProblem.create_conditional_problem` built an EMPTY `MFGComponents` and then
   attached `hamiltonian_func` / `hamiltonian_dm_func` to it. Neither is a field of
   `MFGComponents`, so those lines set attributes nothing reads -- and the constructor validates
   that a Hamiltonian or Lagrangian is present, so it raised before reaching them.
2. Behind that, the default `conditional_solver_factory` returned `prob.solve(verbose=False)` -- a
   RESULT -- while its declared type is `Callable[[MFGProblem], MFGSolverProtocol]` and the caller
   invokes `.solve()` on what it gets back.

Neither was ever observed. Three test files mention common noise; measured, all of them stop at
construction, and #1684 item 4's `result.convergence_achieved` read sat unreached on the same dead
path for ten months.

WHAT THESE CATCH, and why the second test is the load-bearing one. A test that only asserted
`solve()` returns would pass on a conditional problem whose Hamiltonian ignores the noise entirely
-- and that is not hypothetical: the obvious fixture, `lambda x, p, m, theta: 0.5 * p**2 + 0.1 * m`,
has no theta in it, runs clean, and reports `mc_error_u` of exactly 0.0 because every sample
produces the same solve. It would certify a `create_conditional_problem` that silently dropped
theta. So the fixture here makes the Hamiltonian depend on theta and asserts the samples DIFFER.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver
from mfgarchon.core import MFGComponents
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_NX, _NT = 22, 11


def _problem(conditional_hamiltonian):
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.1 * np.asarray(m, dtype=float),
        coupling_dm=lambda m: 0.1 * np.ones_like(np.asarray(m, dtype=float)),
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: np.zeros_like(np.asarray(x, dtype=float)),
        m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
    )
    return StochasticMFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_NX], boundary_conditions=no_flux_bc(dimension=1)),
        T=0.5,
        Nt=_NT,
        noise_process=OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1),
        conditional_hamiltonian=conditional_hamiltonian,
        components=components,
    )


@pytest.mark.slow
def test_solve_completes_and_returns_a_usable_result():
    """The plain fact the class could not deliver: a solve that finishes."""
    problem = _problem(lambda x, p, m, theta: 0.5 * p**2 + 0.1 * m)
    result = CommonNoiseMFGSolver(problem, num_noise_samples=2, variance_reduction=False, parallel=False, seed=7).solve(
        verbose=False
    )

    for name in ("u_mean", "m_mean"):
        arr = np.asarray(getattr(result, name))
        assert arr.shape == (_NT + 1, _NX), f"{name} has shape {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{name} is not finite"


@pytest.mark.slow
def test_the_noise_actually_reaches_the_conditional_solve():
    """LOAD-BEARING. A theta-independent Hamiltonian cannot distinguish this from a broken one.

    `H = 0.5 p^2 + theta * m` makes the coupling strength the noise. Two different realisations of
    an OU path must therefore produce two different value functions, and the Monte-Carlo spread
    must be non-zero. If `create_conditional_problem` dropped theta -- which is exactly what the
    dead `hamiltonian_func` assignment did -- every sample would coincide and `mc_error_u` would be
    0.0 while every other assertion in this file still passed.
    """
    problem = _problem(lambda x, p, m, theta: 0.5 * p**2 + theta * m)
    result = CommonNoiseMFGSolver(
        problem, num_noise_samples=4, variance_reduction=False, parallel=False, seed=11
    ).solve(verbose=False)

    paths = np.asarray(result.noise_paths)
    assert paths.shape[0] == 4
    assert np.ptp(paths) > 0, "the fixture is wrong if every sampled noise path is identical"

    spread = float(np.max(np.asarray(result.u_std)))
    assert spread > 0, (
        "every noise sample produced the same value function, so theta never reached the "
        "conditional solve -- which is what the pre-#2191 code did by attaching the Hamiltonian "
        "to a field MFGComponents does not have"
    )

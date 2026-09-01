"""`CommonNoiseMFGSolver.solve()` completes a conditional solve, and the noise reaches it (#2191).

NOT MARKED `slow`, deliberately (#2197 review). They were, and `scripts/local_ci.sh` -- the
authoritative gate -- runs `-m "not slow ..."`, so the only two tests pinning a priority:high
"cannot run" fix were deselected by it and reached CI only through the nightly. Measured runtimes
are 11.96 s and 6.22 s, against the marker's own definition in `pytest.ini:32`: "may take >30
seconds to complete". A pin the gate does not run is not a pin.

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
has no theta in it, runs clean, and reports an `mc_error_u` indistinguishable from zero (exactly
0.0 for some (K, values) pairs and ~1e-17 for others -- see the note at the assertion) because every sample
produces the same solve. It would certify a `create_conditional_problem` that silently dropped
theta. So the fixture here makes the Hamiltonian depend on theta and asserts the samples DIFFER.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver
from mfgarchon.core import MFGComponents
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_NX, _NT = 22, 11


def _problem(conditional_hamiltonian, u_terminal=None):
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.1 * np.asarray(m, dtype=float),
        coupling_dm=lambda m: 0.1 * np.ones_like(np.asarray(m, dtype=float)),
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=u_terminal or (lambda x: np.zeros_like(np.asarray(x, dtype=float))),
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

    # Asserted on the SAMPLES, not on their std, and against a physical threshold rather than
    # zero (#2197 review). `max(u_std) > 0` looked equivalent and was not: it is a float
    # zero-threshold on a sample standard deviation, and over K bit-identical values np.std is
    # EXACTLY 0.0 only when the arithmetic happens to be exact -- which depends on BOTH K and the
    # values being averaged. An earlier version of this comment published a table of "safe" K, and
    # that table was itself measured on one scalar and does not survive a realistic field: K = 5
    # gives 0.0 for one set of values and 7.8e-18 for another. There is no table; that is the point.
    # What is certain is that some (K, values) pairs give exactly 0.0 and some do not, so the old
    # assertion's discriminating power was an accident, and anyone retuning K for runtime or
    # Monte-Carlo smoothness could have disarmed the only guard on the noise coupling, silently.
    #
    # The pairwise sample separation has no such accident: when theta never reaches the conditional
    # problem the samples are bit-identical, so this is exactly 0.0. Measured here on the fix,
    # 0.0357 against 0.0 for a theta-dropping source -- any threshold in (1e-15, 1e-3) separates
    # them, and 1e-3 is chosen so the test says "the solutions genuinely differ", not "the floats
    # are not bit-equal".
    samples = np.asarray(result.u_samples)
    separation = float(np.max(np.abs(samples - samples[0])))
    assert separation > 1e-3, (
        f"the K noise samples produced value functions that differ by at most {separation:.3e}, so "
        f"theta never reached the conditional solve -- which is what the pre-#2191 code did by "
        f"attaching the Hamiltonian to a field MFGComponents does not have"
    )


def test_the_parent_terminal_condition_reaches_the_conditional_solve():
    """The conditional problem inherits u_terminal, and #2197 review found it did not.

    `StochasticMFGProblem.__init__` normalised the terminal cost as `terminal_cost or g` --
    and `MFGProblem` defines NEITHER, measured `hasattr` False for both -- so the chain was
    unconditionally None and `g_conditional` fell through to a literal `0.0`. Every conditional
    solve used u_T == 0 whatever the parent said, and because the solve then completed and
    converged, the result was a confidently wrong number rather than a crash.

    The control is what makes this a defect and not a design choice: `m_initial` propagates from
    the same `components` object and DOES move the answer.
    """
    x = np.linspace(0.0, 1.0, 5)

    def _with_terminal(u_terminal):
        problem = _problem(lambda x_, p, m, theta: 0.5 * p**2 + theta * m, u_terminal=u_terminal)
        conditional = problem.create_conditional_problem(np.linspace(0.0, 1.0, problem.Nt + 1))
        return np.array([float(conditional.components.u_terminal(np.array([xi]))) for xi in x])

    quadratic = _with_terminal(lambda p: 0.5 * float(np.asarray(p).ravel()[0]) ** 2)
    zero = _with_terminal(lambda p: 0.0)

    np.testing.assert_allclose(quadratic, 0.5 * x**2, rtol=0, atol=1e-12)
    # CONTROL: a genuinely zero terminal condition must still come through as zero, so the
    # assertion above is reading the parent's function and not merely finding something non-zero.
    np.testing.assert_allclose(zero, np.zeros_like(x), rtol=0, atol=1e-12)

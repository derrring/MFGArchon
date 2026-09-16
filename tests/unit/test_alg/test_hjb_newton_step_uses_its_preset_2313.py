"""The Newton step's Jacobian is built with the same numerical Hamiltonian as its residual (#2313).

`newton_hjb_step` calls `compute_hjb_residual` and `compute_hjb_jacobian` in turn and returns
``U + spsolve(J, -F)``. Both must read the same preset: a Jacobian of the other preset is a J/r mismatch at every
discrete local maximum, which makes Newton solve the wrong linear system while still looking convergent. Dropping
`numerical_hamiltonian=` from either call left the whole suite green before this test existed (review of #2340,
blocker 2).

Oracle: the definition of a Newton step. ``U_next`` must equal ``U + spsolve(J, -F)`` with ``F`` and ``J`` assembled
from the same preset, and the two presets' predictions must differ on this state -- otherwise the pin is vacuous.

What this oracle does and does not check. It calls the same `compute_hjb_residual`, `compute_hjb_jacobian` and
`spsolve` the implementation calls, so it cannot see an error inside any of them; it sees only which preset each call
was handed, which is what #2340 threads. The discriminating half is `separation`, the distance to the step the other
preset predicts: 1.186e-02 here, so a Jacobian built with the wrong preset moves the step by eight orders of
magnitude more than the 1e-10 tolerance admits (measured at 8debbbbc). The presets' arithmetic is checked against
outside oracles elsewhere -- the integral definition of H in `test_stencils.py`, and the FP transpose in
`test_hjb_fp_adjoint_numerical_hamiltonian_2313.py`.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np
import scipy.sparse as sparse

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import compute_hjb_jacobian, compute_hjb_residual, newton_hjb_step
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_N = 21
_X = np.linspace(0.0, 1.0, _N)
# Interior local maxima along the axis, where the presets differ: cos has maxima at x = 0, 1/2, 1.
_STATE = 0.4 * np.cos(4 * np.pi * _X) + 0.1 * _X
_M = 1.0 + 0.5 * np.cos(2 * np.pi * _X + 1.1)


def _problem() -> MFGProblem:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=0.5,
            sigma=0.3,
            components=MFGComponents(
                m_initial=lambda z: 1.0,
                u_terminal=lambda z: 0.0,
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
                ),
            ),
        )


def _newton_step_by_definition(problem, bc, preset: str) -> np.ndarray:
    residual = np.asarray(
        compute_hjb_residual(
            _STATE, np.zeros(_N), _M, problem, 5, None, 0.3, True, bc=bc, numerical_hamiltonian=preset
        ),
        dtype=float,
    )
    jacobian = compute_hjb_jacobian(
        _STATE, _STATE, _M, problem, 5, None, 0.3, True, bc=bc, numerical_hamiltonian=preset
    )
    return _STATE + sparse.linalg.spsolve(jacobian, -residual)


@pytest.mark.parametrize("preset", ["engquist_osher", "rouy_tourin"])
def test_the_step_is_the_newton_step_of_its_own_preset(preset: str):
    problem = _problem()
    bc = problem.geometry.get_boundary_conditions()
    taken, _step_norm, _residual_norm = newton_hjb_step(
        _STATE, np.zeros(_N), _STATE, _M, problem, 5, None, 0.3, True, bc=bc, numerical_hamiltonian=preset
    )
    expected = _newton_step_by_definition(problem, bc, preset)
    other = "rouy_tourin" if preset == "engquist_osher" else "engquist_osher"
    separation = float(np.abs(expected - _newton_step_by_definition(problem, bc, other)).max())
    assert separation > 1e-3, f"the presets predict the same step ({separation:.3e}); nothing is discriminated"
    np.testing.assert_allclose(np.asarray(taken, dtype=float), expected, rtol=0, atol=1e-10)

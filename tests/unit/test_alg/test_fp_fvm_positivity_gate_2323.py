"""The FVM FP solver stops at the positivity gate the other FP solvers share (#2323).

It warned at an absolute ``min < -1e-12`` and returned the negative density, so whether a coupled result read as
valid depended on the density's units: on #2323's 2-D fixture at 51234208 the same solution, scaled by 0.1, passed
output validation while the unscaled one failed it. `clip_nonnegative_or_raise` measures negative mass as a fraction
of the mass present (#1671): it clips below ``MAX_CLIP_MASS_FABRICATION`` and stops the solve above it.

The fixture is a 2-D Gaussian in the potential ``sin 2 pi x + sin 2 pi y``, with equal speeds on both axes and no
diffusion. At 6000eff0, MUSCL returned min -5.563e-12 (8.1e-14 of the mass) and upwind returned min -1.624e+10, a
negative mass equal to the whole mass present.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _solve(reconstruction: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[17, 17], boundary_conditions=no_flux_bc(dimension=2)
            ),
            Nt=10,
            T=0.5,
            sigma=0.0,
            components=MFGComponents(
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
                ),
            ),
        )
    grid = problem.geometry
    x, y = np.meshgrid(*grid.coordinates, indexing="ij")
    m0 = np.exp(-40 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
    m0 /= float(grid.integrate(m0))
    potential = np.stack([np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)] * 11)
    return FPFVMSolver(problem, reconstruction=reconstruction).solve_fp_system(m0, potential_field=potential)


def test_negative_mass_below_the_gate_is_clipped():
    density = _solve("muscl")
    assert density.min() >= 0.0, f"min {density.min():.3e}: FVM returned a negative density (#2323)"


def test_2d_upwind_with_equal_axis_speeds_stops_at_the_gate():
    """RECORDED DEFECT, not a contract: the upwind sub-step exceeds its 2-D positivity bound (#2323, the scheme half).

    When #2323's scheme half brings this solve within the gate, it stops raising and the first assertion below
    fails: delete this test then.
    """
    error = None
    try:
        _solve("upwind")
    except ValueError as caught:
        error = caught
    assert error is not None, (
        "2-D FVM upwind no longer reaches the positivity gate: #2323's scheme half is fixed, remove this pin"
    )
    assert "would fabricate" in str(error), str(error)

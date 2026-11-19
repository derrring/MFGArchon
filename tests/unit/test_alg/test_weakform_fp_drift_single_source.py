"""Issue #1487 / #1420 (gotcha G-017): the weak-form MFG family's FP drift scale is ``1/control_cost``, the HJB
optimal-control scale. The meshless-Galerkin FP forms its drift from ``H.optimal_control``, and its paired HJB's
streamline diffusion reads ``fp_drift_coefficient``. The G-017 wrong-equilibrium bug was the weak-form family reading a
problem-level ``coupling_coefficient`` instead, which ``MFGProblem`` no longer takes (#2977).
"""

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.factory.scheme_factory import NumericalScheme, create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.pde_coefficients import fp_drift_coefficient


def _problem(control_cost: float) -> MFGProblem:
    geom = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    comp = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=control_cost),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    return MFGProblem(geometry=geom, T=0.2, Nt=5, volatility=0.3, components=comp)


def test_fp_drift_coefficient_sources_from_control_cost():
    # the drift scale is 1/control_cost, read from the Hamiltonian
    assert fp_drift_coefficient(_problem(1.0)) == 1.0
    assert fp_drift_coefficient(_problem(2.0)) == 0.5


def test_meshless_fp_advection_scales_as_one_over_the_control_cost():
    """The meshless-Galerkin FP advection operator (the Weak-GFDM paper's stack) is linear in its drift
    ``-grad(U)/control_cost`` when no streamline diffusion is added, so doubling the control cost halves it exactly.
    Before #1487 it scaled by a problem-level scalar that no control cost moved: wrong equilibrium, and
    A_FP != A_HJB^T."""
    cloud = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    cfg = {
        "collocation_points": cloud,
        "delta": 2.6 / np.sqrt(11),
        "degree": 2,
        "use_newton": True,
        "streamline_diffusion_scale": 0.0,
    }
    u = np.linspace(0.0, 1.0, 11) ** 2
    step = {"t": 0.0, "m_n": np.ones_like(u)}  # the library's control reads neither t nor m (#2903)
    _, fp_one = create_paired_solvers(_problem(1.0), NumericalScheme.MESHLESS_GALERKIN, hjb_config=cfg)
    _, fp_two = create_paired_solvers(_problem(2.0), NumericalScheme.MESHLESS_GALERKIN, hjb_config=cfg)
    a_one = fp_one._build_advection(u, 0.045, **step).toarray()
    a_two = fp_two._build_advection(u, 0.045, **step).toarray()
    assert np.abs(a_one).max() > 0.5
    np.testing.assert_array_equal(a_two, a_one / 2)

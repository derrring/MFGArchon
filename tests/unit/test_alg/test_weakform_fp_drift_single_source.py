"""Issue #1487 / #1420 (gotcha G-017): the weak-form MFG family (FEM + meshless-Galerkin) sources the
FP drift scale from ``fp_drift_coefficient``, which is ``1/control_cost``, the HJB optimal-control
scale. The G-017 wrong-equilibrium bug was the weak-form family reading a problem-level
``coupling_coefficient`` instead, which ``MFGProblem`` no longer takes (#2977).
"""

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
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

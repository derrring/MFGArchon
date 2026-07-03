"""Issue #1487 / #1420 (gotcha G-017): the weak-form MFG family (FEM + meshless-Galerkin) sources the
FP drift scale from ``fp_drift_coefficient`` (= ``1/control_cost``, the HJB optimal-control scale),
single-sourced — NOT the raw ``coupling_coefficient`` (which defaults to 0.5 and silently diverges).

Pins that the FP transports mass at the correct (HJB-control) scale, invariant to a diverging
``coupling_coefficient`` — the G-017 wrong-equilibrium bug was the weak-form family being the lone
holdout still reading ``coupling_coefficient`` directly.
"""

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.factory.scheme_factory import NumericalScheme, create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.pde_coefficients import fp_drift_coefficient


def _problem(coupling_coefficient: float) -> MFGProblem:
    geom = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    comp = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    return MFGProblem(geometry=geom, T=0.2, Nt=5, sigma=0.3, components=comp, coupling_coefficient=coupling_coefficient)


def test_fp_drift_coefficient_sources_from_control_cost_not_coupling():
    # control_cost=1.0 -> correct drift = 1/control_cost = 1.0, regardless of coupling_coefficient.
    assert fp_drift_coefficient(_problem(0.5)) == 1.0
    assert fp_drift_coefficient(_problem(2.0)) == 1.0


def test_meshless_fp_advection_is_single_sourced_from_control_cost():
    """The meshless-Galerkin FP advection operator (the Weak-GFDM paper's stack) must be invariant to
    ``coupling_coefficient`` (single-sourced from ``control_cost``). Before #1487 it scaled by
    ``coupling_coefficient`` -> wrong equilibrium and A_FP != A_HJB^T."""
    u = np.linspace(0.0, 1.0, 11)
    cloud = u.reshape(-1, 1)
    cfg = {
        "collocation_points": cloud,
        "delta": 2.6 / np.sqrt(11),
        "degree": 2,
        "use_newton": True,
        "streamline_diffusion_scale": 1.0,
    }
    _, fp_half = create_paired_solvers(_problem(0.5), NumericalScheme.MESHLESS_GALERKIN, hjb_config=cfg)
    _, fp_double = create_paired_solvers(_problem(2.0), NumericalScheme.MESHLESS_GALERKIN, hjb_config=cfg)
    c_half = fp_half._build_advection(u, 0.045).toarray()
    c_double = fp_double._build_advection(u, 0.045).toarray()
    assert np.abs(c_half).max() > 0.0
    assert np.allclose(c_half, c_double), (
        "FP advection must source the drift from control_cost (single source), not coupling_coefficient"
    )

"""Issue #1489: paired weak-form solvers must share the duality-critical config (esp.
streamline_diffusion_scale), or the SD block is added to one side only and A_FP = A_HJB^T breaks.
The factory fails loud on conflicting keys; the coupling iterator fails loud on a hand-built mismatch."""

from __future__ import annotations

import pytest

import numpy as np


def _problem_and_cloud():
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    geom = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    comp = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
        ),
    )
    prob = MFGProblem(geometry=geom, T=0.2, Nt=5, sigma=0.3, components=comp, coupling_coefficient=1.0)
    return prob, np.linspace(0.0, 1.0, 11).reshape(-1, 1)


def test_factory_fails_loud_on_conflicting_duality_key():
    from mfgarchon.factory.scheme_factory import NumericalScheme, create_paired_solvers

    prob, cloud = _problem_and_cloud()
    with pytest.raises(ValueError, match=r"streamline_diffusion|conflicting"):
        create_paired_solvers(
            prob,
            NumericalScheme.MESHLESS_GALERKIN,
            hjb_config={
                "collocation_points": cloud,
                "delta": 2.6 / np.sqrt(11),
                "use_newton": True,
                "streamline_diffusion_scale": 1.0,
            },
            fp_config={"streamline_diffusion_scale": 0.0},  # conflicts with the HJB value
        )


def test_iterator_fails_loud_on_hand_built_sd_mismatch():
    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
    from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver

    prob, cloud = _problem_and_cloud()
    delta = 2.6 / np.sqrt(11)
    hjb = MeshlessGalerkinHJBSolver(prob, cloud, delta=delta, use_newton=True, streamline_diffusion_scale=1.0)
    fp = MeshlessGalerkinFPSolver(prob, cloud, delta=delta, streamline_diffusion_scale=0.0)  # mismatched
    with pytest.raises(ValueError, match="streamline_diffusion"):
        FixedPointIterator(prob, hjb, fp)

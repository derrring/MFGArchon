"""Tests for the canonical FP drift contract (Issue #1043).

`drift_field` MEANS the advective velocity α* everywhere; solvers that instead take the value
function U expose it through `potential_field` and carry `_drift_convention == VALUE_FUNCTION`.
The weak-form family historically (mis)named its U input `drift_field`; it is renamed to
`potential_field` with a deprecation alias, which the Deprecation Policy requires be proven
equivalent.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.base_fp import DriftConvention
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
from mfgarchon.alg.numerical.weak_form_fp_solver import WeakFormFPSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def test_drift_convention_trait_values():
    """The velocity-taking solvers keep the VELOCITY default; the U-taking solvers are
    explicitly VALUE_FUNCTION (machine-readable contract for a future coupler dispatch).

    FPParticleSolver is VALUE_FUNCTION by default: its 1D path always takes U via `drift_field`
    and computes alpha = -coupling*grad(U), and the nD default does the same; only the per-call
    `drift_is_precomputed=True` (nD) flips it to VELOCITY. The class trait records the default
    (Issue #1043). Previously it silently inherited the base VELOCITY default and was untested."""
    assert FPFDMSolver._drift_convention is DriftConvention.VELOCITY
    assert FPGFDMSolver._drift_convention is DriftConvention.VELOCITY
    for cls in (
        WeakFormFPSolver,
        MeshlessGalerkinFPSolver,
        FPSLJacobianSolver,
        FPSLSolver,
        FPNetworkSolver,
        FPParticleSolver,
    ):
        assert cls._drift_convention is DriftConvention.VALUE_FUNCTION, cls.__name__


def _problem(sigma=0.3, n=15):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(
        hamiltonian=H, m_initial=lambda x: np.exp(-20 * (x - 0.5) ** 2), u_terminal=lambda x: 0.5 * (x - 0.5) ** 2
    )
    return MFGProblem(geometry=grid, components=comp, T=0.5, Nt=10, sigma=sigma, coupling_coefficient=0.5)


def test_weak_form_drift_field_alias_equivalent_to_potential_field():
    """Issue #1043 rename: passing the value function via the deprecated `drift_field` must warn
    and give byte-identical results to the canonical `potential_field` (Deprecation Policy)."""
    x = np.linspace(0.0, 1.0, 15)
    m0 = np.exp(-20 * (x - 0.5) ** 2)
    U = np.tile(0.5 * (x - 0.5) ** 2, (11, 1))  # (Nt+1, n) value function

    fp = MeshlessGalerkinFPSolver(_problem(), collocation_points=x[:, None], delta=3.5 / 14)
    traj_new = fp.solve_fp_system(m0, potential_field=U)
    with pytest.warns(DeprecationWarning, match="drift_field"):
        traj_old = fp.solve_fp_system(m0, drift_field=U)
    assert np.array_equal(traj_new, traj_old)


def test_weak_form_rejects_both_potential_and_drift():
    """Passing both the canonical and deprecated U inputs is a fail-loud error."""
    x = np.linspace(0.0, 1.0, 15)
    m0 = np.exp(-20 * (x - 0.5) ** 2)
    U = np.tile(0.5 * (x - 0.5) ** 2, (11, 1))
    fp = MeshlessGalerkinFPSolver(_problem(), collocation_points=x[:, None], delta=3.5 / 14)
    with pytest.warns(DeprecationWarning, match="drift_field"), pytest.raises(ValueError, match="potential_field"):
        fp.solve_fp_system(m0, potential_field=U, drift_field=U)

"""Issue #1456: BC-capability gate — solvers fail loud on a BCType they do not support.

The `BoundaryCapable` protocol (`geometry/boundary/protocols.py`) lets a solver declare
`_SUPPORTED_BC_TYPES`; `BaseMFGSolver._validate_bc_support` raises on an unsupported type at
construction instead of silently collapsing it to the solver's default (usually Neumann / no-flux)
— the BC-blindness class mapped in #1456. This pins the migrated solvers: `FPParticleSolver`
(already fail-fast → now a declared contract), `FPSLSolver` (silently collapsed Dirichlet/Robin to
its zero-flux Neumann stencil → now fails loud), and `HJBGFDMSolver` (declares
Dirichlet/Neumann/no-flux/Robin/periodic; rejects Reflecting/Extrapolation at construction — the
general-Robin / mixed-periodic sub-cases the row builder still enforces pass the type-level gate).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBGFDMSolver, HJBSemiLagrangianSolver, HJBWENOSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, neumann_bc, no_flux_bc, periodic_bc, robin_bc, uniform_bc
from mfgarchon.geometry.boundary.types import BCType

pytestmark = pytest.mark.filterwarnings("ignore")

N = 21


def _components():
    return MFGComponents(
        m_initial=lambda x: np.ones_like(x),
        u_terminal=lambda x: 0.0 * x,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )


def _problem(bc):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=bc)
    return MFGProblem(geometry=grid, T=0.2, Nt=10, sigma=0.3, components=_components())


def _pts():
    return np.linspace(0.0, 1.0, N).reshape(-1, 1)


# ---------------------------------------------------------------------------
# HJBGFDM — declares Dirichlet/Neumann/no-flux/Robin/periodic; Reflecting/Extrapolation
# (which no other test constructs, and the audit confirms it cannot honor) fail loud at
# construction. Periodic and general-Robin sub-cases pass the type-level gate and remain
# enforced by the row builder at solve time.
# ---------------------------------------------------------------------------


def test_hjb_gfdm_fails_loud_on_reflecting():
    with pytest.raises(NotImplementedError, match="does not support"):
        HJBGFDMSolver(_problem(uniform_bc(BCType.REFLECTING, dimension=1)), collocation_points=_pts(), delta=0.25)


@pytest.mark.parametrize("bc_factory", [no_flux_bc, dirichlet_bc, periodic_bc, robin_bc])
def test_hjb_gfdm_accepts_supported(bc_factory):
    HJBGFDMSolver(_problem(bc_factory(dimension=1)), collocation_points=_pts(), delta=0.25)  # must not raise


# ---------------------------------------------------------------------------
# FPSLSolver — supports no-flux / Neumann / periodic; the silent-Neumann-collapse
# of Dirichlet / Robin now fails loud (the headline #1456 flip).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bc", [dirichlet_bc(dimension=1), robin_bc(dimension=1)])
def test_fp_sl_fails_loud_on_unsupported(bc):
    with pytest.raises(NotImplementedError, match="does not support"):
        FPSLSolver(_problem(bc))


@pytest.mark.parametrize("bc_factory", [no_flux_bc, periodic_bc])
def test_fp_sl_accepts_supported(bc_factory):
    FPSLSolver(_problem(bc_factory(dimension=1)))  # must not raise


# ---------------------------------------------------------------------------
# HJB-WENO — WENO5 ghost buffers handle Dirichlet/Neumann/no-flux/periodic; Robin/Reflecting/
# Extrapolation (silently reflected/degraded in the ghost path) fail loud at construction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bc", [robin_bc(dimension=1), uniform_bc(BCType.REFLECTING, dimension=1)])
def test_hjb_weno_fails_loud_on_unsupported(bc):
    with pytest.raises(NotImplementedError, match="does not support"):
        HJBWENOSolver(_problem(bc))


@pytest.mark.parametrize("bc_factory", [no_flux_bc, neumann_bc, dirichlet_bc, periodic_bc])
def test_hjb_weno_accepts_supported(bc_factory):
    HJBWENOSolver(_problem(bc_factory(dimension=1)))  # must not raise


# ---------------------------------------------------------------------------
# HJB-FDM — assembles Dirichlet/Neumann/no-flux/periodic boundary rows; Robin/Reflecting/
# Extrapolation fail loud at construction (no test constructs HJB-FDM with them).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bc", [robin_bc(dimension=1), uniform_bc(BCType.REFLECTING, dimension=1)])
def test_hjb_fdm_fails_loud_on_unsupported(bc):
    with pytest.raises(NotImplementedError, match="does not support"):
        HJBFDMSolver(_problem(bc))


@pytest.mark.parametrize("bc_factory", [no_flux_bc, neumann_bc, dirichlet_bc, periodic_bc])
def test_hjb_fdm_accepts_supported(bc_factory):
    HJBFDMSolver(_problem(bc_factory(dimension=1)))  # must not raise


# ---------------------------------------------------------------------------
# FP-FVM — conservative FV: no-flux/Neumann/periodic; Robin/Reflecting/Extrapolation fail loud
# via the gate (Dirichlet has its own Issue #422 guard, tested in test_fvm_hjb_coupling).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bc", [robin_bc(dimension=1), uniform_bc(BCType.REFLECTING, dimension=1)])
def test_fp_fvm_fails_loud_on_unsupported(bc):
    with pytest.raises(NotImplementedError, match="does not support"):
        FPFVMSolver(_problem(bc))


@pytest.mark.parametrize("bc_factory", [no_flux_bc, neumann_bc, periodic_bc])
def test_fp_fvm_accepts_supported(bc_factory):
    FPFVMSolver(_problem(bc_factory(dimension=1)))  # must not raise


# ---------------------------------------------------------------------------
# FP-FDM — assembles Dirichlet/Neumann/no-flux/periodic boundary rows; Robin (no stencil,
# #1250) and Reflecting/Extrapolation fail loud at construction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bc", [robin_bc(dimension=1), uniform_bc(BCType.REFLECTING, dimension=1)])
def test_fp_fdm_fails_loud_on_unsupported(bc):
    with pytest.raises(NotImplementedError, match="does not support"):
        FPFDMSolver(_problem(bc))


@pytest.mark.parametrize("bc_factory", [no_flux_bc, neumann_bc, dirichlet_bc, periodic_bc])
def test_fp_fdm_accepts_supported(bc_factory):
    FPFDMSolver(_problem(bc_factory(dimension=1)))  # must not raise


# ---------------------------------------------------------------------------
# FPParticle — reflect (no-flux/Neumann/reflecting), wrap (periodic), absorb (Dirichlet);
# Robin is not represented and fails loud.
# ---------------------------------------------------------------------------


def test_fp_particle_fails_loud_on_robin():
    with pytest.raises(NotImplementedError, match="does not support"):
        FPParticleSolver(_problem(robin_bc(dimension=1)))


@pytest.mark.parametrize("bc_factory", [no_flux_bc, periodic_bc, dirichlet_bc])
def test_fp_particle_accepts_supported(bc_factory):
    FPParticleSolver(_problem(bc_factory(dimension=1)))  # must not raise (Dirichlet = absorbing)


# ---------------------------------------------------------------------------
# HJB-SL / FP-SL-Jacobian / FP-GFDM — zero-flux/periodic; Dirichlet/Robin (silently collapsed to
# Neumann / returned None — the audit's silent-mishandling cases) now fail loud at construction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bc", [dirichlet_bc(dimension=1), robin_bc(dimension=1)])
def test_hjb_sl_fails_loud_on_unsupported(bc):
    with pytest.raises(NotImplementedError, match="does not support"):
        HJBSemiLagrangianSolver(_problem(bc))


@pytest.mark.parametrize("bc_factory", [no_flux_bc, neumann_bc, periodic_bc])
def test_hjb_sl_accepts_supported(bc_factory):
    HJBSemiLagrangianSolver(_problem(bc_factory(dimension=1)))


@pytest.mark.parametrize("bc", [dirichlet_bc(dimension=1), robin_bc(dimension=1)])
def test_fp_sl_jacobian_fails_loud_on_unsupported(bc):
    with pytest.raises(NotImplementedError, match="does not support"):
        FPSLJacobianSolver(_problem(bc))


@pytest.mark.parametrize("bc_factory", [no_flux_bc, neumann_bc, periodic_bc])
def test_fp_sl_jacobian_accepts_supported(bc_factory):
    FPSLJacobianSolver(_problem(bc_factory(dimension=1)))


@pytest.mark.parametrize("bc", [dirichlet_bc(dimension=1), robin_bc(dimension=1)])
def test_fp_gfdm_fails_loud_on_unsupported(bc):
    pts = np.linspace(0.0, 1.0, N).reshape(-1, 1)
    with pytest.raises(NotImplementedError, match="does not support"):
        FPGFDMSolver(_problem(bc), collocation_points=pts, delta=0.25)


@pytest.mark.parametrize("bc_factory", [no_flux_bc, neumann_bc, periodic_bc])
def test_fp_gfdm_accepts_supported(bc_factory):
    pts = np.linspace(0.0, 1.0, N).reshape(-1, 1)
    FPGFDMSolver(_problem(bc_factory(dimension=1)), collocation_points=pts, delta=0.25)


# ---------------------------------------------------------------------------
# The shared gate is a no-op for None / the particle "periodic" string sentinel.
# ---------------------------------------------------------------------------


def test_validate_bc_support_noop_for_none_and_sentinel():
    solver = FPParticleSolver(_problem(no_flux_bc(dimension=1)))
    solver._validate_bc_support(None)  # None -> no-op
    solver._validate_bc_support("periodic")  # string sentinel -> no-op (not a BoundaryConditions)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

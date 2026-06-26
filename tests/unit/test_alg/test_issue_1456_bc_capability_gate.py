"""Issue #1456: BC-capability gate — solvers fail loud on a BCType they do not support.

The `BoundaryCapable` protocol (`geometry/boundary/protocols.py`) lets a solver declare
`_SUPPORTED_BC_TYPES`; `BaseMFGSolver._validate_bc_support` raises on an unsupported type at
construction instead of silently collapsing it to the solver's default (usually Neumann / no-flux)
— the BC-blindness class mapped in #1456. This pins the template solvers wired in the first
increment: `FPParticleSolver` (already fail-fast → now a declared contract) and `FPSLSolver`
(silently collapsed Dirichlet/Robin to its zero-flux Neumann stencil → now fails loud).
(`HJBGFDMSolver` already fails loud at its row builder and has a uniform-vs-mixed periodic nuance
the type-level construction gate cannot express honestly — deferred to the per-solver rollout.)
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc, periodic_bc, robin_bc

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
# The shared gate is a no-op for None / the particle "periodic" string sentinel.
# ---------------------------------------------------------------------------


def test_validate_bc_support_noop_for_none_and_sentinel():
    solver = FPParticleSolver(_problem(no_flux_bc(dimension=1)))
    solver._validate_bc_support(None)  # None -> no-op
    solver._validate_bc_support("periodic")  # string sentinel -> no-op (not a BoundaryConditions)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

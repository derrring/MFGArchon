"""Fail-loud guards for silent-wrong BC handling (Issues #1558, #1559).

Each converts a silent-wrong (a fabricated normal, a wrong-BC default, a silently
no-flux-coerced dirichlet) into an explicit raise. All three paths are off published
numerics (no experiment / shipped example reaches them), so these pin the fail-loud
behavior rather than a numeric result.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def test_sdf_vanishing_gradient_normal_fails_loud():
    """#1558: SDFParticleBCHandler._compute_normal fabricated an arbitrary [1,0,...] normal when
    the finite-difference SDF gradient vanished -- reflecting a particle along a geometry-independent
    direction (silent-wrong). A constant SDF has zero gradient everywhere, so the normal is undefined;
    it must raise (mirroring project_to_domain's #1047 raise)."""
    from mfgarchon.geometry.boundary import SDFParticleBCHandler

    handler = SDFParticleBCHandler(lambda pts: -0.5 * np.ones(np.asarray(pts).shape[0]), dimension=2)
    with pytest.raises(RuntimeError, match="vanishing SDF gradient"):
        handler._compute_normal(np.array([0.3, 0.4]))


def test_tensor_grid_unknown_bc_type_fails_loud():
    """#1558: get_boundary_handler(bc_type) silently defaulted an unrecognized bc_type to periodic
    (1D) / neumann (nD) -- and its docstring advertised periodic_x/periodic_both/mixed keys that were
    never in either factory, so all of them silently became the default BC. An unrecognized key must
    raise, not substitute a different BC. The bc_type factory is reached only when no BC is stored
    (Priority 1 short-circuits otherwise), so clear it first via the documented 'None to clear' setter."""
    grid1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    grid1d.set_boundary_conditions(None)
    with pytest.raises(ValueError, match="Unsupported 1D bc_type"):
        grid1d.get_boundary_handler("bogus")

    grid2d = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
    )
    grid2d.set_boundary_conditions(None)
    with pytest.raises(ValueError, match="Unsupported 2D bc_type"):
        grid2d.get_boundary_handler("periodic_x")  # advertised in the old docstring, never implemented

    # A supported key must still resolve (the raise is scoped to unknown keys only).
    assert grid1d.get_boundary_handler("periodic") is not None
    assert grid2d.get_boundary_handler("no_flux") is not None


def _small_1d_problem(n=11):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    comps = MFGComponents(
        m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        u_terminal=lambda x: 0.0 * np.asarray(x, dtype=float),
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )
    return MFGProblem(geometry=grid, T=0.1, Nt=2, sigma=0.3, components=comps)


def test_legacy_dirichlet_bc_fails_loud():
    """#1559: the FP-FDM time-stepping assembly treated ANY legacy fdm_bc_1d BoundaryConditions as
    no-flux (the except-AttributeError branch). _is_dirichlet_at_point can't see a legacy BC (no
    is_uniform -> returns False), so a legacy dirichlet was silently assembled as no-flux. It must
    raise for legacy dirichlet/robin while legacy periodic/neumann/no_flux still assemble."""
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.geometry.boundary.fdm_bc_1d import BoundaryConditions as LegacyBC

    prob = _small_1d_problem()
    n = 11
    m0 = np.ones(n)
    drift = np.zeros((prob.Nt + 1, n))

    solver = FPFDMSolver(prob)
    solver.boundary_conditions = LegacyBC(type="dirichlet", left_value=0.0, right_value=0.0)
    with pytest.raises(NotImplementedError, match="1559"):
        solver.solve_fp_system(m0.copy(), drift_field=drift, volatility_field=0.3)

    # Legacy periodic must still assemble (it relies on the no-flux assembly + interior wrapping).
    solver.boundary_conditions = LegacyBC(type="periodic")
    M = solver.solve_fp_system(m0.copy(), drift_field=drift, volatility_field=0.3)
    assert np.all(np.isfinite(M))

"""Pinning tests for Issue #1412: FPParticleSolver volatility override no longer mutates problem.sigma.

The grid-drift particle paths (CPU 1D/nD, GPU) read the SDE volatility from
``_get_grid_params()["sigma"]``. Previously a per-solve ``volatility_field`` was applied by
**monkeypatching** the shared object — ``self.problem.sigma = effective_sigma`` before dispatch,
restored in a ``finally`` (Issue #1248). That mutated externally-visible state for the duration of
the solve (a re-entrancy / shared-state hazard) and is the kind of "raw read + private override"
#1412 consolidates.

It is now a solver-local override attribute ``self._effective_sigma_override`` (the #1316 pattern),
resolved in ``_get_grid_params`` through the shared single source
``pde_coefficients.resolve_diffusion_source``. ``problem.sigma`` is never written.

These pin:
  * the override feeds ``_get_grid_params`` and does NOT touch ``problem.sigma``;
  * a seeded solve with ``volatility_field=s`` is identical to a solve with ``problem.sigma=s`` and
    no override (the override is exactly equivalent to the sigma it represents — byte-identical
    physics);
  * a solve leaves ``problem.sigma`` unchanged and clears the transient override.

Refs #1412 (override pattern from #1316; replaces the #1248 monkeypatch).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

pytestmark = pytest.mark.filterwarnings("ignore")

Nx = 20
T = 0.3
Nt = 6


def _problem(sigma: float) -> MFGProblem:
    geo = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1))
    comp = MFGComponents(
        m_initial=lambda x: np.exp(-20.0 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.5 * (x - 0.5) ** 2,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )
    return MFGProblem(geometry=geo, components=comp, T=T, Nt=Nt, sigma=sigma)


def _m0() -> np.ndarray:
    x = np.linspace(0.0, 1.0, Nx)
    m = np.exp(-20.0 * (x - 0.5) ** 2)
    return m / (m.sum() / Nx)


def _drift() -> np.ndarray:
    return np.tile(0.3 * (np.linspace(0.0, 1.0, Nx) - 0.5) ** 2, (Nt + 1, 1))


def test_get_grid_params_uses_override_without_touching_problem_sigma():
    """The deterministic core: _get_grid_params resolves sigma from the solver-local override
    when set, and problem.sigma is never written."""
    problem = _problem(sigma=0.3)
    solver = FPParticleSolver(problem, num_particles=100)

    solver._effective_sigma_override = 0.77
    assert solver._get_grid_params()["sigma"] == pytest.approx(0.77), "override not consumed by _get_grid_params"
    assert problem.sigma == pytest.approx(0.3), "override mutated the shared problem.sigma (monkeypatch regressed)"

    solver._effective_sigma_override = None
    assert solver._get_grid_params()["sigma"] == pytest.approx(0.3), "None override must fall back to problem.sigma"


def test_volatility_override_equiv_to_setting_problem_sigma():
    """A seeded solve with volatility_field=0.5 (problem.sigma=0.1) is identical to a seeded solve
    with problem.sigma=0.5 and no override — the override is exactly the sigma it represents."""
    m0, drift = _m0(), _drift()

    solver_override = FPParticleSolver(_problem(sigma=0.1), num_particles=500)
    np.random.seed(42)
    m_override = solver_override.solve_fp_system(m0, drift_field=drift, volatility_field=0.5)

    solver_direct = FPParticleSolver(_problem(sigma=0.5), num_particles=500)
    np.random.seed(42)
    m_direct = solver_direct.solve_fp_system(m0, drift_field=drift)

    np.testing.assert_allclose(
        m_override,
        m_direct,
        rtol=1e-10,
        atol=1e-10,
        err_msg="volatility_field=0.5 override is not equivalent to problem.sigma=0.5 (physics differs)",
    )


def test_solve_with_override_leaves_problem_sigma_unchanged():
    """After a solve with a custom volatility_field, problem.sigma is unchanged (no shared-state
    leak) and the transient override is cleared."""
    problem = _problem(sigma=0.1)
    solver = FPParticleSolver(problem, num_particles=300)

    np.random.seed(42)
    solver.solve_fp_system(_m0(), drift_field=_drift(), volatility_field=0.5)

    assert problem.sigma == pytest.approx(0.1), "solve mutated problem.sigma (override leaked)"
    assert solver._effective_sigma_override is None, "transient override not cleared after solve"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

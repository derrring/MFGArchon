#!/usr/bin/env python3
"""Issue #1426: solver options that are stored but never applied must fail loud on a non-default
value instead of being silent no-ops. Defaults remain accepted (baseline-safe).

Covers the GFDM ``congestion_mode`` / WENO ``weno_m_parameter`` options (S0-23/24), plus the
solver-specific dead knobs ``FPGFDMSolver.boundary_indices`` / ``domain_bounds`` (S0-26) and
``FPSLJacobianSolver.characteristic_solver`` (S0-27). These last two are guarded on those specific
solvers only — the namesakes are live on other solvers / geometry APIs. (Network knobs S0-25 are
pinned in ``test_fp_network_solver`` / ``test_hjb_network_solver`` alongside their live siblings.)
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver, HJBWENOSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem(nx=21):
    comp = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=domain, T=1.0, Nt=21, sigma=0.5, components=comp)


def _pts(problem):
    bounds = problem.geometry.get_bounds()
    (nx,) = problem.geometry.get_grid_shape()
    return np.linspace(bounds[0][0], bounds[1][0], nx).reshape(-1, 1)


class TestDeadOptionFailLoud:
    def test_gfdm_congestion_mode_multiplicative_raises(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotImplementedError, match="congestion_mode"):
                HJBGFDMSolver(problem, _pts(problem), monotonicity_scheme="none", congestion_mode="multiplicative")

    def test_gfdm_congestion_mode_additive_ok(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            HJBGFDMSolver(problem, _pts(problem), monotonicity_scheme="none", congestion_mode="additive")

    def test_weno_m_parameter_nondefault_raises(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotImplementedError, match="weno_m_parameter"):
                HJBWENOSolver(problem, weno_m_parameter=2.0)

    def test_weno_m_parameter_default_ok(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            HJBWENOSolver(problem, weno_m_parameter=1.0)

    # Issue #1426 S0-26: FPGFDMSolver.boundary_indices / domain_bounds stored, never read.

    def test_fp_gfdm_boundary_indices_raises(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotImplementedError, match="boundary_indices"):
                FPGFDMSolver(problem, collocation_points=_pts(problem), boundary_indices={0, 1})

    def test_fp_gfdm_domain_bounds_raises(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotImplementedError, match="domain_bounds"):
                FPGFDMSolver(problem, collocation_points=_pts(problem), domain_bounds=[(0.0, 1.0)])

    def test_fp_gfdm_boundary_defaults_ok(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            FPGFDMSolver(problem, collocation_points=_pts(problem))

    # Issue #1426 S0-27: FPSLJacobianSolver.characteristic_solver stored, never read.

    def test_fp_sl_characteristic_solver_nondefault_raises(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotImplementedError, match="characteristic_solver"):
                FPSLJacobianSolver(problem, characteristic_solver="rk4")

    def test_fp_sl_characteristic_solver_default_ok(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            FPSLJacobianSolver(problem)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


class TestDeadOptionsAreProvablyDead:
    """The guards refuse options that are "stored but never read". This proves the "never read".

    Issue #1714: `pytest.raises` on a guard's own message records that the guard fires. For a
    dead-option guard it cannot record the thing that makes refusing correct — that the parameter
    genuinely has no effect. "I could not find a reader" and "there is no reader" look identical
    in a test that only asserts the raise, and the first is what a grep gives you.

    Injecting the option AFTER construction bypasses the guard and lets the claim be measured
    directly: set it to two wildly different values and compare the solutions. Byte-identical
    output over a real solve is evidence no grep can produce.
    """

    def test_fp_gfdm_boundary_indices_and_domain_bounds_change_nothing(self):
        # A gentler problem than the shared `_problem()` fixture, which at T=1.0 / sigma=0.5 drives
        # the unstabilised GFDM flux past the mass-fabrication gate (#1752) before the comparison
        # can run. The point here is byte-identity between two runs, so the configuration only has
        # to be one the solver completes.
        comp = MFGComponents(
            m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        )
        domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=domain, T=0.2, Nt=8, sigma=0.3, components=comp)

        points = _pts(problem)
        n = points.shape[0]
        xs = points[:, 0]
        m_initial = np.exp(-((xs - 0.35) ** 2) / (2 * 0.09**2))
        m_initial /= m_initial.sum()
        drift = np.zeros((problem.Nt + 1, n))

        def solve(**injected):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solver = FPGFDMSolver(problem, collocation_points=points)
                for name, value in injected.items():
                    # The solver stores these privately. Setting a public name would create a NEW
                    # attribute nothing reads, and byte-identity would follow trivially -- the test
                    # would pass while measuring nothing. Assert the target exists first.
                    _absent = object()
                    assert getattr(solver, name, _absent) is not _absent, (
                        f"{name!r} is not an attribute of the constructed solver; injecting it "
                        f"would prove nothing about whether the option is read"
                    )
                    setattr(solver, name, value)
                return solver.solve_fp_system(m_initial.copy(), drift_field=drift)

        baseline = solve()
        assert np.all(np.isfinite(baseline)), "the baseline solve must succeed for this to mean anything"
        assert baseline.std() > 1e-6, "a constant solution would make byte-identity vacuous"

        # Two values as far apart as the parameter admits: nothing marked boundary, everything
        # marked boundary, and a domain fifty times the real one.
        assert np.array_equal(solve(_boundary_indices={0, 1}), baseline), (
            "boundary_indices={0, 1} changed the solution -- it is read somewhere, and the guard's "
            "claim that it is stored-but-never-read is wrong"
        )
        assert np.array_equal(solve(_boundary_indices=set(range(n))), baseline), (
            "marking every node as a boundary changed the solution; boundary_indices is live"
        )
        assert np.array_equal(solve(_domain_bounds=[(-50.0, 50.0)]), baseline), (
            "a domain 50x the real one changed the solution; domain_bounds is live"
        )

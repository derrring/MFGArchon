"""The semi-Lagrangian pair structures a step alike on both sides (#1880).

When a characteristic crosses more than one cell per step, the HJB half cuts the step into
sub-steps. The FP half used to make one forward splat. The coupled Picard map then amplified an
antisymmetric perturbation ~3.3x per sweep on a reflection-symmetric problem, where upwind FD damps it
at 0.89 and SL refined in dt at 0.2. The mirror mismatch, with FP sub-stepping while HJB does not, is
unstable too. So the FP half sub-steps exactly when its HJB half does.

The first two tests are mutation-verified pins of Picard stability, a property two independent schemes
agree on for this fixture; they are not a law of the problem. The third is an external oracle for the
FP half alone: the stationary Ornstein-Uhlenbeck density.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.factory.scheme_factory import create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme

_H = SeparableHamiltonian(
    control_cost=QuadraticControlCost(lambda_=1.0), coupling=lambda m: -m, coupling_dm=lambda m: -1.0
)


def _problem(volatility=0.0, eps=0.0):
    """Invariant under x -> 1 - x when eps = 0; eps seeds an antisymmetric perturbation of m0."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    mass = grid.integrate(np.exp(-10 * (grid.coordinates[0] - 0.5) ** 2))

    def m0(x):
        x = np.asarray(x)
        return np.exp(-10 * (x - 0.5) ** 2) * (1 + eps * (x - 0.5)) / mass

    conditions = Conditions(m_initial=m0, u_terminal=lambda x: 0.0, T=1.0)
    return MFGProblem(model=Model(hamiltonian=_H, volatility=volatility), domain=grid, conditions=conditions, Nt=10)


def _asymmetry(M):
    M = np.asarray(M)
    return float(np.linalg.norm(M - M[:, ::-1]) / np.linalg.norm(M))


def test_a_seeded_asymmetry_decays_under_the_sl_pair():
    """CFL ~13 on this fixture. With the FP half unsub-stepped the asymmetry grew ~3.3x per sweep."""

    def asym_after(sweeps):
        result = _problem(eps=1e-6).solve(
            scheme=NumericalScheme.SL_LINEAR, max_iterations=sweeps, tolerance=1e-30, verbose=False
        )
        return _asymmetry(result.M)

    growth = (asym_after(8) / asym_after(4)) ** 0.25
    assert growth < 1.0, f"seeded asymmetry grows {growth:.3f}x per sweep"


def test_the_pair_stays_stable_when_its_hjb_half_does_not_substep():
    """The mirror mismatch: FP sub-stepping while HJB does not. It converged before #1880; forcing the
    FP half to sub-step regardless left it unconverged with asymmetry ~1 after 200 sweeps."""
    problem = _problem(volatility=0.2)
    hjb, fp = create_paired_solvers(
        problem, NumericalScheme.SL_LINEAR, hjb_config={"enable_adaptive_substepping": False}
    )
    result = problem.solve(hjb_solver=hjb, fp_solver=fp, max_iterations=60, tolerance=1e-8, verbose=False)
    assert result.converged
    assert _asymmetry(result.M) < 1e-10


def _ou_error(dimension, cfl):
    """FP half alone, U = a|x - c|^2: the stationary no-flux density is exp(-a|x - c|^2 / D)."""
    sigma, n = 0.4, 21
    if dimension == 1:
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
        c, Nt, T = np.array([0.35]), 10, 1.0
    else:
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[n, n], boundary_conditions=no_flux_bc(dimension=2)
        )
        c, Nt, T = np.array([0.4, 0.55]), 6, 0.6
    axes = np.meshgrid(*grid.coordinates, indexing="ij")
    Q = sum((X - ci) ** 2 for X, ci in zip(axes, c, strict=True))
    d = grid.coordinates[0][1] - grid.coordinates[0][0]
    slope = max(float(np.max(np.abs(g))) for g in np.atleast_1d(np.gradient(Q, *([d] * dimension))))
    a = cfl / (slope * (T / Nt) / d)  # the largest grid velocity crosses `cfl` cells per step
    problem = MFGProblem(
        model=Model(hamiltonian=_H, volatility=sigma),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=T),
        Nt=Nt,
    )
    m0 = np.exp(-30 * sum((X - 0.5) ** 2 for X in axes))
    M = FPSLSolver(problem).solve_fp_system(m0, potential_field=np.broadcast_to(a * Q, (Nt + 1, *Q.shape)))
    stationary = np.exp(-a * Q / (sigma**2 / 2))
    stationary *= grid.integrate(M[-1]) / grid.integrate(stationary)
    return grid.integrate(np.abs(M[-1] - stationary)) / grid.integrate(stationary)


@pytest.mark.parametrize(("dimension", "cfl"), [(1, 10.0), (2, 8.0)])
def test_the_fp_half_relaxes_to_the_stationary_ou_density_at_cfl_above_one(dimension, cfl):
    # Measured relative L1: 1-D 0.07 and 2-D 0.11 sub-stepped, against 0.30 and 0.47 as one splat.
    assert _ou_error(dimension, cfl) < 0.2

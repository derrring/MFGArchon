"""FVM advection keeps the density non-negative, and the solver stops at the shared positivity gate (#2323).

**The scheme.** Each explicit advection sub-step was sized from the largest single-axis speed over the smallest
spacing. That misses the sum over axes and the half control volume a no-flux wall node owns, so a forward-Euler
step could empty a cell and go past it. At b17a2881, upwind reached min -1.25e+02 on a 2-D no-flux flow with equal
speeds on both axes, and MUSCL reached min -2.8e-01 on the potential x + y. The sub-step is now a fraction of the
finite-volume positivity bound `advective_outflow_rate`. Oracle: without a source term, a step inside that bound
maps a non-negative density to a non-negative one, so no step reaches the gate with a negative value.

**The gate.** The solver warned at an absolute ``min < -1e-12`` and returned the negative density, so whether a
coupled result read as valid depended on the density's units. It now goes through `clip_nonnegative_or_raise`
(#1671), which clips below 1e-8 of the mass present and stops the solve above it. A source term is what still
drives the density negative, so the gate is pinned with sinks.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

import mfgarchon.alg.numerical.fp_solvers.fp_fvm as fp_fvm
from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem(dimension: int, n: int) -> MFGProblem:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)] * dimension,
                Nx_points=[n] * dimension,
                boundary_conditions=no_flux_bc(dimension=dimension),
            ),
            Nt=10,
            T=0.5,
            sigma=0.0,
            components=MFGComponents(
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
                ),
            ),
        )


@pytest.mark.parametrize(
    ("dimension", "n", "reconstruction", "potential"),
    [
        (2, 17, "upwind", lambda x, y: np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)),
        (2, 17, "muscl", lambda x, y: x + y),
        (1, 41, "upwind", lambda x: x),
    ],
    ids=["2d_upwind_equal_axis_speeds", "2d_muscl_diagonal_potential", "1d_upwind_wall"],
)
def test_advection_does_not_drive_the_density_negative(monkeypatch, dimension, n, reconstruction, potential):
    """At b17a2881 these went to min -1.25e+02, -2.80e-01 and -3.70e-02 before the gate."""
    reached_the_gate = []
    gate = fp_fvm.clip_nonnegative_or_raise

    def recording_gate(density, **kwargs):
        reached_the_gate.append(float(np.min(density)))
        return gate(density, **kwargs)

    monkeypatch.setattr(fp_fvm, "clip_nonnegative_or_raise", recording_gate)
    problem = _problem(dimension, n)
    grid = problem.geometry
    coordinates = np.meshgrid(*grid.coordinates, indexing="ij")
    m0 = np.ones_like(coordinates[0]) / float(grid.integrate(np.ones_like(coordinates[0])))
    FPFVMSolver(problem, reconstruction=reconstruction).solve_fp_system(
        m0, potential_field=np.stack([potential(*coordinates)] * 11)
    )
    assert len(reached_the_gate) == 10
    assert min(reached_the_gate) >= 0.0, f"a step reached the gate at min {min(reached_the_gate):.3e} (#2323)"


def _solve_with_a_sink(rate: float) -> np.ndarray:
    """No advection, no diffusion: a sink of ``rate`` on x < 0.5, where the density starts at zero."""
    problem = _problem(1, 21)
    x = problem.geometry.coordinates[0]
    m0 = np.where(x >= 0.5, 1.0, 0.0)
    m0 = m0 / float(problem.geometry.integrate(m0))

    def sink(t, points):
        return np.where(np.asarray(points)[:, 0] < 0.5, rate, 0.0)

    return FPFVMSolver(problem, reconstruction="muscl").solve_fp_system(m0, source_term=sink)


def test_negative_mass_below_the_gate_is_clipped():
    density = _solve_with_a_sink(-1e-9)
    assert density.min() >= 0.0, f"min {density.min():.3e}: FVM returned a negative density (#2323)"


def test_negative_mass_above_the_gate_stops_the_solve():
    with pytest.raises(ValueError, match="would fabricate"):
        _solve_with_a_sink(-5.0)

"""FVM advection keeps the density non-negative, and the solver stops at the shared positivity gate (#2323).

**The scheme.** Each explicit advection sub-step was sized from the largest single-axis speed over the smallest
spacing. That misses the sum over axes and the half control volume a no-flux wall node owns, so a forward-Euler
step could empty a cell and go past it. At b17a2881 the first step reached min -6.10e+01 for upwind on a 2-D no-flux
flow with equal speeds on both axes, and -2.80e-01 for MUSCL on the potential x + y. The sub-step is now a fraction
of the finite-volume positivity bound `advective_outflow_rate`: 0.8 of it for upwind and 0.4 for MUSCL, whose bound
is 2/3. Oracle: without a source term, a step inside that bound maps a non-negative density to a non-negative one, so
no step reaches the gate with a negative value. On a periodic grid that repeats its endpoint this needs the repeated
node to equal node 0, which the solver does not enforce (#2336); every fixture here is periodic at the seam.

The fixtures cover the rule's branches: no-flux walls with half control volumes in 1-D and 2-D, the sum over axes, a
periodic wrap and its repeated cell, outflow rather than inflow, and, through a sweep of constant speeds, MUSCL's
halved target and the rounding up of the sub-step count. The review of #2332 found mutants of the last five branches
that no test failed.

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
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc


def _problem(dimension: int, n: int, periodic: bool = False) -> MFGProblem:
    boundary = periodic_bc(dimension=dimension) if periodic else no_flux_bc(dimension=dimension)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)] * dimension, Nx_points=[n] * dimension, boundary_conditions=boundary
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
    ("dimension", "n", "periodic", "reconstruction", "field"),
    [
        (2, 17, False, "upwind", ("potential", lambda x, y: np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))),
        (2, 17, False, "muscl", ("potential", lambda x, y: x + y)),
        (1, 41, False, "upwind", ("potential", lambda x: x)),
        (1, 41, False, "upwind", ("potential", lambda x: 2 * (x - 0.5) ** 2)),
        (1, 41, True, "upwind", ("potential", lambda x: -np.abs(np.sin(np.pi * x)))),
    ],
    ids=[
        "2d_upwind_equal_axis_speeds",
        "2d_muscl_diagonal_potential",
        "1d_upwind_wall",
        "1d_upwind_outflow_not_inflow",
        "1d_periodic_wrap",
    ],
)
def test_advection_does_not_drive_the_density_negative(monkeypatch, dimension, n, periodic, reconstruction, field):
    """The first three went to min -6.10e+01, -2.80e-01 and -3.70e-02 at their first step at b17a2881.

    The last two each go negative under one mutant of the rule (measured at 8a618465): the rate summing inflow instead
    of outflow, and the rate ignoring the periodic wrap faces or dropping `spans`.
    """
    reached_the_gate = []
    gate = fp_fvm.clip_nonnegative_or_raise

    def recording_gate(density, **kwargs):
        reached_the_gate.append(float(np.min(density)))
        return gate(density, **kwargs)

    monkeypatch.setattr(fp_fvm, "clip_nonnegative_or_raise", recording_gate)
    problem = _problem(dimension, n, periodic)
    grid = problem.geometry
    coordinates = np.meshgrid(*grid.coordinates, indexing="ij")
    m0 = np.ones_like(coordinates[0])
    m0 = m0 / float(grid.integrate(m0))
    kind, values = field
    stacked = np.stack([values(*coordinates)] * 11)
    FPFVMSolver(problem, reconstruction=reconstruction).solve_fp_system(
        m0, **({"potential_field": stacked} if kind == "potential" else {"drift_field": stacked})
    )
    assert len(reached_the_gate) == 10
    assert min(reached_the_gate) >= 0.0, f"a step reached the gate at min {min(reached_the_gate):.3e} (#2323)"


@pytest.mark.parametrize("reconstruction", ["muscl", "upwind"])
def test_no_constant_speed_drives_a_periodic_density_negative(monkeypatch, reconstruction):
    """Speeds from 0.3 to 3.2 times h/dt, so the sweep does not depend on how the solver splits a time step.

    Measured at 8a618465 for n in 39, 41, 43 and Nt in 9, 10, 11: with MUSCL's target raised to upwind's 0.8, the MUSCL
    sweep goes negative in all nine settings; with the sub-step count rounded down, both sweeps do.
    """
    reached_the_gate = []
    gate = fp_fvm.clip_nonnegative_or_raise

    def recording_gate(density, **kwargs):
        reached_the_gate.append(float(np.min(density)))
        return gate(density, **kwargs)

    monkeypatch.setattr(fp_fvm, "clip_nonnegative_or_raise", recording_gate)
    problem = _problem(1, 41, periodic=True)
    x = problem.geometry.coordinates[0]
    m0 = np.maximum(np.sin(2 * np.pi * x), 0.0) ** 3
    m0 = m0 / float(problem.geometry.integrate(m0))
    h = float(problem.geometry.get_grid_spacing()[0])
    for fraction in (0.3, 0.5, 0.7, 0.79, 0.9, 1.2, 1.58, 2.4, 3.2):
        speed = fraction * h / problem.dt
        FPFVMSolver(problem, reconstruction=reconstruction).solve_fp_system(
            m0, drift_field=np.full((11, x.size), speed)
        )
    assert len(reached_the_gate) == 90
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

"""`mass_conservation_error` must be measured, and `None` must mean "not measured" (Issue #1672).

The field is documented as `max|integral(m) - 1| over time steps`. It defaulted to `0.0` and no
solver on the coupling path ever wrote it, so every coupled solve reported perfect conservation --
including the FDM_CENTERED case whose mass reached 6378 (Issue #1671). That is why #1673 had to
assert mass conservation by hand: `result.mass_conservation_error` could not be used, because it
read `0.0` for the divergent solve and for a healthy one alike.

`0.0` is a valid measurement. An unmeasured field must therefore not be `0.0`.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme
from mfgarchon.utils.solver_result import SolverResult


def _problem(**kwargs):
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents

    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1)),
        Nt=10,
        T=1.0,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        ),
        **kwargs,
    )


def test_an_unmeasured_field_is_none_not_zero():
    """A bare SolverResult must not claim perfect conservation it never measured."""
    result = SolverResult(
        U=np.zeros((2, 3)),
        M=np.zeros((2, 3)),
        iterations=1,
        error_history_U=np.zeros(1),
        error_history_M=np.zeros(1),
        solver_name="stub",
        converged=False,
    )

    assert result.mass_conservation_error is None, (
        "an unwritten field read 0.0, which is indistinguishable from a solve that conserved "
        "mass exactly -- the defect this pins"
    )


@pytest.mark.parametrize("scheme", [NumericalScheme.FDM_UPWIND, NumericalScheme.FDM_CENTERED])
def test_the_coupled_solve_reports_the_quantity_it_documents(scheme):
    """The reported value must equal `max|integral(m) - 1|` computed independently."""
    problem = _problem(sigma=1.0)
    result = problem.solve(scheme=scheme, max_iterations=5, verbose=False)

    cell_volume = problem.geometry.volume_element()
    spatial_axes = tuple(range(1, result.M.ndim))
    expected = float(np.max(np.abs(np.sum(result.M, axis=spatial_axes) * cell_volume - 1.0)))

    assert result.mass_conservation_error is not None, "the coupling path did not measure it"
    assert result.mass_conservation_error == pytest.approx(expected, rel=1e-12), (
        f"reported {result.mass_conservation_error!r} against an independently computed {expected!r}"
    )


def test_the_measurement_uses_the_geometry_volume_element():
    """Pin the source of the cell measure, not just the number.

    A second hand-rolled `dx` would agree on a unit-interval grid and diverge elsewhere, which is
    the repo's dominant defect class. Using a non-unit domain makes the two disagree.
    """
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents

    problem = MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 4.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1)),
        Nt=10,
        T=1.0,
        sigma=1.0,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 2.0) ** 2).squeeze(),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        ),
    )
    result = problem.solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=3, verbose=False)

    assert problem.geometry.volume_element() == pytest.approx(0.2), "fixture assumption"
    naive = float(np.max(np.abs(np.sum(result.M, axis=1) * (1.0 / 20) - 1.0)))
    correct = float(np.max(np.abs(np.sum(result.M, axis=1) * 0.2 - 1.0)))

    assert naive != pytest.approx(correct, rel=1e-6), "fixture cannot discriminate the two"
    assert result.mass_conservation_error == pytest.approx(correct, rel=1e-12)

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
    """The reported value must equal the drift computed independently.

    ``abs=0`` is load-bearing. ``pytest.approx``'s default absolute floor is 1e-12, and these
    fixtures conserve mass to ~7e-15, so with the default every value in [0, 1e-12] passes --
    including the literal ``0.0`` that is bug #1672 itself. Review of this PR caught that: all four
    tests stayed green with the measurement replaced by a constant zero.
    """
    problem = _problem(sigma=1.0)
    result = problem.solve(scheme=scheme, max_iterations=5, verbose=False)

    spatial_axes = tuple(range(1, result.M.ndim))
    mass = np.sum(result.M, axis=spatial_axes)
    expected = float(np.max(np.abs(mass / mass[0] - 1.0)))

    assert result.mass_conservation_error is not None, "the coupling path did not measure it"
    assert result.mass_conservation_error == pytest.approx(expected, rel=1e-12, abs=0)


def test_a_solve_that_loses_mass_reports_an_order_one_error():
    """The positive control. Without it every assertion sits within the noise floor.

    A Dirichlet-0 boundary absorbs mass, so the true value is O(1) rather than O(1e-15) and the
    assertion has room to fail. This is the fixture that discriminates; the machine-precision ones
    above only pin agreement with an independent computation.
    """
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.geometry.boundary import dirichlet_bc

    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=dirichlet_bc(dimension=1, value=0.0)
        ),
        Nt=10,
        T=1.0,
        sigma=1.0,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        ),
    )
    result = problem.solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=3, verbose=False)

    assert result.mass_conservation_error is not None
    assert result.mass_conservation_error > 0.1, (
        f"an absorbing boundary must show up as lost mass; got "
        f"{result.mass_conservation_error!r}, which is within the noise floor of a conserving solve"
    )


def test_the_metric_is_drift_not_deviation_from_one():
    """Why the target is the initial mass rather than 1.0.

    Two owners disagree on the cell measure: ``MFGProblem._initialize_functions`` normalises with
    ``prod(L_i / Nx_points_i)`` while ``volume_element()`` returns ``prod(L_i / (Nx_points_i - 1))``
    -- points against intervals. Measured against 1.0 that fork reports ``(N/(N-1))**d - 1``: 21%
    on an 11-point 2D grid whose mass is flat to 4e-16, shrinking like d/N so it reads as a
    first-order-convergent error. A ratio is invariant to the measure. The fork itself is
    pre-existing and out of scope here.
    """
    n, d = 11, 2
    flat = np.ones((5,) + (n,) * d)
    flat /= flat[0].sum() * (1.0 / n) ** d  # normalised the way the problem normalises m0
    mass = np.sum(flat, axis=tuple(range(1, flat.ndim))) * (1.0 / (n - 1)) ** d  # measured the other way

    against_one = float(np.max(np.abs(mass - 1.0)))
    drift = float(np.max(np.abs(mass / mass[0] - 1.0)))

    assert against_one == pytest.approx((n / (n - 1)) ** d - 1, rel=1e-12)
    assert against_one > 0.2, "the fork is large, not a rounding artefact"
    assert drift == 0.0, "a perfectly flat mass must report exactly zero drift"


def test_a_real_drift_is_still_reported():
    """The ratio must not launder an actual failure into zero."""
    mass = np.array([1.0, 1.0, 1.0, 6378.0, 6378.0])

    assert float(np.max(np.abs(mass / mass[0] - 1.0))) == pytest.approx(6377.0, rel=1e-12, abs=0)

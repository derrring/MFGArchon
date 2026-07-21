#!/usr/bin/env python3
"""
Unit tests for runtime safety validation (Issue #688).

Tests that:
- check_finite detects NaN/Inf with location info
- check_bounds detects out-of-range values
- validate_solver_output catches NaN in U/M and negative density
- FixedPointIterator terminates early on NaN (integration)

Follows the pattern of test_array_field_validation.py.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary.conditions import no_flux_bc
from mfgarchon.utils.validation.runtime import (
    check_bounds,
    check_finite,
    validate_solver_output,
)

_NX = 11


def _divergence_probe_problem():
    """The one problem all the FixedPointIterator divergence tests below run on.

    Extracted after an independent review flagged five byte-identical copies of this
    construction in this file (Issue #1717 review, MINOR-7). Nothing here is tuned per
    test -- each test varies only what the mocked HJB/FP solvers return.
    """
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[_NX],
            boundary_conditions=no_flux_bc(dimension=1),
        ),
        components=MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: -(m**2),
                coupling_dm=lambda m: -2 * m,
            ),
            m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=lambda x: x**2,
        ),
        Nt=10,
    )


# ===========================================================================
# check_finite
# ===========================================================================


@pytest.mark.unit
def test_check_finite_clean_array():
    """Clean array should pass finiteness check."""
    arr = np.linspace(0.0, 1.0, 20)
    result = check_finite(arr, "test", raise_on_error=False)
    assert result.is_valid


@pytest.mark.unit
def test_check_finite_nan_detected():
    """Array with NaN should fail with location info."""
    arr = np.ones((10, 5))
    arr[3, 2] = np.nan
    result = check_finite(arr, "U", location="timestep 3", raise_on_error=False)
    assert not result.is_valid
    assert result.context.get("n_nan") == 1
    assert any("NaN" in str(issue) for issue in result.issues)


@pytest.mark.unit
def test_check_finite_raise_on_error():
    """check_finite with raise_on_error=True should raise ValueError."""
    arr = np.ones(10)
    arr[5] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        check_finite(arr, "M", raise_on_error=True)


# ===========================================================================
# check_bounds
# ===========================================================================


@pytest.mark.unit
def test_check_bounds_within():
    """Array within bounds should pass."""
    arr = np.linspace(0.0, 1.0, 20)
    result = check_bounds(arr, "density", lower=0.0, upper=1.0)
    assert result.is_valid


@pytest.mark.unit
def test_check_bounds_violation():
    """Array with values outside bounds should fail."""
    arr = np.array([0.5, 1.5, -0.1, 0.8])
    result = check_bounds(arr, "density", lower=0.0, upper=1.0)
    assert not result.is_valid
    assert any("above" in str(issue).lower() for issue in result.issues)
    assert any("below" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# validate_solver_output
# ===========================================================================


@pytest.mark.unit
def test_validate_solver_output_valid():
    """Clean U and M should pass output validation."""
    Nt, Nx = 10, 20
    U = np.random.randn(Nt, Nx)
    M = np.abs(np.random.randn(Nt, Nx))  # Non-negative
    result = validate_solver_output(U, M)
    assert result.is_valid


@pytest.mark.unit
def test_validate_solver_output_nan_u():
    """NaN in U should be detected."""
    Nt, Nx = 10, 20
    U = np.ones((Nt, Nx))
    U[5, 10] = np.nan
    M = np.ones((Nt, Nx))
    result = validate_solver_output(U, M)
    assert not result.is_valid
    assert any("NaN" in str(issue) for issue in result.issues)


@pytest.mark.unit
def test_validate_solver_output_negative_density():
    """Negative density should be detected."""
    Nt, Nx = 10, 20
    U = np.ones((Nt, Nx))
    M = np.ones((Nt, Nx))
    M[3, 5] = -0.5
    result = validate_solver_output(U, M, check_finite=False, check_density_positive=True)
    assert not result.is_valid
    assert any("negative" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# Integration: FixedPointIterator NaN early termination
# ===========================================================================


@pytest.mark.unit
def test_fixed_point_nan_early_termination():
    """FixedPointIterator should terminate early when NaN appears in iteration."""
    from unittest.mock import Mock

    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import (
        FixedPointIterator,
    )
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary.conditions import no_flux_bc

    # Pre-existing construction, left inline: folding it into
    # _divergence_probe_problem() is a change to a test this PR does not otherwise
    # touch, so it belongs to whoever next edits #688's coverage, not here.
    Nx = 11
    geom = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[Nx],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )
    components = MFGComponents(
        hamiltonian=H,
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: x**2,
    )
    problem = MFGProblem(geometry=geom, components=components, Nt=10)

    Nt = problem.Nt
    num_time_steps = Nt + 1
    spatial_shape = problem.spatial_shape

    # Mock HJB solver: returns NaN on second call
    hjb_solver = Mock()
    call_count = [0]

    def hjb_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] >= 2:
            return np.full((num_time_steps, *spatial_shape), np.nan)
        return np.zeros((num_time_steps, *spatial_shape))

    hjb_solver.solve_hjb_system.side_effect = hjb_side_effect

    # Mock FP solver: returns valid density
    fp_solver = Mock()
    fp_solver.solve_fp_system.return_value = np.ones((num_time_steps, *spatial_shape)) / Nx

    # Create iterator and solve
    iterator = FixedPointIterator(
        problem=problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        relaxation=0.5,
    )
    result = iterator.solve(max_iterations=10, tolerance=1e-6)

    # Verify early termination
    assert not result.converged
    assert result.iterations < 10
    assert result.metadata.get("convergence_reason") == "diverged_nan"

    # Verify output validation detected the NaN
    output_val = result.metadata.get("output_validation")
    assert output_val is not None
    assert output_val["is_valid"] is False


@pytest.mark.unit
def test_fp_is_not_solved_after_hjb_returns_nonfinite():
    """A diverged HJB must stop the iteration before FP runs (Issue #1717).

    The sibling test above mocks FP to return a clean density regardless of input, so it
    cannot see whether FP was handed a non-finite U. Here FP records what it received.
    Without the guard, the FP source and drift are composed from a NaN U and the real FP
    solver raises a CFL diagnostic -- an FP error for an HJB failure -- and that exception
    escapes before the #1078 HJB-vs-FP attribution can run.
    """
    from unittest.mock import Mock

    problem = _divergence_probe_problem()
    Nx = _NX
    shape = (problem.Nt + 1, *problem.spatial_shape)

    hjb_calls = [0]

    def hjb_side_effect(*args, **kwargs):
        hjb_calls[0] += 1
        # Second HJB solve diverges.
        return np.full(shape, np.nan) if hjb_calls[0] >= 2 else np.zeros(shape)

    hjb_solver = Mock()
    hjb_solver.solve_hjb_system.side_effect = hjb_side_effect

    u_seen_by_fp = []

    def fp_side_effect(*args, **kwargs):
        # The value function reaches FP through one of several channels depending on the
        # solver's signature -- positionally with a Mock, but as potential_field= or
        # drift_field= with a real solver (resolve_fp_drift_kwargs). Scan every array
        # argument rather than one named channel, so hardening this double into
        # Mock(spec=FPFDMSolver) cannot silently disarm the assertion.
        candidates = [a for a in args if isinstance(a, np.ndarray)]
        candidates += [v for v in kwargs.values() if isinstance(v, np.ndarray)]
        u_seen_by_fp.append(any(not np.all(np.isfinite(c)) for c in candidates))
        return np.ones(shape) / Nx

    fp_solver = Mock()
    fp_solver.solve_fp_system.side_effect = fp_side_effect

    FixedPointIterator(problem=problem, hjb_solver=hjb_solver, fp_solver=fp_solver, relaxation=0.5).solve(
        max_iterations=10, tolerance=1e-6
    )

    assert hjb_calls[0] == 2, "the second HJB solve is the one that diverges"
    assert not any(u_seen_by_fp), "FP was handed a non-finite array"
    assert len(u_seen_by_fp) == 1, (
        f"FP ran {len(u_seen_by_fp)} times; it must not run in the iteration whose HJB diverged"
    )
    # Not asserted here: convergence_reason == "diverged_nan". The pre-existing post-damping
    # check sets the same string, so it passes with or without the guard and pins nothing
    # (Issue #1701). The mass sentinel below does discriminate.


@pytest.mark.unit
def test_mass_conservation_is_unmeasured_when_fp_never_ran():
    """A solve that dies in HJB must not report perfect mass conservation (Issue #1717).

    When the first HJB solve diverges, FP never runs and `self.M` still holds the cold start,
    whose mass is constant across time by construction -- so measuring it yields exactly 0.0,
    the best possible value, for the worst possible solve. Issue #1672 added the None sentinel
    so that "not measured" cannot masquerade as a zero; this pins that it is used here.
    """
    from unittest.mock import Mock

    problem = _divergence_probe_problem()
    Nx = _NX
    shape = (problem.Nt + 1, *problem.spatial_shape)

    hjb_solver = Mock()
    hjb_solver.solve_hjb_system.side_effect = lambda *a, **k: np.full(shape, np.nan)
    fp_solver = Mock()
    fp_solver.solve_fp_system.side_effect = lambda *a, **k: np.ones(shape) / Nx

    result = FixedPointIterator(problem=problem, hjb_solver=hjb_solver, fp_solver=fp_solver, relaxation=0.5).solve(
        max_iterations=10, tolerance=1e-6
    )

    assert fp_solver.solve_fp_system.call_count == 0, "FP must not run when the first HJB diverges"
    assert result.mass_conservation_error is None, (
        f"mass conservation reported {result.mass_conservation_error!r} for a solve where FP "
        "never produced a density; it must be None (not measured)"
    )


@pytest.mark.unit
def test_mass_conservation_is_still_measured_when_fp_ran_before_the_divergence():
    """The mirror of the test above, and the harder direction (Issue #1717).

    Suppressing the measurement whenever the solve diverges would be the same defect as
    fabricating a zero, pointed the other way: if FP completed a step before the HJB blew
    up, `self.M` is a density this solve produced and its mass error is a real number. Here
    FP loses half the mass, so `None` would be hiding a ~50% error behind "not measured".
    """
    from unittest.mock import Mock

    problem = _divergence_probe_problem()
    Nx = _NX
    shape = (problem.Nt + 1, *problem.spatial_shape)

    hjb_calls = [0]

    def hjb_side_effect(*args, **kwargs):
        hjb_calls[0] += 1
        # Diverge on the SECOND solve, so one full FP step has already completed.
        return np.full(shape, np.nan) if hjb_calls[0] >= 2 else np.zeros(shape)

    hjb_solver = Mock()
    hjb_solver.solve_hjb_system.side_effect = hjb_side_effect
    # A density that sheds mass down the time axis: a real, measurable conservation defect.
    losing_mass = np.ones(shape) / Nx * np.linspace(1.0, 0.1, shape[0])[:, None]
    fp_solver = Mock()
    fp_solver.solve_fp_system.side_effect = lambda *a, **k: losing_mass

    result = FixedPointIterator(problem=problem, hjb_solver=hjb_solver, fp_solver=fp_solver, relaxation=0.5).solve(
        max_iterations=10, tolerance=1e-6
    )

    assert fp_solver.solve_fp_system.call_count == 1, "one FP step must have completed"
    assert result.mass_conservation_error is not None, (
        "FP produced a density in this solve, so its mass error is measurable; reporting None "
        "hides a real defect behind 'not measured'"
    )
    assert result.mass_conservation_error > 0.1, (
        f"expected the injected mass loss to be reported, got {result.mass_conservation_error!r}"
    )


@pytest.mark.unit
def test_terminal_condition_survives_an_hjb_divergence():
    """The terminal row is boundary data, not solver output (Issue #1717).

    The normal path and the two pre-existing divergence breaks all pass through
    preserve_terminal_condition; the new guard must not be the one that returns U[-1] as NaN.
    """
    from unittest.mock import Mock

    problem = _divergence_probe_problem()
    Nx = _NX
    shape = (problem.Nt + 1, *problem.spatial_shape)

    hjb_solver = Mock()
    hjb_solver.solve_hjb_system.side_effect = lambda *a, **k: np.full(shape, np.nan)
    fp_solver = Mock()
    fp_solver.solve_fp_system.side_effect = lambda *a, **k: np.ones(shape) / Nx

    result = FixedPointIterator(problem=problem, hjb_solver=hjb_solver, fp_solver=fp_solver, relaxation=0.5).solve(
        max_iterations=10, tolerance=1e-6
    )

    x = problem.geometry.get_spatial_grid().ravel()
    np.testing.assert_allclose(np.asarray(result.U)[-1], x**2, rtol=0, atol=1e-12)

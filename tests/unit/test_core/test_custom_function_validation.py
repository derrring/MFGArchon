#!/usr/bin/env python3
"""
Unit tests for custom function validation (Issue #686).

Tests that:
- Valid HamiltonianBase instances pass validation
- NaN-producing Hamiltonians are caught
- Hamiltonian derivative consistency checking works
- Drift functions with correct/wrong signatures are handled
- Running cost functions with correct/wrong signatures are handled

Follows the pattern of test_ic_bc_validation.py.
"""

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import (
    BoundedControlCost,
    HamiltonianBase,
    L1ControlCost,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary.conditions import no_flux_bc
from mfgarchon.utils.validation import (
    ValidationError,
    ValidationSeverity,
    validate_custom_functions,
    validate_drift,
    validate_hamiltonian,
    validate_hamiltonian_consistency,
    validate_running_cost,
)

# ===========================================================================
# Test Helpers
# ===========================================================================


def _geometry(Nx_points=11, dimension=1):
    """Create a test geometry."""
    bounds = [(0.0, 1.0)] * dimension
    nx = [Nx_points] * dimension
    return TensorProductGrid(
        bounds=bounds,
        Nx_points=nx,
        boundary_conditions=no_flux_bc(dimension=dimension),
    )


def _hamiltonian():
    """Create a standard test Hamiltonian."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )


def _problem(m_initial, u_terminal, hamiltonian=None, Nx_points=11, **kwargs):
    """Create a test MFGProblem."""
    geom = _geometry(Nx_points=Nx_points)
    components = MFGComponents(
        hamiltonian=hamiltonian or _hamiltonian(),
        m_initial=m_initial,
        u_terminal=u_terminal,
    )
    return MFGProblem(geometry=geom, components=components, **kwargs)


# ===========================================================================
# Hamiltonian validation (standalone)
# ===========================================================================


@pytest.mark.unit
def test_valid_hamiltonian_passes():
    """A standard SeparableHamiltonian should pass validation."""
    H = _hamiltonian()
    geom = _geometry()
    result = validate_hamiltonian(H, geom)
    assert result.is_valid, f"Unexpected issues: {result.issues}"


@pytest.mark.unit
def test_hamiltonian_returning_nan_raises():
    """A Hamiltonian that returns NaN should fail validation."""

    class NaNHamiltonian(HamiltonianBase):
        @property
        def dimension(self):
            return 1

        def __call__(self, x, m, p, t=0.0):
            return float("nan")

    H = NaNHamiltonian()
    geom = _geometry()
    result = validate_hamiltonian(H, geom)
    assert not result.is_valid
    assert any("NaN" in str(issue) for issue in result.issues)


@pytest.mark.unit
def test_hamiltonian_wrong_signature_raises():
    """An object that doesn't accept (x, m, p, t) should fail validation."""

    class BadHamiltonian:
        def __call__(self, x):
            return 0.0

    geom = _geometry()
    result = validate_hamiltonian(BadHamiltonian(), geom)
    assert not result.is_valid
    assert any("signature" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# Hamiltonian consistency check
# ===========================================================================


@pytest.mark.unit
def test_hamiltonian_consistency_passes():
    """SeparableHamiltonian with correct coupling_dm should pass consistency."""
    H = _hamiltonian()
    geom = _geometry()
    result = validate_hamiltonian_consistency(H, H.dm, geom)
    # Should have no warnings about inconsistency
    inconsistent_warnings = [i for i in result.issues if "inconsistent" in str(i).lower()]
    assert len(inconsistent_warnings) == 0


@pytest.mark.unit
def test_hamiltonian_consistency_gates_on_dm_witness():
    """A grossly wrong dH_dm invalidates the result (Issue #1642, C1).

    Catches: the validator regressing to warning-only, which made the caller's
    `if not result.is_valid: raise` branch structurally dead.
    """
    H = _hamiltonian()
    geom = _geometry()

    def wrong_dm(x, m, p, t=0.0):
        return 42.0

    result = validate_hamiltonian_consistency(H, wrong_dm, geom)
    assert not result.is_valid
    errors = [i for i in result.issues if i.severity is ValidationSeverity.ERROR]
    assert len(errors) == 1
    assert errors[0].location == "dH_dm"
    # The diagnostic must exhibit the witness, not just say "inconsistent".
    message = errors[0].message
    assert "42" in message
    assert "x=" in message
    assert "m=" in message
    assert "p=" in message


@pytest.mark.unit
@pytest.mark.parametrize("dimension", [1, 2])
def test_hamiltonian_dp_consistency_passes(dimension):
    """Correct dH_dp should pass consistency check, in 1-D and 2-D."""
    H = _hamiltonian()
    geom = _geometry(dimension=dimension)
    result = validate_hamiltonian_consistency(H, H.dm, geom, dH_dp=H.dp)
    assert result.is_valid
    assert result.issues == []


@pytest.mark.unit
@pytest.mark.parametrize("dimension", [1, 2])
def test_hamiltonian_consistency_gates_on_dp_witness(dimension):
    """A grossly wrong dH_dp invalidates the result and names the values.

    Only the LAST component is wrong, so in 2-D the per-component loop has to
    compare component 1 against component 1. Catches a loop that compares every
    component against `dp_analytical[0]` -- which is indistinguishable from the
    correct loop in 1-D, the only dimension the rest of this module exercises.
    """
    H = _hamiltonian()
    geom = _geometry(dimension=dimension)
    wrong_index = dimension - 1

    def wrong_dp(x, m, p, t=0.0):
        claimed = np.atleast_1d(H.dp(x, m, p, t)).astype(float).copy()
        claimed[wrong_index] = 99.0
        return claimed

    result = validate_hamiltonian_consistency(H, H.dm, geom, dH_dp=wrong_dp)
    assert not result.is_valid
    errors = [i for i in result.issues if i.severity is ValidationSeverity.ERROR]
    assert [i.location for i in errors] == [f"dH_dp[{wrong_index}]"]
    assert "99" in errors[0].message


@pytest.mark.unit
def test_error_reports_the_witness_probe_not_an_earlier_warning():
    """The ERROR must carry the witness probe's values, not an earlier probe's.

    `_report` grades on `witness is not None` but formats whichever comparison it
    selected; if those decouple, the message asserts "exceeds 0.01 relative"
    while quoting a 0.5% discrepancy -- a diagnostic that names innocent values.
    Probe order is m=0.5 before m=1.0, so the warning-grade point is seen first.
    """
    H = _hamiltonian()
    geom = _geometry()

    def wrong_dm(x, m, p, t=0.0):
        if m == 1.0:
            return -42.0  # witness grade: 21x off
        return -2.0 * m * 1.005  # warning grade: 0.5% off, seen first

    result = validate_hamiltonian_consistency(H, wrong_dm, geom)
    assert not result.is_valid
    errors = [i for i in result.issues if i.severity is ValidationSeverity.ERROR]
    assert [i.location for i in errors] == ["dH_dm"]

    message = errors[0].message
    assert "analytical=-42" in message
    assert "m=1," in message
    assert "-1.005" not in message
    assert result.context["dH_dm_analytical"] == -42.0
    assert result.context["dH_dm_witness_m"] == 1.0


@pytest.mark.unit
def test_consistency_context_reaches_validate_custom_functions():
    """The witness values must survive the merge into the aggregate result.

    Catches dropping the consistency result's `context`, which silently strips
    the machine-readable witness from every caller that goes through
    `validate_custom_functions` -- the only path `MFGProblem` uses.
    """
    H = _hamiltonian()
    geom = _geometry()

    def wrong_dm(x, m, p, t=0.0):
        return 42.0

    result = validate_custom_functions(hamiltonian=H, dH_dm=wrong_dm, dH_dp=H.dp, geometry=geom)
    assert not result.is_valid
    assert result.context["dH_dm_analytical"] == 42.0
    assert result.context["dH_dm_witness_m"] in (0.5, 1.0)
    assert "dH_dm_witness_p" in result.context


@pytest.mark.unit
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_regularized_hamiltonian_constructs_in_every_dimension(dimension):
    """A Moreau-Yosida-regularized control cost must survive construction.

    The gate is live on this path, so any dH_dp/H disagreement in the regularized
    Hamiltonian becomes a hard construction failure rather than wrong numbers.
    Catches a per-component Moreau penalty, which is exact at d=1 and off by a
    factor of d above it.
    """
    base = SeparableHamiltonian(
        control_cost=BoundedControlCost(control_cost=1.0, max_control=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )
    H = base.regularize(0.1)
    geom = _geometry(dimension=dimension)

    result = validate_custom_functions(hamiltonian=H, dH_dm=H.dm, dH_dp=H.dp, geometry=geom)
    assert result.is_valid, [str(i) for i in result.issues]

    components = MFGComponents(
        hamiltonian=H,
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    problem = MFGProblem(geometry=geom, components=components, T=0.1, Nt=3)
    assert problem is not None


@pytest.mark.unit
def test_small_discrepancy_warns_but_does_not_gate():
    """A 0.1% derivative discrepancy is a warning, never a gate.

    Catches: widening the witness predicate until it fires on discrepancies a
    finite-difference stencil cannot distinguish from a modelling choice. Raising
    on a false positive is worse than warning.
    """
    H = _hamiltonian()
    geom = _geometry()

    def slightly_off_dm(x, m, p, t=0.0):
        return -2.0 * m * 1.001

    result = validate_hamiltonian_consistency(H, slightly_off_dm, geom)
    assert result.is_valid
    warnings = [i for i in result.issues if i.severity is ValidationSeverity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].location == "dH_dm"


@pytest.mark.unit
def test_kink_at_probe_point_warns_but_does_not_gate():
    """A subgradient at a kink of H must not gate.

    H(p) = |p - 0.4157| is non-differentiable exactly at one probe magnitude; the
    claimed derivative is exact away from the kink and picks the +1 subgradient at
    it. Catches: dropping the one-sided-slope smoothness guard, which would make
    every non-smooth control cost (L1, bounded) refusable at construction.
    """
    kink = 0.4157

    def H(x, m, p, t=0.0):
        return float(abs(np.atleast_1d(p)[0] - kink))

    def dH_dm(x, m, p, t=0.0):
        return 0.0

    def dH_dp(x, m, p, t=0.0):
        return np.array([1.0 if np.atleast_1d(p)[0] >= kink else -1.0])

    result = validate_hamiltonian_consistency(H, dH_dm, _geometry(), dH_dp=dH_dp)
    assert result.is_valid
    assert [i.severity for i in result.issues] == [ValidationSeverity.WARNING]
    assert "differentiable" in result.issues[0].message


@pytest.mark.unit
@pytest.mark.parametrize("potential_magnitude", [1e8, 1e14], ids=["warn-floor", "witness-floor"])
def test_large_magnitude_hamiltonian_is_silent(potential_magnitude):
    """A correct H with a huge potential must not warn or gate.

    Central-difference roundoff is ~ eps_mach * |H| / step, so problem magnitude
    alone manufactures an apparent discrepancy: ~1e-3 relative at |H|=1e8 (which
    the warning floor must absorb) and O(1) absolute at |H|=1e14, where the m-term
    is lost to cancellation entirely (which the witness floor must absorb).
    Catches: dropping either roundoff floor, which turns problem magnitude into a
    spurious inconsistency report -- and at 1e14 into a refused construction.
    """
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
        potential=lambda x, t=0.0, magnitude=potential_magnitude: magnitude,
    )
    result = validate_hamiltonian_consistency(H, H.dm, _geometry(), dH_dp=H.dp)
    assert result.is_valid
    assert result.issues == []


@pytest.mark.unit
def test_probe_grid_catches_derivative_correct_only_at_m_equals_one():
    """dH_dm = -2 is exact at m=1 and wrong elsewhere; the m-probe must catch it.

    Catches: shrinking the probe grid back to the single point (m=1, p=0), which
    made this class of error invisible.
    """
    H = _hamiltonian()

    def dm_right_only_at_one(x, m, p, t=0.0):
        return -2.0

    result = validate_hamiltonian_consistency(H, dm_right_only_at_one, _geometry())
    assert not result.is_valid


@pytest.mark.unit
def test_probe_grid_catches_derivative_correct_only_at_p_zero():
    """A dH_dm error carried by the |p|^2 term is invisible at p=0.

    Congestion Hamiltonians have dH_dm proportional to |p|^2, so probing only p=0
    cannot see a wrong congestion derivative. Catches: dropping the nonzero-p probes.
    """

    def H(x, m, p, t=0.0):
        return float(np.dot(np.atleast_1d(p), np.atleast_1d(p))) / (2.0 * (1.0 + 3.0 * m))

    def dH_dm_wrong(x, m, p, t=0.0):
        # Correct value is -3|p|^2 / (2 (1+3m)^2); zero is right only at p = 0.
        return 0.0

    result = validate_hamiltonian_consistency(H, dH_dm_wrong, _geometry())
    assert not result.is_valid
    errors = [i for i in result.issues if i.severity is ValidationSeverity.ERROR]
    assert errors[0].location == "dH_dm"
    # The witness must be at nonzero momentum -- p = 0 cannot expose this error.
    assert any(component != 0.0 for component in result.context["dH_dm_witness_p"])


# ===========================================================================
# validate_custom_functions (aggregate)
# ===========================================================================


@pytest.mark.unit
def test_validate_custom_functions_all_valid():
    """All functions valid should produce no errors."""
    H = _hamiltonian()
    geom = _geometry()
    result = validate_custom_functions(
        hamiltonian=H,
        dH_dm=H.dm,
        dH_dp=H.dp,
        geometry=geom,
    )
    assert result.is_valid


@pytest.mark.unit
def test_validate_custom_functions_with_consistency():
    """Consistency check enabled with correct derivatives should pass."""
    H = _hamiltonian()
    geom = _geometry()
    result = validate_custom_functions(
        hamiltonian=H,
        dH_dm=H.dm,
        dH_dp=H.dp,
        geometry=geom,
        check_consistency=True,
    )
    assert result.is_valid


@pytest.mark.unit
def test_validate_custom_functions_defaults_to_checking_consistency():
    """The aggregate runs the consistency check unless explicitly told not to.

    Catches: the call site or the default reverting to check_consistency=False,
    which leaves the capability unreachable even when the validator can gate.
    """
    H = _hamiltonian()
    geom = _geometry()

    def wrong_dm(x, m, p, t=0.0):
        return 42.0

    assert not validate_custom_functions(hamiltonian=H, dH_dm=wrong_dm, dH_dp=H.dp, geometry=geom).is_valid
    assert validate_custom_functions(
        hamiltonian=H, dH_dm=wrong_dm, dH_dp=H.dp, geometry=geom, check_consistency=False
    ).is_valid


@pytest.mark.unit
def test_validate_custom_functions_propagates_consistency_invalidity():
    """The aggregate must forward is_valid, not only the issue list.

    Catches: the aggregate extending cons_result.issues while dropping
    cons_result.is_valid -- a second, independent way for the raise branch to die.
    """
    H = _hamiltonian()
    geom = _geometry()

    def wrong_dm(x, m, p, t=0.0):
        return 42.0

    result = validate_custom_functions(
        hamiltonian=H,
        dH_dm=wrong_dm,
        dH_dp=H.dp,
        geometry=geom,
        check_consistency=True,
    )
    assert not result.is_valid
    assert any(i.severity is ValidationSeverity.ERROR for i in result.issues)


# ===========================================================================
# Drift validation
# ===========================================================================


@pytest.mark.unit
def test_valid_drift_passes():
    """A drift with signature drift(x, m) should pass."""
    geom = _geometry()

    def my_drift(x, m):
        return -x

    result = validate_drift(my_drift, geom)
    assert result.is_valid


@pytest.mark.unit
def test_drift_wrong_signature_raises():
    """A drift with wrong arity should fail."""
    geom = _geometry()

    def bad_drift():
        return 0.0

    result = validate_drift(bad_drift, geom)
    assert not result.is_valid
    assert any("signature" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# Running cost validation
# ===========================================================================


@pytest.mark.unit
def test_valid_running_cost_passes():
    """A running cost with signature f(x, m) should pass."""
    geom = _geometry()

    def my_cost(x, m):
        return float(np.sum(x**2)) + m

    result = validate_running_cost(my_cost, geom)
    assert result.is_valid


@pytest.mark.unit
def test_running_cost_wrong_signature_raises():
    """A running cost with wrong arity should fail."""
    geom = _geometry()

    def bad_cost():
        return 0.0

    result = validate_running_cost(bad_cost, geom)
    assert not result.is_valid
    assert any("signature" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# Integration: MFGProblem construction triggers validation
# ===========================================================================


@pytest.mark.unit
def test_mfg_problem_valid_hamiltonian_accepted():
    """MFGProblem with valid Hamiltonian should construct without error."""
    problem = _problem(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: x**2,
    )
    assert problem is not None


@pytest.mark.unit
def test_mfg_problem_nan_hamiltonian_rejected():
    """MFGProblem with NaN-producing Hamiltonian should raise ValidationError."""

    class NaNHamiltonian(HamiltonianBase):
        @property
        def dimension(self):
            return 1

        def __call__(self, x, m, p, t=0.0):
            return float("nan")

    with pytest.raises(ValidationError, match="NaN"):
        _problem(
            m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=lambda x: x**2,
            hamiltonian=NaNHamiltonian(),
        )


@pytest.mark.unit
def test_mfg_problem_rejects_hamiltonian_with_wrong_dm():
    """End-to-end: construction refuses a Hamiltonian whose dm is not dH/dm.

    This is the live consumer the capability exists for (mfg_problem.py). Catches:
    the call site passing check_consistency=False again, the aggregate dropping
    is_valid, or the validator regressing to warning-only -- any one of which
    re-kills the raise branch.
    """

    class WrongDmHamiltonian(SeparableHamiltonian):
        def dm(self, x, m, p, t=0.0):
            return 42.0

    with pytest.raises(ValidationError, match="dH_dm"):
        _problem(
            m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=lambda x: x**2,
            hamiltonian=WrongDmHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: -(m**2),
                coupling_dm=lambda m: -2 * m,
            ),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "control_cost",
    [
        QuadraticControlCost(control_cost=1.0),
        L1ControlCost(control_cost=1.0),
        BoundedControlCost(control_cost=1.0, max_control=2.0),
    ],
    ids=["quadratic", "l1", "bounded"],
)
def test_mfg_problem_accepts_every_shipped_control_cost(control_cost):
    """Turning the gate on must not refuse any Hamiltonian the library ships.

    L1 and bounded costs are non-smooth; this is the false-positive regression
    test for the gate as a whole.
    """
    problem = _problem(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: x**2,
        hamiltonian=SeparableHamiltonian(
            control_cost=control_cost,
            coupling=lambda m: -(m**2),
            coupling_dm=lambda m: -2 * m,
        ),
    )
    assert problem is not None

"""
Validation for custom MFG functions (Hamiltonian, drift, running_cost).

This module validates user-provided mathematical functions to catch
errors before they propagate into solver iterations.

Issue #686: Custom function validation (Hamiltonian, drift, running_cost)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.validation.protocol import ValidationResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from mfgarchon.geometry.protocol import GeometryProtocol

# --- Finite-difference consistency checking (Issue #1642, capability C1) -----
#
# The consistency check is allowed to *gate* (set is_valid False, so the caller's
# `raise ValidationError(result)` fires), but only on an exhibited witness: a
# concrete (x, m, p) where the derivative identity measurably fails by a margin
# no finite-difference artifact can produce. Anything weaker stays a warning,
# because a numerical check that raises on a false positive is worse than one
# that warns.

_FD_STEP = 1e-6

# Warning threshold: relative discrepancy above this is reported, not gated.
_WARN_RTOL = 1e-4

# Witness threshold: a wrong derivative (sign flip, missing term, wrong factor)
# is O(1) relative. 1e-2 sits two orders above the warning tier and ~8 orders
# above the measured relative noise of this stencil on the shipped Hamiltonians.
_WITNESS_RTOL = 1e-2

# Central differencing has roundoff error ~ eps_mach * |H| / step. A witness must
# clear that bound by this factor before it can be called a defect rather than noise.
_WITNESS_NOISE_SAFETY = 1e4

# One-sided slopes of a smooth H agree to O(step); at a kink they disagree by O(1).
# Above this relative disagreement the derivative does not exist at the probe, so
# no witness can be exhibited there (L1 and bounded control costs both have kinks).
_SMOOTHNESS_RTOL = 1e-3


def _get_sample_inputs(
    geometry: GeometryProtocol,
    location: str,
) -> tuple[np.ndarray, np.ndarray, float, int, ValidationResult | None]:
    """Extract a sample point (x, p, m) from geometry for validation.

    Returns:
        (x_sample, p_sample, m_sample, dimension, error_result)
        If error_result is not None, the caller should return it immediately.
    """
    try:
        grid = geometry.get_spatial_grid()
        if isinstance(grid, np.ndarray) and grid.ndim == 2:
            # (N, d) array from TensorProductGrid, ImplicitDomain, etc.
            mid = grid.shape[0] // 2
            x_sample = grid[mid]  # shape (d,)
            dimension = grid.shape[1]
        elif isinstance(grid, np.ndarray) and grid.ndim == 1:
            # 1D flat grid
            x_sample = grid[len(grid) // 2]
            dimension = 1
        elif isinstance(grid, (list, tuple)):
            # Legacy: tuple of 1D arrays (meshgrid-style)
            x_sample = np.array([g[len(g) // 2] for g in grid])
            dimension = len(grid)
        else:
            x_sample = np.atleast_1d(grid)[len(np.atleast_1d(grid)) // 2]
            dimension = 1
    except Exception as e:
        result = ValidationResult()
        result.add_error(
            f"Could not get sample point from geometry: {e}",
            location=location,
        )
        return np.array([]), np.array([]), 1.0, 1, result

    x_sample = np.atleast_1d(x_sample).astype(float)
    p_sample = np.zeros(dimension, dtype=float)
    m_sample = 1.0
    return x_sample, p_sample, m_sample, dimension, None


def validate_custom_functions(
    hamiltonian: Any | None,
    dH_dm: Callable | None,
    dH_dp: Callable | None,
    geometry: GeometryProtocol,
    *,
    check_consistency: bool = True,
) -> ValidationResult:
    """
    Validate all custom Hamiltonian-related functions.

    Supports HamiltonianBase instances (preferred) and raw callables.

    Args:
        hamiltonian: HamiltonianBase instance or callable H(x, m, p, t)
        dH_dm: Derivative dH/dm (bound method or callable(x, m, p, t))
        dH_dp: Derivative dH/dp (bound method or callable(x, m, p, t))
        geometry: Geometry for sample point generation
        check_consistency: If True (default), verify dH_dm/dH_dp are the
            derivatives of H by finite differences. Costs O(dimension) extra
            scalar H evaluations; see validate_hamiltonian_consistency for when
            that check gates versus warns.

    Returns:
        ValidationResult with any issues found

    Issue #686: Custom function validation
    Issue #1642 (C1): consistency findings propagate to is_valid
    """
    result = ValidationResult()

    if hamiltonian is not None:
        h_result = validate_hamiltonian(hamiltonian, geometry)
        result.issues.extend(h_result.issues)
        if not h_result.is_valid:
            result.is_valid = False

    if dH_dm is not None:
        dm_result = validate_hamiltonian_derivative(dH_dm, geometry, "dH_dm")
        result.issues.extend(dm_result.issues)
        if not dm_result.is_valid:
            result.is_valid = False

    if dH_dp is not None:
        dp_result = validate_hamiltonian_derivative(dH_dp, geometry, "dH_dp")
        result.issues.extend(dp_result.issues)
        if not dp_result.is_valid:
            result.is_valid = False

    # Check consistency if requested and all functions provided
    if check_consistency and hamiltonian is not None and dH_dm is not None:
        cons_result = validate_hamiltonian_consistency(hamiltonian, dH_dm, geometry, dH_dp=dH_dp)
        result.issues.extend(cons_result.issues)
        result.context.update(cons_result.context)
        if not cons_result.is_valid:
            result.is_valid = False

    return result


def validate_hamiltonian(
    hamiltonian: Any,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate Hamiltonian function H(x, m, p, t).

    Supports HamiltonianBase instances (called as H(x, m, p, t))
    and raw callables (tried with same signature).

    Checks:
    - Callable with correct signature
    - Returns float or array
    - No NaN/Inf in output

    Args:
        hamiltonian: HamiltonianBase instance or callable
        geometry: Geometry for sample point

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    x_sample, p_sample, m_sample, _dim, err = _get_sample_inputs(geometry, "hamiltonian")
    if err is not None:
        return err

    # Evaluate: HamiltonianBase.__call__ signature is (x, m, p, t=0.0)
    try:
        value = hamiltonian(x_sample, m_sample, p_sample, 0.0)
    except TypeError as e:
        result.add_error(
            f"Hamiltonian has wrong signature: {e}",
            location="hamiltonian",
            suggestion="Hamiltonian should have signature H(x, m, p, t)",
        )
        return result
    except Exception as e:
        result.add_warning(
            f"Hamiltonian raised exception at sample point: {e}",
            location="hamiltonian",
        )
        return result

    # Check return type
    if not isinstance(value, (int, float, np.integer, np.floating, np.ndarray)):
        result.add_error(
            f"Hamiltonian must return float or ndarray, got {type(value).__name__}",
            location="hamiltonian",
        )
        return result

    # Check for NaN/Inf
    if np.isscalar(value):
        if not np.isfinite(value):
            result.add_error(
                "Hamiltonian returned NaN or Inf at sample point",
                location="hamiltonian",
            )
    elif isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
        result.add_error(
            "Hamiltonian returned array with NaN or Inf values",
            location="hamiltonian",
        )

    return result


def validate_hamiltonian_derivative(
    derivative_func: Callable,
    geometry: GeometryProtocol,
    name: str,
) -> ValidationResult:
    """
    Validate a Hamiltonian derivative function (dH_dm or dH_dp).

    The derivative should have signature f(x, m, p, t) matching
    HamiltonianBase.dm() / HamiltonianBase.dp().

    Args:
        derivative_func: Derivative function (bound method or callable)
        geometry: Geometry for sample point
        name: Name for error messages ("dH_dm" or "dH_dp")

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    x_sample, p_sample, m_sample, _dim, err = _get_sample_inputs(geometry, name)
    if err is not None:
        return err

    # Evaluate: derivative signature is (x, m, p, t=0.0)
    try:
        value = derivative_func(x_sample, m_sample, p_sample, 0.0)
    except TypeError as e:
        result.add_error(
            f"{name} has wrong signature: {e}",
            location=name,
            suggestion=f"{name} should have signature {name}(x, m, p, t)",
        )
        return result
    except Exception as e:
        result.add_warning(
            f"{name} raised exception at sample point: {e}",
            location=name,
        )
        return result

    # Check return type
    if not isinstance(value, (int, float, np.integer, np.floating, np.ndarray)):
        result.add_error(
            f"{name} must return float or ndarray, got {type(value).__name__}",
            location=name,
        )

    return result


@dataclass(frozen=True)
class _FDComparison:
    """One finite-difference vs analytical comparison at one probe point."""

    m: float
    p: np.ndarray
    numerical: float
    analytical: float
    diff: float
    noise_bound: float
    smooth: bool
    finite: bool
    warn_rtol: float

    @property
    def is_inconsistent(self) -> bool:
        """Warning grade: relative discrepancy above the reporting tolerance.

        Also required to clear the differencing roundoff bound. Without that, a
        correct Hamiltonian of large magnitude (a big potential term) reports a
        discrepancy the stencil cannot resolve at this step size.
        """
        if not self.finite:
            return True
        if self.diff <= self.noise_bound:
            return False
        return self.diff / max(abs(self.analytical), 1e-10) > self.warn_rtol

    @property
    def is_witness(self) -> bool:
        """Error grade: a discrepancy no finite-difference artifact can produce.

        All three conditions are required. Smoothness rules out a kink (where the
        derivative simply does not exist); the noise bound rules out roundoff; the
        relative margin rules out a merely imprecise analytical derivative.
        """
        if not (self.finite and self.smooth):
            return False
        return self.diff > _WITNESS_NOISE_SAFETY * self.noise_bound and self.diff > _WITNESS_RTOL * max(
            abs(self.numerical), abs(self.analytical)
        )


def _compare_derivative(
    h_minus: float,
    h_center: float,
    h_plus: float,
    step: float,
    analytical: float,
    m: float,
    p: np.ndarray,
    warn_rtol: float,
) -> _FDComparison:
    """Compare a central difference of H against a claimed analytical derivative."""
    numerical = (h_plus - h_minus) / (2.0 * step)
    forward = (h_plus - h_center) / step
    backward = (h_center - h_minus) / step

    finite = bool(np.isfinite([h_minus, h_center, h_plus, analytical]).all())
    smooth = finite and abs(forward - backward) <= _SMOOTHNESS_RTOL * max(abs(forward), abs(backward), 1.0)
    noise_bound = float(np.finfo(float).eps) * max(abs(h_minus), abs(h_center), abs(h_plus), 1.0) / step

    return _FDComparison(
        m=m,
        p=p.copy(),
        numerical=numerical,
        analytical=analytical,
        diff=abs(numerical - analytical) if finite else float("inf"),
        noise_bound=noise_bound,
        smooth=smooth,
        finite=finite,
        warn_rtol=warn_rtol,
    )


def _consistency_probes(dimension: int) -> Iterator[tuple[float, np.ndarray]]:
    """Probe points (m, p) for the derivative-consistency search.

    Two densities because a wrong dH_dm can still be right at a single m (a user
    writing a constant where the derivative is affine). Three momenta because
    dH_dp of any Hamiltonian even in p vanishes at p=0, and CongestionHamiltonian's
    dH_dm is proportional to |p|^2 and therefore also vanishes there — probing only
    p=0 makes both checks structurally blind.

    The nonzero magnitudes are deliberately not round numbers. L1ControlCost has a
    kink at |p| = control_cost and BoundedControlCost at |p| = control_cost *
    max_control, and users pick round values for those; a probe landing exactly on
    a kink produces a true-but-useless warning. The smoothness guard in
    _FDComparison.is_witness is the backstop that keeps such a probe from gating.
    """
    for m in (0.5, 1.0):
        for p_magnitude in (0.0, 0.4157, 1.3691):
            yield m, np.full(dimension, p_magnitude, dtype=float)


def _report(
    result: ValidationResult,
    location: str,
    quantity: str,
    x_sample: np.ndarray,
    witness: _FDComparison | None,
    inconsistent: _FDComparison | None,
) -> None:
    """Emit at most one issue for `location`, at the highest grade observed."""
    worst = witness if witness is not None else inconsistent
    if worst is None:
        return

    point = f"x={np.array2string(x_sample, precision=4)}, m={worst.m:.4g}, p={np.array2string(worst.p, precision=4)}"

    if not worst.finite:
        result.add_warning(
            f"{quantity} consistency could not be evaluated at {point}: "
            f"non-finite values (analytical={worst.analytical}, finite-difference={worst.numerical})",
            location=location,
            suggestion=f"Ensure H and {quantity} are finite on the problem's admissible (m, p) range",
        )
        return

    detail = (
        f"at {point}: analytical={worst.analytical:.8g}, "
        f"finite-difference={worst.numerical:.8g} (step={_FD_STEP:g}), "
        f"|difference|={worst.diff:.6g}"
    )
    result.context[f"{location}_numerical"] = worst.numerical
    result.context[f"{location}_analytical"] = worst.analytical
    result.context[f"{location}_witness_m"] = worst.m
    result.context[f"{location}_witness_p"] = worst.p.tolist()

    if witness is not None:
        result.add_error(
            f"{quantity} is not the derivative of H {detail}. This exceeds "
            f"{_WITNESS_RTOL:g} relative and {_WITNESS_NOISE_SAFETY:g}x the "
            f"finite-difference roundoff bound {worst.noise_bound:.3g}, and H is "
            f"smooth at this point, so the discrepancy is not a differencing artifact.",
            location=location,
            suggestion=f"Correct {quantity} so it equals the derivative of H, or correct H",
        )
    elif not worst.smooth:
        result.add_warning(
            f"{quantity} may be inconsistent with H {detail}. H is not "
            f"differentiable at this probe (its one-sided slopes disagree), so the "
            f"central difference does not approximate a derivative here.",
            location=location,
            suggestion=f"Ignore if {quantity} is a subgradient at a kink of H; otherwise verify it",
        )
    else:
        result.add_warning(
            f"{quantity} may be inconsistent with H {detail}. Relative discrepancy "
            f"exceeds {worst.warn_rtol:g} but is not large enough to rule out a "
            f"finite-difference artifact.",
            location=location,
            suggestion=f"Verify {quantity} is the correct derivative of H",
        )


def validate_hamiltonian_consistency(
    hamiltonian: Any,
    dH_dm: Callable,
    geometry: GeometryProtocol,
    tolerance: float = _WARN_RTOL,
    dH_dp: Callable | None = None,
) -> ValidationResult:
    """
    Check that dH_dm and dH_dp are the derivatives of H, by finite differences.

    Numerical checks, over a small grid of probe points (m, p):
        dH_dm_numerical = (H(x, m+eps, p, t) - H(x, m-eps, p, t)) / (2*eps)
        dH_dp_numerical[i] = (H(x, m, p+eps*e_i, t) - H(x, m, p-eps*e_i, t)) / (2*eps)

    Severity (Issue #1642, capability C1). This validator can invalidate its
    result, so a caller doing `if not result.is_valid: raise ValidationError(...)`
    will fire. It does so only on an **exhibited witness** — a probe point where
    H is smooth, all values are finite, and the discrepancy clears both a relative
    margin and the differencing roundoff bound by a wide factor. Every weaker
    signal (small discrepancy, kink at the probe, non-finite value, failure to
    evaluate) is a warning, because a false positive here would refuse a correct
    problem at construction time.

    Args:
        hamiltonian: HamiltonianBase instance or callable H(x, m, p, t)
        dH_dm: Claimed derivative dH/dm with signature (x, m, p, t)
        geometry: Geometry for sample point
        tolerance: Relative tolerance for the warning tier
        dH_dp: Claimed gradient dH/dp with signature (x, m, p, t). Optional.

    Returns:
        ValidationResult; is_valid is False when a witness was exhibited.
    """
    result = ValidationResult()

    x_sample, _p_sample, _m_sample, dimension, err = _get_sample_inputs(geometry, "hamiltonian")
    if err is not None:
        return err

    step = _FD_STEP
    dm_witness: _FDComparison | None = None
    dm_inconsistent: _FDComparison | None = None
    dp_witness: list[_FDComparison | None] = [None] * dimension
    dp_inconsistent: list[_FDComparison | None] = [None] * dimension

    try:
        for m, p in _consistency_probes(dimension):
            h_center = float(hamiltonian(x_sample, m, p, 0.0))

            # --- dH_dm ---
            h_m_plus = float(hamiltonian(x_sample, m + step, p, 0.0))
            h_m_minus = float(hamiltonian(x_sample, m - step, p, 0.0))
            comparison = _compare_derivative(
                h_m_minus,
                h_center,
                h_m_plus,
                step,
                float(dH_dm(x_sample, m, p, 0.0)),
                m,
                p,
                tolerance,
            )
            if comparison.is_witness and dm_witness is None:
                dm_witness = comparison
            if comparison.is_inconsistent and dm_inconsistent is None:
                dm_inconsistent = comparison

            # --- dH_dp, per component ---
            if dH_dp is None:
                continue
            dp_analytical = np.atleast_1d(dH_dp(x_sample, m, p, 0.0)).astype(float)
            for i in range(dimension):
                p_plus = p.copy()
                p_minus = p.copy()
                p_plus[i] += step
                p_minus[i] -= step
                comparison = _compare_derivative(
                    float(hamiltonian(x_sample, m, p_minus, 0.0)),
                    h_center,
                    float(hamiltonian(x_sample, m, p_plus, 0.0)),
                    step,
                    float(dp_analytical[i]),
                    m,
                    p,
                    tolerance,
                )
                if comparison.is_witness and dp_witness[i] is None:
                    dp_witness[i] = comparison
                if comparison.is_inconsistent and dp_inconsistent[i] is None:
                    dp_inconsistent[i] = comparison

    except Exception as e:
        # Evaluation failed, so no witness can be exhibited: warn, never gate.
        result.add_warning(
            f"Could not verify Hamiltonian consistency: {e}",
            location="hamiltonian",
        )
        return result

    _report(result, "dH_dm", "dH_dm", x_sample, dm_witness, dm_inconsistent)
    for i in range(dimension):
        _report(result, f"dH_dp[{i}]", f"dH_dp[{i}]", x_sample, dp_witness[i], dp_inconsistent[i])

    return result


def validate_drift(
    drift: Callable,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate drift function for FP equation.

    The drift should have signature drift(x), drift(x, m), or drift(t, x, m)
    and return a vector of the same dimension as x.

    Args:
        drift: Drift function
        geometry: Geometry for sample point

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    x_sample, _p_sample, _m_sample, dimension, err = _get_sample_inputs(geometry, "drift")
    if err is not None:
        return err

    m_sample = np.ones(dimension if dimension > 1 else 1)

    # Try different signatures: drift(x, m), drift(t, x, m), drift(x)
    value = None
    for args in [(x_sample, m_sample), (0.0, x_sample, m_sample), (x_sample,)]:
        try:
            value = drift(*args)
            break
        except TypeError:
            continue

    if value is None:
        result.add_error(
            "Drift has wrong signature",
            location="drift",
            suggestion="Drift should have signature drift(x), drift(x, m), or drift(t, x, m)",
        )
        return result

    # Check return shape
    if isinstance(value, np.ndarray):
        if value.shape != (dimension,) and value.shape != ():
            result.add_warning(
                f"Drift returned shape {value.shape}, expected ({dimension},)",
                location="drift",
            )

    return result


def validate_running_cost(
    running_cost: Callable,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate running cost function.

    The running cost should have signature f(x), f(x, m), or f(t, x, m)
    and return a scalar.

    Args:
        running_cost: Running cost function
        geometry: Geometry for sample point

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    x_sample, _p_sample, _m_sample, _dimension, err = _get_sample_inputs(geometry, "running_cost")
    if err is not None:
        return err

    m_sample = 1.0

    # Try different signatures: f(x, m), f(t, x, m), f(x)
    value = None
    for args in [(x_sample, m_sample), (0.0, x_sample, m_sample), (x_sample,)]:
        try:
            value = running_cost(*args)
            break
        except TypeError:
            continue

    if value is None:
        result.add_error(
            "Running cost has wrong signature",
            location="running_cost",
            suggestion="Running cost should have signature f(x), f(x, m), or f(t, x, m)",
        )
        return result

    # Check return type
    if not np.isscalar(value):
        result.add_warning(
            f"Running cost should return scalar, got {type(value).__name__}",
            location="running_cost",
        )

    return result

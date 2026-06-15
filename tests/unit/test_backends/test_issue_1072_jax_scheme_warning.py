"""Tests for Issue #1072 interim mitigation: JAX backend scheme-downgrade warning.

The JAX backend's ``hjb_step`` / ``fpk_step`` ghost-implement 2nd-order central
differences (``jax_backend.py`` ``_hjb_step_impl`` / ``_fpk_step_impl``) rather
than the high-order operators (WENO5 / upwind) used by the NumPy path. Selecting
JAX for a high-order scheme therefore silently solves *different* math, making
cross-backend benchmarks apples-to-oranges. The interim fix
(:func:`mfgarchon.backends.warn_if_jax_scheme_downgraded`) emits a one-time
warning at the seam where both the backend and the scheme are known.

Gate (Issue #1072):
- Warning FIRES on JAX + high-order scheme.
- Warning does NOT fire on JAX + natively-2nd-order-central scheme.
- Warning does NOT fire on the NumPy path (any scheme).
- Warning is one-time per scheme value.

These tests target the warning helper directly so they do not depend on JAX
being installed (the helper is pure dispatch over the backend name + scheme).
"""

from __future__ import annotations

import warnings

import pytest

from mfgarchon.backends import warn_if_jax_scheme_downgraded
import mfgarchon.backends as backends_pkg
from mfgarchon.types import NumericalScheme


@pytest.fixture(autouse=True)
def _reset_warned_guard():
    """Clear the one-time guard so each test starts from a clean slate."""
    backends_pkg._JAX_SCHEME_DOWNGRADE_WARNED.clear()
    yield
    backends_pkg._JAX_SCHEME_DOWNGRADE_WARNED.clear()


# --------------------------------------------------------------------------- #
# Warning FIRES: JAX + high-order scheme
# --------------------------------------------------------------------------- #


def test_jax_high_order_scheme_warns():
    """JAX + a high-order scheme (SL_CUBIC, O(h^4)) emits the warning."""
    with pytest.warns(UserWarning, match="JAX backend"):
        fired = warn_if_jax_scheme_downgraded("jax", NumericalScheme.SL_CUBIC)
    assert fired is True


def test_jax_upwind_scheme_warns():
    """JAX + FDM_UPWIND warns: JAX uses central, not the requested upwind stencil."""
    with pytest.warns(UserWarning, match="2nd-order central"):
        fired = warn_if_jax_scheme_downgraded("jax", NumericalScheme.FDM_UPWIND)
    assert fired is True


def test_warning_message_mentions_issue_and_weno():
    """The warning is actionable: names the cause and the interim-fix issue."""
    with pytest.warns(UserWarning) as record:
        warn_if_jax_scheme_downgraded("jax", NumericalScheme.FDM_UPWIND)
    msg = str(record[0].message)
    assert "Issue #1072" in msg
    assert "WENO5" in msg or "high-order" in msg


def test_scheme_accepts_string_value():
    """The helper accepts a bare scheme string (not only the enum)."""
    with pytest.warns(UserWarning, match="JAX backend"):
        fired = warn_if_jax_scheme_downgraded("jax", "fdm_upwind")
    assert fired is True


# --------------------------------------------------------------------------- #
# Warning does NOT fire: JAX + natively-2nd-order-central scheme
# --------------------------------------------------------------------------- #


def test_jax_fdm_centered_does_not_warn():
    """FDM_CENTERED is exactly what the JAX 2nd-order central path implements."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        fired = warn_if_jax_scheme_downgraded("jax", NumericalScheme.FDM_CENTERED)
    assert fired is False


def test_jax_fdm_centered_string_does_not_warn():
    """Same no-warn behavior for the bare 'fdm_centered' string."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fired = warn_if_jax_scheme_downgraded("jax", "fdm_centered")
    assert fired is False


# --------------------------------------------------------------------------- #
# Warning does NOT fire: NumPy / other backends, or no scheme
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", list(NumericalScheme))
def test_numpy_backend_never_warns(scheme):
    """NumPy path is the high-order reference — it never warns, for any scheme."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fired = warn_if_jax_scheme_downgraded("numpy", scheme)
    assert fired is False


def test_default_backend_none_does_not_warn():
    """A None backend name (config default = numpy) never warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fired = warn_if_jax_scheme_downgraded(None, NumericalScheme.FDM_UPWIND)
    assert fired is False


def test_pytorch_backend_does_not_warn():
    """The warning is JAX-specific; other non-default backends are out of scope."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fired = warn_if_jax_scheme_downgraded("pytorch", NumericalScheme.FDM_UPWIND)
    assert fired is False


def test_no_scheme_does_not_warn():
    """Expert Mode (no resolvable scheme) cannot be classified -> no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fired = warn_if_jax_scheme_downgraded("jax", None)
    assert fired is False


# --------------------------------------------------------------------------- #
# One-time semantics
# --------------------------------------------------------------------------- #


def test_warning_fires_only_once_per_scheme():
    """Repeated identical calls warn once; the guard suppresses the rest."""
    with pytest.warns(UserWarning):
        first = warn_if_jax_scheme_downgraded("jax", NumericalScheme.FDM_UPWIND)
    assert first is True

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # second call must be silent
        second = warn_if_jax_scheme_downgraded("jax", NumericalScheme.FDM_UPWIND)
    assert second is False


def test_distinct_schemes_warn_independently():
    """The guard is keyed per scheme: a different scheme still warns once."""
    with pytest.warns(UserWarning):
        warn_if_jax_scheme_downgraded("jax", NumericalScheme.FDM_UPWIND)
    with pytest.warns(UserWarning):
        fired = warn_if_jax_scheme_downgraded("jax", NumericalScheme.SL_CUBIC)
    assert fired is True

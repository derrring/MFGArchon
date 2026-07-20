"""Every backend must agree on the reduction conventions (Issue #1663).

`BaseBackend` declares `mean`, `std`, `max`, `min` and `trapezoid` as abstract, but its
docstrings originally said only what each computes, never under which convention. Three
implementers each inherited their own library's default, and `TorchBackend.std` silently
returned the Bessel-corrected `N-1` quantity where NumPy and JAX returned `N`.

That divergence survived because the test meant to catch it --
`tests/integration/test_cross_backend_consistency.py::test_backend_factory_integration` --
called `backend.sum()`, a method that has never been part of the contract. The AttributeError
was swallowed by an `except Exception: print(...)`, leaving its results dict empty, so the
comparison block was skipped with the message "Only one backend available" while all three
backends had in fact been constructed successfully.

This file is the pin that was missing: same input, every backend, every contract reduction.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.backends import create_backend

REDUCTIONS = ("mean", "std", "max", "min")

# A backend may legitimately run at float32 -- on Apple Silicon the torch backend is forced to
# it, with a warning, because MPS has no float64. So the tolerance must admit float32 epsilon
# (~1e-7) while still catching a convention divergence. The one this file exists to pin was
# 5e-3 relative (Bessel N-1 vs N), roughly 50000x larger than float32 noise, so 1e-5 separates
# them with four orders of margin either way.
RTOL = 1e-5


def _available() -> list[str]:
    out = []
    for name in ("numpy", "torch", "jax"):
        try:
            create_backend(name)
            out.append(name)
        except Exception:
            pass  # absence of an optional backend is not a failure here
    return out


@pytest.fixture(scope="module")
def field() -> np.ndarray:
    """A deterministic non-symmetric field. Non-constant, so std cannot be trivially zero."""
    return np.sin(np.linspace(0.0, np.pi, 101)) + 0.3 * np.linspace(0.0, 1.0, 101)


def test_at_least_one_backend_is_constructible():
    """Guard the guard: if this list is ever empty the comparisons below are vacuous."""
    assert _available(), "no backend could be constructed; every comparison below would be empty"


@pytest.mark.parametrize("op", REDUCTIONS)
def test_reductions_agree_across_backends(op, field):
    """All constructible backends must return the same value for the same input.

    `std` is the one that failed before this pin: torch returned 0.314315110 against
    0.312755227 for NumPy and JAX, a ratio of exactly sqrt(N/(N-1)).
    """
    names = _available()
    if len(names) < 2:
        pytest.skip(f"only {names} constructible; cross-backend comparison needs at least two")

    values = {}
    for name in names:
        backend = create_backend(name)
        values[name] = float(getattr(backend, op)(backend.array(field)))

    reference = values[names[0]]
    for name, value in values.items():
        assert value == pytest.approx(reference, rel=RTOL), (
            f"backend {name!r} returns {op}={value!r} against {names[0]!r}'s {reference!r} "
            f"(ratio {value / reference:.9f}); backends must share one convention -- see "
            f"BaseBackend.{op}"
        )


def test_std_is_the_population_convention(field):
    """Pin the convention itself, not merely agreement -- three backends agreeing on the
    WRONG convention would satisfy the test above."""
    expected = float(np.std(field, ddof=0))
    bessel = float(np.std(field, ddof=1))
    assert expected != pytest.approx(bessel), "fixture too degenerate to discriminate the two"

    for name in _available():
        backend = create_backend(name)
        got = float(backend.std(backend.array(field)))
        assert got == pytest.approx(expected, rel=RTOL), (
            f"backend {name!r} std={got!r}; expected the population value {expected!r} "
            f"(ddof=0), not the Bessel-corrected {bessel!r} (ddof=1)"
        )

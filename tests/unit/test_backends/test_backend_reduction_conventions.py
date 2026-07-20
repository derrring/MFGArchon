"""Every backend must agree on the reduction conventions (Issue #1663).

`BaseBackend` declares `mean`, `std`, `max`, `min` and `trapezoid` as abstract, but its
docstrings originally said only what each computes, never under which convention. Four
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

import pathlib

import pytest

import numpy as np

from mfgarchon.backends import create_backend

# Hand-maintained, deliberately. The obvious self-deriving fix does not work:
# BaseBackend.__subclasses__() returns only three after a plain `import mfgarchon.backends`,
# because numba_backend is imported lazily -- so deriving from it would silently reproduce the
# very undercount this file exists to catch. test_candidates_covers_every_backend_module below
# guards the tuple against the filesystem instead, which is independent of both.
_CANDIDATES = ("numpy", "torch", "jax", "numba")

REDUCTIONS = ("mean", "std", "max", "min")

# A backend may legitimately run at float32 -- on Apple Silicon the torch backend is forced to
# it, with a warning, because MPS has no float64. So the tolerance must admit float32 epsilon
# (~1e-7) while still catching a convention divergence. The one this file exists to pin was
# 5e-3 relative (Bessel N-1 vs N). Measured on this file's fixture: float32 noise reaches
# 7.06e-8, so 1e-5 sits 2.15 orders above it, and the divergence is 4.9876e-3, 2.70 orders
# below. Margin either way, but two orders, not four.
RTOL = 1e-5


def _available() -> list[str]:
    out = []
    for name in _CANDIDATES:
        try:
            create_backend(name)
            out.append(name)
        except ImportError:
            # Not installed. Anything ELSE -- a backend that is present but fails to
            # construct -- must propagate: swallowing it is the defect this PR removes
            # from test_cross_backend_consistency.py.
            pass
    return out


@pytest.fixture(scope="module")
def field() -> np.ndarray:
    """A deterministic non-symmetric field. Non-constant, so std cannot be trivially zero."""
    # The +0.5 keeps every reduction's reference away from zero. Without it min() is exactly
    # 0.0, which makes pytest.approx(0.0, rel=...) degenerate to an absolute 1e-12 -- a
    # different tolerance than documented -- and makes the failure message's value/reference
    # raise ZeroDivisionError instead of printing. std is translation-invariant, so the
    # convention pin below is unaffected. This is the same degenerate-reference defect this
    # PR removes from test_cross_backend_consistency.py, reintroduced here and caught in review.
    return np.sin(np.linspace(0.0, np.pi, 101)) + 0.3 * np.linspace(0.0, 1.0, 101) + 0.5


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
    """Pin the convention itself, not merely agreement -- every backend agreeing on the
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


def test_candidates_covers_every_backend_module():
    """A fifth backend must fail here until _CANDIDATES names it.

    This is what licenses the hand-maintained tuple: the filesystem is an independent
    source, unlike __subclasses__(), which under-reports lazily imported backends.
    """
    import mfgarchon.backends

    pkg = pathlib.Path(mfgarchon.backends.__file__).parent
    found = {p.stem.removesuffix("_backend") for p in pkg.glob("*_backend.py")} - {"base"}
    assert found == set(_CANDIDATES), (
        f"backend modules on disk {sorted(found)} do not match _CANDIDATES "
        f"{sorted(_CANDIDATES)}; add the new backend to _CANDIDATES so it gets pinned"
    )

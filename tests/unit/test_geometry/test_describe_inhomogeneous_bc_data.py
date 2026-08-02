"""Direct tests for the single owner of "is this boundary datum verifiably zero?" (#1802).

The predicate had no unit test of its own: it was covered only through its two callers, the
#1686 Neumann capability gate and the #1681 regime guard. That is thin for a shared owner --
the callers exercise a slice of the input space each, and the branches they do not reach
(providers, arrays, unrecognised types, the malformed-object refusal) were unpinned.

Two properties matter more than the individual verdicts:

- **Anything not provably zero is described**, never assumed homogeneous. ``isinstance(v,
  (int, float))`` alone would accept ``neumann_bc(value=lambda t: 5.0)`` and drop it
  silently -- the behaviour both callers exist to stop.
- **A malformed object raises rather than reporting "nothing disagrees"**, matching
  ``geometric_operations`` (#1691). A capability gate that answers "clean" for an object it
  cannot read is a pass it did not earn.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import numpy as np

from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.bc_utils import describe_inhomogeneous_bc_data
from mfgarchon.geometry.boundary.providers import AdjointConsistentProvider


def _bc(*, segments=(), default_bc=None, default_value=None):
    return SimpleNamespace(segments=list(segments), default_bc=default_bc, default_value=default_value)


def _seg(value, bc_type=BCType.NEUMANN):
    return SimpleNamespace(bc_type=bc_type, value=value)


class TestZeroIsAccepted:
    @pytest.mark.parametrize(
        "value",
        [None, 0, 0.0, -0.0, np.float64(0.0), np.zeros(5), np.zeros((2, 3))],
        ids=["none", "int0", "float0", "neg0", "np0", "zeros1d", "zeros2d"],
    )
    def test_verifiably_zero_data_is_not_reported(self, value):
        assert describe_inhomogeneous_bc_data(_bc(segments=[_seg(value)])) == []


class TestNonZeroIsDescribed:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.2, [0.2]),
            (-3, [-3.0]),
            (np.array([0.0, 1e-12]), ["<array>"]),
            (lambda t: 5.0, ["<callable>"]),
            # The provider branch: is_provider is True and callable is False, so without
            # the branch this falls through to float() and reads as "<unrecognised ...>".
            # Deleting the branch -- or returning None from it, the exact fail-silent --
            # left all 21 tests green before this case existed.
            (AdjointConsistentProvider(side="left", sigma=0.1), ["<provider>"]),
            ("g", ["<unrecognised str>"]),
        ],
        ids=["float", "negative", "array-with-one-nonzero", "callable", "provider", "unrecognised"],
    )
    def test_anything_not_provably_zero_is_described(self, value, expected):
        assert describe_inhomogeneous_bc_data(_bc(segments=[_seg(value)])) == expected

    def test_a_callable_is_refused_rather_than_evaluated(self):
        """A time-dependent value cannot be compared to zero here, so it must not be assumed."""
        assert describe_inhomogeneous_bc_data(_bc(segments=[_seg(lambda t: 0.0)])) == ["<callable>"]


class TestBothChannels:
    def test_the_fall_through_default_is_a_value_too(self):
        """#1686's hole, and #1802's first guard reproduced it: segments-only misses this."""
        bc = _bc(segments=[_seg(0.0)], default_bc=BCType.NEUMANN, default_value=0.7)
        assert describe_inhomogeneous_bc_data(bc) == [0.7]

    def test_bc_types_filters_both_channels(self):
        bc = _bc(segments=[_seg(0.2, BCType.DIRICHLET)], default_bc=BCType.NEUMANN, default_value=0.7)
        assert describe_inhomogeneous_bc_data(bc, bc_types={BCType.NEUMANN}) == [0.7]
        assert describe_inhomogeneous_bc_data(bc, bc_types={BCType.DIRICHLET}) == [0.2]
        assert describe_inhomogeneous_bc_data(bc, bc_types=None) == [0.2, 0.7]

    def test_results_are_deduplicated_and_ordered(self):
        bc = _bc(segments=[_seg(0.2), _seg(0.2), _seg(0.1)])
        assert describe_inhomogeneous_bc_data(bc) == [0.1, 0.2]


class TestShapeContract:
    def test_a_non_bc_object_is_not_a_finding(self):
        for not_a_bc in (None, "periodic", 42):
            assert describe_inhomogeneous_bc_data(not_a_bc) == []

    @pytest.mark.parametrize(
        "obj",
        [SimpleNamespace(default_bc=BCType.NEUMANN, default_value=0.7), SimpleNamespace(segments=[])],
        ids=["default-without-segments", "segments-without-default"],
    )
    def test_half_a_segmented_bc_raises_instead_of_reporting_clean(self, obj):
        """#1691: exactly one of the pair is the signature of a rename, not of a clean BC."""
        with pytest.raises(AttributeError, match=r"has .* but no "):
            describe_inhomogeneous_bc_data(obj)


class TestAgainstTheRealBoundaryConditions:
    """The SimpleNamespace fixtures above are convenient; these use the shipped class."""

    def test_real_homogeneous_bc_is_clean(self):
        bc = BoundaryConditions(
            dimension=1,
            segments=[BCSegment(name="l", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min")],
            domain_bounds=[[0.0, 1.0]],
        )
        assert describe_inhomogeneous_bc_data(bc) == []

    def test_real_inhomogeneous_bc_is_described(self):
        bc = BoundaryConditions(
            dimension=1,
            segments=[BCSegment(name="l", bc_type=BCType.DIRICHLET, value=0.25, boundary="x_min")],
            domain_bounds=[[0.0, 1.0]],
        )
        assert describe_inhomogeneous_bc_data(bc) == [0.25]

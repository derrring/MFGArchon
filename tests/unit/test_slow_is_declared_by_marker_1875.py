"""A test is slow because it carries the marker, never because of its name (#1875).

`tests/conftest.py` used to mark any test whose name contained "large", "slow" or "benchmark" as `slow`,
and the gate deselects `slow`. So a descriptive name removed a fast test from the gate with nothing at
the test to show it. At `bfdab257` it held 31 cases out of the gate, which run in 0.40 s together; one is
#1684's pin of `MultiPopulationResult.errors`. #2570's max-over-fields pins were held out the same way
until #2570 renamed them. The rule is gone. A name may still say slow or
benchmark, and then it must carry the marker: `pytest_collection_modifyitems` refuses the session
otherwise, so these tests' own names avoid both words.
"""

from __future__ import annotations

import pytest

from tests.conftest import pytest_collection_modifyitems, undeclared_slow_names


class _Item:
    """The two things the check reads from a collected pytest item."""

    def __init__(self, name: str, markers: tuple[str, ...] = ()):
        self.originalname = name
        self.name = f"{name}[case]"
        self.nodeid = f"tests/unit/test_x.py::{self.name}"
        self._markers = markers

    def get_closest_marker(self, marker: str):
        return object() if marker in self._markers else None


@pytest.mark.parametrize("name", ["test_a_slow_solve", "test_benchmark_the_kernel"])
def test_a_name_that_announces_the_marker_must_carry_it(name):
    assert undeclared_slow_names([_Item(name)]) == [f"tests/unit/test_x.py::{name}[case]"]
    assert undeclared_slow_names([_Item(name, ("slow",))]) == []
    assert undeclared_slow_names([_Item(name, ("benchmark",))]) == []


def test_a_name_no_longer_decides_the_marker():
    """The words that used to deselect a test by name now leave it in the gate, and need no marker."""
    names = ["test_the_largest_field_change_decides", "test_the_reported_error_is_the_larger_of_the_two_fields"]
    assert undeclared_slow_names([_Item(name) for name in names]) == []


def test_collection_refuses_an_undeclared_name():
    """The check is wired into collection, so an undeclared name fails the session, not just this helper."""
    with pytest.raises(pytest.UsageError, match=r"named as slow or benchmark but carry neither marker"):
        pytest_collection_modifyitems(config=None, items=[_Item("test_a_slow_solve")])

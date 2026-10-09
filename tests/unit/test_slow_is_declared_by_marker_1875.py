"""A test is slow because it carries the marker, never because of its name (#1875).

`tests/conftest.py` used to mark any test whose name contained "large", "slow" or "benchmark" as `slow`,
and the gate deselects `slow`. So a descriptive name removed a fast test from the gate with nothing at
the test to show it. At `bfdab257` it held 31 cases out of the gate, which run in 0.40 s together; one is
#1684's pin of `MultiPopulationResult.errors`. The rule is gone. A name may still say slow, as a whole
underscore-separated word, and then it must carry the marker: the test fails at setup otherwise, so
these tests' own names avoid that word.
"""

from __future__ import annotations

import pytest

from tests.conftest import pytest_collection_modifyitems, pytest_runtest_setup, says_slow, undeclared_slow_names

pytest_plugins = ("pytester",)


class _Item:
    """The parts of a collected pytest item the conftest hooks read and write."""

    def __init__(self, name: str, markers: tuple[str, ...] = ()):
        self.originalname = name
        self.name = f"{name}[case]"
        self.nodeid = f"tests/unit/test_x.py::{self.name}"
        self.fspath = "tests/unit/test_x.py"
        self.markers = list(markers)
        self.stash: dict = {}

    def get_closest_marker(self, marker: str):
        return object() if marker in self.markers else None

    def add_marker(self, marker) -> None:
        self.markers.append(marker.name)


@pytest.mark.parametrize(
    ("name", "says"),
    [
        ("test_a_slow_solve", True),
        ("test_slow", True),
        ("test_a_slowdown_is_reported", False),
        ("test_benchmark_the_kernel", False),
        ("test_the_largest_field_change_decides", False),
    ],
)
def test_only_the_whole_word_counts(name, says):
    assert says_slow(name) is says


def test_a_name_that_says_it_must_carry_the_marker():
    assert undeclared_slow_names([_Item("test_a_slow_solve")]) == ["tests/unit/test_x.py::test_a_slow_solve[case]"]
    assert undeclared_slow_names([_Item("test_a_slow_solve", ("slow",))]) == []


def test_a_name_no_longer_decides_the_marker():
    """The words that used to deselect a test by name now leave it in the gate; the path marker is the control."""
    items = [_Item(name) for name in ("test_a_large_grid", "test_benchmark_the_kernel", "test_the_larger_field")]
    pytest_collection_modifyitems(config=None, items=items)
    assert [item.markers for item in items] == [["unit"]] * 3
    assert [item.stash for item in items] == [{}] * 3


def test_an_undeclared_name_fails_where_it_runs():
    """Collection flags the test and its own setup fails it; a declared one runs."""
    undeclared, declared = _Item("test_a_slow_solve"), _Item("test_a_slow_march", ("slow",))
    pytest_collection_modifyitems(config=None, items=[undeclared, declared])
    with pytest.raises(pytest.fail.Exception, match=r"^rename this test, or declare it with @pytest\.mark\.slow"):
        pytest_runtest_setup(undeclared)
    assert pytest_runtest_setup(declared) is None


def test_an_undeclared_name_fails_in_a_real_session_even_when_marked_xfail(pytester):
    """The hooks in a real pytest session, where other plugins act on the same test.

    pytest's skipping plugin turns a setup failure into an expected failure once it has read an xfail
    marker, so the conftest hook must run first. A fake item cannot see that interplay.
    """
    pytester.makeconftest("from tests.conftest import pytest_collection_modifyitems, pytest_runtest_setup\n")
    pytester.makepyfile(
        "import pytest\n\n\n"
        "@pytest.mark.xfail(reason='an xfail marker must not absorb the refusal')\n"
        "def test_a_slow_solve():\n    pass\n\n\n"
        "def test_a_slowdown_is_reported():\n    pass\n"
    )
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:xdist", "-p", "no:benchmark")
    result.assert_outcomes(errors=1, passed=1)
    result.stdout.fnmatch_lines(["*ERROR at setup of test_a_slow_solve*", "*rename this test*"])

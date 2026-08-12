"""The assertion-strength scan must classify by structure, not by name or vibe.

Its number is a review queue, not a delete list -- which is the lesson that produced it. The first
selector tried was "inert under the six convention mutations"; every one of the five tests #1715
named that way turned out to be a genuine cross-path pin, because inertness selects for *tests
something else*. This selector is structural: an assertion that only checks well-formedness cannot
separate right from wrong for ANY input.
"""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "cas", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "check_assertion_strength.py"
)
cas = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cas)


def _classify(body: str) -> bool:
    """True if the scan flags a test with this body."""
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "test_probe.py").write_text(f"def test_probe():\n    {body}\n")
        weak, total = cas.scan(pathlib.Path(d))
        assert total == 1, f"the probe was not collected as exactly one test ({total})"
        return bool(weak)


@pytest.mark.parametrize(
    "body",
    [
        "assert r is not None",
        "assert np.isfinite(u).all()",
        "assert u.shape == (5,)",
        "assert len(xs) == 3",
        "assert isinstance(r, dict)",
        "pass",
    ],
)
def test_well_formedness_only_is_flagged(body: str):
    """A wrong answer of the right shape passes every one of these."""
    assert _classify(body), f"not flagged: {body!r}"


@pytest.mark.parametrize(
    "body",
    [
        "assert abs(x - 1.234) < 1e-9",
        "np.testing.assert_allclose(a, b, rtol=1e-9)",
        "with pytest.raises(ValueError):\n        f()",
        "with pytest.warns(UserWarning):\n        f()",
        "assert solve(a) < solve(b)",
        "assert x == pytest.approx(2.5)",
    ],
)
def test_a_real_assertion_is_not_flagged(body: str):
    """Control. `raises` and `warns` carry the assertion themselves and have no `assert` node --
    omitting them from the strong list flagged every fail-loud guard in the tree."""
    assert not _classify(body), f"wrongly flagged: {body!r}"


def test_a_helper_nested_inside_a_test_is_not_counted():
    """pytest does not collect it, so counting it inflates both numerator and denominator.

    Measured before the fix: three files contributed duplicate rows (`test_progress.py::test_func`
    twice, `test_parameter_migration.py::test_function` three times) for functions that never run.
    """
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "test_nested.py").write_text(
            "def test_outer():\n    def test_helper():\n        pass\n    assert abs(1.0 - 1.0) < 1e-9\n"
        )
        weak, total = cas.scan(pathlib.Path(d))
    assert total == 1, f"nested helper counted: total={total}"
    assert not weak


def test_class_level_tests_are_counted():
    """Half this repo's integration tests live in classes; missing them would halve the denominator."""
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "test_cls.py").write_text(
            "class TestThing:\n    def test_inside(self):\n        assert result is not None\n"
        )
        weak, total = cas.scan(pathlib.Path(d))
    assert total == 1
    assert len(weak) == 1


def test_the_frozen_paradigms_are_out_of_scope():
    """`alg/neural` and `alg/reinforcement` are frozen; a sweep must state their exclusion."""
    assert "alg/neural" in cas.FROZEN
    assert "alg/reinforcement" in cas.FROZEN


def test_the_repo_scan_reports_a_denominator():
    """Never a bare count: '0 of unknown' and '0 of all' render identically (#1901 class 2)."""
    weak, total = cas.scan(pathlib.Path(cas.REPO) / "tests")
    assert total > 4000, f"the scan collected only {total} tests; the denominator looks wrong"
    assert 0 < len(weak) < total

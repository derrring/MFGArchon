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


def test_the_frozen_paradigms_are_actually_excluded_not_merely_named():
    """A membership assertion is a tautology: the constant contains itself, always.

    The first version of this asserted `"alg/neural" in cas.FROZEN` and passed while the filter
    matched ZERO files under `tests/` -- those are SOURCE paths, and the filter was matching test
    FILENAMES. Found by review (#1905), which named it as the same tautological shape this script
    exists to count.

    ~~131 frozen test functions sat in the denominator with a 47% flag rate against 20.6%
    overall~~ [CORRECTED 2026-08-13] -- neither figure is reproducible from any state of the
    committed code, and re-review found this line still asserting them. Re-measured under the
    single owner (`check_frozen_areas._references`): the frozen set is **14 files, 145 test
    functions**, of which 66 would be flagged = **46%**, against 19.5% over the 5393 that remain.
    A wrong number inside a test docstring is worse than one in prose, because the file around it
    reads as verified.

    So assert the BEHAVIOUR: a frozen-named file must be absent from the scan, and a non-frozen
    one present.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / "test_reaches_frozen.py").write_text(
            "from mfgarchon.alg.neural.nn import feedforward\n\ndef test_a():\n    assert feedforward is not None\n"
        )
        (root / "test_ordinary.py").write_text("def test_b():\n    assert y is not None\n")
        weak, total = cas.scan(root)
    assert total == 1, f"the frozen file was not excluded from the denominator (total={total})"
    names = [n for _, n, _ in weak]
    assert names == ["test_b"], f"scan returned {names}; the frozen file leaked in"


def test_a_file_named_after_a_frozen_paradigm_but_not_reaching_one_stays_in_the_denominator():
    """The specific error the single-owner change fixes, as a fixture.

    A file whose PATH contains `test_neural` while it imports nothing frozen is not frozen.
    `tests/unit/test_utils/test_neural/test_normalization.py` is the real instance: 36 test
    functions were dropped from the denominator because a DIRECTORY is spelled that way, while
    CLAUDE.md freezes `alg/neural` and `alg/reinforcement` only. Found by re-review (#1905).
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d) / "test_neural"
        root.mkdir()
        (root / "test_normalization.py").write_text(
            "from mfgarchon.utils.numerical import integration\n\ndef test_a():\n    assert integration is not None\n"
        )
        _weak, total = cas.scan(pathlib.Path(d))
    assert total == 1, (
        "a file named after a frozen paradigm was excluded although it reaches none; "
        "the decider is back to matching names"
    )


def test_frozen_membership_has_one_owner_and_this_module_is_not_it():
    """Positive control against the real tree, plus the invariant that keeps them from diverging.

    `check_frozen_areas.py` decides frozen membership by AST -- imports and string literals --
    and carries the argument for why a name match is insufficient. This module answered the same
    question by filename substring; the two disagreed on six files, 40 test functions wrongly out
    and 20 wrongly in. Re-review (#1905) also found that the previous repair had entrenched the
    wrong set by pinning its file count in a test, so the pin is now on the OWNER, not the answer.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_frozen_areas_probe", pathlib.Path(cas.REPO) / "scripts" / "check_frozen_areas.py"
    )
    cfa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfa)

    tree = list((pathlib.Path(cas.REPO) / "tests").rglob("test_*.py"))
    frozen = [f for f in tree if cfa._references(f)]
    assert frozen, "no test file reaches a frozen package; the exclusion is inert or the tree moved"
    # The two must not be able to disagree, which is what a shared decider buys. Asserting it
    # here means a future local shortcut in check_assertion_strength reddens rather than drifts.
    assert [f for f in tree if cas._is_frozen(f)] == frozen, (
        "check_assertion_strength and check_frozen_areas disagree about frozen membership; "
        "there must be exactly one decider"
    )
    weak, _total = cas.scan(pathlib.Path(cas.REPO) / "tests")
    assert not any(cfa._references(f) for f, _, _ in weak), "a frozen-paradigm test appears in the flagged set"


def test_a_separation_assertion_is_the_strongest_class_not_the_weakest():
    """`assert not allclose(a, b)` says two things must DIFFER.

    That is this repo's own doctrine -- assert on disagreement, not validity; byte-identity is the
    defect, not the pass. Calling every `not` weak inverted it on 70 tests, among them
    `test_coupling_affects_solution` and `test_fp_velocity_consumes_cross_density_1071`.
    Found by review (#1905).
    """
    for body in (
        "assert not np.allclose(a, b)",
        "assert not np.array_equal(a, b)",
        "assert not np.isclose(a, b).all()",
    ):
        assert not _classify(body), f"separation assertion called weak: {body!r}"
    # ...while a bare truthiness assert stays weak.
    assert _classify("assert converged")


def test_the_repo_scan_reports_a_denominator():
    """Never a bare count: '0 of unknown' and '0 of all' render identically (#1901 class 2)."""
    weak, total = cas.scan(pathlib.Path(cas.REPO) / "tests")
    assert total > 4000, f"the scan collected only {total} tests; the denominator looks wrong"
    assert 0 < len(weak) < total


def test_the_printed_line_carries_its_numerator_denominator_and_percentage(capsys):
    """`main()` was never called by any test, so nothing about the reported line was pinned.

    Review (#1905) walked two mutations straight through: inverting the fraction to
    `(total - len(weak)) / total` turned 20.6% into 79.4%, and dropping `of {total}` removed the
    denominator entirely -- which is the exact defect (#1901 class 2) this line exists to avoid.
    """
    cas.main()
    out = capsys.readouterr().out
    weak, total = cas.scan(pathlib.Path(cas.REPO) / "tests")
    assert f"{len(weak)} of {total} defined test functions" in out, f"numerator/denominator pair missing: {out!r}"
    assert f"{100 * len(weak) / total:.1f}%" in out, "the percentage does not match the scan"
    # The complement must NOT be what is printed: 20.6% and 79.4% are both 'a percentage'.
    assert f"{100 * (total - len(weak)) / total:.1f}%" not in out, "the fraction is inverted"
    assert "WRONG answer" in out, "the line no longer says what the number means"


def test_the_review_queue_caveat_is_printed_with_the_number():
    """The count is not a delete list, and saying so is part of the number's meaning.

    37 capability cells, ~37-42 fail-loud negative controls and dependency probes sit inside it,
    all assertion-free by nature. A bare percentage invites the deletion PR that was withdrawn.
    """
    import inspect

    body = inspect.getsource(cas.main)
    assert "review queue, not a delete list" in body

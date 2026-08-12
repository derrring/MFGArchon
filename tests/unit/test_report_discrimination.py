"""The gate's discriminating-fraction line must carry its denominator and say when it moved.

"N passed" has never been the quantity worth growing, and printing it alone invites growing it.
This reports the one that says whether green means anything -- and reports it the way #1901's
class 2 requires every count to be reported, beside the population it was taken over, because
"0 of unknown" and "0 of all" render identically.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "report_discrimination.py"
_spec = importlib.util.spec_from_file_location("report_discrimination", _SCRIPT)
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)


def test_it_reports_the_denominator_and_the_commit(capsys, monkeypatch):
    """A fraction without its population is the defect this line exists to avoid printing."""
    monkeypatch.setattr(rd, "_current_collected", lambda *a, **k: 5872)
    rd.main()
    out = capsys.readouterr().out
    baseline = json.loads(rd.BASELINE.read_text())
    then = baseline["_measured_at"]["collected"]
    commit = baseline["_measured_at"]["commit"]
    assert f"of {then} tests" in out, f"the denominator is missing from: {out!r}"
    assert commit in out, "the measurement commit is missing, so staleness is unjudgeable"
    assert "%" in out


def test_a_moved_suite_size_is_called_stale(capsys, monkeypatch):
    """The whole point: the recorded fraction silently ages as the suite grows."""
    monkeypatch.setattr(rd, "_current_collected", lambda *a, **k: 9999)
    rd.main()
    out = capsys.readouterr().out
    assert "STALE" in out, f"a moved denominator was not reported: {out!r}"
    assert "9999" in out


def test_an_unmoved_suite_size_is_not_called_stale(capsys, monkeypatch):
    """Control: the warning must not fire when nothing moved, or it is noise."""
    then = json.loads(rd.BASELINE.read_text())["_measured_at"]["collected"]
    monkeypatch.setattr(rd, "_current_collected", lambda *a, **k: then)
    rd.main()
    assert "STALE" not in capsys.readouterr().out


def test_an_undeterminable_suite_size_says_so_rather_than_guessing(capsys, monkeypatch):
    monkeypatch.setattr(rd, "_current_collected", lambda *a, **k: None)
    rd.main()
    assert "unknown" in capsys.readouterr().out


def test_a_convention_no_test_notices_is_named(capsys, monkeypatch, tmp_path):
    """An UNCOVERED convention is the finding this whole instrument exists to surface."""
    baseline = json.loads(rd.BASELINE.read_text())
    baseline["mutations"]["a_convention_nothing_notices"] = {"kill_count": 0, "status": "UNCOVERED", "owner": "x"}
    fake = tmp_path / "b.json"
    fake.write_text(json.dumps(baseline))
    monkeypatch.setattr(rd, "BASELINE", fake)
    monkeypatch.setattr(rd, "_current_collected", lambda *a, **k: baseline["_measured_at"]["collected"])
    rd.main()
    out = capsys.readouterr().out
    assert "a_convention_nothing_notices" in out, f"an uncovered convention was not named: {out!r}"


def test_missing_artifacts_report_cannot_measure(capsys, monkeypatch, tmp_path):
    """Never a silent zero: absent inputs are 'could not measure', not 'nothing to report'."""
    monkeypatch.setattr(rd, "BASELINE", tmp_path / "nope.json")
    rd.main()
    assert "CANNOT MEASURE" in capsys.readouterr().out


def test_the_collect_parser_handles_both_pytest_summary_forms():
    """`N/M tests collected (K deselected)` is what the CI marker set actually prints.

    The first version read `line.split()[0]` and returned None on that form, printing
    "current suite size unknown" while the number was on screen. Parsed against a real run.
    """
    import subprocess

    real = subprocess.run
    for line, expected in (
        ("6142/6551 tests collected (409 deselected) in 1.72s", 6142),
        ("6551 tests collected in 1.70s", 6551),
        ("1 test collected in 0.01s", 1),
    ):

        class _Fake:
            stdout = line

        subprocess.run = lambda *a, **k: _Fake()
        try:
            assert rd._current_collected() == expected, f"failed to parse: {line!r}"
        finally:
            subprocess.run = real


@pytest.mark.parametrize("bad", ["", "no summary here", "collected but not tests"])
def test_an_unparseable_summary_returns_none_rather_than_a_guess(bad):
    import subprocess

    real = subprocess.run

    class _Fake:
        stdout = bad

    subprocess.run = lambda *a, **k: _Fake()
    try:
        assert rd._current_collected() is None
    finally:
        subprocess.run = real


# --- the printed line itself, which 18 mutations walked through (#1905) -------------------


def test_the_printed_line_carries_the_numerator_the_denominator_and_the_percentage(capsys, monkeypatch):
    """Every one of these was movable without a test noticing: the numerator could switch from the
    union to the sum (212 -> 220), the percentage could be halved, the convention count could read
    99, and `of {then} tests` could be dropped entirely -- taking the denominator with it."""
    monkeypatch.setattr(rd, "_current_collected", lambda *a, **k: 5872)
    rd.main()
    out = capsys.readouterr().out
    baseline = json.loads(rd.BASELINE.read_text())
    matrix = json.loads((rd.BASELINE.parent / "discrimination_killmatrix.json").read_text())
    union = {t for e in matrix["mutations"].values() for t in e.get("killed", ())}
    then = baseline["_measured_at"]["collected"]
    assert f"{len(union)} of {then}" in out, f"numerator/denominator pair missing from: {out!r}"
    assert f"{len(baseline['mutations'])} conventions" in out, "the convention count is wrong or absent"
    assert f"{100 * len(union) / then:.2f}%" in out, "the percentage does not match the artifacts"
    # The union, not the sum: 8 tests kill more than one mutation, so they differ.
    assert sum(v["kill_count"] for v in baseline["mutations"].values()) != len(union), (
        "sum and union coincide on this baseline, so this test cannot tell them apart"
    )
    assert f"{sum(v['kill_count'] for v in baseline['mutations'].values())} of" not in out


def test_the_staleness_check_uses_the_same_exclusion_the_baseline_recorded(monkeypatch):
    """The baseline ignores its own self-test file; comparing against a count that includes it
    reported +5.0% where the like-for-like figure is +4.6% -- a verdict over a denominator
    different from the thing it judges, inside the instrument built to report exactly that."""
    seen = {}

    class _Fake:
        stdout = "6141/6551 tests collected (410 deselected) in 1.7s"

    def spy(*args, **kwargs):
        seen["argv"] = args[0]
        return _Fake()

    import subprocess

    real = subprocess.run
    subprocess.run = spy
    try:
        excluded = json.loads(rd.BASELINE.read_text())["_measured_at"]["excluded"]
        rd._current_collected(excluded)
    finally:
        subprocess.run = real
    assert "--ignore" in seen["argv"], "the exclusion is not passed to the collect"
    assert excluded in seen["argv"], f"{excluded} not in {seen['argv']}"


def test_a_shrinking_suite_is_also_called_stale(capsys, monkeypatch):
    """`now != then`, not `now > then`: a suite that shrank has moved just as much."""
    then = json.loads(rd.BASELINE.read_text())["_measured_at"]["collected"]
    monkeypatch.setattr(rd, "_current_collected", lambda *a, **k: then - 500)
    rd.main()
    assert "STALE" in capsys.readouterr().out

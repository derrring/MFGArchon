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
    monkeypatch.setattr(rd, "_current_collected", lambda: 5872)
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
    monkeypatch.setattr(rd, "_current_collected", lambda: 9999)
    rd.main()
    out = capsys.readouterr().out
    assert "STALE" in out, f"a moved denominator was not reported: {out!r}"
    assert "9999" in out


def test_an_unmoved_suite_size_is_not_called_stale(capsys, monkeypatch):
    """Control: the warning must not fire when nothing moved, or it is noise."""
    then = json.loads(rd.BASELINE.read_text())["_measured_at"]["collected"]
    monkeypatch.setattr(rd, "_current_collected", lambda: then)
    rd.main()
    assert "STALE" not in capsys.readouterr().out


def test_an_undeterminable_suite_size_says_so_rather_than_guessing(capsys, monkeypatch):
    monkeypatch.setattr(rd, "_current_collected", lambda: None)
    rd.main()
    assert "unknown" in capsys.readouterr().out


def test_a_convention_no_test_notices_is_named(capsys, monkeypatch, tmp_path):
    """An UNCOVERED convention is the finding this whole instrument exists to surface."""
    baseline = json.loads(rd.BASELINE.read_text())
    baseline["mutations"]["a_convention_nothing_notices"] = {"kill_count": 0, "status": "UNCOVERED", "owner": "x"}
    fake = tmp_path / "b.json"
    fake.write_text(json.dumps(baseline))
    monkeypatch.setattr(rd, "BASELINE", fake)
    monkeypatch.setattr(rd, "_current_collected", lambda: baseline["_measured_at"]["collected"])
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

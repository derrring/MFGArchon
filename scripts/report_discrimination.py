#!/usr/bin/env python3
"""Print the discriminating fraction beside its denominator, and say when the denominator moved.

The gate prints "N passed". That number has never been the thing worth growing, and printing it
alone invites growing it: measured 2026-08-12 over the whole corpus (#1901), what recurs in this
repository is not a wrong formula but a green light, and 5,481 of 5,665 collected tests noticed
nothing when any of six load-bearing conventions was broken.

So print the fraction that does matter, and print it the way #1901's class 2 says every count must
be printed -- with the population it was taken over, because "0 of unknown" and "0 of all" render
identically. This reports rather than gates: the gating is `test_discrimination.py --check-baseline`
in the weekly tier, because measuring it costs one full suite run per mutation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "scripts" / "discrimination_baseline.json"
MATRIX = REPO / "scripts" / "discrimination_killmatrix.json"


def _current_collected(excluded: str | None = None) -> int | None:
    """Collected today over the SAME population the baseline measured.

    None if it cannot be determined -- never a guess. `excluded` is `_measured_at["excluded"]`:
    the sweep ignores its own self-test file, so a count that includes it compared 6168 against a
    like-for-like 6141 and reported +5.0% where the truth is +4.6%. A staleness verdict taken over
    a different denominator than the thing it judges is exactly the defect this line exists to
    report. Found by review (#1905).
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "--collect-only",
            "-q",
            "--color=no",
            "-p",
            "no:randomly",
            "-o",
            "addopts=",
            *(["--ignore", excluded] if excluded else []),
            "-m",
            "not slow and not benchmark and not experimental and not optional_torch and not environment",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )
    # pytest prints either "N tests collected" or "N/M tests collected (K deselected)". Take the
    # SELECTED count, N, which is the population the sweep actually runs. Parsed against a real
    # run rather than assumed -- the first version of this read `line.split()[0]` and silently
    # returned None on the "N/M" form, printing "current suite size unknown" while the number
    # was right there.
    for line in reversed(proc.stdout.splitlines()):
        if "tests collected" in line or "test collected" in line:
            head = line.split()[0].split("/")[0].replace(",", "")
            if head.isdigit():
                return int(head)
    return None


def main() -> int:
    if not BASELINE.exists() or not MATRIX.exists():
        print("discrimination : CANNOT MEASURE -- baseline or kill matrix absent")
        return 0

    baseline = json.loads(BASELINE.read_text())
    matrix = json.loads(MATRIX.read_text())
    measured = baseline["_measured_at"]
    then = int(measured["collected"])
    killers = {t for entry in matrix["mutations"].values() for t in entry.get("killed", ())}
    conventions = len(baseline["mutations"])
    uncovered = sorted(n for n, v in baseline["mutations"].items() if v["kill_count"] == 0)

    print(
        f"discrimination : {len(killers)} of {then} tests notice at least one of {conventions} "
        f"conventions = {100 * len(killers) / then:.2f}%  (measured at {measured['commit']})"
    )
    if uncovered:
        print(f"                 {len(uncovered)} convention(s) NO test notices: {', '.join(uncovered)}")

    now = _current_collected(measured.get("excluded"))
    if now is None:
        print("                 current suite size unknown -- the fraction above may be stale")
    elif now != then:
        drift = 100 * (now - then) / then
        print(
            f"                 the suite has since moved {then} -> {now} ({drift:+.1f}%): the "
            f"fraction above is STALE, and adding tests lowers it unless they discriminate"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

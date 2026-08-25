#!/usr/bin/env python3
"""Ratchet the warnings the suite emits, keyed on IDENTITY rather than count (#2119).

WHY A RATCHET AND NOT A NUMBER. The gate used to print all 6,030 lines of pytest's warnings
summary -- 95.7% of everything it emitted -- and did so every run for a year while none of the 456
calls it was reporting were retired. #2118 stopped printing the listing, which makes ignoring them
cheaper unless something counts them. "The count is still in the tail" is not that something: a
number scrolls past exactly the way the listing did.

WHY IDENTITY AND NOT OCCURRENCES, and why THIS identity. Measured on one full-suite run, every
key computed from the same raw records so the rows are comparable:

    occurrences                 5022    jitters 5021-5023 parallel, 5002 serial -- an exact gate
                                        flakes, a banded one admits new warnings silently
    (file, line, kind)           609    stable across runs, useless across edits: inserting one
                                        line moves every identity in that file
    raw text[:60]                318    messages embed measurements ("Hybrid neighborhood: 4/21
                                        points (19.0%)"), so each count is its own identity
    digits->N, text[:60]         230    called STABLE here on two agreeing runs; the third gave a
                                        different number. That sentence shipped in this docstring.
    digits->N, text[:40]         225    <- what this gates on

WHAT JUSTIFIES 40, HONESTLY. Eleven agreeing full-suite runs, including a fully serial one, plus
ONE understood channel: the embedded measurements, which the digit normalisation closes. A second
channel named in an earlier version of this file -- a `Reason: ...` suffix said to render
inconsistently -- was measured FALSE; both forms appear in the same run from two distinct decorated
`__init__`s. **The 60-character key's instability channel remains unexplained.** So this is samples
plus a partial mechanism, not a proof, and it is written that way because the alternative already
put a falsified "stable" into this docstring once.

AND IT IS STABLE BY BEING COARSER. Against the 60-character key it merges 5 groups, three of them
real distinctions -- `signature 'legacy'` with `'neural'`, Newton's "iteration budget" with
"residual stopped decreasing", and `ZeroFluxCalculator` with `ZeroGradientCalculator`. A new
warning differing from an existing one only past character 40 of the same file and category raises
no new identity. Verified by construction, not inferred.

WHAT AN IDENTITY IS, because "225" reads as 225 distinct warnings and is not. The key is per
CALL SITE: one deprecated API called from 153 test files is 153 identities. The 225 are 44 distinct
(kind, message) pairs across 189 files, the largest group being 153 sharing
`Legacy MFGProblem(geometry=, components=`. That is the intended behaviour, not an artifact -- a
new test reaching for a deprecated API should turn this red, and it only can if the caller is part
of the key.

BIDIRECTIONAL, like the other four baselines. A new identity is a regression. A DISAPPEARED one is
progress that must be recorded, so a fix cannot land silently and a later regression cannot hide
behind it.

The census itself is written by `tests/conftest.py`'s `pytest_terminal_summary` during the gate's
own suite run -- no second run, and no per-worker merge, because the controller's
`terminalreporter.stats["warnings"]` is complete even under `-n auto` and `--disable-warnings`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parent / "warning_baseline.json"

#: A full run is ~6,600 outcomes. The floor only has to separate a real run from a collect-only
#: one (which reports 0) or a run that died in the first seconds; it is deliberately far below the
#: real number so that legitimately shrinking the suite does not trip it.
MIN_TESTS = 500


def _load_census(path: Path) -> dict:
    if not path.is_file():
        print(
            f"CANNOT RUN: no warnings census at {path}.\n"
            "It is written by tests/conftest.py during a pytest run with MFGARCHON_WARNING_CENSUS\n"
            "set. Nothing was measured, so this says nothing about whether warnings changed.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return json.loads(path.read_text())


def _self_test() -> int:
    """Positive control, run through the PRODUCTION path -- exit code and printed text both.

    An earlier version routed its cases through a `quiet=True` shortcut that returned before every
    line the real path executes. It was green under three separate mutations that broke the gate:
    turning the production `return 1` into `return 0` (the ratchet then PRINTS a regression and
    exits 0, so `check $?` reports PASS); swapping `appeared` with `left` (a regression reported as
    progress, with an instruction that records it into the baseline); and comparing the baseline to
    itself. A control that does not execute the return the caller reads is not a control.

    So: `contextlib.redirect_stdout` over the real comparison, assert the code AND the direction,
    and drive one case through `main()` so nothing between census-load and comparison is uncovered.
    This is the shape `check_citations.py` already uses in this directory.
    """
    import contextlib
    import io
    import tempfile

    if not BASELINE.is_file():
        print("self-test CANNOT RUN: no baseline to build cases from", file=sys.stderr)
        return 2
    base = json.loads(BASELINE.read_text())["identities"]
    if len(base) < 2:
        print(f"self-test CANNOT RUN: baseline has {len(base)} identities, need 2", file=sys.stderr)
        return 2

    probes = [f"mfgarchon/_probe{n}.py\tUserWarning\tself-test injected identity {n}" for n in (1, 2)]
    # TWO of each, not one. With every case moving a single identity, `len(appeared)` is never
    # above 1 and anything conditioned on the count is uncovered -- review demonstrated a
    # fail-open BAND (`return 0 if len(appeared) + len(left) >= 2 else 1`) and a truncated listing
    # (`appeared[:0]`) that both passed a one-at-a-time control while printing the regression.
    cases = [
        ("unchanged", base, 0, [], []),
        ("two appeared", [*base, *probes], 1, ["NEW warning identities (2)"], probes),
        ("two vanished", base[2:], 1, ["Warning identities GONE (2)"], sorted(base)[:2]),
    ]
    failures = []
    for label, identities, want_rc, want_texts, want_named in cases:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            rc = _compare(set(identities), set(base), 0)
        out = sink.getvalue()
        if rc != want_rc:
            failures.append(f"{label}: expected exit {want_rc}, got {rc}")
        for want in want_texts:
            if want not in out:
                failures.append(f"{label}: expected {want!r} in the output, got {out.strip()[:90]!r}")
        # EVERY identity must be NAMED, not just counted. A ratchet that reports "2 appeared"
        # without saying which is a counter, which is what this whole file exists to replace.
        for identity in want_named:
            if identity.split("\t")[0] not in out:
                failures.append(f"{label}: {identity.split(chr(9))[0]} was counted but not named")
        if not want_texts and out.strip() and "ratchet OK" not in out:
            failures.append(f"{label}: expected an OK line, got {out.strip()[:90]!r}")

    # The REMEDY is the actionable half, and inverting it turns the ratchet into a baseline
    # eraser: a regression carrying "re-baseline with --write-baseline" gets recorded, not fixed.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        _compare({*base, *probes}, set(base), 0)
    appeared_out = sink.getvalue()
    if "Fix the cause" not in appeared_out:
        failures.append("a NEW identity must be told to fix the cause, not to re-baseline")
    if "Re-baseline with" in appeared_out.split("GONE")[0]:
        failures.append("a NEW identity is carrying the re-baseline instruction, which would record it")

    # One case through main(), so census loading and the baseline read are covered too.
    with tempfile.TemporaryDirectory() as tmp:
        census = Path(tmp) / "census.json"
        # A realistic `tests_run`, so the floor can be strict. An earlier version omitted it, which
        # forced `census.get("tests_run")` to tolerate a missing key -- the control's shape making
        # the guard accept exactly the class of file the guard exists to reject.
        census.write_text(json.dumps({"identities": sorted([*base, *probes]), "occurrences": 0, "tests_run": 6639}))
        argv = sys.argv
        try:
            sys.argv = ["check_warnings.py", "--census", str(census)]
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main()
        finally:
            sys.argv = argv
        if rc != 1:
            failures.append(f"through main(): an appeared identity must exit 1, got {rc}")

        # Below the floor: a collect-only census must be REFUSED, not reported as 225 vanished.
        census.write_text(json.dumps({"identities": [], "occurrences": 0, "tests_run": 0}))
        try:
            sys.argv = ["check_warnings.py", "--census", str(census)]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = main()
        finally:
            sys.argv = argv
        if rc != 2:
            failures.append(f"through main(): a census below the {MIN_TESTS} floor must exit 2, got {rc}")

    if failures:
        for line in failures:
            print(f"self-test FAILED: {line}", file=sys.stderr)
        return 1
    print(
        f"self-test OK: both directions fire with the right message, no-change is silent, main() agrees ({len(base)} identities)"
    )
    return 0


def _compare(now: set[str], was: set[str], occurrences: int) -> int:
    """The one comparison. No quiet mode: a control that skips the caller's return is not one."""
    appeared = sorted(now - was)
    left = sorted(was - now)
    if not appeared and not left:
        print(f"warnings ratchet OK: {len(now)} identities, {occurrences} occurrences (not gated)")
        return 0
    if appeared:
        print(f"\nNEW warning identities ({len(appeared)}) -- something started warning that did not:")
        for identity in appeared:
            print(f"    {identity.replace(chr(9), '  |  ')}")
        print("    Fix the cause, or record it with --write-baseline and say why in the commit.")
    if left:
        print(f"\nWarning identities GONE ({len(left)}) -- progress, and it must be recorded:")
        for identity in left:
            print(f"    {identity.replace(chr(9), '  |  ')}")
        print("    Re-baseline with --write-baseline so a later regression cannot hide behind this.")
    print(f"\n{len(now)} identities now, {len(was)} in the baseline. Occurrences reported, not gated.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", type=Path, default=None, help="census JSON; defaults to $MFGARCHON_WARNING_CENSUS")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--show", action="store_true", help="list every identity and exit")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Positive control: assert the comparison fires on an appeared identity AND on a vanished one, and stays silent on no change.",
    )
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    census_path = args.census or Path(os.environ.get("MFGARCHON_WARNING_CENSUS", ""))
    census = _load_census(census_path)
    now = set(census["identities"])

    # A census from a collect-only or truncated run looks well-formed and is not a measurement.
    # Without this the ratchet reports hundreds of identities GONE and instructs the reader to
    # re-baseline -- which would write that non-measurement into the artifact.
    ran = census.get("tests_run", 0)
    if ran < MIN_TESTS:
        print(
            f"CANNOT RUN: the census records {ran} test outcomes, below the {MIN_TESTS} floor.\n"
            "That is a collect-only or truncated run, not a measurement of the suite. Refusing\n"
            "rather than reporting every identity as vanished.",
            file=sys.stderr,
        )
        return 2

    if args.show:
        for identity in sorted(now):
            print(identity.replace("\t", "  |  "))
        return 0

    if args.write_baseline:
        BASELINE.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Warning identities the suite emits (#2119). Keyed (file, kind, digits-normalised "
                        "text[:40]) -- NOT "
                        "line numbers, which move under any edit, and NOT occurrence counts, which jitter "
                        "run to run. Bidirectional: a new identity is a regression, a removed one is "
                        "progress that must be recorded here. `occurrences` is reported, never gated. "
                        "Regenerate with --write-baseline."
                    ),
                    "occurrences_when_written": census["occurrences"],
                    "identities": sorted(now),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {BASELINE.name}: {len(now)} identities, {census['occurrences']} occurrences")
        return 0

    if not BASELINE.is_file():
        print(f"CANNOT RUN: no baseline at {BASELINE}. Write one with --write-baseline.", file=sys.stderr)
        return 2

    was = set(json.loads(BASELINE.read_text())["identities"])
    return _compare(now, was, census["occurrences"])


if __name__ == "__main__":
    sys.exit(main())

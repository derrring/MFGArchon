#!/usr/bin/env python3
"""Ratchet the warnings the suite emits, keyed on IDENTITY rather than count (#2119).

WHY A RATCHET AND NOT A NUMBER. The gate used to print all 6,030 lines of pytest's warnings
summary -- 95.7% of everything it emitted -- and did so every run for a year while none of the 456
calls it was reporting were retired. #2118 stopped printing the listing, which makes ignoring them
cheaper unless something counts them. "The count is still in the tail" is not that something: a
number scrolls past exactly the way the listing did.

WHY IDENTITY AND NOT OCCURRENCES. Measured on two runs of the real suite at one commit:

    occurrences               5022  vs  5023   <- jitters, so an exact gate flakes and a banded
                                                  one lets new warnings in silently
    (file, line, kind)         608  vs   608   <- stable across runs, useless across edits:
                                                  inserting a line moves every identity in a file
    (file, kind, text[:60])    315  vs   315   <- stable, and survives an edit

So this gates on the third and REPORTS the first. It is the move `check_citations.py` made when it
went from counting drifted citations to naming them, for the same reason.

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
    """Positive control: the ratchet must fire in BOTH directions and stay silent on no change.

    A baseline is only evidence if the comparison still works. Each case is built from the shipped
    baseline itself, so this exercises the real identities rather than a toy set.
    """
    import tempfile

    if not BASELINE.is_file():
        print("self-test CANNOT RUN: no baseline to build cases from", file=sys.stderr)
        return 2
    base = json.loads(BASELINE.read_text())["identities"]
    if len(base) < 2:
        print(f"self-test CANNOT RUN: baseline has {len(base)} identities, need 2 to remove one", file=sys.stderr)
        return 2

    cases = {
        "unchanged": (base, 0),
        "one appeared": ([*base, "mfgarchon/_probe.py\tUserWarning\tself-test injected identity"], 1),
        "one vanished": (base[1:], 1),
    }
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (identities, expected) in cases.items():
            census = Path(tmp) / "census.json"
            census.write_text(json.dumps({"identities": sorted(identities), "occurrences": 0}))
            got = _compare(set(json.loads(census.read_text())["identities"]), set(base), 0, quiet=True)
            if got != expected:
                failures.append(f"{name}: expected exit {expected}, got {got}")

    if failures:
        for line in failures:
            print(f"self-test FAILED: {line}", file=sys.stderr)
        return 1
    print(f"self-test OK: appeared and vanished both fire, no-change stays silent ({len(base)} identities)")
    return 0


def _compare(now: set[str], was: set[str], occurrences: int, *, quiet: bool = False) -> int:
    """Shared by the ratchet and its self-test, so the control exercises the real comparison."""
    appeared = sorted(now - was)
    left = sorted(was - now)
    if not appeared and not left:
        if not quiet:
            print(f"warnings ratchet OK: {len(now)} identities, {occurrences} occurrences (not gated)")
        return 0
    if quiet:
        return 1
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

    if args.show:
        for identity in sorted(now):
            print(identity.replace("\t", "  |  "))
        return 0

    if args.write_baseline:
        BASELINE.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Warning identities the suite emits (#2119). Keyed (file, kind, text[:60]) -- NOT "
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

#!/usr/bin/env python3
"""Ratchet the type-error count per top-level package, and refuse to read an unscanned package as 0.

The gate's mypy step checks ONE of this package's fourteen top-level subpackages (`mfgarchon/config`,
scope copied from ci.yml). That narrowing is deliberate and ci.yml states why, so this instrument
does not widen it: the gate step still gates only what it gated. What was missing is that the other
thirteen had no number at all, so nothing said whether they were getting worse. This records one per
package and fails when any of them moves.

WHY A RATCHET AND NOT `mypy mfgarchon`: turning the gate on everywhere would fail immediately and
permanently, which is a gate nobody can keep. A per-package baseline that fails on INCREASE makes
the debt visible and stops it growing; failing on DECREASE too means an improvement cannot land
without being recorded, so the number stays true.

THE CONTROL THIS EXISTS FOR, and it is not hypothetical -- it is the mistake that produced this
script. `mypy mfgarchon/utils` prints "There are no .py[i] files in directory 'mfgarchon/utils'",
because `[tool.mypy] exclude` in pyproject.toml drops four packages outright. A census that greps
mypy's output for "Found N errors" gets nothing back for those, and a `default 0` turns "the
instrument did not look" into "the package is clean". Measured 2026-09-10: that reading reported
five clean packages where only one -- `mfgarchon/config`, 6 source files, 0 errors -- is actually
clean. The other four are unmeasured.

So every count here is stored beside a STATUS, and a package that leaves the scan fails the ratchet
rather than reporting zero. `Success: no issues found in N source files` and `There are no .py[i]
files in directory` are different sentences and this script never conflates them.

TWO MEASUREMENT CHOICES, both forced by measurement rather than taste:

- ONE invocation over `mfgarchon`, attributed by path prefix, not fourteen invocations. Measured:
  one run is ~10 s against ~23 s cold for a single package, and the two agree exactly -- 1304 error
  lines whole-package against 1302 summed per-package, the difference being the two `_root` errors
  in files directly under `mfgarchon/` that no `mfgarchon/<pkg>/` prefix matches. That bucket is
  real, it is small, and a naive prefix rule drops it silently, so it is carried explicitly.
- `--no-pretty`. `pretty = true` in pyproject.toml wraps mypy's messages across lines at a width
  that moves with COLUMNS, so a line count taken without it is a function of the terminal.

Exit 0 clean, 1 the tree moved against the baseline, 2 the instrument could not measure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().parent / "mypy_scope_baseline.json"

#: Files directly under `mfgarchon/` match no `mfgarchon/<pkg>/` prefix. Named rather than dropped:
#: a bucket that silently disappears reads as clean.
ROOT_BUCKET = "_root"

MEASURED = "measured"
EXCLUDED = "excluded"

EXIT_OK = 0
EXIT_MOVED = 1
EXIT_INSTRUMENT_BROKEN = 2

#: `path:LINE: error:` OR `path:LINE:COL: error:` -- this repo has `show_column_numbers`, and a
#: pattern without the optional column matched 23 of 1304 lines and dropped the rest in silence.
#: `scan()` cross-checks this against a raw substring count for exactly that reason.
_ERROR_LINE = re.compile(r"^(?P<path>[^:]+):\d+:(?:\d+:)? error:")
_CHECKED = re.compile(r"checked (\d+) source file")
_SUCCESS = re.compile(r"no issues found in (\d+) source file")
#: mypy counting its OWN findings. Independent of any pattern of ours over its body, which is
#: what makes it the authority rather than a third opinion -- see `scan()`.
_FOUND = re.compile(r"Found (\d+) error")
_NO_FILES = "no .py[i] files in directory"


def _mypy(target: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-P", "-m", "mypy", target, "--follow-imports=silent", "--no-pretty", *extra],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def packages() -> list[str]:
    """Top-level subpackages, from the filesystem.

    This is the population the baseline is keyed on. It is deliberately NOT the set mypy happens to
    report: a package that stops being scanned must still appear here, or its disappearance is what
    the ratchet cannot see.
    """
    return sorted(
        p.name
        for p in (REPO / "mfgarchon").iterdir()
        if p.is_dir() and p.name != "__pycache__" and not p.name.startswith(".")
    )


def attribute(stdout: str) -> dict[str, int]:
    """Error lines bucketed by top-level package, with files under `mfgarchon/` in ROOT_BUCKET."""
    counts: dict[str, int] = {}
    for line in stdout.splitlines():
        m = _ERROR_LINE.match(line)
        if not m:
            continue
        parts = Path(m.group("path")).parts
        key = parts[1] if len(parts) >= 3 and parts[0] == "mfgarchon" else ROOT_BUCKET
        counts[key] = counts.get(key, 0) + 1
    return counts


def probe_status(package: str) -> str | None:
    """`measured` or `excluded` for one package, or None if mypy said neither.

    The whole-package run cannot answer this: a package with zero errors and a package that is not
    scanned at all both contribute no lines to it. This is the only question that needs its own
    invocation, and it is asked ONLY of packages whose count is zero -- currently one.
    """
    result = _mypy(f"mfgarchon/{package}")
    out = result.stdout + result.stderr
    if _NO_FILES in out:
        return EXCLUDED
    if _SUCCESS.search(out) or _CHECKED.search(out) or "error:" in out:
        return MEASURED
    return None


def scan() -> tuple[dict[str, int], int]:
    """(counts by package, total source files mypy checked). Raises RuntimeError if it cannot tell."""
    result = _mypy("mfgarchon")
    out = result.stdout
    m = _CHECKED.search(out) or _SUCCESS.search(out)
    if not m:
        raise RuntimeError(
            f"mypy printed no file-count summary, so the population is unknown.\n"
            f"stdout tail: {out.strip().splitlines()[-3:] if out.strip() else '(empty)'}\n"
            f"stderr tail: {result.stderr.strip().splitlines()[-3:] if result.stderr.strip() else '(empty)'}"
        )
    counts = attribute(out)

    # THREE counts of one run, and they are not three opinions. `attribute()` and the raw substring
    # count are both OUR patterns over mypy's body, so they share a population and can be wrong
    # together; `Found N errors` is mypy counting its own findings and is the only one whose
    # population we did not choose. It is therefore the authority, and the other two are checked
    # against it.
    #
    # Measured while writing this: a pattern missing mypy's optional column attributed 23 of 1304
    # and would have pinned a baseline blind to the rest. Separately, a peer's census of the same
    # tree read 924 from a grep over the body with `pretty = true` still on -- the wrapping moves
    # with COLUMNS -- while their reading of `Found N errors` was correct at the same moment. Same
    # tool, same invocation, two instruments; only the one with a self-chosen population was wrong.
    raw = sum(1 for line in out.splitlines() if ": error:" in line)
    declared = int(f.group(1)) if (f := _FOUND.search(out)) else (0 if _SUCCESS.search(out) else None)
    if declared is None:
        raise RuntimeError(
            "mypy printed neither 'Found N errors' nor a success line, so it did not report a total "
            "of its own and there is nothing to check this script's parsing against."
        )
    if sum(counts.values()) != declared or raw != declared:
        raise RuntimeError(
            f"mypy reports {declared} errors; this script attributed {sum(counts.values())} and a "
            f"raw substring count of the same output found {raw}. mypy's own total is the authority "
            f"and this script's pattern no longer matches its output format; the difference would "
            f"otherwise be reported as an improvement."
        )
    return counts, int(m.group(1))


def compare(baseline: dict, counts: dict[str, int], checked: int, statuses: dict[str, str]) -> list[str]:
    """Every way the tree can have moved against the baseline. Empty list means clean."""
    problems: list[str] = []
    was_counts: dict[str, int] = baseline["counts"]
    was_status: dict[str, str] = baseline["status"]

    for key in sorted(set(was_counts) | set(counts)):
        before, after = was_counts.get(key), counts.get(key, 0)
        if before is None:
            problems.append(f"{key}: NEW bucket with {after} errors; it is not in the baseline at all")
        elif after > before:
            problems.append(f"{key}: {before} -> {after} (+{after - before}) type errors")
        elif after < before:
            problems.append(f"{key}: {before} -> {after} (-{before - after}) -- an improvement, record it")

    for pkg in sorted(set(was_status) | set(statuses)):
        before, after = was_status.get(pkg), statuses.get(pkg)
        if before == after:
            continue
        if after is None:
            problems.append(f"{pkg}: was in the baseline and is no longer a package at all")
        elif before is None:
            problems.append(f"{pkg}: NEW package, status {after}; decide its number and record it")
        elif after == EXCLUDED:
            problems.append(
                f"{pkg}: LEFT THE SCAN ({before} -> {after}). Its count is now 0 because nothing "
                f"looked at it, not because it is clean. This is the failure this ratchet exists for."
            )
        else:
            problems.append(f"{pkg}: entered the scan ({before} -> {after}); it now has a number, record it")

    if checked != baseline["checked_source_files"]:
        problems.append(
            f"the scan itself changed size: mypy checked {baseline['checked_source_files']} source "
            f"files when the baseline was written and {checked} now"
        )
    return problems


def _toolchain() -> dict[str, str]:
    out = subprocess.run([sys.executable, "-P", "-m", "mypy", "--version"], capture_output=True, text=True).stdout
    return {"python": ".".join(str(v) for v in sys.version_info[:3]), "mypy": out.strip()}


def write_baseline(path: Path) -> int:
    try:
        counts, checked = scan()
    except RuntimeError as exc:
        print(f"CANNOT RUN: {exc}", file=sys.stderr)
        return EXIT_INSTRUMENT_BROKEN

    statuses = {}
    for pkg in packages():
        status = probe_status(pkg)
        if status is None:
            print(f"CANNOT RUN: mypy said neither 'scanned' nor 'excluded' for mfgarchon/{pkg}.", file=sys.stderr)
            return EXIT_INSTRUMENT_BROKEN
        statuses[pkg] = status

    excluded_with_counts = [p for p, s in statuses.items() if s == EXCLUDED and counts.get(p)]
    if excluded_with_counts:
        print(
            f"CANNOT RUN: {excluded_with_counts} are reported excluded yet contributed error lines. "
            f"The two measurements disagree, so neither is trustworthy.",
            file=sys.stderr,
        )
        return EXIT_INSTRUMENT_BROKEN

    for pkg, status in statuses.items():
        if status == MEASURED:
            counts.setdefault(pkg, 0)

    path.write_text(
        json.dumps(
            {
                "_comment": (
                    "Type errors per top-level package (#2295). The gate's own mypy step checks "
                    "mfgarchon/config only; this records the other thirteen so they cannot get worse "
                    "unnoticed. Bidirectional: an improvement fails until it is recorded here. "
                    "`status` is the half that matters -- an EXCLUDED package is not scanned by mypy "
                    "at all, so its absence from `counts` means 'not measured', never 'clean'. "
                    "Regenerate with: python scripts/check_mypy_scope.py --write-baseline"
                ),
                "checked_source_files": checked,
                "toolchain_when_written": _toolchain(),
                "status": dict(sorted(statuses.items())),
                "counts": dict(sorted(counts.items())),
            },
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )
    measured = sum(1 for s in statuses.values() if s == MEASURED)
    print(
        f"wrote {path.name}: {sum(counts.values())} errors over {measured} measured packages "
        f"({len(statuses) - measured} excluded), {checked} source files checked"
    )
    return EXIT_OK


def check_baseline(path: Path) -> int:
    if not path.exists():
        print(f"CANNOT RUN: no baseline at {path}. Write one with --write-baseline.", file=sys.stderr)
        return EXIT_INSTRUMENT_BROKEN
    baseline = json.loads(path.read_text())

    try:
        counts, checked = scan()
    except RuntimeError as exc:
        print(f"CANNOT RUN: {exc}", file=sys.stderr)
        return EXIT_INSTRUMENT_BROKEN

    # Only ZERO-count measured packages need their own invocation: a package with errors is
    # self-evidently scanned. Currently that is one package, so the control costs one probe.
    statuses = dict(baseline["status"])
    for pkg, was in baseline["status"].items():
        if was == MEASURED and not counts.get(pkg):
            live = probe_status(pkg)
            if live is None:
                print(f"CANNOT RUN: mypy said neither 'scanned' nor 'excluded' for mfgarchon/{pkg}.", file=sys.stderr)
                return EXIT_INSTRUMENT_BROKEN
            statuses[pkg] = live
        elif was == EXCLUDED and counts.get(pkg):
            statuses[pkg] = MEASURED
    for pkg in packages():
        statuses.setdefault(pkg, MEASURED if counts.get(pkg) else probe_status(pkg) or MEASURED)
    for pkg in list(statuses):
        if pkg not in packages():
            del statuses[pkg]

    problems = compare(baseline, counts, checked, statuses)
    if not problems:
        measured = sum(1 for s in statuses.values() if s == MEASURED)
        print(
            f"mypy scope ratchet OK: {sum(counts.values())} errors over {measured} measured packages, "
            f"{len(statuses) - measured} excluded, {checked} source files checked"
        )
        return EXIT_OK

    print(f"mypy scope moved against {path.name} ({len(problems)} change(s)):")
    for line in problems:
        print(f"    {line}")
    print(
        "\n  Fix the cause, or -- if the change is intended -- record it:\n"
        "      python scripts/check_mypy_scope.py --write-baseline\n"
        "  and say in the commit why the number moved. A package that LEFT THE SCAN is not a\n"
        "  number to re-record: find what removed it from [tool.mypy] exclude's complement first."
    )
    return EXIT_MOVED


def _self_test() -> int:
    """Positive control. An inert ratchet reports a stable number and reads exactly like success.

    Two-sided, and it drives `main()` as well as `compare()` -- a control that does not execute the
    return the caller reads is not a control (`check_warnings.py` records the case where a `return 1`
    turned into `return 0` while the ratchet still PRINTED its regression).

    The fourth case is the one this script exists for and no sibling ratchet has: a package that
    leaves the scan must fail, not read as zero.
    """
    failures: list[str] = []

    def case(name: str, baseline: dict, counts: dict, checked: int, statuses: dict, want: bool) -> None:
        try:
            fired = bool(compare(baseline, counts, checked, statuses))
        except Exception as exc:  # a case that explodes is a failure, not a stopped run
            failures.append(f"{name}: raised {type(exc).__name__}: {exc}")
            return
        if fired != want:
            failures.append(f"{name}: expected {'a finding' if want else 'silence'}, got the opposite")

    base = {
        "checked_source_files": 100,
        "status": {"alpha": MEASURED, "beta": MEASURED, "gamma": EXCLUDED},
        "counts": {"alpha": 5, "beta": 0},
    }
    live = {"alpha": 5}

    case("no change is silent", base, live, 100, {"alpha": MEASURED, "beta": MEASURED, "gamma": EXCLUDED}, False)
    case("an increase fires", base, {"alpha": 6}, 100, {"alpha": MEASURED, "beta": MEASURED, "gamma": EXCLUDED}, True)
    case("a decrease fires", base, {"alpha": 4}, 100, {"alpha": MEASURED, "beta": MEASURED, "gamma": EXCLUDED}, True)
    case(
        "a package LEAVING THE SCAN fires rather than reading as zero",
        base,
        live,
        100,
        {"alpha": MEASURED, "beta": EXCLUDED, "gamma": EXCLUDED},
        True,
    )
    case(
        "a package entering the scan fires",
        base,
        live,
        100,
        {"alpha": MEASURED, "beta": MEASURED, "gamma": MEASURED},
        True,
    )
    case("a new bucket fires", base, {"alpha": 5, ROOT_BUCKET: 2}, 100, base["status"], True)
    case("the scan changing size fires", base, live, 90, base["status"], True)

    # The zero-count package is the trap: `beta` has 0 errors in both the baseline and the live
    # counts, so ONLY the status half can tell these two cases apart. Assert that directly.
    silent = compare(base, live, 100, {"alpha": MEASURED, "beta": MEASURED, "gamma": EXCLUDED})
    fires = compare(base, live, 100, {"alpha": MEASURED, "beta": EXCLUDED, "gamma": EXCLUDED})
    if silent or not fires:
        failures.append("the zero-count discriminator is inert: identical counts must part on status alone")
    elif "LEFT THE SCAN" not in " ".join(fires):
        failures.append("a package leaving the scan fired, but the message does not say so")

    # The PARSER, not just the comparison. The comparison was fully controlled while the regex
    # silently dropped 98% of mypy's lines, because every case above hands `compare` a dict that
    # `attribute` never produced. Both of mypy's line formats appear here.
    sample = "\n".join(
        [
            "mfgarchon/alg/x.py:12: error: bare line format  [attr-defined]",
            "mfgarchon/alg/y.py:12:34: error: with a column  [attr-defined]",
            "mfgarchon/geometry/z.py:9:1: error: another package  [misc]",
            "mfgarchon/top.py:3:1: error: directly under the package  [misc]",
            "mfgarchon/alg/x.py:12:34: note: notes are not errors",
            "Found 4 errors in 4 files (checked 10 source files)",
        ]
    )
    got = attribute(sample)
    want = {"alg": 2, "geometry": 1, ROOT_BUCKET: 1}
    if got != want:
        failures.append(f"attribute() parsed {got}, expected {want} -- both mypy line formats must count")
    if _FOUND.search("Found 4 errors in 4 files (checked 10 source files)") is None:
        failures.append("_FOUND no longer matches mypy's own total line, so the authority is unreadable")
    if _FOUND.search("Success: no issues found in 6 source files") is not None:
        failures.append("_FOUND matches a success line, so a clean run would be read as an error total")

    # Drive the caller's return, not just the comparison.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "absent.json"
        if check_baseline(missing) != EXIT_INSTRUMENT_BROKEN:
            failures.append("a missing baseline must exit 2 (cannot measure), not 0 or 1")

    if failures:
        for line in failures:
            print(f"self-test FAILED: {line}", file=sys.stderr)
        return EXIT_MOVED
    print(
        "self-test OK: increase, decrease, a package leaving the scan, one entering, a new bucket and "
        "a resized scan all fire; no change is silent; identical zero counts part on status alone; "
        "attribute() counts both of mypy's line formats and the _root bucket; a missing baseline exits 2"
    )
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-baseline", type=Path, default=None, metavar="FILE")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--show", action="store_true", help="print the live per-package counts and exit")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Positive control: assert the comparison fires on an increase, a decrease, and a package that left the scan.",
    )
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if args.write_baseline:
        return write_baseline(BASELINE)
    if args.show:
        try:
            counts, checked = scan()
        except RuntimeError as exc:
            print(f"CANNOT RUN: {exc}", file=sys.stderr)
            return EXIT_INSTRUMENT_BROKEN
        for pkg in packages():
            status = probe_status(pkg) if not counts.get(pkg) else MEASURED
            shown = counts.get(pkg, 0) if status == MEASURED else "--"
            print(f"  {pkg:<16} {shown:>6}  {status}")
        if ROOT_BUCKET in counts:
            print(f"  {ROOT_BUCKET:<16} {counts[ROOT_BUCKET]:>6}  {MEASURED}")
        print(f"  {checked} source files checked")
        return EXIT_OK
    return check_baseline(args.check_baseline or BASELINE)


if __name__ == "__main__":
    sys.exit(main())

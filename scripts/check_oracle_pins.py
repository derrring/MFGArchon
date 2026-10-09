#!/usr/bin/env python3
"""Fail if a test that #2512's progress counts did not pass in the gate's own run (#1875).

`scripts/oracle_pins.txt` declares the oracle cells, (b) rows, conventions and pins #2512 counts.
`scripts/local_ci.sh` runs the suite with `-o junit_family=xunit1 -o junit_suite_name=<nonce>
--junitxml=<file>`, and this reads that run's own results: a target counts only if it PASSED there.
It reports, per target:

- ``ok``: passed in this run;
- ``SKIPPED``: skipped or xfailed in this run;
- ``FAILED``: failed or errored;
- ``NOT RUN``: collected by pytest, but not in this run: deselected by a marker, or excluded by an
  ``--ignore``, a ``norecursedirs`` or a ``-k``;
- ``MISSING``: not collected at all, through a rename, a deleted file or a broken import.

It exits 1 if any target is not ``ok``. It also exits 1, with its own message, if the results file is
missing, empty, unparseable, lists no tests, or carries another run's nonce, so a leftover file from
an earlier green run cannot stand in for a run that crashed before writing. It prints how many
declared items of each kind are met; a declared item with no target (``-``) counts in the
denominator and is never met. These lines are what a merge report quotes for #2512.

Usage:
    python scripts/check_oracle_pins.py --junit FILE --nonce NONCE   # the gate step, after the suite
    python scripts/check_oracle_pins.py --self-test                  # offline; no pytest run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "scripts" / "oracle_pins.txt"
KIND_ORDER = ("oracle", "row", "convention", "pin")
UNPINNED = "-"


class ResultsError(Exception):
    """The run's results cannot be read as this run's results."""


def read_manifest(path: Path) -> list[tuple[str, str, str]]:
    """``(kind, label, target)`` per non-comment line; ``target`` is ``-`` for a declared, unpinned item."""
    entries = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            raise SystemExit(f"{path}:{number}: expected '<kind> <label> <target>', got {line!r}")
        entries.append((parts[0], parts[1], parts[2]))
    return entries


def read_results(path: Path, nonce: str) -> dict[str, str]:
    """Node ID -> ``passed`` / ``skipped`` / ``failed`` from an xunit1 JUnit file written by this run."""
    if not path.is_file():
        raise ResultsError(f"no results file at {path}: the suite did not write one")
    if path.stat().st_size == 0:
        raise ResultsError(f"the results file {path} is empty")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ResultsError(f"the results file {path} does not parse: {exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    names = {suite.get("name") for suite in suites}
    if names != {nonce}:
        raise ResultsError(
            f"the results file {path} belongs to another run (suite name {sorted(map(str, names))}, expected {nonce!r})"
        )
    results: dict[str, str] = {}
    for case in root.iter("testcase"):
        file, classname, name = case.get("file"), case.get("classname") or "", case.get("name")
        if not file or not name:
            raise ResultsError(f"a test case in {path} has no file or name: is junit_family xunit1?")
        module = file[: -len(".py")].replace("/", ".")
        classes = classname[len(module) :].lstrip(".").split(".") if classname.startswith(module) else []
        nodeid = "::".join([file, *[c for c in classes if c], name])
        tags = {child.tag for child in case}
        results[nodeid] = "failed" if tags & {"failure", "error"} else ("skipped" if "skipped" in tags else "passed")
    if not results:
        raise ResultsError(f"the results file {path} lists no tests")
    return results


def collect_population(files: list[str]) -> set[str]:
    """Every node ID pytest collects from ``files``, with no marker selection, as the gate's suite line runs."""
    existing = [f for f in files if (REPO / f).is_file()]
    if not existing:
        return set()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONSAFEPATH"] = "1"
    # An explicit level, not -q: pytest.ini's addopts adds --verbose, and only -1 prints one node ID a line.
    cmd = [
        sys.executable,
        "-P",
        "-m",
        "pytest",
        "--collect-only",
        "--verbosity=-1",
        "-p",
        "no:cacheprovider",
        *existing,
    ]
    run = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    if run.returncode not in (0, 5):
        raise ResultsError(
            f"collecting the manifest's files failed (pytest exit {run.returncode}):\n{run.stdout[-1500:]}"
        )
    population = {line.strip() for line in run.stdout.splitlines() if "::" in line}
    if run.returncode == 0 and not population:
        raise ResultsError(
            f"pytest collected from {len(existing)} files but no node ID could be read:\n{run.stdout[-1500:]}"
        )
    return population


def evaluate(entries, results: dict[str, str], population: set[str]) -> tuple[int, list[str]]:
    """Exit status and report lines for ``entries`` against one run's ``results``."""
    words = {"passed": "ok", "skipped": "SKIPPED", "failed": "FAILED"}
    status_of: list[tuple[tuple[str, str, str], str]] = []
    for entry in entries:
        target = entry[2]
        if target == UNPINNED:
            status_of.append((entry, UNPINNED))
            continue
        if "::" in target:
            if target in results:
                status = words[results[target]]
            else:
                status = "NOT RUN" if target in population else "MISSING"
        else:
            tests = sorted(n for n in population if n.startswith(target + "::"))
            if not tests:
                status = "MISSING"
            else:
                outcomes = [results.get(n, "not run") for n in tests]
                for outcome, word in (("failed", "FAILED"), ("not run", "NOT RUN"), ("skipped", "SKIPPED")):
                    if outcome in outcomes:
                        status = f"{word} ({outcomes.count(outcome)} of {len(tests)} tests)"
                        break
                else:
                    status = "ok"
        status_of.append((entry, status))

    # A label is met only when it has at least one pinned line and every pinned line is ok; a declared,
    # unpinned label counts in the denominator and is never met.
    pinned_ok: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for (kind, label, _), status in status_of:
        pinned_ok[kind][label]  # declare the label even when it has no pinned line
        if status != UNPINNED:
            pinned_ok[kind][label].append(status == "ok")
    lines = [
        f"{status:12s} {kind} {label}  {target}"
        for (kind, label, target), status in status_of
        if status not in ("ok", UNPINNED)
    ]
    for kind in sorted(pinned_ok, key=lambda k: (KIND_ORDER.index(k) if k in KIND_ORDER else len(KIND_ORDER), k)):
        met = {label: bool(oks) and all(oks) for label, oks in pinned_ok[kind].items()}
        families: dict[str, list[bool]] = defaultdict(list)
        for label, ok in met.items():
            if "/" in label:
                families[label.split("/", 1)[0]].append(ok)
        detail = (
            " (" + ", ".join(f"{fam} {sum(v)}/{len(v)}" for fam, v in sorted(families.items())) + ")"
            if families
            else ""
        )
        lines.append(f"{kind}: {sum(met.values())}/{len(met)}{detail}")
    bad = any(status not in ("ok", UNPINNED) for _, status in status_of)
    return (1 if bad else 0), lines


def check(manifest: Path, junit: Path, nonce: str) -> tuple[int, list[str]]:
    entries = read_manifest(manifest)
    try:
        results = read_results(junit, nonce)
        files = sorted({t.split("::", 1)[0] for _, _, t in entries if t != UNPINNED})
        population = collect_population(files)
    except ResultsError as exc:
        return 1, [f"the instrument did not run: {exc}"]
    return evaluate(entries, results, population)


def _junit(cases: str, name: str = "run-1") -> str:
    return f'<testsuites><testsuite name="{name}" tests="1">{cases}</testsuite></testsuites>'


def self_test() -> int:
    """Every status, the whole-file arm, the cell rule, the denominators, and every carrier failure."""
    failures: list[str] = []
    passed = '<testcase file="t/a.py" classname="t.a" name="test_p"/>'
    in_class = '<testcase file="t/a.py" classname="t.a.TestK" name="test_k[1]"/>'
    skipped = '<testcase file="t/a.py" classname="t.a" name="test_s"><skipped message="x"/></testcase>'
    failed = '<testcase file="t/a.py" classname="t.a" name="test_f"><failure message="x"/></testcase>'
    population = {
        "t/a.py::test_p",
        "t/a.py::TestK::test_k[1]",
        "t/a.py::test_s",
        "t/a.py::test_f",
        "t/a.py::test_d",
        "t/b.py::test_q",
    }
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.xml"
        good.write_text(_junit(passed + in_class + skipped + failed))
        results = read_results(good, "run-1")

        def expect(target: str, word: str) -> None:
            status, lines = evaluate([("oracle", "c/x", target)], results, population)
            if status != (0 if word == "ok" else 1) or (word != "ok" and not lines[0].startswith(word)):
                failures.append(f"{target}: expected {word}, got status {status}, {lines}")

        expect("t/a.py::test_p", "ok")
        expect("t/a.py::TestK::test_k[1]", "ok")  # a class in classname rebuilds the node ID
        expect("t/a.py::test_s", "SKIPPED")
        expect("t/a.py::test_f", "FAILED")
        expect("t/a.py::test_d", "NOT RUN")  # collected, absent from the run: deselected
        expect("t/a.py::gone", "MISSING")
        expect("t/a.py", "FAILED")  # the whole-file arm: one failed test of five
        expect("t/b.py", "NOT RUN")  # the whole-file arm: its one test deselected
        expect("t/c.py", "MISSING")
        cell = [
            ("oracle", "c/x", "t/a.py::test_p"),
            ("oracle", "c/x", "t/a.py::test_d"),
            ("oracle", "c/y", UNPINNED),
        ]
        _, lines = evaluate(cell, results, population)
        if lines[-1] != "oracle: 0/2 (c 0/2)":
            failures.append(f"a cell with one line not run, and a declared cell, must count 0/2: {lines}")
        status, lines = evaluate(
            [("convention", "C1", "t/a.py::test_p"), ("convention", "C2", UNPINNED)], results, population
        )
        if status != 0 or lines != ["convention: 1/2"]:
            failures.append(f"a declared, unpinned item counts in the denominator and never fails: {status}, {lines}")

        (Path(tmp) / "empty.xml").write_text("")
        (Path(tmp) / "bad.xml").write_text("<testsuites><testsuite")
        (Path(tmp) / "none.xml").write_text(_junit(""))
        # Each carrier failure must reach its own arm: an empty file would otherwise read as unparseable.
        carriers = {
            "missing": (Path(tmp) / "nope.xml", "run-1", "no results file"),
            "empty": (Path(tmp) / "empty.xml", "run-1", "is empty"),
            "unparseable": (Path(tmp) / "bad.xml", "run-1", "does not parse"),
            "no tests": (Path(tmp) / "none.xml", "run-1", "lists no tests"),
            "another run's nonce": (good, "run-2", "belongs to another run"),
        }
        for case, (path, nonce, says) in carriers.items():
            try:
                read_results(path, nonce)
            except ResultsError as exc:
                if says not in str(exc):
                    failures.append(f"a results file that is {case} was refused, but not as {case}: {exc}")
                continue
            failures.append(f"a results file that is {case} was accepted")
    if failures:
        print("self-test: FAILED\n  " + "\n  ".join(failures))
        return 1
    print("self-test: ok (every status, the whole-file arm, the cell rule, the denominators, five carrier failures)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--nonce")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.junit is None or args.nonce is None:
        parser.error("--junit and --nonce are required: this reads the gate's own run")
    status, lines = check(args.manifest, args.junit, args.nonce)
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(main())

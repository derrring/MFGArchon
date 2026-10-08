#!/usr/bin/env python3
"""Fail if a test that #2512's progress counts is not collected by the gate (#1875).

`scripts/oracle_pins.txt` lists the oracle cells, conventions and pins the progress count rests on.
This collects each target under the gate's own marker expression, `scripts/ci_markers.txt`, and
reports, per target:

- ``ok``: collected under the gate;
- ``DESELECTED``: collected without the markers, but not with them; or
- ``MISSING``: not collected at all, through a rename, a deleted file or a broken import.

It exits 1 if any target is DESELECTED or MISSING, and prints how many cells of each kind still
count. A pin that leaves the gate, through a marker, a rename or a skip at collection, then turns
the gate red instead of quietly lowering the count. #2570's max-over-fields pins were deselected
this way, by a name rule that marked any test called "large" as slow.

Usage:
    python scripts/check_oracle_pins.py                 # the gate step
    python scripts/check_oracle_pins.py --self-test     # proves it can see a deselected and a missing pin
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "scripts" / "oracle_pins.txt"
MARKERS = REPO / "scripts" / "ci_markers.txt"


def read_manifest(path: Path) -> list[tuple[str, str, str]]:
    """``(kind, label, target)`` per non-comment line."""
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


def collect(files: list[str], markers: str | None) -> set[str]:
    """Node IDs pytest collects from ``files``, under ``markers`` if given."""
    existing = [f for f in files if (REPO / f).is_file()]
    if not existing:
        return set()
    # Collected the way `scripts/local_ci.sh` runs the suite: `-P` with PYTHONSAFEPATH=1, under its markers.
    cmd = [sys.executable, "-P", "-m", "pytest", "--collect-only", "-qq", "-p", "no:cacheprovider", *existing]
    if markers is not None:
        cmd += ["-m", markers]
    env = {**os.environ, "PYTHONSAFEPATH": "1"}
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env).stdout
    return {line.strip() for line in out.splitlines() if "::" in line}


def check(manifest: Path) -> tuple[int, list[str]]:
    """Exit status and report lines for ``manifest``."""
    entries = read_manifest(manifest)
    files = sorted({target.split("::", 1)[0] for _, _, target in entries})
    markers = MARKERS.read_text().strip()
    everything, gated = collect(files, None), collect(files, markers)
    if not everything and files:
        return 1, [f"pytest collected nothing from {len(files)} file(s): the instrument did not run"]

    status_of: dict[tuple[str, str, str], str] = {}
    for entry in entries:
        target = entry[2]
        if "::" in target:
            status = "ok" if target in gated else ("DESELECTED" if target in everything else "MISSING")
        else:
            in_file = {n for n in everything if n.startswith(target + "::")}
            if not in_file:
                status = "MISSING"
            elif not in_file <= gated:
                status = f"DESELECTED ({len(in_file - gated)} of {len(in_file)} tests)"
            else:
                status = "ok"
        status_of[entry] = status

    labels: dict[str, dict[str, bool]] = defaultdict(dict)
    for (kind, label, _), status in status_of.items():
        labels[kind][label] = labels[kind].get(label, True) and status == "ok"
    lines = [
        f"{status:12s} {kind} {label}  {target}"
        for (kind, label, target), status in status_of.items()
        if status != "ok"
    ]
    for kind in sorted(labels):
        held = sum(labels[kind].values())
        families: dict[str, list[bool]] = defaultdict(list)
        for label, ok in labels[kind].items():
            families[label.split("/", 1)[0] if "/" in label else label].append(ok)
        detail = ", ".join(f"{fam} {sum(v)}/{len(v)}" for fam, v in sorted(families.items()))
        lines.append(f"{kind}: {held}/{len(labels[kind])} collect under the gate ({detail})")
    bad = any(status != "ok" for status in status_of.values())
    return (1 if bad else 0), lines


def self_test() -> int:
    """The real manifest passes; a deselected target and a missing one each fail, and are named."""
    status, lines = check(MANIFEST)
    if status != 0:
        print("self-test: the real manifest does not pass:\n  " + "\n  ".join(lines))
        return 1
    # A deselected target, found rather than assumed: a test that carries @pytest.mark.slow.
    probe = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-qq",
        "-p",
        "no:cacheprovider",
        "-m",
        "slow",
        "tests/unit",
    ]
    slow = sorted(
        line.strip()
        for line in subprocess.run(probe, cwd=REPO, capture_output=True, text=True).stdout.splitlines()
        if "::" in line
    )
    if not slow:
        print("self-test: found no @pytest.mark.slow test to use as the deselected control")
        return 1
    cases = {
        "DESELECTED": f"oracle control/slow {slow[0]}\n",
        "MISSING": "oracle control/missing tests/unit/test_alg/test_hjb_fdm_bc_oracles_2512.py::no_such_test\n",
    }
    for expected, text in cases.items():
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(text)
        status, lines = check(Path(handle.name))
        Path(handle.name).unlink()
        if status != 1 or not any(line.startswith(expected) for line in lines):
            print(f"self-test: a {expected} target was not reported as such: status {status}, {lines}")
            return 1
    print(f"self-test: ok (the real manifest passes; {slow[0]} is reported DESELECTED, a missing node MISSING)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    status, lines = check(args.manifest)
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(main())

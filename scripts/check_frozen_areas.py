#!/usr/bin/env python3
"""Fail when a test is added against a frozen prototype paradigm.

`mfgarchon/alg/neural/` and `mfgarchon/alg/reinforcement/` are design prototypes, not under
development (CLAUDE.md, "FROZEN"). Adding tests to a placeholder is the counter-intuitive half of
that freeze: coverage reads as a promise that the behaviour is intended and load-bearing, so a
later reader starts preserving decisions nobody made.

Prose does not stop this. `hasattr` is banned by the project's Python conventions and was written
into a test anyway on 2026-07-30, because the fail-fast ratchet scans `mfgarchon/` and not
`tests/` (#1780) -- a rule the checker cannot see is a rule the surrounding code stops teaching.
So the freeze gets a counter, not just a paragraph.

WHAT THIS DOES NOT DO. It counts test FILES that import a frozen package; it does not read intent
and it does not look at line counts. Deleting a frozen test lowers the count and the baseline must
be regenerated, which is the same ratchet shape as `check_fail_fast.py`. The explicitly allowed
actions -- keeping the packages importable, a one-line build fix, filing an issue -- add no test
file and cannot trip it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FROZEN_PACKAGES = ("mfgarchon.alg.neural", "mfgarchon.alg.reinforcement")

# Matches `from mfgarchon.alg.neural...` and `import mfgarchon.alg.neural...`, including
# submodules, and tolerates leading whitespace for imports inside a function or fixture.
_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(" + "|".join(re.escape(p) for p in FROZEN_PACKAGES) + r")\b",
    re.MULTILINE,
)


def offending_files(tests_root: Path) -> dict[str, list[str]]:
    """Test files importing each frozen package, keyed by package."""
    found: dict[str, list[str]] = {p: [] for p in FROZEN_PACKAGES}
    for path in sorted(tests_root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # A file that cannot be read cannot be classified; surfacing it is more useful than
            # skipping it silently, which would let the count drift down for the wrong reason.
            raise
        for package in {m.group(1) for m in _IMPORT.finditer(text)}:
            found[package].append(str(path))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", default="tests", help="Test root to scan")
    parser.add_argument("--write-baseline", metavar="FILE")
    parser.add_argument("--check-baseline", metavar="FILE")
    parser.add_argument("--self-test", action="store_true", help="Prove the check can fail")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    found = offending_files(Path(args.tests))
    counts = {package: len(files) for package, files in found.items()}

    if args.write_baseline:
        Path(args.write_baseline).write_text(
            json.dumps(
                {
                    "_comment": (
                        "Test files importing a FROZEN prototype paradigm (CLAUDE.md). The count "
                        "must not grow: adding tests to a placeholder makes it read as a pinned "
                        "contract. Lowering it by deleting such a test is welcome -- regenerate "
                        "with --write-baseline in the same change."
                    ),
                    "counts": counts,
                    "files": found,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote frozen-area baseline: {counts}")
        return 0

    if args.check_baseline:
        baseline = json.loads(Path(args.check_baseline).read_text())
        recorded = baseline["counts"]
        grew = {p: (recorded.get(p, 0), counts[p]) for p in counts if counts[p] > recorded.get(p, 0)}
        if grew:
            print("FAIL: new tests added against a FROZEN prototype paradigm.")
            for package, (was, now) in sorted(grew.items()):
                print(f"  {package}: {was} -> {now} test file(s)")
                for path in sorted(set(found[package]) - set(baseline["files"].get(package, []))):
                    print(f"      + {path}")
            print(
                "\nalg/neural and alg/reinforcement are prototypes, not under development "
                "(CLAUDE.md).\nA placeholder with tests reads as a pinned contract. File an issue "
                "instead, or get an\nexplicit instruction naming the paradigm and regenerate:\n"
                "  python scripts/check_frozen_areas.py --write-baseline "
                f"{args.check_baseline}"
            )
            return 1
        shrank = {p: (recorded.get(p, 0), counts[p]) for p in counts if counts[p] < recorded.get(p, 0)}
        if shrank:
            print(f"FAIL: frozen-area test count DROPPED {shrank} -- regenerate the baseline.")
            return 1
        print(f"OK: no new tests against frozen paradigms {counts}")
        return 0

    print(json.dumps({"counts": counts, "files": found}, indent=2))
    return 0


def self_test() -> int:
    """Prove the check fires. A gate that has never been seen to fail is not known to work."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "tests"
        (root / "unit").mkdir(parents=True)
        (root / "unit" / "test_clean.py").write_text("from mfgarchon.core import mfg_problem\n")
        baseline = Path(tmp) / "baseline.json"

        before = offending_files(root)
        if any(before.values()):
            print("SELF-TEST FAILED: the clean fixture already reports offenders")
            return 1
        baseline.write_text(json.dumps({"counts": dict.fromkeys(FROZEN_PACKAGES, 0), "files": before}) + "\n")

        # The change this exists to catch, in each of the three shapes a test can take.
        for name, body in (
            ("test_from.py", "from mfgarchon.alg.reinforcement.algorithms import x\n"),
            ("test_import.py", "import mfgarchon.alg.neural.nn\n"),
            ("test_indented.py", "def test_a():\n    from mfgarchon.alg.neural import core\n"),
        ):
            (root / "unit" / name).write_text(body)
            after = offending_files(root)
            if not any(after.values()):
                print(f"SELF-TEST FAILED: {name} was not detected")
                return 1
            (root / "unit" / name).unlink()

        print("SELF-TEST PASSED: from-import, plain import, and function-scoped import all detected")
        return 0


if __name__ == "__main__":
    sys.exit(main())

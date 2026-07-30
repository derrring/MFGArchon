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
and it does not look at line counts. Deleting a frozen test drops it from the set and the baseline
must be regenerated in the same change -- the shape `check_doc_api.py` uses, which hard-fails on
a drop. (`check_fail_fast.py` only nudges and exits 0; citing it here was wrong.) The explicitly allowed
actions -- keeping the packages importable, a one-line build fix, filing an issue -- add no test
file and cannot trip it.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

FROZEN_PACKAGES = ("mfgarchon.alg.neural", "mfgarchon.alg.reinforcement")

# Matches `from mfgarchon.alg.neural...` and `import mfgarchon.alg.neural...`, including
# submodules, and tolerates leading whitespace for imports inside a function or fixture.
_PREFIXES = tuple(FROZEN_PACKAGES)


def _names_a_frozen_package(text: str) -> bool:
    """Whether this string names a frozen package or one of its submodules."""
    return any(text == p or text.startswith(p + ".") for p in _PREFIXES)


def _references(path: Path) -> set[str]:
    """Frozen packages this file reaches, by AST -- imports AND string literals.

    Regex over `from|import <dotted>` was the first form and it missed the idiom this very
    baseline depends on. `test_mean_field_rl_requires_pop_state_1508.py` reaches three frozen RL
    algorithms through `@pytest.mark.parametrize` + `importlib.import_module(name)`, and was
    counted only because of one unrelated static import elsewhere in the file. Lifting its
    parametrised half into a new file passed the gate.

    So string literals count too. `importlib.import_module("mfgarchon.alg.neural.nn")`,
    `pytest.importorskip(...)` (20+ uses in this suite) and `patch("mfgarchon.alg.neural.nn.MLP")`
    are all house style here, not evasion.

    Also handles `from mfgarchon.alg import neural`, where the frozen name is the imported symbol
    rather than part of the module path -- the shape `mfgarchon/alg/__init__.py` exports and
    `from mfgarchon.alg import SchemeFamily` already uses in six test files.
    """
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _names_a_frozen_package(alias.name):
                    found.add(next(p for p in _PREFIXES if alias.name.startswith(p)))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if _names_a_frozen_package(node.module):
                found.add(next(p for p in _PREFIXES if node.module.startswith(p)))
            else:
                # `from mfgarchon.alg import neural`
                for alias in node.names:
                    dotted = f"{node.module}.{alias.name}"
                    if _names_a_frozen_package(dotted):
                        found.add(next(p for p in _PREFIXES if dotted.startswith(p)))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _names_a_frozen_package(node.value):
                found.add(next(p for p in _PREFIXES if node.value.startswith(p)))
    return found


def offending_files(tests_root: Path) -> dict[str, list[str]]:
    """Test files reaching each frozen package, keyed by package."""
    found: dict[str, list[str]] = {p: [] for p in FROZEN_PACKAGES}
    for path in sorted(tests_root.rglob("*.py")):
        for package in _references(path):
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
        # As check_doc_api.py does: a gate whose self-test is an opt-in flag has its verification
        # happen once, by hand, and never again.
        if self_test() != 0:
            return 1
        baseline = json.loads(Path(args.check_baseline).read_text())
        recorded = {p: set(baseline["files"].get(p, [])) for p in FROZEN_PACKAGES}
        # Sets, not counts. A count nets to zero on delete-one-add-one, which is exactly the shape
        # of "I removed a frozen test and added a different one" -- the second half is the thing
        # this gate exists to stop.
        added = {p: sorted(set(found[p]) - recorded[p]) for p in FROZEN_PACKAGES}
        added = {p: f for p, f in added.items() if f}
        if added:
            print("FAIL: new tests added against a FROZEN prototype paradigm.")
            for package, files in sorted(added.items()):
                print(f"  {package}:")
                for path in files:
                    print(f"      + {path}")
            print(
                "\nalg/neural and alg/reinforcement are prototypes, not under development "
                "(CLAUDE.md).\nA placeholder with tests reads as a pinned contract. File an issue "
                "instead, or get an\nexplicit instruction naming the paradigm and regenerate:\n"
                "  python scripts/check_frozen_areas.py --write-baseline "
                f"{args.check_baseline}"
            )
            return 1
        removed = {p: sorted(recorded[p] - set(found[p])) for p in FROZEN_PACKAGES}
        removed = {p: f for p, f in removed.items() if f}
        if removed:
            print(f"FAIL: frozen-area tests disappeared {removed} -- regenerate the baseline.")
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
        # Positive control on the NEAREST false-positive class: a sibling package under alg/.
        # A mutant broadening the match to `mfgarchon.alg.*` would flag every numerical test, and
        # a clean fixture that only imports mfgarchon.core would not notice.
        (root / "unit" / "test_clean.py").write_text(
            "from mfgarchon.core import mfg_problem\n"
            "from mfgarchon.alg.numerical.fp_solvers import fp_fdm\n"
            'import importlib; importlib.import_module("mfgarchon.alg.numerical")\n'
        )
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

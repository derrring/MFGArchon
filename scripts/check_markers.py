#!/usr/bin/env python3
"""Every declared pytest marker must do something, and must not promise a schedule it lacks.

Two invariants, both mechanical:

**Reachable** -- a declared marker is applied by at least one test, or referenced by at least one
selector (a `-m` expression in a workflow or in `scripts/local_ci.sh`). A marker that is neither is
dead weight that `--strict-markers` cannot catch: that flag rejects *undeclared* markers, so a
declaration is exactly what makes a meaningless marker look legitimate.

**Honest** -- a marker whose description promises a schedule ("run on every commit", "run weekly")
must be referenced by a selector. This is the dangerous case rather than the merely useless one. The
declaration `tier4: Performance/stress tests - run weekly or manually` existed here with zero
selectors referencing it, so a test marked `tier4` in the belief it would be deferred would in fact
have run in *every* tier -- the opposite of the promise. Names alone carry the same suggestion, which
is why the tier1-4 family was removed rather than re-described.

Deliberately NOT checked: whether a marker is *useful*. `unit` (1404 uses) and `fast` (154) route
nothing automatically and are still legitimate -- they are ad-hoc selectors a developer types by
hand. Their descriptions are descriptive, not promissory, so they pass the second invariant. Judging
usefulness would need a standard this script does not have.

Usage:
    python scripts/check_markers.py                  # report; exit 1 on any violation
    python scripts/check_markers.py --json           # machine-readable census
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys

# Wording that claims a marker changes WHEN a test runs. Kept small and literal on purpose: a
# broader vocabulary would start flagging descriptive text like "may take >30 seconds".
SCHEDULE_PROMISES = (
    "run on",
    "run every",
    "run weekly",
    "run daily",
    "run nightly",
    "on every commit",
    "on prs",
    "on merge",
    "skipped on",
)

SELECTOR_FILES = (".github/workflows/*.yml", ".github/workflows/*.yaml", "scripts/local_ci.sh")


def declared_markers(root: pathlib.Path) -> dict[str, str]:
    """Marker name -> description, from pytest.ini's `markers` linelist."""
    ini = (root / "pytest.ini").read_text()
    if "markers =" not in ini:
        raise SystemExit("pytest.ini has no `markers =` block; this checker assumes one exists")
    block = ini.split("markers =", 1)[1]
    out = {}
    for line in block.splitlines():
        if line.strip() and not line.startswith((" ", "\t")):
            break  # left the indented block
        m = re.match(r"\s+(\w+)\s*:\s*(.*)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def applied_markers(root: pathlib.Path) -> dict[str, int]:
    """Marker name -> number of `pytest.mark.<name>` applications under tests/.

    AST-based: a textual scan would count the name inside docstrings and inside this file's own
    prose, which is how a marker can look used while being applied nowhere.
    """
    counts: dict[str, int] = {}
    for path in sorted((root / "tests").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr == "mark":
                counts[node.attr] = counts.get(node.attr, 0) + 1
    return counts


def selector_references(root: pathlib.Path) -> dict[str, list[str]]:
    """Marker name -> files whose `-m` expressions mention it."""
    refs: dict[str, list[str]] = {}
    names = declared_markers(root)
    for pattern in SELECTOR_FILES:
        for path in sorted(root.glob(pattern)):
            text = path.read_text()
            for name in names:
                if re.search(rf"\bnot {re.escape(name)}\b|-m [\"']?{re.escape(name)}\b", text):
                    refs.setdefault(name, []).append(str(path.relative_to(root)))
    return refs


def census(root: pathlib.Path) -> dict:
    declared = declared_markers(root)
    applied = applied_markers(root)
    refs = selector_references(root)
    rows = {}
    for name, description in declared.items():
        uses = applied.get(name, 0)
        selectors = refs.get(name, [])
        promises = [p for p in SCHEDULE_PROMISES if p in description.lower()]
        rows[name] = {
            "description": description,
            "uses": uses,
            "selectors": selectors,
            "unreachable": uses == 0 and not selectors,
            "false_promise": bool(promises) and not selectors,
            "promise_words": promises,
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=".", help="repository root")
    parser.add_argument("--json", action="store_true", help="emit the census as JSON")
    args = parser.parse_args()

    root = pathlib.Path(args.path).resolve()
    rows = census(root)

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))

    unreachable = sorted(n for n, r in rows.items() if r["unreachable"])
    false_promise = sorted(n for n, r in rows.items() if r["false_promise"])

    if not args.json:
        print(f"{len(rows)} declared markers\n")
        print(f"{'marker':22s} {'uses':>5s}  selectors")
        for name, row in rows.items():
            sel = ", ".join(row["selectors"]) or "--"
            print(f"{name:22s} {row['uses']:5d}  {sel}")
        print()

    if unreachable:
        print(f"UNREACHABLE ({len(unreachable)}): {', '.join(unreachable)}")
        print("  Applied by no test and named by no selector. Delete the declaration, or use it.")
    if false_promise:
        print(f"FALSE PROMISE ({len(false_promise)}):")
        for name in false_promise:
            print(f"  {name}: description claims {rows[name]['promise_words']} but no selector routes it")
        print("  A test marked with one of these runs in EVERY tier, which is the opposite of the")
        print("  description. Either add the selector or drop the scheduling claim.")

    if unreachable or false_promise:
        return 1
    if not args.json:
        print("OK -- every declared marker is reachable, and none promises an unimplemented schedule.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

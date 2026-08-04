#!/usr/bin/env python3
"""Count the deprecations that actually exist, and refuse to let the count drift unnoticed.

Two checks, and the first one is why this file was rewritten (Issue #1713).

**The count.** The previous version discovered deprecated symbols by walking the AST for
``@deprecated`` decorators and calling ``_extract_metadata``, which returned ``None`` unless the
decorator literally passed ``removal_blockers=``. That kwarg is optional and **no call site in the
package passes it**, so the detector found nothing by construction: it printed ``Total deprecated
symbols: 0`` and exited 0 while ``audit_all_deprecations`` reported 72 live in the same tree, and
``.github/workflows/deprecation-check.yml`` was green over it.

This now asks the **runtime registry** instead. ``audit_all_deprecations`` reads what the decorators
actually registered, so it cannot under-count the way a syntax guess can -- the same class of error
that made ``check_fail_fast``'s ``broad_except`` read 11 against a true 115 for 32 days, because
``except\\s+Exception\\s*:`` cannot match ``except Exception as e:`` (#1706).

**Boundary, and it is real:** this covers deprecations *declared through the decorators*. Something
retired without one is invisible here too. Asking the registry removes the under-counting that came
from guessing syntax; it does not make the count a census of everything ever deprecated.

**The usage check.** The original purpose: production code must not call a deprecated symbol whose
``internal_usage`` blocker has been cleared. It is kept, and now runs over a real symbol list. Note
it is vacuous today by construction -- 0 of 72 have that blocker cleared -- so it starts doing work
the day someone clears one, and not before. Reporting the count is what carries information now.

The baseline fails in BOTH directions, like ``fail_fast_baseline.json`` and
``capability_baseline.json``: a count that drops must be recorded too, so an improvement cannot land
silently and a baseline cannot be lowered without doing the work.

Usage:
    python scripts/check_internal_deprecation.py                    # check against the shipped baseline
    python scripts/check_internal_deprecation.py --write-baseline   # record the current counts
    python scripts/check_internal_deprecation.py --show             # list every live deprecation
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().parent / "deprecation_baseline.json"


def live_deprecations() -> list[dict[str, Any]]:
    """Every deprecation the decorators registered, flattened across the audit's buckets."""
    import logging

    logging.disable(logging.INFO)  # importing the package emits workflow setup chatter
    import mfgarchon
    from mfgarchon.utils.deprecation import audit_all_deprecations

    buckets = audit_all_deprecations(mfgarchon)
    return [entry for bucket in buckets.values() for entry in bucket]


def counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    """The numbers the baseline pins.

    Coarse on purpose: a per-symbol baseline would churn on every rename and stop being read.
    """
    by_type: dict[str, int] = {}
    for entry in entries:
        by_type[entry["type"]] = by_type.get(entry["type"], 0) + 1
    cleared = sum(1 for e in entries if not _blocked_on_internal_usage(e))
    return {
        "total": len(entries),
        "internal_usage_cleared": cleared,
        **{f"type:{k}": v for k, v in sorted(by_type.items())},
    }


def _blocked_on_internal_usage(entry: dict[str, Any]) -> bool:
    blockers = entry.get("remaining_blockers") or entry.get("removal_blockers") or []
    return "internal_usage" in blockers


def production_uses(symbols: set[str]) -> list[str]:
    """Call sites in ``mfgarchon/`` naming one of ``symbols``, excluding the places that must.

    ``deprecation.py`` defines the decorators and ``compat/`` exists to hold backward-compatible
    wrappers. A hit anywhere else is production code depending on something declared ready to go.
    """
    if not symbols:
        return []
    hits = []
    for path in sorted((REPO / "mfgarchon").rglob("*.py")):
        if path.name == "deprecation.py" or "compat" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name in symbols:
                hits.append(f"{path.relative_to(REPO)}:{node.lineno}: {name}")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-baseline", action="store_true", help="record current counts and exit")
    parser.add_argument("--show", action="store_true", help="list every live deprecation")
    args = parser.parse_args()

    entries = live_deprecations()
    current = counts(entries)

    if args.show:
        for entry in sorted(entries, key=lambda x: (x["type"], x["name"])):
            blockers = ",".join(entry.get("remaining_blockers") or []) or "-"
            print(
                f"  {entry['type']:10s} {entry['name']:58s} since {entry.get('since')!s:9s} "
                f"removal {entry.get('removal')!s:9s} [{blockers}]"
            )
        print()

    print(f"Live deprecations (from the runtime registry): {current['total']}")
    for key, value in current.items():
        if key != "total":
            print(f"  {key}: {value}")

    if args.write_baseline:
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"\nBaseline written to {BASELINE.relative_to(REPO)}")
        return 0

    if not BASELINE.exists():
        print(f"\nNo baseline at {BASELINE.relative_to(REPO)}; write one with --write-baseline.", file=sys.stderr)
        return 2

    baseline = json.loads(BASELINE.read_text())
    drift = [
        f"  {key}: {baseline.get(key, 0)} -> {current.get(key, 0)}"
        for key in sorted(set(baseline) | set(current))
        if baseline.get(key, 0) != current.get(key, 0)
    ]
    if drift:
        print("\nDeprecation counts moved against the baseline:")
        print("\n".join(drift))
        print(
            "\nBoth directions fail. A count that ROSE means a deprecation was added without "
            "recording it; a count that FELL means one was retired while the baseline still claims "
            "it exists. Regenerate with --write-baseline in the commit that does the work."
        )
        return 1

    ready = {e["name"].split(".")[-1] for e in entries if not _blocked_on_internal_usage(e)}
    violations = production_uses(ready)
    if violations:
        print(f"\nProduction code calls {len(violations)} deprecated symbol(s) with internal_usage cleared:")
        for violation in violations[:40]:
            print(f"  {violation}")
        return 1

    print(f"\nMatches baseline. {len(ready)} symbol(s) have internal_usage cleared; no production call sites.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

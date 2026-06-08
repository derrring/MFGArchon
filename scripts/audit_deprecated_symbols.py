#!/usr/bin/env python3
"""
Audit deprecated symbols for removal readiness.

Scans the IMPORTED ``mfgarchon`` package for live ``@deprecated`` /
``@deprecated_parameter`` / ``@deprecated_value`` metadata -- the source of truth, read off
the decorated objects rather than fragile source-text parsing -- and categorizes each unique
deprecation against the removal policy (3 minor versions OR 6 months; CLAUDE.md):

  - READY     : age-eligible AND all removal blockers cleared -> safe to delete
  - NOT READY : age-eligible but removal blockers remain
  - ACTIVE    : not yet old enough to remove

Removal blockers (``internal_usage``, ``equivalence_test``, ``migration_docs``) are checklist
items you mark cleared with ``--cleared`` once verified; the audit does NOT auto-check live
usage, so always confirm a READY symbol's call sites before deleting.

Usage:
    python scripts/audit_deprecated_symbols.py
    python scripts/audit_deprecated_symbols.py --cleared internal_usage equivalence_test migration_docs
    python scripts/audit_deprecated_symbols.py --current-version v0.19.8 --ready-only

Created: 2026-01-20 (Issue #616); reworked to use the runtime deprecation registry.
Reference: docs/development/DEPRECATION_LIFECYCLE_POLICY.md
"""

from __future__ import annotations

import argparse
import sys


def _emit(title: str, items: list[dict]) -> None:
    print(f"\n{title}  ({len(items)})")
    for it in sorted(items, key=lambda x: (x.get("since", ""), x.get("name", ""))):
        line = f"  - {it['name']}  [{it.get('since', '?')}, {it.get('type', '?')}]"
        if it.get("remaining_blockers"):
            line += f"  blockers: {', '.join(it['remaining_blockers'])}"
        print(line)
        if it.get("age_reason"):
            print(f"      {it['age_reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit deprecated symbols for removal readiness")
    parser.add_argument(
        "--current-version",
        default=None,
        help="Version to judge age against (default: installed mfgarchon version)",
    )
    parser.add_argument(
        "--cleared",
        nargs="+",
        default=[],
        help="Removal blockers cleared globally (e.g. internal_usage equivalence_test migration_docs)",
    )
    parser.add_argument("--ready-only", action="store_true", help="List only the READY-for-removal symbols")
    args = parser.parse_args()

    import mfgarchon
    from mfgarchon.utils.deprecation import audit_all_deprecations

    report = audit_all_deprecations(mfgarchon, current_version=args.current_version, completed_blockers=args.cleared)
    all_items = report["ready"] + report["not_ready"] + report["active"]
    cur_ver = all_items[0]["current_version"] if all_items else "(unknown)"

    print("=" * 72)
    print(f"Deprecated Symbol Audit   current={cur_ver}   cleared blockers={args.cleared or 'none'}")
    print("=" * 72)

    _emit("READY FOR REMOVAL (age-eligible + blockers cleared)", report["ready"])
    if not args.ready_only:
        _emit("NOT READY (age-eligible, blockers remain)", report["not_ready"])
        print(f"\nACTIVE (not yet old enough): {len(report['active'])} symbol(s)")

    print(
        f"\nSummary: {len(report['ready'])} ready, {len(report['not_ready'])} not-ready, "
        f"{len(report['active'])} active."
    )
    print(
        "\nNote: 'ready' means age-eligible with the listed blockers cleared via --cleared; "
        "the audit does not scan live usage -- verify call sites before deleting."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

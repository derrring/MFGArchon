#!/usr/bin/env python3
"""Collate changelog.d/ fragments into a Keep-a-Changelog section (Issue #1521).

Each fragment is ``changelog.d/<slug>.<category>.md`` where <category> is one of
added/changed/deprecated/removed/fixed; the file holds the markdown bullet(s) for that change.
Every PR adds its OWN fragment file, so there is no append-only merge conflict on CHANGELOG.md.

Usage:
  python scripts/collate_changelog.py                       # print the collated section to stdout
  python scripts/collate_changelog.py --check               # validate fragment filenames (CI); exit 1 on a bad name
  python scripts/collate_changelog.py --version X.Y.Z --date YYYY-MM-DD   # print a full "## [X.Y.Z] - date" section
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CATEGORIES = ["added", "changed", "deprecated", "removed", "fixed"]
HEADINGS = {
    "added": "Added",
    "changed": "Changed",
    "deprecated": "Deprecated",
    "removed": "Removed (BREAKING)",
    "fixed": "Fixed",
}
FRAG_DIR = Path(__file__).resolve().parent.parent / "changelog.d"


def _collect() -> tuple[dict[str, list[str]], list[str]]:
    grouped: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    bad: list[str] = []
    for frag in sorted(FRAG_DIR.glob("*.md")):
        if frag.name == "README.md":
            continue
        parts = frag.stem.rsplit(".", 1)  # <slug>.<category>
        if len(parts) != 2 or parts[1] not in CATEGORIES:
            bad.append(frag.name)
            continue
        grouped[parts[1]].append(frag.read_text().strip())
    return grouped, bad


def render(version: str | None = None, date: str | None = None) -> str:
    grouped, bad = _collect()
    if bad:
        raise SystemExit(f"invalid fragment filename(s): {bad}; expected <slug>.<{'|'.join(CATEGORIES)}>.md")
    out: list[str] = []
    if version:
        out.append(f"## [{version}] - {date}\n")
    for cat in CATEGORIES:
        if grouped[cat]:
            out.append(f"### {HEADINGS[cat]}\n")
            out.append("\n".join(grouped[cat]))
            out.append("")
    return ("\n".join(out).rstrip() + "\n") if out else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate fragment filenames only")
    ap.add_argument("--version")
    ap.add_argument("--date")
    args = ap.parse_args()
    if args.check:
        _, bad = _collect()
        if bad:
            print(f"invalid changelog.d fragment filename(s): {bad}", file=sys.stderr)
            sys.exit(1)
        print("changelog.d fragments OK")
        return
    sys.stdout.write(render(args.version, args.date))


if __name__ == "__main__":
    main()

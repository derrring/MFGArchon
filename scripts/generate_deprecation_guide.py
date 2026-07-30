#!/usr/bin/env python3
"""Auto-generate DEPRECATION_MODERNIZATION_GUIDE.md from decorator metadata.

Scans mfgarchon/ for @deprecated and @deprecated_parameter decorators,
extracts metadata, and generates a user-facing migration guide.

Usage:
    python scripts/generate_deprecation_guide.py           # Generate
    python scripts/generate_deprecation_guide.py --check   # Check if up-to-date

Issue #989: Auto-generate deprecation guide.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def scan_all_deprecations() -> list[dict]:
    """Scan mfgarchon for all deprecated items.

    NOTE: this walks the package by IMPORTING it, so the result depends on which optional extras
    are installed -- `scan_deprecated` skips an unimportable submodule and everything below it.
    Measured: 72 deduplicated items with torch present, 41 without. That is why this is not wired
    to a gate; see Issue #1774.
    """
    import mfgarchon
    from mfgarchon.utils.deprecation import scan_deprecated

    return scan_deprecated(mfgarchon, recursive=True)


def group_by_version(items: list[dict]) -> dict[str, list[dict]]:
    """Group deprecation items by 'since' version."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        version = item.get("since", "unknown")
        groups[version].append(item)
    return dict(sorted(groups.items(), reverse=True))


def deduplicate(items: list[dict]) -> list[dict]:
    """Remove duplicate entries (same name + type + since + replacement)."""
    seen = set()
    unique = []
    for item in items:
        key = (
            item.get("name", ""),
            item.get("type", ""),
            item.get("since", ""),
            item.get("replacement", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def format_item(item: dict) -> str:
    """Format a single deprecation item as markdown."""
    name = item.get("name", "unknown")
    replacement = item.get("replacement", "N/A")
    removal = item.get("removal", "v1.0.0")
    item_type = item.get("type", "unknown")

    if item_type == "parameter":
        # "ClassName.method.param_name" -> extract parts
        parts = name.split(".")
        if len(parts) >= 2:
            func_name = ".".join(parts[:-1])
            param = parts[-1]
            return f"- **`{param}`** in `{func_name}()` — use `{replacement}` instead (remove by {removal})"
        return f"- **`{name}`** — use `{replacement}` instead (remove by {removal})"
    elif item_type == "function":
        return f"- **`{name}()`** — use `{replacement}` instead (remove by {removal})"
    elif item_type == "property":
        return f"- **`{name}`** (property) — use `{replacement}` instead (remove by {removal})"
    elif item_type == "alias":
        return f"- **`{name}`** (import alias) — use `{replacement}` instead (remove by {removal})"
    else:
        return f"- **`{name}`** ({item_type}) — use `{replacement}` instead (remove by {removal})"


def generate_guide(items: list[dict]) -> str:
    """Generate the full markdown guide."""
    items = deduplicate(items)
    groups = group_by_version(items)

    lines = [
        "# Deprecation Modernization Guide",
        "",
        "**Auto-generated** by `scripts/generate_deprecation_guide.py`",
        f"**Total deprecated items**: {len(items)}",
        f"**Versions covered**: {', '.join(groups.keys())}",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This guide documents deprecated usage patterns in MFGArchon and provides",
        "migration paths to modern APIs. All deprecated patterns emit warnings at",
        "runtime and will be removed at the version specified.",
        "",
        "To find deprecated usage in your code:",
        "```bash",
        "python -W error::DeprecationWarning -c 'import mfgarchon; ...'",
        "```",
        "",
        "---",
        "",
    ]

    for version, version_items in groups.items():
        # Sub-group by type
        by_type: dict[str, list[dict]] = defaultdict(list)
        for item in version_items:
            by_type[item.get("type", "unknown")].append(item)

        lines.append(f"## Deprecated since {version}")
        lines.append("")
        lines.append(f"*{len(version_items)} items*")
        lines.append("")

        type_order = ["parameter", "function", "property", "alias"]
        type_labels = {
            "parameter": "Parameters",
            "function": "Functions / Classes",
            "property": "Properties",
            "alias": "Import Aliases",
        }

        for t in type_order:
            if t in by_type:
                type_items = sorted(by_type[t], key=lambda x: x.get("name", ""))
                lines.append(f"### {type_labels.get(t, t.title())}")
                lines.append("")
                for item in type_items:
                    lines.append(format_item(item))
                lines.append("")

        # Any remaining types
        for t, type_items in by_type.items():
            if t not in type_order:
                type_items = sorted(type_items, key=lambda x: x.get("name", ""))
                lines.append(f"### {t.title()}")
                lines.append("")
                for item in type_items:
                    lines.append(format_item(item))
                lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## Migration Help")
    lines.append("")
    lines.append("If you encounter a deprecation warning not listed here,")
    lines.append("please file an issue at https://github.com/derrring/MFGArchon/issues")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate deprecation guide")
    parser.add_argument("--check", action="store_true", help="Check if guide is up-to-date")
    parser.add_argument(
        "--output",
        default="docs/user/DEPRECATION_MODERNIZATION_GUIDE.md",
        help="Output file path",
    )
    args = parser.parse_args()

    items = scan_all_deprecations()
    guide = generate_guide(items)

    output_path = Path(args.output)

    if args.check:
        if not output_path.exists():
            print(f"FAIL: {output_path} does not exist")
            sys.exit(1)
        existing = output_path.read_text()
        if existing.strip() == guide.strip():
            print(f"OK: {output_path} is up-to-date ({len(deduplicate(items))} items)")
            sys.exit(0)
        else:
            print(f"FAIL: {output_path} is out-of-date. Run: python scripts/generate_deprecation_guide.py")
            sys.exit(1)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(guide)
        print(f"Generated {output_path} ({len(deduplicate(items))} items, {len(guide)} chars)")


if __name__ == "__main__":
    # main() returns an exit code; discarding it made a refusal to regenerate look like
    # a success to every caller, including the pre-push gate.
    sys.exit(main() or 0)

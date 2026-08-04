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

    This walks the package by IMPORTING it, so what it can see depends on which optional extras
    are installed, and a guide generated from a partial walk teaches a partial API as if it were
    the whole one. `scan_deprecated` now refuses rather than returning the smaller number
    (Issue #1713), so this raises here instead of writing a wrong document.

    Unlike the ratchet, this is NOT scoped to the live library: the guide is user-facing, and a
    user who installs `[nn]` meets the frozen paradigms' deprecations too. So generating it needs
    a complete environment, torch included. See Issue #1774.
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


def _short_name(item: dict) -> str:
    """The bare identifier a user types, without the owning class/method path."""
    return item.get("name", "").split(".")[-1]


def find_name_collisions(items: list[dict]) -> dict[str, dict[str, list[str]]]:
    """Identifiers that are deprecated in one place and the recommended replacement in another.

    A guide that lists both without saying so tells a reader to migrate TO a name on one line and
    AWAY from it on the next, with nothing marking the two as different quantities. The reader's
    reasonable conclusion -- that the name is being phased out everywhere -- is the wrong one, and
    the API accepts the wrong migration without a warning because both parameters exist.

    Measured instance, the one that motivated this (Issue #1043 / #1044): `drift_field` is the
    replacement for `velocity_field` on `FPFDMSolver`, where it means the optimal control a*, and
    is simultaneously deprecated in favour of `potential_field` on seven solvers, where it means
    the value function U. A correct FDM caller who follows the seven rows migrates
    `drift_field=alpha` to `potential_field=alpha`; the solver then computes -c*grad(alpha), which
    for a constant control is zero, so the advection vanishes silently. On a 21-point 1D problem
    with alpha = 1.0 the density centroid moves 0.3 -> 0.5055 correctly and 0.3 -> 0.3151 after
    the migration: a 37.7% error, no exception, no warning.

    Returns ``{name: {"deprecated_in": [...], "replaces": [...]}}``, each entry carrying the owning
    ``method`` and the ``other`` name on that side, so a reader can act on one row without
    reconstructing the pairing. Empty when the guide is unambiguous, which is the state to aim for
    -- a collision is a legitimate transient, not a permanent design, and this surfaces it rather
    than deciding it.
    """
    collisions: dict[str, dict[str, list[dict]]] = {}
    replaced_names = {i.get("replacement", "") for i in items}
    for name in {_short_name(i) for i in items if _short_name(i)}:
        if name not in replaced_names:
            continue
        collisions[name] = {
            # Where this name is itself on the way out, and what it points at.
            "deprecated_in": sorted(
                (
                    {"method": i.get("name", "").rsplit(".", 1)[0], "other": i.get("replacement", "")}
                    for i in items
                    if _short_name(i) == name
                ),
                key=lambda d: d["method"],
            ),
            # Where this name is the destination, and which name it replaces there.
            "replaces": sorted(
                (
                    {"method": i.get("name", "").rsplit(".", 1)[0], "other": _short_name(i)}
                    for i in items
                    if i.get("replacement", "") == name
                ),
                key=lambda d: d["method"],
            ),
        }
    return collisions


def format_collisions(collisions: dict[str, dict[str, list[dict]]]) -> list[str]:
    """The warning section. Placed before the version listings, not after."""
    if not collisions:
        return []
    lines = [
        "## Do not migrate these across solvers",
        "",
        "The identifiers below are **deprecated in one place and the recommended replacement in "
        "another**. That is not a mistake in this guide: the same word names different quantities "
        "on different solvers, and each row is correct for the API it names.",
        "",
        "It does mean a migration you read on one row **does not transfer** to another solver. "
        "Both parameters usually exist on both solvers, so applying the wrong one is accepted "
        "silently and changes the answer rather than raising. Check the target solver's `solve_*` "
        "docstring for what the parameter means there before renaming anything.",
        "",
    ]
    for name, sides in sorted(collisions.items()):
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("| in this API | `" + name + "` is | migration on that row |")
        lines.append("|---|---|---|")
        for row in sides["replaces"]:
            lines.append(f"| `{row['method']}()` | the destination | `{row['other']}` -> `{name}` |")
        for row in sides["deprecated_in"]:
            lines.append(f"| `{row['method']}()` | itself deprecated | `{name}` -> `{row['other']}` |")
        lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def format_item(item: dict, collisions: dict[str, dict] | None = None) -> str:
    """Format a single deprecation item as markdown."""
    name = item.get("name", "unknown")
    replacement = item.get("replacement", "N/A")
    removal = item.get("removal", "v1.0.0")
    item_type = item.get("type", "unknown")
    # A reader scanning one row must see the ambiguity without reading the whole guide.
    flag = ""
    if collisions:
        hit = {_short_name(item), replacement} & set(collisions)
        if hit:
            flag = " [see *Do not migrate these across solvers*: "
            flag += ", ".join(f"`{h}`" for h in sorted(hit)) + "]"

    if item_type == "parameter":
        # "ClassName.method.param_name" -> extract parts
        parts = name.split(".")
        if len(parts) >= 2:
            func_name = ".".join(parts[:-1])
            param = parts[-1]
            return f"- **`{param}`** in `{func_name}()` — use `{replacement}` instead (remove by {removal}){flag}"
        return f"- **`{name}`** — use `{replacement}` instead (remove by {removal}){flag}"
    elif item_type == "function":
        return f"- **`{name}()`** — use `{replacement}` instead (remove by {removal}){flag}"
    elif item_type == "property":
        return f"- **`{name}`** (property) — use `{replacement}` instead (remove by {removal}){flag}"
    elif item_type == "alias":
        return f"- **`{name}`** (import alias) — use `{replacement}` instead (remove by {removal}){flag}"
    else:
        return f"- **`{name}`** ({item_type}) — use `{replacement}` instead (remove by {removal}){flag}"


def generate_guide(items: list[dict]) -> str:
    """Generate the full markdown guide."""
    items = deduplicate(items)
    groups = group_by_version(items)
    collisions = find_name_collisions(items)

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

    lines.extend(format_collisions(collisions))

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
                    lines.append(format_item(item, collisions))
                lines.append("")

        # Any remaining types
        for t, type_items in by_type.items():
            if t not in type_order:
                type_items = sorted(type_items, key=lambda x: x.get("name", ""))
                lines.append(f"### {t.title()}")
                lines.append("")
                for item in type_items:
                    lines.append(format_item(item, collisions))
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

    from mfgarchon.utils.deprecation import IncompleteScanError

    try:
        items = scan_all_deprecations()
    except IncompleteScanError as exc:
        print(f"FAIL: cannot read the whole package here, so the guide would be wrong: {exc}", file=sys.stderr)
        for module, why in sorted(exc.unimportable.items()):
            print(f"  {module}: {why}", file=sys.stderr)
        return 2

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

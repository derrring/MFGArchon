import argparse
import json
import os
import re
import sys


def check_fail_fast_violations(start_path="."):
    """
    Scans the codebase for violations of 'Fail Fast' principles:
    1. hasattr() usage (should be replaced by explicit interfaces/try-except)
    2. Silent 'pass' in except blocks
    3. Bare 'except:' (catches everything, including SystemExit)
    4. Broad 'except Exception:' (hides bugs)
    """

    # Patterns
    hasattr_pattern = re.compile(r"hasattr\s*\(")
    silent_pass_pattern = re.compile(r"except\s*(?:[a-zA-Z0-9_,\\s().]+)?:\s*(?:\n\s*)?pass\b", re.MULTILINE)
    bare_except_pattern = re.compile(r"except\s*:")
    broad_except_pattern = re.compile(r"except\s+Exception\s*:")

    # Counters and storage
    issues = {"hasattr": [], "silent_pass": [], "bare_except": [], "broad_except": []}

    # Walk directories
    for root, dirs, files in os.walk(start_path):
        # Ignore hidden dirs and venv
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "venv" and d != "__pycache__"]

        for file in files:
            if not file.endswith(".py"):
                continue

            # Skip this script
            if file == os.path.basename(__file__):
                continue

            path = os.path.join(root, file)
            try:
                with open(path, encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue  # Skip files we can't read

            # Check hasattr (line by line for context)
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if hasattr_pattern.search(line):
                    issues["hasattr"].append(f"{path}:{i + 1}: {line.strip()}")

            # Check regexes against full content
            for match in silent_pass_pattern.finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                issues["silent_pass"].append(f"{path}:{line_num}: Silent 'pass' in except block")

            for match in bare_except_pattern.finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                issues["bare_except"].append(f"{path}:{line_num}: Bare 'except:'")

            for match in broad_except_pattern.finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                issues["broad_except"].append(f"{path}:{line_num}: Broad 'except Exception:'")

    return issues


def print_section(title, items, limit=None):
    if not items:
        return

    print(f"\n{'=' * len(title)}")
    print(title)
    print(f"{'=' * len(title)}")
    print(f"Total count: {len(items)}")

    display_items = items[:limit] if limit else items
    for item in display_items:
        print(item)

    if limit and len(items) > limit:
        print(f"... and {len(items) - limit} more.")


def _counts(results: dict) -> dict:
    """Per-category violation counts (the ratchet's comparison surface)."""
    return {category: len(items) for category, items in results.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for 'Fail Fast' principle violations.")
    parser.add_argument("--path", default=".", help="Root directory to scan")
    parser.add_argument("--limit", type=int, default=20, help="Limit lines printed per category")
    parser.add_argument("--all", action="store_true", help="Show all violations (no limit)")
    parser.add_argument("--write-baseline", metavar="FILE", help="Write current per-category counts to FILE and exit")
    parser.add_argument(
        "--check-baseline",
        metavar="FILE",
        help=(
            "Ratchet mode (CI guard): compare current counts to FILE and exit 1 ONLY if any "
            "category increased (a new fail-fast violation was introduced). Counts may ratchet "
            "down freely; regenerate the baseline with --write-baseline after fixing violations."
        ),
    )

    args = parser.parse_args()
    results = check_fail_fast_violations(args.path)
    counts = _counts(results)

    # --- Ratchet modes (the durable CI guard) ---
    if args.write_baseline:
        with open(args.write_baseline, "w") as fh:
            json.dump(counts, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"Wrote fail-fast baseline to {args.write_baseline}: {counts}")
        sys.exit(0)

    if args.check_baseline:
        with open(args.check_baseline) as fh:
            baseline = json.load(fh)
        categories = sorted(set(counts) | set(baseline))
        regressed = [
            (c, counts.get(c, 0), baseline.get(c, 0)) for c in categories if counts.get(c, 0) > baseline.get(c, 0)
        ]
        improved = [
            (c, counts.get(c, 0), baseline.get(c, 0)) for c in categories if counts.get(c, 0) < baseline.get(c, 0)
        ]
        if improved:
            print("Fail-fast violations DECREASED — please tighten the baseline (run --write-baseline):")
            for c, cur, base in improved:
                print(f"  {c}: {base} -> {cur} ({cur - base})")
        if regressed:
            print("FAIL: new fail-fast violations introduced (no new broad/bare except, silent pass, or hasattr):")
            for c, cur, base in regressed:
                print(f"  {c}: {base} -> {cur} (+{cur - base})")
            print(
                "If a decrease is expected, regenerate the baseline: python scripts/check_fail_fast.py "
                "--path mfgarchon --write-baseline scripts/fail_fast_baseline.json"
            )
            sys.exit(1)
        print(f"OK: no new fail-fast violations vs baseline (counts: {counts})")
        sys.exit(0)

    # --- Human report mode ---
    print(f"Scanning '{args.path}' for Fail Fast violations...")
    limit = None if args.all else args.limit
    print_section("SILENT FALLBACKS (Critical)", results["silent_pass"], limit)
    print_section("BARE EXCEPTS (Critical)", results["bare_except"], limit)
    print_section("BROAD EXCEPTIONS (Warning)", results["broad_except"], limit)
    print_section("HASATTR USAGE (Forbidden)", results["hasattr"], limit)

    total_issues = sum(len(v) for v in results.values())
    print(f"\nTotal Violations Found: {total_issues}")

    if total_issues > 0:
        sys.exit(1)

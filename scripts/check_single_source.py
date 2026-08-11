#!/usr/bin/env python3
"""Ratchet on restated quantities: how many sites compute something that has one owner.

The axiom's single-source rule triggers on "the same quantity, convention, or dispatch
decision restated in >= 2 places", and CLAUDE.md names three owners by hand
(`diffusion_from_volatility`, `fp_drift_coefficient`, and since #1894 `hjb_residual_norm`).
Nothing measured whether the restatements were growing. This does: each registry entry
records how many sites currently restate a quantity, and the check fails in BOTH
directions -- growth is a regression, shrink is progress that must be written down.

Why every entry carries sentinels
---------------------------------
A broken search pattern returns 0 hits, and 0 hits reads exactly like clean code. That
is not hypothetical: `% *Nx\\b` and `np\\.roll\\b` both returned 0 from `git grep -E` on
this machine on 2026-08-11, because that grep does not implement `\\b` -- the true counts
were 18 and 18. A ratchet that can silently read "clean" is worse than no ratchet, so
each entry declares two sentinels covering the two ways the instrument breaks:

* `sentinel_text` -- a literal string the pattern MUST match. Catches a pattern that
  compiles but means nothing (the `\\b` case, a typo'd escape, a dialect assumption).
* `sentinel_file` -- a path that MUST be among the scanned files. Catches include/exclude
  globs that select nothing after a directory move.

Either failing exits 2 (instrument broken), never 0 and never 1.

Comments and strings are stripped before matching, via `tokenize`. `check_fail_fast.py`
records the same trap from the other side: its regex era counted `hasattr` mentions
inside docstrings as calls, 40 of 164. Here the noise is heavier -- 6 of the 18 `% Nx`
hits are prose about the periodic fallback, not the fallback.
"""

import argparse
import fnmatch
import io
import json
import re
import sys
import tokenize
from pathlib import Path

DEFAULT_BASELINE = Path(__file__).with_name("single_source_baseline.json")

# Instrument failures. Kept distinct from a count change so a broken pattern can never be
# mistaken for a clean tree by a caller reading the exit code.
EXIT_OK = 0
EXIT_COUNT_CHANGED = 1
EXIT_INSTRUMENT_BROKEN = 2


class InstrumentError(RuntimeError):
    """The check could not measure -- the pattern or the file selection is broken."""


def code_only_lines(path: Path) -> list[str]:
    """File contents with comments and string literals blanked, one entry per line.

    Line numbering is preserved so reported sites are navigable. A file that does not
    tokenize (syntax error, exotic encoding) raises `InstrumentError` rather than falling
    back to raw text: a silent fallback here would count docstring prose as code on
    exactly the files most likely to be malformed, and letting the raw `TokenError` escape
    would exit 1 -- which the gate reads as "the count changed", a verdict about the tree.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InstrumentError(f"cannot read {path}: {exc}") from exc
    lines = source.splitlines()
    blanked = list(lines)
    readline = io.StringIO(source).readline
    try:
        tokens = list(tokenize.generate_tokens(readline))
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        raise InstrumentError(f"cannot tokenize {path}: {exc}") from exc
    for tok in tokens:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (srow, scol), (erow, ecol) = tok.start, tok.end
        for row in range(srow, erow + 1):
            idx = row - 1
            if idx >= len(blanked):
                continue
            line = blanked[idx]
            start = scol if row == srow else 0
            end = ecol if row == erow else len(line)
            blanked[idx] = line[:start] + " " * (end - start) + line[end:]
    return blanked


def scan_files(root: Path, include: list[str], exclude: list[str]) -> list[Path]:
    """Files selected by the include globs and not knocked out by the exclude globs."""
    selected: dict[str, Path] = {}
    for pattern in include:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, ex) for ex in exclude):
                continue
            selected[rel] = path
    return [selected[rel] for rel in sorted(selected)]


def measure(entry: dict, root: Path) -> tuple[int, list[str]]:
    """Site count and site list for one registry entry, after both sentinel checks."""
    name = entry["name"]
    try:
        regex = re.compile(entry["pattern"])
    except re.error as exc:
        raise InstrumentError(f"{name}: pattern does not compile: {exc}") from exc

    sentinel_text = entry["sentinel_text"]
    if not regex.search(sentinel_text):
        raise InstrumentError(
            f"{name}: pattern {entry['pattern']!r} does not match its own sentinel text "
            f"{sentinel_text!r}. The pattern is broken, not the tree -- a zero count here "
            f"would read as clean code."
        )

    files = scan_files(root, entry["include"], entry.get("exclude", []))
    if not files:
        raise InstrumentError(f"{name}: include globs {entry['include']} selected no files under {root}")

    rels = {p.relative_to(root).as_posix() for p in files}
    sentinel_file = entry["sentinel_file"]
    if sentinel_file not in rels:
        raise InstrumentError(
            f"{name}: sentinel file {sentinel_file} is not among the {len(rels)} scanned files. "
            f"The include/exclude globs no longer reach it."
        )

    sites = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(code_only_lines(path), start=1):
            if regex.search(line):
                sites.append(f"{rel}:{lineno}")
    return len(sites), sites


def load_registry(baseline_path: Path) -> dict:
    with baseline_path.open() as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", default=".", help="Repository root to scan")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="Registry + baseline counts (JSON)")
    parser.add_argument("--list", action="store_true", help="Print every site, then exit 0")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Rewrite the baseline counts in place. Deliberate act: the counts are a claim about the tree.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    baseline_path = Path(args.baseline)
    registry = load_registry(baseline_path)

    try:
        measured = {e["name"]: measure(e, root) for e in registry["entries"]}
    except InstrumentError as exc:
        print(f"INSTRUMENT BROKEN: {exc}")
        print("Not a verdict about the code. Fix the pattern or the globs, then re-run.")
        return EXIT_INSTRUMENT_BROKEN

    if args.list:
        for entry in registry["entries"]:
            count, sites = measured[entry["name"]]
            print(f"\n{entry['name']} -- {entry['quantity']}")
            print(f"  owner: {entry['owner']}")
            print(f"  {count} site(s):")
            for site in sites:
                print(f"    {site}")
        return EXIT_OK

    if args.write_baseline:
        for entry in registry["entries"]:
            entry["count"] = measured[entry["name"]][0]
        with baseline_path.open("w") as fh:
            json.dump(registry, fh, indent=2)
            fh.write("\n")
        print(f"Wrote {baseline_path}: " + ", ".join(f"{e['name']}={e['count']}" for e in registry["entries"]))
        return EXIT_OK

    changed = []
    for entry in registry["entries"]:
        count, sites = measured[entry["name"]]
        if count != entry["count"]:
            changed.append((entry, count, sites))

    if changed:
        print("FAIL: the number of sites restating a single-owner quantity changed.")
        for entry, count, sites in changed:
            direction = "GREW" if count > entry["count"] else "SHRANK"
            print(f"\n  {entry['name']}: {entry['count']} -> {count}  ({direction})")
            print(f"    quantity: {entry['quantity']}")
            print(f"    owner:    {entry['owner']}")
            for site in sites:
                print(f"      {site}")
        print(
            "\nGrowth is a regression: route the new site through the owner instead.\n"
            "Shrink is progress and must be recorded: python scripts/check_single_source.py --write-baseline"
        )
        return EXIT_COUNT_CHANGED

    counts = ", ".join(f"{e['name']}={e['count']}" for e in registry["entries"])
    print(f"OK: single-source site counts unchanged ({counts})")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

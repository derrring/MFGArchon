#!/usr/bin/env python3
"""Ratchet on restated quantities: how many sites compute something that has one owner.

The axiom's single-source rule triggers on "the same quantity, convention, or dispatch
decision restated in >= 2 places", and CLAUDE.md names three owners by hand
(`diffusion_from_volatility`, `fp_drift_coefficient`, and since #1894 `hjb_residual_norm`).
Nothing measured whether the restatements were growing. This does: each registry entry
records how many sites currently restate a quantity, and the check fails in BOTH
directions -- growth is a regression, shrink is progress that must be written down.

Why every entry carries a sentinel, and why it is a live site
-------------------------------------------------------------
A broken search pattern returns 0 hits, and 0 hits reads exactly like clean code. That
is not hypothetical: `% *Nx\\b` and `np\\.roll\\b` both returned 0 from `git grep -E` on
this machine on 2026-08-11, because that grep does not implement `\\b` -- the true counts
were 18 and 18.

So each entry names a `sentinel_file` that MUST be scanned and in which the pattern MUST
match at least once. Choose the quantity's OWNER file: it contains the quantity by
definition, so the sentinel stays live for as long as the entry is meaningful.

The first version of this checker used a `sentinel_text` literal stored in the JSON
instead, and that is not strong enough -- it proves the pattern intersects one hardcoded
string, never that it still describes the TREE. Measured on 2026-08-12: entry 1's pattern
had four alternatives of which three were structurally unreachable, because `ruff format`
(gated at `local_ci.sh:214`) hugs `**` and no file under `mfgarchon/` can contain
`sigma ** 2`. The literal sentinel matched the one live alternative and stayed green while
12 real sites went uncounted. Worse, respelling the counted sites to the ruff-canonical
`0.5 * sigma**2` -- which removes nothing -- dropped the count 6 -> 0, printed SHRANK, and
the tool instructed the operator to run `--write-baseline`, after which the entry read
clean forever. A live-site sentinel closes that: the owner's own line stops matching too,
so the same drift raises InstrumentError instead of reporting progress.

Either sentinel condition failing exits 2 (instrument broken), never 0 and never 1.

Comments and strings are stripped before matching, via `tokenize`. `check_fail_fast.py`
records the same trap from the other side: its regex era counted `hasattr` mentions
inside docstrings as calls, 40 of 164. Here the noise is heavier -- 6 of the 18 `% Nx`
hits are prose about the periodic fallback, not the fallback. f-string bodies need their
own handling: on Python 3.12+ they tokenize as FSTRING_MIDDLE rather than STRING, so
blanking only `tokenize.STRING` leaves their text visible to the matcher -- which counted
`pde_coefficients.py:330`, a log message, as a site.
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
    # FSTRING_MIDDLE carries the literal text of an f-string on Python 3.12+, where an
    # f-string is no longer one STRING token. Its interpolated `{...}` expressions arrive as
    # ordinary NAME/OP tokens and are real code, so they are correctly left alone.
    blank_types = {tokenize.COMMENT, tokenize.STRING}
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    if fstring_middle is not None:
        blank_types.add(fstring_middle)
    for tok in tokens:
        if tok.type not in blank_types:
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

    if not any(s.startswith(f"{sentinel_file}:") for s in sites):
        raise InstrumentError(
            f"{name}: pattern {entry['pattern']!r} matches nothing in its sentinel file "
            f"{sentinel_file}, which owns this quantity and must contain it. The pattern no "
            f"longer describes the tree -- the {len(sites)} site(s) it did find are not a "
            f"verdict about the code. Re-read the pattern against the tree's actual spelling "
            f"(ruff normalises `x ** 2` to `x**2`, so a pattern written the spaced way is dead)."
        )
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

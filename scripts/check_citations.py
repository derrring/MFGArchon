#!/usr/bin/env python3
"""Measure `path.py:NNN` citations in durable prose that no longer point at what they name.

A line number in a document is a claim with an expiry date, and nothing marks it when it stops
being true. Measured at the time of writing: 44% of the adjudicable citations in this repository's
live prose named a symbol that was not near the cited line (Issue #2102).

WHY THE SYMBOL IS THE DISCRIMINATOR
-----------------------------------
Checking "is the line number inside the file" is almost useless. Of 208 citations exactly ONE
pointed past EOF, and two of three "in range" citations sampled at random landed on blank lines. A
drifted citation almost always still points somewhere.

So a citation is adjudicable exactly when the prose around it names a symbol in backticks. Then the
question has an answer: is that symbol within +/-WINDOW lines of the cited line? A citation with no
symbol nearby is **recorded as unadjudicable, never counted as passing** -- an unjudged row reads
exactly like a clean one, which is the failure this repository files under #1918.

WHAT THIS SCRIPT IS NOT
-----------------------
It is a measurement, not a ratchet. There is no baseline and it exits 0 whatever it finds, unless
the instrument itself cannot report (exit 2). The ratchet is deliberately a separate change: an
instrument that will gate merges should first be read, and its own error modes seen, on numbers
nobody has to act on.

KNOWN FAILURE MODE OF THE RESOLVER, PAID FOR ONCE
-------------------------------------------------
The first version of this measurement reported "184 citations point at a file that does not exist".
Every one was an artefact: citations are usually written as a BARE BASENAME (`mfg_problem.py:2534`)
while the file lives at `mfgarchon/core/mfg_problem.py`, and resolving the string as a path misses
it. That produced a clean, plausible, wrong answer. Resolution now tries, in order: the string as a
repo-relative path, as a path under `mfgarchon/`, and finally as a basename over every tracked
`.py`, narrowed by suffix when the citation carries directories. A basename matching more than one
file is reported AMBIGUOUS rather than guessed.

`CHANGELOG.md` is exempt, on the same reasoning `scripts/check_doc_api.py:82` uses: an entry
describing a v0.16 line is correct as of then. `archive/` likewise.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

EXEMPT_DIRS = {"archive", ".git", "node_modules", ".venv", "build", "dist"}
EXEMPT_FILES = {Path("CHANGELOG.md")}

#: `path.py:123` or `pkg/path.py:123-145`. The trailing range is captured and ignored: the start
#: line is what the prose points a reader at.
CITE = re.compile(r"\b([A-Za-z0-9_][A-Za-z0-9_./-]*\.py):(\d+)(?:-\d+)?\b")

#: A backticked identifier. Dotted names are kept whole and compared by their last component, so
#: `FDMApplicator.apply` matches a definition of `apply`.
IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")

#: How far from the cited line the named symbol may sit. Wide enough that a citation pointing at a
#: docstring still anchors to the def above it; narrow enough that a drifted one does not.
WINDOW = 12

#: Identifiers too generic to discriminate, and the file's own stem, which appears in every citation.
NOISE = {"self", "cls", "None", "True", "False", "return", "import", "from", "def", "class"}


class InstrumentError(RuntimeError):
    """The measurement could not be made. Distinct from 'the measurement found nothing'."""


def tracked_files(root: Path) -> list[str]:
    """`git ls-files`, not `rglob`: an untracked scratch file is not durable prose, and a deleted
    file that `rm` left behind would otherwise be read as a source of citations."""
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise InstrumentError(f"git ls-files failed in {root}: {out.stderr.strip()}")
    files = [f for f in out.stdout.split("\0") if f]
    if not files:
        raise InstrumentError(f"git ls-files returned nothing in {root} -- not a repository?")
    return files


def _index_by_basename(files: list[str]) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if f.endswith(".py"):
            idx[Path(f).name].append(f)
    return idx


def resolve(rel: str, root: Path, by_base: dict[str, list[str]]) -> tuple[Path | None, str]:
    """Return (path, status) where status is 'ok', 'ambiguous' or 'missing'."""
    for candidate in (root / rel, root / "mfgarchon" / rel):
        if candidate.is_file():
            return candidate, "ok"
    matches = by_base.get(Path(rel).name, [])
    if "/" in rel:
        matches = [m for m in matches if m.endswith(rel)]
    if len(matches) == 1:
        return root / matches[0], "ok"
    if len(matches) > 1:
        return None, "ambiguous"
    return None, "missing"


def _symbols_near(lines: list[str], i: int, cited_stem: str) -> list[str]:
    """Symbols in the citation's own line, plus adjacent lines within the same paragraph.

    The paragraph is the unit, and the walk STOPS AT A BLANK LINE. A flat +/-1 window instead lets a
    citation borrow a symbol from the next sentence, which reports a drifted citation as anchored --
    caught by this file's own self-test, where seven unrelated claims on seven consecutive lines all
    anchored to each other's symbols. Real prose wraps inside a paragraph, so adjacent non-blank
    lines are genuinely the same claim; a blank line is where that stops being true.
    """
    span = [lines[i]]
    if i > 0 and lines[i - 1].strip():
        span.append(lines[i - 1])
    if i + 1 < len(lines) and lines[i + 1].strip():
        span.append(lines[i + 1])
    context = " ".join(span)
    out = []
    for s in IDENT.findall(context):
        if s.endswith(".py") or s in NOISE or s == cited_stem or len(s) <= 3:
            continue
        out.append(s)
    return out


def measure(root: Path) -> dict:
    files = tracked_files(root)
    by_base = _index_by_basename(files)
    prose = [
        f
        for f in files
        if f.endswith((".md", ".py")) and Path(f) not in EXEMPT_FILES and not set(Path(f).parts) & EXEMPT_DIRS
    ]
    result: dict[str, list[dict]] = {
        "anchored": [],
        "drifted": [],
        "unadjudicable": [],
        "ambiguous": [],
        "missing": [],
    }
    for f in prose:
        try:
            lines = (root / f).read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            for m in CITE.finditer(line):
                rel, ln = m.group(1), int(m.group(2))
                target, status = resolve(rel, root, by_base)
                row = {"file": f, "line": i + 1, "cites": rel, "cited_line": ln}
                if status != "ok":
                    result[status].append(row)
                    continue
                symbols = _symbols_near(lines, i, Path(rel).stem)
                if not symbols:
                    result["unadjudicable"].append(row)
                    continue
                row["symbols"] = symbols
                target_lines = (target or Path()).read_text(errors="replace").splitlines()
                if ln > len(target_lines):
                    row["why"] = f"past EOF ({len(target_lines)} lines)"
                    result["drifted"].append(row)
                    continue
                window = "\n".join(target_lines[max(0, ln - 1 - WINDOW) : ln - 1 + WINDOW])
                if any(s.split(".")[-1] in window for s in symbols):
                    result["anchored"].append(row)
                else:
                    row["at_cited_line"] = target_lines[ln - 1].strip()[:70]
                    result["drifted"].append(row)
    return result


def self_test(root: Path) -> int:
    """Build a tree whose every category MUST be reported, and require each to fire.

    Several distinct shapes per category, not one: a narrowing that still fires once on a
    single-shape control passes it while having gone half blind (#1761, quoted in
    `scripts/check_doc_api.py`). The negative control is the two `anchored` shapes -- a checker that
    reports everything as drifted is as useless as one that reports nothing.
    """
    import tempfile

    expected = {
        "drifted": 2,
        "anchored": 2,
        "unadjudicable": 1,
        "missing": 1,
        "ambiguous": 1,
    }
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        subprocess.run(["git", "-C", str(d), "init", "-q"], check=True)
        (d / "pkg").mkdir()
        (d / "pkg" / "target.py").write_text(
            "\n".join(
                ["# line 1"]
                + [f"# filler {n}" for n in range(2, 30)]
                + ["def anchored_symbol():", "    pass"]
                + [f"# filler {n}" for n in range(32, 90)]
                + ["def far_away_symbol():", "    pass"]
            )
        )
        # A second file with the same basename, so a bare-basename citation is ambiguous.
        (d / "other").mkdir()
        (d / "other" / "dup.py").write_text("x = 1\n")
        (d / "pkg" / "dup.py").write_text("y = 2\n")
        # Blank lines between claims: each is its own paragraph, which is what the walk above
        # treats as the unit. Written this way deliberately -- the first version had them on
        # consecutive lines and every claim anchored to its neighbour's symbol.
        (d / "doc.md").write_text(
            "\n\n".join(
                [
                    # anchored, symbol on the cited line
                    "`anchored_symbol` is defined at pkg/target.py:30.",
                    # anchored, symbol a few lines off but inside the window
                    "See `anchored_symbol` -- pkg/target.py:36 is inside it.",
                    # drifted, symbol is 60 lines away
                    "`far_away_symbol` lives at pkg/target.py:30, it says.",
                    # drifted, past EOF
                    "`anchored_symbol` at pkg/target.py:9999.",
                    # unadjudicable: a citation with no backticked symbol beside it
                    "Something happens at pkg/target.py:30 for reasons.",
                    # missing: no such file
                    "`anchored_symbol` at pkg/no_such_file.py:12.",
                    # ambiguous: bare basename matching two tracked files
                    "`anchored_symbol` at dup.py:1.",
                ]
            )
        )
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
        got = measure(d)
        counts = {k: len(v) for k, v in got.items()}

    failures = [f"{k}: expected {n}, got {counts.get(k, 0)}" for k, n in expected.items() if counts.get(k, 0) != n]
    if failures:
        print("SELF-TEST FAILED -- the instrument cannot see what it counts:")
        for f in failures:
            print(f"  {f}")
        print(f"  full counts: {counts}")
        return 1
    print(f"self-test OK: every category fires  {counts}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", default=".", help="Repository root to scan")
    parser.add_argument("--list", action="store_true", help="Print every drifted citation")
    parser.add_argument("--json", metavar="FILE", help="Write the full result to FILE")
    parser.add_argument("--self-test", action="store_true", help="Prove the instrument still fires")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.self_test:
        return self_test(root)

    try:
        result = measure(root)
    except InstrumentError as exc:
        print(f"CANNOT MEASURE: {exc}", file=sys.stderr)
        return 2

    counts = {k: len(v) for k, v in result.items()}
    adjudicable = counts["anchored"] + counts["drifted"]
    print(f"citations        : {sum(counts.values())}")
    print(f"  adjudicable    : {adjudicable}  (the prose names a symbol next to the citation)")
    print(f"    anchored     : {counts['anchored']}")
    print(f"    drifted      : {counts['drifted']}", end="")
    if adjudicable:
        print(f"   = {counts['drifted'] / adjudicable * 100:.0f}%")
    else:
        print()
    print(f"  unadjudicable  : {counts['unadjudicable']}  (no symbol named -- RECORDED, not passing)")
    print(f"  ambiguous      : {counts['ambiguous']}  (basename matches several tracked files)")
    print(f"  missing        : {counts['missing']}  (no such file)")

    if args.list:
        print("\ndrifted:")
        for row in result["drifted"]:
            print(f"  {row['file']}:{row['line']}")
            print(f"      cites {row['cites']}:{row['cited_line']} for {row.get('symbols')}")
            print(f"      at that line: {row.get('at_cited_line', row.get('why'))!r}")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

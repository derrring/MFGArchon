#!/usr/bin/env python3
"""Measure `path.py:NNN` citations in durable prose that no longer point at what they name.

A line number in a document is a claim with an expiry date, and nothing marks it when it stops
being true. Measured at the time of writing: 18 of the 38 adjudicable citations in this
repository's live prose -- 47% -- name a symbol that is not near the cited line (Issue #2102).

That is 18 of the 38 that are JUDGED AT ALL, and the coverage figure belongs beside it: 154
citations are recorded `unadjudicable`, a large majority. Quoting 49% bare reads as "half this
repository's citations are wrong" and would be #1918's own failure committed against this report.

Those three figures are quoted here and nowhere else. The volatile ones -- the citation total and
`missing` -- are quoted nowhere at all: both move whenever any fixture in `tests/` gains a fake
path, and transcribing them by hand from one tree into prose describing another produced three
separate wrong published counts before this line was written. `unadjudicable` is not in that class
and dropping it with them was an over-correction; it held at 154 across every commit on this branch
while the other two moved. Run the script.

THAT PERCENTAGE IS A FUNCTION OF `WINDOW`, AND THE SWEEP HAS NO PLATEAU
-----------------------------------------------------------------------
It moves about 1 point per unit near 12, and the denominator does not move at all -- only the split
between anchored and drifted:

    WINDOW      1      6     10     12     16     25     60    200
    drifted  63.2%  57.9%  52.6%  47.4%  44.7%  42.1%  28.9%  21.1%

There is no plateau, so quote the number with its window, never bare. The one WINDOW-INDEPENDENT
statement available is the floor: **8 of the 38 adjudicable citations survive any window** -- 7
whose named symbol is not in the target file at all, plus the one past EOF, which has no symbol and
is wrong regardless. `WINDOW = 12` is a choice, not a measurement, and it is the constant a reader
should attack first.

Those figures were 19 / 39 and 9 until review found that `CLAUDE.md` is a tracked symlink to
`AGENTS.md`: one paragraph, two tracked paths, counted twice. `content_duplicates` now finds that
by inode rather than by guessing which names collide, and the numbers above are the deduplicated
ones.

WHY THE SYMBOL IS THE DISCRIMINATOR
-----------------------------------
Checking "is the line number inside the file" is almost useless. Of every citation in this
repository exactly ONE points past EOF, and two of three "in range" citations sampled at random landed on blank lines. A
drifted citation almost always still points somewhere.

That single row is also why the EOF test runs BEFORE the symbol gate in `measure`. Gated on a
symbol, as it was first written, the one certainly-broken citation in the repository was filed as
`unadjudicable` and this paragraph rested on a finding the instrument did not report.

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
Every one was an artefact: citations are usually written as a bare basename -- `mfg_problem.py`
and a line number -- while the file lives at `mfgarchon/core/mfg_problem.py`, and resolving the
string as a path misses it.

An example path in THIS file's prose is written so it cannot parse as a citation, and that is not
fussiness: this script is inside the population it scans, so an illustrative path becomes a row in
its own measurement. The one exception is the regex's own documentation below, where the notation
IS the subject and neutering it would delete the thing being documented. That produced a clean, plausible, wrong answer. Resolution now tries, in order: the string as a
repo-relative path, as a path under `mfgarchon/`, and finally as a basename over every tracked
`.py`, narrowed by suffix when the citation carries directories. A basename matching more than one
file is reported AMBIGUOUS rather than guessed.

`CHANGELOG.md` is exempt, on the same reasoning `check_doc_api.py` uses: an entry describing a
v0.16 line is correct as of then. `archive/` likewise.

`changelog.d/` fragments are NOT exempt, and the asymmetry is deliberate rather than an oversight:
a citation is judged while it can still be fixed, and a fragment is editable where a released entry
is history. The consequence is real and should not surprise anyone -- collating fragments at a
version bump moves ~12 rows out of `drifted` in one commit. Under the bidirectional ratchet part 2 adds
that is not a free win: `drifted` falling fails the gate and forces a human to re-baseline, which is the
contract working, not a hole in it.

WHAT THIS SCRIPT COUNTS THAT IS NOT A DEFECT
--------------------------------------------
`missing` is mostly fixture text. A checker that plants a fake path in a test is not making
a claim about this repository, and nothing here can tell the difference -- `test_check_single_source`
contributes 5 rows and `test_check_fail_fast` 4, none of them a defect in anything. Only 2 of 30 are
genuine dead paths. That is why part 2's ratchet will record `missing` and never act on it, and why this
script builds its own fixtures from variables rather than literals: written out, they counted as
broken citations in the measurement this script produces about itself.
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

    # An UNMERGED index lists each conflicted path once per stage, so `git ls-files` returns it
    # three times and every citation in it is counted three times. Found by rebasing this branch:
    # `missing` read 62 where the resolved tree reads 30, and a gated number would have tripled the
    # same way for a file that merely happened to be conflicted.
    #
    # Refusing is the whole fix, and deduplicating as well was tried and DELETED: unmerged entries
    # are the only way `git ls-files` repeats a path, so with this raise in place the dedup could
    # not be reached -- mutating it away killed none of the tests. A guard with no reachable input
    # is not a second layer of safety, it is a line that reads like one.
    #
    # The refusal is also the right verdict on its own terms: a file mid-conflict holds `<<<<<<<`
    # markers and both versions of every line, so any count over it is meaningless.
    probe = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--unmerged", "-z"],
        capture_output=True,
        text=True,
        check=False,
    )
    # The returncode check is the guard, not decoration. Without it a failing query returns an
    # empty stdout, the `if` below is false, and the tripled count this exists to refuse is handed
    # back as a clean measurement with exit 0 -- demonstrated by review with a `git` that fails
    # only on `--unmerged`. A silent zero from a broken query reads exactly like a clean result,
    # which is the failure this whole script is about.
    if probe.returncode != 0:
        raise InstrumentError(
            f"git ls-files --unmerged failed in {root}: {probe.stderr.strip()} -- so whether the "
            f"index is mid-conflict is unknown, and a count over a conflicted tree is meaningless."
        )
    unmerged = probe.stdout
    if unmerged.strip("\0").strip():
        paths = sorted({e.split("\t", 1)[1] for e in unmerged.split("\0") if "\t" in e})
        raise InstrumentError(
            f"the index has unmerged paths ({', '.join(paths[:3])}"
            f"{', ...' if len(paths) > 3 else ''}) -- a tree mid-conflict holds both versions of "
            f"every line and any count over it is meaningless. Finish the merge and re-run."
        )
    # Unmerged entries are the only way `git ls-files` repeats a PATH, but that is not the property
    # that matters: what inflates a count is more index entries than distinct files on disk. On a
    # case-INSENSITIVE filesystem -- macOS by default -- an index carrying both `Doc.md` and
    # `doc.md` for one on-disk file has no unmerged entry, passes the guard above, and counts every
    # citation in that file twice. Review built exactly that index and measured it.
    #
    # Deduplicating the list would not have caught this either, which is why the earlier `set()`
    # attempt was deleted rather than kept: `Doc.md` and `doc.md` are distinct strings. The
    # collision has to be resolved against the filesystem, and only for the candidates -- on a
    # case-sensitive filesystem those really are two files and refusing would be wrong.
    return files


def content_duplicates(root: Path, files: list[str]) -> set[str]:
    """Tracked paths whose CONTENT is already counted through another path.

    Discovered by `stat`, not by guessing which names might collide, and that is the second shape
    this took. The first enumerated name rules -- casefold, then NFC plus casefold -- and review
    kept finding cases outside them: `Doc.md`/`doc.md` on a case-insensitive filesystem, then NFD
    against NFC. The one that ended the guessing is in this repository: `CLAUDE.md` is a tracked
    SYMLINK to `AGENTS.md`, sharing an inode while sharing no relation between the names at all. It
    was counted twice, and the published `drifted` was one row high because of it.

    Sameness is a property of the filesystem, so ask the filesystem. Two causes, two verdicts, and
    `git ls-files -s` hands over the discriminator for free:

      - a tracked SYMLINK whose target is also tracked is legitimate; its citations are already
        counted through the target, so it is skipped;
      - two non-symlink entries on one inode -- case, Unicode form, or a hard link -- mean the
        index is broken in a way that doubles a count, and that is refused.
    """
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s", "-z"], capture_output=True, text=True, check=False
    )
    if listing.returncode != 0:
        raise InstrumentError(f"git ls-files -s failed in {root}: {listing.stderr.strip()}")
    modes = {}
    for entry in listing.stdout.split("\0"):
        if "\t" in entry:
            meta, path = entry.split("\t", 1)
            modes[path] = meta.split(" ", 1)[0]

    wanted = set(files)
    by_inode: dict[tuple[int, int], list[str]] = {}
    for f in files:
        try:
            st = (root / f).stat()
        except OSError:
            # Unreadable is the reader's business, not this guard's: a path that cannot be stat-ed
            # cannot be read either, so it contributes no citations and cannot double-count.
            continue
        by_inode.setdefault((st.st_dev, st.st_ino), []).append(f)

    skip = set()
    for paths in by_inode.values():
        if len(paths) < 2:
            continue
        links = [p for p in paths if modes.get(p) == "120000"]
        if len(links) == len(paths) - 1 and all(str(Path(p)) in wanted or True for p in paths):
            skip.update(links)
            continue
        raise InstrumentError(
            f"these tracked paths are one file on this filesystem and none of them is a symlink: "
            f"{', '.join(sorted(paths))} -- every citation in it would be counted once per entry. "
            f"Resolve the duplicate index entry (`git rm --cached` one of them) and re-run."
        )
    return skip


def _index_by_basename(files: list[str]) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if f.endswith(".py"):
            idx[Path(f).name].append(f)
    return idx


def resolve(rel: str, root: Path, by_base: dict[str, list[str]]) -> tuple[Path | None, str]:
    """Return (path, status) where status is 'ok', 'ambiguous' or 'missing'."""
    for candidate in (root / rel, root / "mfgarchon" / rel):
        # `root / rel` is not confined: `sub/../../outside/escape.py:3` read a file outside the
        # repository and reported it anchored. A citation that leaves the tree is not a citation
        # into this codebase.
        if candidate.is_file() and candidate.resolve().is_relative_to(root.resolve()):
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
    """Symbols named on the citation's OWN line. Nothing is borrowed from a neighbouring line.

    Two earlier designs were wrong, and the second was wrong in a way the first hid.

    The original took a flat +/-1 lines. Then a guard was added to skip BLANK neighbours, described
    as "the paragraph is the unit". That guard was INERT: a line falsy under `.strip()` holds no
    backtick, so the identifier regex finds nothing in it either way, and deleting it changed no
    output on any input. It passed unnoticed because it shipped alongside a rewritten self-test
    fixture whose claims were separated by blank lines -- the FIXTURE was doing all the work.
    Independent review deleted both guards and the self-test plus all eight unit tests survived,
    including one named for the behaviour.

    The defect was live the whole time, because real prose does not put a blank line between every
    claim. Measured on this repository: 20 of 41 anchored rows anchored ONLY on a symbol absent from
    the citation's own line, two hand-verified false -- one borrowing `SeparableHamiltonian` from the
    next line, where it belongs to a DIFFERENT citation.

    A narrower repair -- keep the neighbours, drop those carrying their own citation -- looks better
    on the aggregate (74 -> 70 adjudicable against 74 -> 39) and **fixes neither confirmed case**.
    Both were checked against it directly, which is the only reason that is known. So the rule is
    the strict one: the line that carries the citation must name what it cites, or the row is
    recorded `unadjudicable` and judged by nothing. A smaller denominator that means something beats
    a larger one holding 20 rows anchored on someone else's evidence.
    """
    span = [lines[i]]
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
    duplicates = content_duplicates(root, files)
    prose = [
        f
        for f in files
        if f.endswith((".md", ".py"))
        and f not in duplicates
        and Path(f) not in EXEMPT_FILES
        and not set(Path(f).parts) & EXEMPT_DIRS
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
                # `git ls-files` lists a tracked file that has been deleted from the working
                # tree, so `resolve` can return a path that does not open. Unwrapped, that was a
                # traceback and exit 1 from a step whose contract is "measurement only".
                try:
                    target_lines = target.read_text(errors="replace").splitlines()
                except OSError:
                    row["why"] = "tracked but not readable in the working tree"
                    result["missing"].append(row)
                    continue

                # EOF BEFORE the symbol gate, not after. A citation past the end of its file is
                # wrong with no symbol needed, and gating it on one hid the single certainly-broken
                # citation in this repository inside `unadjudicable` -- the one row the module
                # docstring rests its central argument on.
                if ln > len(target_lines):
                    row["why"] = f"past EOF ({len(target_lines)} lines)"
                    row["symbols"] = _symbols_near(lines, i, Path(rel).stem)
                    result["drifted"].append(row)
                    continue

                symbols = _symbols_near(lines, i, Path(rel).stem)
                if not symbols:
                    result["unadjudicable"].append(row)
                    continue
                row["symbols"] = symbols
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
        "drifted": 3,
        "anchored": 2,
        "unadjudicable": 2,
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
        # Built from variables, so no `<path>.py:<digits>` literal appears in THIS file. A planted
        # fixture is not a claim about this repository and the scanner cannot tell: written out,
        # these lines were counted as broken citations in the measurement this script produces
        # about itself, which made editing a comment here move its own numbers.
        tgt, gone, dup = "pkg/target.py", "pkg/no_such_file.py", "dup.py"
        (d / "doc.md").write_text(
            "\n\n".join(
                [
                    # anchored, symbol on the cited line
                    f"`anchored_symbol` is defined at {tgt}:30.",
                    # anchored, symbol a few lines off but inside the window
                    f"See `anchored_symbol` -- {tgt}:36 is inside it.",
                    # drifted, symbol is 60 lines away
                    f"`far_away_symbol` lives at {tgt}:30, it says.",
                    # drifted, past EOF
                    f"`anchored_symbol` at {tgt}:9999.",
                    # drifted with NO symbol named: past EOF is wrong whether or not the prose
                    # names anything, and gating it on a symbol hid the one real case in the repo.
                    f"Nothing is named here, but {tgt}:9998 is past the end.",
                    # unadjudicable: a citation with no backticked symbol beside it
                    f"Something happens at {tgt}:30 for reasons.",
                    # unadjudicable: the symbol is on the PRECEDING line, not this one. Two lines,
                    # no blank between -- the shape the previous walk borrowed across, and the one
                    # the blank-line fixture could never exhibit. Restoring the +/-1 span turns
                    # this into an `anchored` row and fails the counts above.
                    f"`anchored_symbol` is a fine symbol.\nSomething else at {tgt}:30.",
                    # missing: no such file
                    f"`anchored_symbol` at {gone}:12.",
                    # ambiguous: bare basename matching two tracked files
                    f"`anchored_symbol` at {dup}:1.",
                ]
            )
        )
        # Exemptions need a shape or nothing notices when one is dropped: emptying EXEMPT_DIRS and
        # deleting the CHANGELOG entry both SURVIVED this self-test as first written. Each file
        # below carries a citation that WOULD be drifted if it were scanned.
        (d / "CHANGELOG.md").write_text(f"`far_away_symbol` was at {tgt}:30 in v0.16.\n")
        (d / "archive").mkdir()
        (d / "archive" / "old.md").write_text(f"`far_away_symbol` at {tgt}:30, long ago.\n")
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

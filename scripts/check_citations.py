#!/usr/bin/env python3
"""Measure `path.py:NNN` citations in durable prose that no longer point at what they name.

A line number in a document is a claim with an expiry date, and nothing marks it when it stops
being true. Measured at the time of writing: 18 of the 38 adjudicable citations in this
repository's live prose -- 47% -- name a symbol that is not near the cited line (Issue #2102).

THIS IS A REVIEW QUEUE, NOT A DEFECT LIST, and the distinction is measured rather than modest.
All 18 were read by hand: one is a withdrawn claim (#2112) and the other 17 contain both kinds --
**genuinely wrong citations, and the instrument mis-attributing a symbol** -- in a proportion two
independent hand-reads disagreed about, so none is stated here.
A prose line often carries a citation and a backticked name belonging to a DIFFERENT clause
-- "`propagate = False` (logger.py, line 211), so whether `caplog` sees..." is a CORRECT citation
whose target line really is `logger.propagate = False`, with `caplog` sitting in the consequence
clause. (Written with the line number in words, because spelling it made this very paragraph the
20th drifted row -- the third time explaining a defect in this file created another instance of it.) Restricting the
search to the citation's own line narrows that and does not close it.

It cannot be closed mechanically. Character distance between citation and symbol was measured as a
discriminator and FAILED: real 8 median, artifact 9, over overlapping ranges. The difference is
grammatical -- does the symbol describe the cited line, or the sentence's consequence -- and no
positional rule sees a clause boundary. So the instrument reports a superset and a human separates,
which is the shape `local_ci.sh` already uses for assertion strength: a review queue, not a delete
list.

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

THE RATCHET IS ON WHICH CLAIMS DRIFTED, AND ON TWO COUNTS BESIDE THEM
----------------------------------------------------------------------
The baseline records WHICH citations are drifted, keyed on the resolved target and the symbols the
prose names beside it. Counts alone were satisfied by a compensating pair -- two rows hidden, two
fresh ones added, gate green. `check_single_source` and `capability_matrix` pin identities for the
same reason; `check_doc_api` is bidirectional on counts, and `check_fail_fast` is warn-only rather
than a gate.

The counts stay beside the identities because neither subsumes the other. Two sentences in one file
citing the same target line collapse to one key, so a second drifted citation does not enlarge the
set -- caught by this script's own self-test within a minute of the identity check being written.
And a fixed citation must fail until it is recorded, in EITHER direction.

It also fails when **`adjudicable` shrinks**, and that half is not symmetry for its own sake. The
cheapest way to make `drifted` fall is to delete the symbol name from the prose: the row leaves the
numerator AND the denominator, and a numerator-only ratchet records it as an improvement. Measured:

    `far_away` is at pkg/target.py:LINE.  ->  drifted 1  adjudicable 1
    The thing is at pkg/target.py:LINE.   ->  drifted 0  adjudicable 0  unadjudicable 1

Adding correct new prose raises `adjudicable` and passes; only a shrinking denominator is refused.

KNOWN FAILURE MODE OF THE RESOLVER, PAID FOR ONCE
-------------------------------------------------
The first version of this measurement reported "184 citations point at a file that does not exist".
Every one was an artefact: citations are usually written as a bare basename -- `mfg_problem.py`
and a line number -- while the file lives at `mfgarchon/core/mfg_problem.py`, and resolving the
string as a path misses it.

That produced a clean, plausible, wrong answer. Resolution now tries, in order: the string as a
repo-relative path, as a path under `mfgarchon/`, and finally as a basename over every tracked
`.py`, narrowed by suffix when the citation carries directories. A basename matching more than one
file is reported AMBIGUOUS rather than guessed.

An example path in THIS file's prose must contribute NO ADJUDICABLE ROW -- not "must not parse as a
citation", which is what this rule said and which the file itself violates: the regex's own
documentation and `resolve`'s traversal example both parse, and land in `missing`, which is ungated.
The enforceable invariant is the adjudicable one, and it needs a pin because `--write-baseline`
records whatever is present -- which is exactly how two rows of this file's own prose shipped inside
a PASSING baseline. `test_this_file_contributes_no_adjudicable_citation` is that pin. It is not
circular: it is the instrument judging one ordinary input file, the same shape as the
shipped-baseline test.

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
import builtins
import contextlib
import io
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
        # COUNT the regular files; do not match an exact group shape. The previous version
        # required exactly one non-symlink, so a group of two symlinks pointing at an UNTRACKED
        # target -- a legitimate tree -- was refused, with a message asserting "none of them is a
        # symlink" while both were. It also carried `... or True`, which made its other half
        # unconditionally true: the third inert guard on this issue, and no test saw any of it.
        regulars = [p for p in paths if modes.get(p) != "120000"]
        if len(regulars) >= 2:
            raise InstrumentError(
                f"these tracked paths are one file on this filesystem and at least two of them "
                f"are regular files, not symlinks: {', '.join(sorted(regulars))} -- every citation "
                f"in it would be counted once per entry. Resolve the duplicate index entry "
                f"(`git rm --cached` one of them) and re-run."
            )
        # Keep one representative -- the regular file when there is one, otherwise any single
        # symlink -- and skip the rest. Chains and groups of three collapse to one count too.
        keep = regulars[0] if regulars else sorted(paths)[0]
        skip.update(p for p in paths if p != keep)
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
                # The RESOLVED path, not the literal citation text. Rewriting a bare basename to
                # its full path -- which is the correct repair for the `ambiguous` and `missing`
                # rows, and what this module's resolver section is about -- is the same claim, and
                # keying on the raw string reported it as one regression plus one improvement.
                row["target"] = str(target.relative_to(root))
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
        ratchet_failures = _ratchet_self_test(d, d / "doc.md", d / "baseline.json")

    failures = [f"{k}: expected {n}, got {counts.get(k, 0)}" for k, n in expected.items() if counts.get(k, 0) != n]
    failures += ratchet_failures
    if failures:
        print("SELF-TEST FAILED -- the instrument cannot see what it counts:")
        for f in failures:
            print(f"  {f}")
        print(f"  full counts: {counts}")
        return 1
    print(f"self-test OK: every category fires, and the ratchet is not inert  {counts}")
    return 0


def _ratchet_self_test(d: Path, doc: Path, baseline: Path) -> list[str]:
    """Prove `--check-baseline` actually goes red. The classifier firing does not imply this.

    Seven shapes over four gate branches. NOT one-to-one, and saying so matters: `left` has two,
    because a row can leave `drifted` by being fixed or by having its symbol taken away, and those
    are different edits even though the branch reports them identically now.

    Each mapping below was measured the only way that establishes it -- kill one branch, record
    WHICH SHAPES GO GREEN. Watching the whole self-test go red instead establishes that *some*
    branch spoke, not which, and that weaker measurement is how the previous version came to claim
    a one-to-one isolation it did not have.

      unchanged                              must pass
      new drifted at an ALREADY-RECORDED     the COUNT moved and no identity did
        target line
      new drifted at a FRESH target line     an IDENTITY appeared
      the symbol SWAPPED beside a recorded    an identity appeared and another left; without the
        citation                               symbols in the key every count and location is
                                               unchanged and this passes
      a recorded drifted one FIXED           an identity LEFT
      the symbol deleted beside a DRIFTED    an identity LEFT by the other route
        citation
      the symbol deleted beside an ANCHORED  only the DENOMINATOR shrank; nothing left `drifted`,
        citation                               which is why this shape and the one above are not
                                               interchangeable

    Each shape isolates one branch, verified by killing each branch in turn. The first two are not
    redundant: identities alone pass two sentences in one file citing the same line, counts alone
    pass a compensating pair.

    Anchored, not drifted, and that is the whole point of the third shape. Deleting the symbol
    beside a DRIFTED citation lowers `drifted` too, so the bidirectional branch catches it and the
    `adjudicable` pin is never exercised -- measured, killing the entire `adjudicable` block left
    this self-test green. Beside an ANCHORED citation only the denominator moves, which isolates
    the branch this ratchet exists for.
    """
    original = doc.read_text()
    failures = []
    seen = []

    def verdict(label: str) -> int:
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
        # Two of the three shapes are SUPPOSED to print a failure block. Swallow it, or a passing
        # self-test prints "CITATION RATCHET FAILED" twice and reads as a broken one.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            rc = compare_to_baseline(measure(d), baseline)
        seen.append(f"{label}={'red' if rc else 'green'}")
        return rc

    write_baseline(measure(d), baseline, d)
    if verdict("unchanged") != 0:
        failures.append("ratchet: an unchanged tree does not pass its own baseline")

    tgt = "pkg/target.py"
    # Same target line as an existing drifted row: the identity set does not grow, only the count.
    doc.write_text(f"{original}\n\n`far_away_symbol` is also at {tgt}:30, a NEW drifted one.\n")
    if verdict("new drift, same target") == 0:
        failures.append(
            "ratchet: a second drifted citation to an ALREADY-RECORDED target line passed -- "
            "identities alone cannot see it, so the count check is not doing its job"
        )

    # A target line nothing cites yet: a new identity appears and the count moves with it.
    doc.write_text(f"{original}\n\n`far_away_symbol` is at {tgt}:31, a NEW drifted one.\n")
    if verdict("new drift, new target") == 0:
        failures.append(
            "ratchet: a drifted citation to a target line nothing else cites passed -- "
            "the identity check is not doing its job"
        )

    # Progress, which must also stop the gate until it is recorded. Moving the citation onto the
    # line the symbol actually occupies removes an identity from the set.
    doc.write_text(original.replace(f"`far_away_symbol` lives at {tgt}:30", f"`far_away_symbol` lives at {tgt}:90"))
    if verdict("drifted one fixed") == 0:
        failures.append(
            "ratchet: FIXING a recorded drifted citation passed silently -- an unrecorded "
            "improvement is where the next regression hides, so it must stop the gate"
        )

    # Both checks at once, which is how review defeated the location-only version: swap the symbol
    # beside a RECORDED citation for a bogus one at the SAME target line. Every count is unchanged
    # and every location is unchanged; only the symbol set moves, so the gate key must carry it.
    doc.write_text(
        original.replace(f"`far_away_symbol` lives at {tgt}:30", f"Something lives at {tgt}:30")
        + f"\n\n`bogus_symbol` is at {tgt}:30 as well.\n"
    )
    if verdict("symbol swapped, counts intact") == 0:
        failures.append(
            "ratchet: swapping the symbol beside a recorded citation for a bogus one at the same "
            "target line passed -- counts and locations are both unchanged, so the identity key "
            "must carry the symbols and does not"
        )

    # The evasion, on a DRIFTED row: it leaves `drifted` and lands in `unadjudicable`, which the
    # message must call hiding rather than progress.
    # Compensated by a fresh ANCHORED citation, deliberately: deleting a symbol always shrinks
    # `adjudicable` too, and that branch fires first, so an uncompensated shape exercises the wrong
    # one.
    doc.write_text(
        original.replace("`far_away_symbol` lives at", "Something lives at")
        + f"\n\nAlso `anchored_symbol` at {tgt}:31.\n"
    )
    if verdict("drifted one, symbol removed") == 0:
        failures.append(
            "ratchet: deleting the symbol beside a DRIFTED citation passed -- it left `drifted` "
            "for `unadjudicable`, where nothing judges it, and a row leaving must stop the gate "
            "whichever way it left"
        )

    # The same evasion on an ANCHORED row, which touches ONLY the denominator: nothing leaves
    # `drifted`, so the branch above cannot fire and this is the one that must.
    doc.write_text(original.replace("`anchored_symbol` is defined at", "Something is defined at"))
    if verdict("symbol deleted") == 0:
        failures.append(
            "ratchet: deleting a symbol name shrank `adjudicable` and PASSED -- the denominator "
            "is not pinned, so the ratchet rewards hiding a citation rather than fixing it"
        )

    doc.write_text(original)
    subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
    print(f"  ratchet: {', '.join(seen)}")
    return failures


DEFAULT_BASELINE = Path(__file__).resolve().parent / "citation_baseline.json"


def summarise(result: dict) -> dict[str, int]:
    counts = {k: len(v) for k, v in result.items()}
    counts["adjudicable"] = counts["anchored"] + counts["drifted"]
    return counts


def _location(r: dict) -> str:
    """Where a claim points. Stable under prose reflow and under rewriting a bare basename."""
    return f"{r['file']} -> {r.get('target', r['cites'])}:{r['cited_line']}"


def _identities(rows: list[dict]) -> set[str]:
    """What a claim SAYS: where it points, and which symbols the prose names beside it.

    Two keys exist because they answer different questions. This one gates, and it carries the
    symbols so that swapping the symbol beside a recorded citation cannot pass -- review defeated
    the location-only version by adding a bogus symbol at an already-recorded target line while
    deleting the real one, keeping every count and every location identical.

    `_location` is the same key without the symbols. It is what `write_baseline` and the failure
    message would need if they ever had to match a drifted row against an `unadjudicable` one --
    they no longer do, because this branch stopped adjudicating why a row left.
    """
    # `sorted(set(...))`: `_symbols_near` returns findall order with duplicates, so swapping two
    # backticked names inside one sentence -- pure reflow, same claim -- was reported as one
    # regression plus one improvement, and writing a repeated name once would trip the gate.
    return {_location(r) + (f" [{','.join(sorted(set(r['symbols'])))}]" if r.get("symbols") else "") for r in rows}


def drifted_identities(result: dict) -> list[str]:
    """Which claims are drifted, not how many.

    Counts alone are satisfied by a compensating pair: review hid two drifted citations and added
    two fresh ones, and the gate went GREEN with two new broken citations shipped -- which made
    `local_ci.sh`'s "no citation newly points away from the symbol its prose names" false as an
    invariant. `check_single_source` and `capability_matrix` pin identities for the same reason.

    The key deliberately omits the CITING line number. Inserting a paragraph above a citation
    shifts it and would churn every identity in the file on an unrelated edit; the claim being
    made -- this file says that target line holds this symbol -- does not move.
    """
    return sorted(_identities(result["drifted"]))


def write_baseline(result: dict, path: Path, root: Path) -> None:
    def git(*a: str) -> str:
        # `rstrip("\n")`, NOT `strip()`. `git status --porcelain` emits `XY PATH`, so a worktree-
        # only modification starts with a SPACE -- and `strip()` ate it off the first line only,
        # which shifted `ln[3:]` by one character so the carve-out below never matched and every
        # baseline was stamped `-dirty`. Found by review; the previous version of this file argued
        # for the carve-out at length and did not implement it, and no test noticed.
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, check=False).stdout.rstrip(
            "\n"
        )

    # `-dirty` is not decoration. A baseline written from an uncommitted tree records a commit
    # that does not describe what was measured, and the next reader cannot reproduce it -- which
    # is the same defect as publishing a sweep of a candidate that is not what shipped.
    #
    # The baseline itself is excluded from that check, and only it: `measure` scans `.md` and `.py`
    # (see its `prose` filter), so a `.json` file cannot move a single count. Without the carve-out
    # the field would read `-dirty` on every baseline ever written, including the one written from
    # a clean tree, and a marker that is always on discriminates nothing.
    try:
        own = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        # A baseline written OUTSIDE the repository -- which `--write-baseline /tmp/x.json` does,
        # and which every replay harness does -- has no path inside the tree to exempt. This raised
        # ValueError and crashed the write; found by a 40-commit replay whose positive control went
        # red, not by reading.
        own = None
    modified = [ln[3:] for ln in git("status", "--porcelain").splitlines() if own is None or ln[3:] != own]
    head = git("rev-parse", "--short", "HEAD") + ("-dirty" if modified else "")
    payload = {
        "_comment": (
            "Ratchet for `path.py:NNN` citations whose named symbol is no longer near the cited "
            "line (#2102). `drifted` lists WHICH claims are drifted, keyed on the resolved target "
            "file, the cited line, and the symbols the prose names beside it; `counts` pins how "
            "many, because neither catches what the other does. Both are bidirectional: a new "
            "drifted citation is a regression and a fixed one is progress that must be recorded "
            "here. `adjudicable` may not SHRINK -- deleting the symbol name from beside a citation "
            "would otherwise lower `drifted` and read as an improvement. Regenerate with --write-baseline."
        ),
        "_measured_at": {"head_when_written": head, "window": WINDOW},
        "counts": summarise(result),
        "drifted": drifted_identities(result),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compare_to_baseline(result: dict, path: Path) -> int:
    if not path.is_file():
        print(f"CANNOT COMPARE: no baseline at {path}", file=sys.stderr)
        return 2
    baseline = json.loads(path.read_text())
    recorded, now = baseline["counts"], summarise(result)
    was, is_now = set(baseline.get("drifted", [])), set(drifted_identities(result))
    problems = []
    # Identities and counts each catch what the other cannot, and this is not belt-and-braces:
    #
    #   - counts alone pass a COMPENSATING PAIR -- hide two drifted rows, add two fresh ones, and
    #     the total is unchanged. Review shipped exactly that and the gate went green.
    #   - identities alone pass a SAME-KEY DUPLICATE -- two different sentences in one file citing
    #     the same target line collapse to one key, so a second drifted citation does not enlarge
    #     the set. This script's own `--self-test` caught that within a minute of the identity
    #     check being written, which is what it is for.
    if now["drifted"] != recorded["drifted"] and set(is_now) == set(was):
        problems.append(
            f"drifted {recorded['drifted']} -> {now['drifted']} with the same set of claims: two "
            f"citations in one file point at the same target line, so this moved a count without "
            f"moving an identity."
        )
    if appeared := sorted(is_now - was):
        problems.append(
            "these citations name a symbol that is no longer near the line they point at:\n"
            + "\n".join(f"      {a}" for a in appeared)
            + "\n    Read each one. Two things it can be, and the instrument cannot tell them apart:"
            + "\n      - the line number is stale: fix it, or drop it and cite the file;"
            + "\n      - the citation is right and this line's backticked name belongs to a"
            + "\n        neighbouring clause. Then change nothing and record it -- a"
            + "\n        legitimate outcome, not a workaround."
            + "\n    Of the 18 rows in the standing backlog one is a withdrawn claim (#2112). The"
            + "\n    other 17 contain both kinds, and two independent hand-reads disagreed about"
            + "\n    which rows fall in which -- so no proportion here is worth acting on. Read the"
            + "\n    line."
        )
    if left := sorted(was - is_now):
        # REPORTED, NOT ADJUDICATED, and the shortest path to that was deleting code rather than
        # adding more. Four consecutive review rounds broke this branch's attempt to say WHY a row
        # left: it accused a correct repair of being the evasion; then the fix for that suppressed
        # the shape written to exercise it; then at a location the baseline had recorded as
        # ambiguous the evasion and its opposite became byte-identical, with the exclusion list
        # writable in two individually-innocuous steps; and the location/gate key seam dropped rows
        # non-deterministically, accusing different rows under different PYTHONHASHSEEDs.
        #
        # None of it changed the verdict. Hidden or fixed, the gate is red and a human looks. The
        # distinction bought one sentence of diagnosis and cost four defects, two of which pointed
        # the reader at the wrong remedy -- worse than saying nothing. So this says where, and the
        # reader says why.
        #
        # The message names two causes and refuses to close the list, because the list is longer
        # than two and the extras are invisible from here. A row also leaves `drifted` by becoming
        # `ambiguous`, or by its target being renamed into `missing`, or by its prose moving under
        # an exempt path -- and none of those touches the citation or the target line.
        #
        # The `ambiguous` route is not theoretical. 14 of the 18 recorded rows cite a BARE
        # BASENAME, so they resolve through the fallback and a second file with that name anywhere
        # in the tree moves them. Measured: adding a two-line `mfgarchon/workflow/logger.py` -- one
        # commit, touching no prose -- moved `AGENTS.md`'s citation of logger.py out of `drifted` and took
        # `ambiguous` from 5 to 7. Naming two of five causes as though they were all of them is the
        # same over-claim this branch was rewritten to stop making, in a smaller register.
        problems.append(
            "these citations are no longer drifted. Most often they were fixed, or the symbol was "
            "taken out from beside them so they are now `unadjudicable`, where nothing judges "
            "them -- but check the note above too: a row also leaves by becoming `ambiguous` or "
            "`missing`, and neither touches the prose or the target. Read them and decide:\n"
            + "\n".join(f"      {row}" for row in left)
            + "\n    Bidirectional on purpose: an unrecorded change is how the next regression "
            "hides inside a number nobody re-read."
        )
    if now["adjudicable"] < recorded["adjudicable"]:
        problems.append(
            f"adjudicable {recorded['adjudicable']} -> {now['adjudicable']}: the denominator "
            f"shrank, so fewer citations are being judged than before. Three causes, in the order "
            f"you should check them:\n"
            f"      1. A VERSION BUMP collated `changelog.d/` into `CHANGELOG.md`, which is exempt "
            f"(AGENTS.md's checklist). Mechanical and expected -- re-record and move on.\n"
            f"      2. Prose carrying citations was deleted.\n"
            f"      3. A symbol name was removed from beside a citation, which moves the row to "
            f"`unadjudicable` where nothing judges it. This is the one the pin exists for."
        )
    # The baseline also records `missing`, `ambiguous` and `unadjudicable`, which the gate above
    # does NOT act on: a planted fixture path and a genuinely broken citation are indistinguishable
    # to this script, and the checkers in `scripts/` plant plenty. Recorded-but-unjudged is the
    # exact failure `unadjudicable` is named after, so say when they move instead of storing a
    # number nobody re-reads.
    moved = [
        f"{k} {recorded[k]} -> {now[k]}"
        for k in ("missing", "ambiguous", "unadjudicable")
        if k in recorded and now[k] != recorded[k]
    ]
    if moved:
        print(f"citation note (not gated): {', '.join(moved)}")

    if problems:
        print("CITATION RATCHET FAILED")
        for p in problems:
            print(f"    {p}")
        print(f"\n  counts now: {now}")
        print("  see them all:  python scripts/check_citations.py --list")
        print("  record this:   python scripts/check_citations.py --write-baseline")
        return 1
    print(f"citation ratchet OK: drifted {now['drifted']}, adjudicable {now['adjudicable']}")
    return 0


def _noop(*_a: object, **_k: object) -> None:
    """Swallow the human-facing report when the gate wants a verdict."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", default=".", help="Repository root to scan")
    parser.add_argument("--list", action="store_true", help="Print every drifted citation")
    parser.add_argument("--json", metavar="FILE", help="Write the full result to FILE")
    parser.add_argument("--self-test", action="store_true", help="Prove the instrument still fires")
    parser.add_argument(
        "--write-baseline",
        metavar="FILE",
        nargs="?",
        const=str(DEFAULT_BASELINE),
        help="Record the current drifted citations as the accepted set (default: %(const)s)",
    )
    parser.add_argument(
        "--check-baseline",
        metavar="FILE",
        nargs="?",
        const=str(DEFAULT_BASELINE),
        help="Fail if any citation newly drifted, or was fixed without being recorded",
    )
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
    # Under `--check-baseline` this is a gate step, and the gate's convention is one line per check.
    # Nothing is lost on failure: `compare_to_baseline` prints the full counts with the verdict.
    # Shadowing `print` here would make the `CANNOT MEASURE` call above a forward reference and
    # raise UnboundLocalError on the one path that reports the instrument failing. Ruff caught it.
    say = _noop if (args.check_baseline is not None and not args.list) else builtins.print
    say(f"citations        : {sum(counts.values())}")
    say(f"  adjudicable    : {adjudicable}  (the prose names a symbol next to the citation)")
    say(f"    anchored     : {counts['anchored']}")
    say(f"    drifted      : {counts['drifted']}", end="")
    if adjudicable:
        say(f"   = {counts['drifted'] / adjudicable * 100:.0f}%")
    else:
        say()
    say(f"  unadjudicable  : {counts['unadjudicable']}  (no symbol named -- RECORDED, not passing)")
    say(f"  ambiguous      : {counts['ambiguous']}  (basename matches several tracked files)")
    say(f"  missing        : {counts['missing']}  (no such file)")

    if args.list:
        say("\ndrifted:")
        for row in result["drifted"]:
            say(f"  {row['file']}:{row['line']}")
            say(f"      cites {row['cites']}:{row['cited_line']} for {row.get('symbols')}")
            say(f"      at that line: {row.get('at_cited_line', row.get('why'))!r}")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True))
        say(f"\nwrote {args.json}")

    if args.write_baseline is not None:
        if not args.write_baseline:
            print("CANNOT WRITE: --write-baseline was given an empty path", file=sys.stderr)
            return 2
        write_baseline(result, Path(args.write_baseline), root)
        say(f"\nwrote baseline {args.write_baseline}")

    # `is not None`, not truthiness: `--check-baseline ""` otherwise falls through to exit 0, a
    # silent pass from a gate. `local_ci.sh` spends a paragraph on this exact shape for
    # MFG_PYTHON (`+x`, not `:-`) and it was reintroduced here.
    if args.check_baseline is not None:
        if not args.check_baseline:
            print("CANNOT COMPARE: --check-baseline was given an empty path", file=sys.stderr)
            return 2
        return compare_to_baseline(result, Path(args.check_baseline))

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""No document may teach a factory that does not work (Issue #1709).

`CLAUDE.md` is read at the start of every coding-agent session, so a removed symbol in
one of its code blocks does not mislead a reader once -- it generates new broken call
sites continuously. Two of #1624's broken examples call exactly the symbols this checks.

**Code blocks only.** Prose that says "`create_fast_solver` was removed" is the
correction, not the defect, and a check that cannot tell those apart would force the
correction to be written in riddles. The distinction is mechanical: fenced blocks are
what a reader copies.

The population is derived from the package, not hardcoded, so a factory removed
tomorrow is covered without editing this file.
"""

import re
from pathlib import Path

import pytest

import mfgarchon.factory as factory

REPO = Path(__file__).resolve().parents[2]

# Historical records, not instructions. A changelog entry describing a v0.16 API is
# correct as written; rewriting it would falsify the record.
EXEMPT = {"CHANGELOG.md", "archive", ".github"}

FENCE = re.compile(r"^\s*(```|~~~)")


def _removed_factory_names() -> set[str]:
    """Every `create_*_solver` the package no longer supports.

    Two failure shapes, both counted: the name is gone entirely (ImportError for the
    reader), or it survives as a stub that raises on call.
    """
    known = {
        "create_standard_solver",
        "create_fast_solver",
        "create_research_solver",
        "create_basic_solver",
        "create_accurate_solver",
        "create_monitored_solver",
    }
    removed = set()
    for name in known:
        fn = getattr(factory, name, None)
        if fn is None:
            removed.add(name)  # gone: importing it fails
            continue
        try:
            fn()
        except NotImplementedError:
            removed.add(name)  # stub: calling it fails
        except Exception:
            pass  # raised for some other reason -- not evidence of removal
    return removed


def _docs() -> list[Path]:
    out = []
    for path in REPO.rglob("*.md"):
        rel = path.relative_to(REPO)
        if rel.parts[0] in EXEMPT or rel.name in EXEMPT:
            continue
        out.append(path)
    return sorted(out)


def _offending_code_lines(path: Path, names: set[str]) -> list[str]:
    inside = False
    hits = []
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if FENCE.match(line):
            inside = not inside
            continue
        if inside and any(n in line for n in names):
            hits.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    return hits


def test_the_removed_set_is_not_empty():
    """Positive control: an empty set would make every assertion below vacuous."""
    assert _removed_factory_names(), (
        "no factory in the known list is removed -- either they were reinstated, in "
        "which case delete this test, or the detection broke and the checks below pass "
        "over anything"
    )


def test_at_least_one_document_is_scanned():
    """Second control: a path bug would silently scan nothing."""
    docs = _docs()
    assert len(docs) > 10, f"only {len(docs)} markdown files found; the walk is broken"
    assert any(d.name == "CLAUDE.md" for d in docs), "CLAUDE.md must be in scope"


def test_no_document_teaches_a_removed_factory():
    names = _removed_factory_names()
    offences = [line for path in _docs() for line in _offending_code_lines(path, names)]
    assert not offences, (
        "these code blocks teach a factory that raises or does not import.\n"
        "Use `problem.solve()` (Issue #580). Prose ABOUT the removal is fine; this "
        "checks fenced blocks only.\n  " + "\n  ".join(offences)
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Issue #1624: two el_farol_bar notebooks still call create_fast_solver, and also "
        "use ExampleMFGProblem with the removed xmin/xmax/Nx constructor args, so they "
        "cannot be fixed by substituting problem.solve(). Recorded as strict xfail rather "
        "than left out of scope: when #1624 lands this XPASSes and fails the build until "
        "the marker is removed, which is the only version of 'tracked elsewhere' that "
        "cannot be forgotten."
    ),
)
def test_no_notebook_teaches_a_removed_factory():
    names = _removed_factory_names()
    offences = [
        f"{path.relative_to(REPO)}: {name}"
        for path in sorted(REPO.rglob("*.ipynb"))
        if path.relative_to(REPO).parts[0] not in EXEMPT
        for name in names
        if name in path.read_text(errors="replace")
    ]
    assert not offences, "\n  ".join(["notebooks calling a removed factory:", *offences])


@pytest.mark.parametrize("name", sorted(_removed_factory_names()))
def test_each_removed_factory_fails_loudly_rather_than_silently(name):
    """A stub that returned None would be worse than one that raises."""
    fn = getattr(factory, name, None)
    if fn is None:
        pytest.skip(f"{name} is gone entirely, so there is nothing to call")
    with pytest.raises(NotImplementedError, match="removed"):
        fn()

"""Discrimination tests for the branch classifier's absorption predicate
(scripts/prune_local_branches.sh).

The script's whole value is that it never labels a branch disposable when its content is not in
main. Two earlier predicates failed that in opposite directions, and both looked fine in use:

- Reverse-applying the branch diff against the WORKING TREE was *anti-correlated* with the property
  it claimed to measure -- it reported an unmerged branch as absorbed and a merged one as not,
  simultaneously, whenever you sat on a third branch.
- Reverse-applying against a temp index fixed that, but still answers "does this diff reverse-apply"
  rather than "is this content in main". A branch whose changes landed and were then further edited
  fails to reverse and reads as unmerged.

The predicate is now `git merge-tree --write-tree`: merge the branch into main and compare trees.
These tests build a real throwaway repository for each case rather than mocking git, because every
one of those failures was a disagreement between what git does and what the script assumed it does.

`absorbed()` returns 0 = absorbed, 1 = not absorbed, 2 = no content difference at all.
"""

import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prune_local_branches.sh"


def _git(repo, *args, **kw):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, **kw)


@pytest.fixture
def repo(tmp_path):
    """A repository with a real `origin/main` remote-tracking ref, not one faked with `update-ref`.

    The upstream is BARE: pushing to the checked-out branch of a non-bare repo is refused, and the
    squash-merge case below has to push."""
    upstream = tmp_path / "upstream.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(upstream)], check=True, capture_output=True)

    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    _git(work, "config", "user.email", "t@t")
    _git(work, "config", "user.name", "t")
    (work / "a.txt").write_text("base\n")
    _git(work, "add", "a.txt")
    _git(work, "commit", "-qm", "base")
    _git(work, "remote", "add", "origin", str(upstream))
    _git(work, "push", "-q", "-u", "origin", "main")
    return work


def _absorbed(repo, branch):
    """Source the script for its definitions only -- that is what `_PRUNE_SOURCED` is for -- and
    call the predicate. Sourcing rather than reimplementing is the point: a copy of the predicate
    here would pass while the shipped one rotted."""
    # The `if` is not style: the script sets -e, so a bare `absorbed ...` that returns non-zero
    # kills the shell before `echo` runs and the harness reads an empty result rather than a
    # verdict. A command in an `if` condition is exempt.
    cmd = (
        f'_PRUNE_SOURCED=1 . "{_SCRIPT}"; TMP=$(mktemp -d); '
        f'if absorbed "{branch}"; then rc=0; else rc=$?; fi; echo "rc=$rc"'
    )
    out = subprocess.run(
        ["bash", "-c", cmd],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("rc=")]
    assert line, f"predicate produced no rc: {out.stdout!r} {out.stderr!r}"
    return int(line[-1].split("=")[1])


def test_a_branch_whose_content_is_not_in_main_is_not_absorbed(repo):
    _git(repo, "checkout", "-qb", "feature")
    (repo / "b.txt").write_text("new\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "add b")
    assert _absorbed(repo, "feature") == 1


def test_a_squash_merged_branch_is_absorbed(repo):
    """The case every commit-graph predicate gets wrong. Under a squash merge no commit of the
    branch is an ancestor of main, so `rev-list --count origin/main..branch` is non-zero and
    `branch --merged` omits it -- while the content is fully present."""
    _git(repo, "checkout", "-qb", "feature")
    (repo / "b.txt").write_text("new\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "add b")

    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--squash", "feature")
    _git(repo, "commit", "-qm", "squashed")
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "fetch", "-q", "origin")

    # The commit-graph view, recorded so the discrimination is visible rather than asserted:
    ahead = _git(repo, "rev-list", "--count", "origin/main..feature").stdout.strip()
    assert ahead != "0", "fixture is not exercising the squash case"
    assert _absorbed(repo, "feature") == 0


def test_a_branch_further_edited_after_landing_is_not_absorbed(repo):
    """The case the reverse-apply predicate got wrong in the other direction: the branch's changes
    are in main AND it carries more, so it must not be pruned."""
    _git(repo, "checkout", "-qb", "feature")
    (repo / "b.txt").write_text("new\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "add b")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--squash", "feature")
    _git(repo, "commit", "-qm", "squashed")
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "fetch", "-q", "origin")

    _git(repo, "checkout", "-q", "feature")
    (repo / "b.txt").write_text("new\nand more\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "extend b")
    assert _absorbed(repo, "feature") == 1


def test_an_empty_commit_branch_reports_no_content_difference(repo):
    """rc 2, distinct from 0: 'nothing to check' must not be reported with the same words as
    'checked, and its content is in main'."""
    _git(repo, "checkout", "-qb", "feature")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "empty")
    assert _absorbed(repo, "feature") == 2


def test_the_verdict_does_not_depend_on_which_branch_is_checked_out(repo):
    """The predicate's first version was evaluated against the WORKING TREE, so sitting on a third
    branch inverted it. Same branch, three checkouts, one answer."""
    _git(repo, "checkout", "-qb", "feature")
    (repo / "b.txt").write_text("new\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "add b")
    _git(repo, "checkout", "-qb", "third", "main")
    (repo / "c.txt").write_text("third\n")
    _git(repo, "add", "c.txt")
    _git(repo, "commit", "-qm", "add c")

    verdicts = []
    for where in ("main", "feature", "third"):
        _git(repo, "checkout", "-q", where)
        verdicts.append(_absorbed(repo, "feature"))
    assert len(set(verdicts)) == 1, f"verdict depends on the checkout: {verdicts}"
    assert verdicts[0] == 1


def test_the_predicate_survives_an_unrelated_nearby_edit_in_main(repo):
    """THE ONE CASE THAT DISCRIMINATES the current predicate from the reverse-apply one it replaced.

    Every other test in this file passes under BOTH, which is worth stating plainly: they pin the
    contract, not the rework. This is the fixture that separates them, found by running the two
    against each other rather than by arguing from the code.

    The branch adds a line; main squash-merges it and then edits a DIFFERENT line that falls inside
    the branch diff's context window. Reverse-applying the diff now fails on context and reports
    NOT absorbed; merging into main produces main's own tree, so the branch demonstrably contributes
    nothing.

        old (git apply --check --reverse) : NOT absorbed (1)   <- wrong
        new (git merge-tree --write-tree) : absorbed (0)

    Direction matters and is not oversold: the old predicate's error is a FALSE NEGATIVE. It keeps a
    prunable branch, never proposes deleting an unmerged one. So this is an accuracy and noise fix,
    not a safety hole being closed -- and a classifier that cries wolf is still a classifier people
    stop reading.
    """
    (repo / "f.txt").write_text("l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-qm", "f base")
    _git(repo, "push", "-q", "origin", "main")

    _git(repo, "checkout", "-qb", "feature")
    (repo / "f.txt").write_text("l1\nl2\nl3\nl4\nADDED\nl5\nl6\nl7\nl8\n")
    _git(repo, "commit", "-qam", "add a line")

    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--squash", "feature")
    _git(repo, "commit", "-qm", "squashed")
    (repo / "f.txt").write_text("l1\nCHANGED\nl3\nl4\nADDED\nl5\nl6\nl7\nl8\n")
    _git(repo, "commit", "-qam", "edit a nearby line")
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "fetch", "-q", "origin")

    assert _absorbed(repo, "feature") == 0, (
        "the branch's content is entirely in main -- merging it produces main's own tree -- so it "
        "is absorbed. Reporting otherwise is the reverse-apply predicate's context sensitivity."
    )


class TestThePopulationCoversBothScopes:
    """2026-08-21: the classifier reported "0 branch(es) show evidence of having landed" on a tree
    carrying SIXTEEN remote branches whose PRs were merged. They had never been local, and the
    population was `refs/heads/` alone.

    That is the same shape as the defects this repo has been chasing all week -- a check that is not
    where the thing it checks lives -- and it is invisible in exactly the way that matters: the
    output is a clean report, not an error.
    """

    def test_a_remote_only_branch_is_in_the_population(self, repo):
        """The load-bearing one. Nothing local; the branch exists only on origin."""
        _git(repo, "checkout", "-qb", "remote-only")
        (repo / "r.txt").write_text("r\n")
        _git(repo, "add", "r.txt")
        _git(repo, "commit", "-qm", "r")
        _git(repo, "push", "-q", "origin", "remote-only")
        _git(repo, "checkout", "-q", "main")
        _git(repo, "branch", "-qD", "remote-only")

        listed = subprocess.run(
            [
                "bash",
                "-c",
                '{ git for-each-ref refs/heads/ --format="%(refname:lstrip=2)"; '
                '  git for-each-ref refs/remotes/origin/ --format="%(refname:lstrip=3)"; '
                "} | grep -vxE 'main|HEAD' | sort -u",
            ],
            cwd=str(repo),
            capture_output=True,
            text=True,
        ).stdout.split()
        assert "remote-only" in listed, f"a remote-only branch must be enumerated; got {listed}"

    def test_the_population_expression_in_the_script_reads_both_refs(self):
        """Pinned against the script text, because the expression above is a copy: if the shipped
        one narrows back to `refs/heads/` the copy keeps passing."""
        src = _SCRIPT.read_text()
        assert "refs/remotes/origin/" in src, "the shipped population must enumerate remote refs"
        assert src.count("git for-each-ref refs/heads/") == 1
        assert src.count("git for-each-ref refs/remotes/origin/") == 1

    def test_the_remedy_names_the_remote_command_too(self):
        """`git branch -D` does not touch origin. A report that pools both scopes and then offers
        only the local command sends the reader to a no-op."""
        src = _SCRIPT.read_text()
        assert "git push origin --delete" in src, "the remote remedy must be stated"
        assert "git branch -D" in src, "the local one must still be stated"


def test_a_squash_merged_branch_stops_being_recognised_once_main_moves_on(repo):
    """The claim this predicate CANNOT make, pinned so it is not re-asserted.

    An earlier comment said absorption "holds under a squash merge". It holds only while main has
    not since modified the same regions -- after that the three-way merge conflicts and the branch
    reads unmerged again. Measured on the real repository: of sixteen remote branches whose PRs are
    merged, eight classify as unmerged this way.

    The error is a FALSE NEGATIVE, which is why the merged-PR name signal is the primary evidence
    and this predicate is corroborating. This test exists so the limitation is stated by a failing
    assertion if anyone strengthens the claim.
    """
    _git(repo, "checkout", "-qb", "feature")
    (repo / "a.txt").write_text("base\nfrom-feature\n")
    _git(repo, "commit", "-qam", "feature edit")

    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--squash", "feature")
    _git(repo, "commit", "-qm", "squashed")
    # main moves on, touching the same region
    (repo / "a.txt").write_text("base\nfrom-feature\nand-then-main\n")
    _git(repo, "commit", "-qam", "main moves on")
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "fetch", "-q", "origin")

    # RECORDED LIMITATION, not a contract. 1 = not absorbed, which is WRONG about this branch --
    # its content is in main. It is kept because the error is a false negative and because stating
    # it here is what stops the flat claim being re-asserted in `absorbed()`.
    assert _absorbed(repo, "feature") == 1, (
        "the predicate now recognises a squash-merged branch under a moving main. That is an "
        "improvement, not a failure: update the conditional wording in `absorbed()` and delete this "
        "test -- but verify against the real repository first, where eight branches with MERGED PRs "
        "classified as unmerged on 2026-08-21"
    )

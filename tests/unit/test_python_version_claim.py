"""The versions the package claims to support are the versions something actually tests.

`pyproject.toml`'s `Programming Language :: Python :: X.Y` classifiers are a public promise, and
`.github/workflows/python-compat.yml` is the only thing that checks it. They were kept in step by
a comment in each file pointing at the other, which is not a mechanism: either can move alone and
the promise silently stops being backed.

The trigger for writing this: 3.15-dev sat in the matrix as an allow-failure entry and was removed
2026-08-13 (scipy publishes no 3.15 wheel, so the job never reached a test). The classifiers did
not list 3.15, so removing it happened to leave them agreeing -- but nothing would have said so if
it had not, and the comment left behind in `pyproject.toml` asserts a two-way correspondence that
until now had no enforcement behind it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "python-compat.yml"
PYPROJECT = REPO / "pyproject.toml"

_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (\d+\.\d+)$")


def _claimed() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    out = set()
    for c in data["project"]["classifiers"]:
        if m := _CLASSIFIER.match(c):
            out.add(m.group(1))
    return out


def _tested() -> set[str]:
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["compat"]
    versions = set(job["strategy"]["matrix"]["python-version"])
    # An `include:` row may add an entry the plain list does not carry.
    for extra in job["strategy"]["matrix"].get("include", ()):
        versions.add(extra["python-version"])
    return versions


def test_every_claimed_version_is_tested():
    """A classifier is a promise to users; an untested one is a promise nothing checks."""
    claimed, tested = _claimed(), _tested()
    assert claimed, "no Python version classifiers found; the parser or the file moved"
    assert claimed <= tested, (
        f"pyproject claims {sorted(claimed - tested)} but python-compat.yml does not test "
        f"{sorted(claimed - tested)}. Either add the version to the matrix or drop the classifier."
    )


def test_every_tested_version_is_claimed():
    """The other direction, which is the one that rots quietly.

    A version can be added to the matrix and pass for months while the classifiers still tell
    users it is unsupported. Dropping a version from the matrix without dropping its classifier
    is the same defect wearing the other face.

    This direction also encodes a policy, so it is the one to argue with rather than edit around:
    **a version belongs in the matrix only if it is required to pass.** "Test it early without
    claiming support" was the old `experimental: true` + `continue-on-error` arrangement, and it
    produced a permanent red check that every reader had to learn to ignore -- which is a worse
    early-warning signal than none, because the ignoring generalises. If a version can pass, claim
    it; if it cannot yet, it is not ready to be in here at all.

    Neither this test nor its sibling hardcodes a version: both sides are parsed, so 3.16 needs
    two file edits and no change here. What needs a change here is *reversing the policy* -- and
    that should cost a deliberate edit with a reason, which is the point.
    """
    claimed, tested = _claimed(), _tested()
    assert tested, "the compat matrix is empty; the parser or the workflow moved"
    assert tested <= claimed, (
        f"python-compat.yml tests {sorted(tested - claimed)} but pyproject does not claim it.\n"
        f"  - if it passes: add the classifier, in this same commit.\n"
        f"  - if it does not pass yet: take it out of the matrix. Carrying a version that cannot\n"
        f"    pass means a permanently red check, which is what 3.15-dev did until 2026-08-13.\n"
        f"  - if you want a non-blocking trial entry back: that is a policy change, not a matrix\n"
        f"    edit. Change this test and say why, and see the header of python-compat.yml."
    )


def test_no_matrix_entry_is_allowed_to_fail_without_saying_so():
    """`continue-on-error` keeps the WORKFLOW green while the check run still reports FAILURE.

    That is what 3.15-dev did: every PR carried a red `py3.15-dev` indistinguishable from a real
    one, and telling them apart meant opening the run and reading the workflow-level conclusion.
    If a non-blocking entry is ever wanted again, it has to be a decision someone makes and
    writes down -- not something a matrix row inherits.
    """
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["compat"]
    assert "continue-on-error" not in job, (
        "python-compat's jobs are all required. Re-introducing continue-on-error means some "
        "version's red no longer blocks; update this test in the same commit and say why."
    )


def test_requires_python_does_not_exceed_the_oldest_claimed_version():
    """`requires-python` is what pip enforces; the classifiers are what humans read."""
    data = tomllib.loads(PYPROJECT.read_text())
    floor = data["project"]["requires-python"]
    m = re.search(r">=\s*(\d+)\.(\d+)", floor)
    assert m, f"could not parse requires-python: {floor!r}"
    declared_floor = (int(m.group(1)), int(m.group(2)))
    oldest_claimed = min(tuple(int(p) for p in v.split(".")) for v in _claimed())
    assert declared_floor == oldest_claimed, (
        f"requires-python says {declared_floor} but the oldest classifier is {oldest_claimed}; "
        f"pip and the README would disagree about the same package."
    )

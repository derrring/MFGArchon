"""The marker declaration list must stay reachable and honest (Issue #1706).

`--strict-markers` rejects *undeclared* markers, which means a declaration is precisely what makes a
meaningless marker look legitimate. Two failures it cannot see:

- **Unreachable** -- declared, applied by no test, named by no selector. Dead weight that reads as
  policy. Five existed: `regression`, `tier4`, `network`, `stochastic`, `numerical`.
- **False promise** -- the description claims a schedule no selector implements. `tier1`-`tier4`
  declared "run on every commit / on PRs / on merge to main / weekly or manually" with **zero**
  selectors referencing any of them, so a test marked `tier4` to defer it would have run in every
  tier instead. The whole family was removed rather than re-described, because the names carry the
  same suggestion as the descriptions.

This is the general shape of a defect an additive change produces: a *modification* is checked
against what was there before, an *addition* regresses nothing and so passes every gate. The same
class already produced two junk markers here: a three-line description in `pytest.ini`'s linelist
`markers` option registered `@pytest.mark.with` and `@pytest.mark.worse` (#1706).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_markers.py"


def _run(cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--path", str(cwd)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_repository_satisfies_both_invariants():
    result = _run(REPO)

    assert result.returncode == 0, (
        f"scripts/check_markers.py rejects the current pytest.ini:\n{result.stdout}\n{result.stderr}"
    )
    assert result.stdout.strip(), "a silent pass is indistinguishable from a checker that died"


def test_no_marker_declaration_spans_multiple_lines():
    """`markers` is a linelist: a continued line registers as its own marker.

    A three-line description produced `@pytest.mark.with` and `@pytest.mark.worse`, which
    `--strict-markers` would then have accepted forever (#1706).
    """
    ini = (REPO / "pytest.ini").read_text()
    block = ini.split("markers =", 1)[1]

    offenders = []
    for line in block.splitlines():
        if line.strip() and not line.startswith((" ", "\t")):
            break
        if line.strip() and ":" not in line:
            offenders.append(line.strip())

    assert not offenders, (
        f"continuation line(s) in the markers block will register as markers: {offenders}. "
        "Each declaration must be one line."
    )


@pytest.mark.parametrize(
    ("injected", "expect_in_output"),
    [
        pytest.param("    ghost_marker: A marker nothing uses.\n", "UNREACHABLE", id="unreachable"),
        pytest.param("    weekly_thing: Heavy tests - run weekly.\n", "FALSE PROMISE", id="false-promise"),
    ],
)
def test_the_checker_rejects_what_it_claims_to(tmp_path, injected, expect_in_output):
    """The positive control. A checker that passes on a corpus it should reject reports nothing.

    Both injections are the real defect shapes, not synthetic ones: `ghost_marker` is `network`
    before it was deleted, `weekly_thing` is `tier4`.
    """
    (tmp_path / "tests").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: Slow tests.\n" + injected)
    clean = _run(tmp_path)

    assert clean.returncode == 1, (
        f"the checker accepted a corpus containing {expect_in_output.lower()}:\n{clean.stdout}"
    )
    assert expect_in_output in clean.stdout


def test_a_marker_applied_only_in_prose_does_not_count_as_used(tmp_path):
    """Detection is AST-based, so a mention inside a docstring must not rescue a dead marker.

    A textual scan reports the marker as used and the declaration survives -- the circularity that
    made an earlier version of a related checker report a false zero.
    """
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        '"""This module used to use pytest.mark.ghost_marker."""\n\n\ndef test_x():\n    pass\n'
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    slow: Slow tests.\n    ghost_marker: Nothing applies this.\n"
    )

    result = _run(tmp_path)

    assert result.returncode == 1, "a docstring mention must not count as an application"
    assert "ghost_marker" in result.stdout


def test_an_applied_marker_with_a_schedule_claim_is_still_rejected(tmp_path):
    """The control that was missing, and the shape `tier1`-`tier3` actually had.

    Both injections above are *unapplied*, so both also trip the unreachable invariant -- meaning
    the false-promise branch could stop failing the build with no test noticing. Review of #1708
    demonstrated exactly that: deleting the branch from the exit condition left the suite green.
    This corpus applies the marker, so unreachable cannot fire and only the promise can.
    """
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    # `slow` is applied too: leaving it bare makes the SCAFFOLD unreachable, and the assertions
    # below would then be satisfied by the scaffold rather than by the marker under test.
    (tmp_path / "tests" / "test_x.py").write_text(
        "import pytest\n\n\n@pytest.mark.weekly_thing\n@pytest.mark.slow\ndef test_x():\n    pass\n"
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    slow: Slow tests.\n    weekly_thing: Heavy tests - run weekly.\n"
    )

    result = _run(tmp_path)

    assert result.returncode == 1, "an applied marker promising a schedule must still be rejected"
    assert "FALSE PROMISE" in result.stdout
    assert "UNREACHABLE" not in result.stdout, "the marker IS applied; only the promise is at fault"


def test_a_marker_registered_only_in_conftest_is_still_seen(tmp_path):
    """`pytest.ini` is not the only registration path, and reading only it made this checker lie.

    `config.addinivalue_line("markers", ...)` is what `--strict-markers` consults. Seven markers --
    the whole promissory tier family -- survived a deletion from the ini and the census reported
    clean over them (#1708 review).
    """
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text(
        'def pytest_configure(config):\n    config.addinivalue_line("markers", "ghost_marker: Nothing applies this.")\n'
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: Slow tests.\n")

    result = _run(tmp_path)

    assert result.returncode == 1, "a marker registered in conftest must be censused too"
    assert "ghost_marker" in result.stdout

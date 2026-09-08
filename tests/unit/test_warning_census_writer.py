"""The warning census must not be destroyed by a session that measured nothing (#2287).

`scripts/local_ci.sh` exports `MFGARCHON_WARNING_CENSUS` and leaves it exported, so every later
pytest invocation in that run inherits it. `scripts/report_discrimination.py` issues
`pytest --collect-only` **after** the warning ratchet has read the census — and the writer in
`tests/conftest.py` then overwrote a real measurement with an empty one.

Measured before the fix: the gate's own census went from 225 identities to 0 identities / 294 bytes.
The consequence is not the gate's verdict, which is decided before the clobber. It is that
`check_warnings.py --write-baseline` — the recovery the ratchet's own failure message instructs you
to run — could not be fed from the artifact the gate had just produced, so re-baselining cost a
second full suite run.

`check_warnings.py`'s `MIN_TESTS` floor already refuses a census this thin. What a reader-side floor
cannot do is prevent the overwrite, which happens before it ever runs; that is why the guard is on
the writer.

ADMISSION (#2287). Class 4 — it guards an instrument, the census that `check_warnings.py` consumes.
Two tests, not four: the defect, and the control without which a writer that never writes would
satisfy it. Two more were written and cut — that `tests_run` sums every outcome kind, and that an
unset census path means no write. Both exercise behaviour this change did not touch, which is the
"should have one" shape § *Admission* refuses, and this package is pre-1.0 with the surface still
moving.

The writer is called directly with a stub reporter rather than by spawning pytest inside pytest: the
defect is a branch in the writer, and a subprocess would pay two interpreter starts for the same
line.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _writer():
    """The hook under test, imported from the root conftest that defines it."""
    from tests.conftest import pytest_terminal_summary

    return pytest_terminal_summary


def _reporter(**stats):
    """A terminal reporter carrying only what the writer reads: `.stats`."""
    return SimpleNamespace(stats=stats)


@pytest.fixture
def census(tmp_path, monkeypatch):
    path = tmp_path / "census.json"
    monkeypatch.setenv("MFGARCHON_WARNING_CENSUS", str(path))
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    return path


def test_a_session_that_ran_tests_writes_the_census(census):
    """The control. Without it, a writer that never writes would satisfy the test below."""
    _writer()(_reporter(passed=[1, 2, 3]), 0, None)

    assert census.exists(), "a session with outcomes must write a census"
    assert json.loads(census.read_text())["tests_run"] == 3


def test_a_collect_only_session_does_not_overwrite_a_real_census(census):
    """The defect. A zero-outcome session must leave an existing measurement alone.

    Mutation, measured for #2287: replacing the `payload["tests_run"] == 0` guard in
    `tests/conftest.py` with `if False:` turns a pinned census of 4 outcomes / 1 identity into
    0 / 0 after one `pytest --collect-only`, and kills this test.
    """
    _writer()(_reporter(passed=[1, 2, 3, 4]), 0, None)
    before = census.read_text()

    _writer()(_reporter(warnings=[]), 0, None)  # collect-only: no outcomes at all

    assert census.read_text() == before, "a zero-outcome session overwrote a real census"

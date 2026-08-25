"""`scripts/gate_hook.sh` is the carrier the gate's verdict travels through, so it gets the same
checks as the gate (#2117).

pre-commit writes a hook's captured stdout in one `output_stream.write(s)`, guarded by
`verbose or hook.verbose or retcode or files_modified` (`run.py:217`). For this hook the first two
are False, so the write happens on a RED gate or on a run during which tracked files changed --
and either way the payload is ~805 KB against a 64 KB non-blocking pipe. Reproduced directly: an
`io.BufferedWriter` over such a pipe completes at 1 KB and raises `[Errno 35]` at 805 KB, the error
in `~/.cache/pre-commit/pre-commit.log`. pre-commit then dies with exit 120 instead of printing why
the gate went red.

#2118 also cuts the volume at source -- 95.7% of it is pytest's warnings summary -- so this adapter
is defence in depth. A red gate's pytest output can grow past the buffer again on its own.

Two bugs were written into the adapter before these tests existed, and both are pinned below:

- `rc=$?` inside `if ! cmd; then` reads the status of the NEGATION. Every red gate came back green.
  An adapter that always reports success is strictly worse than the defect it was written to fix.
- the summary grep matched `FAIL ` on a stream where the gate's own ANSI escape sits between `FAIL`
  and the space, so it dropped every per-check verdict while still matching pytest's uncoloured
  `FAILED` lines -- a summary that looks populated having omitted what it exists to surface.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ADAPTER = Path(__file__).resolve().parents[2] / "scripts" / "gate_hook.sh"

#: The gate colours its verdicts. A stub that printed plain text would pass a grep that the real
#: gate's output defeats, so the stub emits the same escapes `local_ci.sh` does.
STUB = """#!/usr/bin/env bash
printf 'gate interpreter : /stub/python (Python 9.9.9)\\n'
for i in $(seq 1 {noise}); do echo "noise line $i"; done
{body}
printf 'gate interpreter : /stub/python (Python 9.9.9)\\n'
{verdict}
exit {rc}
"""


def _run(tmp_path: Path, *, rc: int, body: str = "", noise: int = 2000, verdict: str | None = None):
    """Run the real adapter against a stub gate, and return (exit code, summary, log)."""
    shutil.copy(ADAPTER, tmp_path / "gate_hook.sh")
    green = "printf '\\033[32mGATE GREEN\\033[0m -- safe to push.\\n'"
    red = "printf '\\033[31mGATE RED\\033[0m -- do not push.\\n'"
    # rc=2 emits the real CANNOT RUN shape -- a headline plus a body. The docstring below says
    # "2 is GATE CANNOT RUN", and until this stub said so too that was a claim the test did not
    # deliver: an auditor would have credited the branch as covered. It also exercises the
    # `sed -n '/^GATE CANNOT RUN/,$p'` range, which nothing else does.
    cannot = (
        "printf '\\033[31mGATE CANNOT RUN\\033[0m -- MFG_PYTHON is unusable\\n'\n"
        "printf 'This is an ENVIRONMENT failure, not a code failure: nothing was measured.\\n'\n"
        "printf 'Activate the env or set MFG_PYTHON to an interpreter with that tooling.\\n'"
    )
    stub = tmp_path / "local_ci.sh"
    stub.write_text(
        STUB.format(
            noise=noise,
            body=body,
            rc=rc,
            verdict=verdict if verdict is not None else {0: green, 2: cannot}.get(rc, red),
        )
    )
    stub.chmod(0o755)
    log = tmp_path / "gate.log"
    proc = subprocess.run(
        [str(tmp_path / "gate_hook.sh")],
        capture_output=True,
        text=True,
        env={**os.environ, "MFGARCHON_GATE_LOG": str(log)},
    )
    return proc.returncode, proc.stdout, log.read_text()


@pytest.mark.parametrize("rc", [0, 1, 2])
def test_the_gates_exit_code_survives_the_adapter(tmp_path, rc):
    """0 is green, 1 is red, 2 is `GATE CANNOT RUN`. All three must arrive unchanged.

    The bug this pins returned 0 for every one of them, so the hook would have reported a red gate
    as passing -- the failure mode a pre-push gate exists to prevent.
    """
    got, summary, _ = _run(tmp_path, rc=rc)
    assert got == rc, f"stub gate exited {rc}; the adapter reported {got}"
    if rc == 2:
        # The BODY, not just the headline. Summarising CANNOT RUN to one clause dropped the
        # environment-vs-code distinction and the remedy, in the one case where the whole log is
        # under 1 KB and there is no volume to summarise away.
        assert "nothing was measured" in summary, f"CANNOT RUN lost its body:\n{summary}"
        assert "Activate the env" in summary, f"CANNOT RUN lost its remedy:\n{summary}"


def test_a_coloured_per_check_FAIL_reaches_the_summary(tmp_path):
    """The gate prints `\\033[31mFAIL\\033[0m <name>`, so the escape sits where a space would be."""
    body = r"""printf '\033[31mFAIL\033[0m ruff check mfgarchon/\n'
printf '\033[33mWARN\033[0m ruff 0.1.0 ran, but the config pins 0.16.0\n'
printf 'FAILED tests/unit/test_x.py::test_y\n'"""
    _, summary, _ = _run(tmp_path, rc=1, body=body)

    assert "FAIL ruff check mfgarchon/" in summary, f"coloured FAIL was dropped:\n{summary}"
    assert "WARN ruff 0.1.0 ran" in summary, f"coloured WARN was dropped:\n{summary}"
    assert "FAILED tests/unit/test_x.py::test_y" in summary
    assert "GATE RED" in summary


def test_the_verdict_survives_more_failures_than_the_cap(tmp_path):
    """The verdict is printed outside the cap, so 200 failing tests cannot push it off the end.

    Its absence is what started #2117: a summary with no verdict reads as a broken gate.
    """
    body = 'for i in $(seq 1 200); do printf "FAILED tests/unit/test_%s.py::test_z\\n" "$i"; done'
    got, summary, _ = _run(tmp_path, rc=1, body=body)

    assert got == 1
    assert "GATE RED -- do not push." in summary, f"verdict lost behind the cap:\n{summary[-500:]}"
    assert summary.count("FAILED tests/unit/") <= 40, "the cap is what keeps this bounded"


def test_a_gate_that_never_reaches_a_verdict_says_so(tmp_path):
    """Killed mid-run, the log has no `GATE` line. Silence there would read as success."""
    got, summary, _ = _run(tmp_path, rc=1, verdict="", body="")
    assert got == 1
    assert "GATE VERDICT MISSING" in summary, f"a verdict-less gate must be named as one:\n{summary}"


def test_the_summary_fits_where_the_gate_did_not(tmp_path):
    """The whole point: what pre-commit writes in one call must fit a non-blocking pipe.

    64 KB is the measured buffer on this platform. The control shows the reproduction is live --
    the gate's real volume raises the exact `[Errno 35]` from pre-commit's own traceback.
    """
    _, summary, log = _run(tmp_path, rc=0, noise=20000)
    assert len(log) > 65536, f"the stub must exceed the pipe buffer or this proves nothing; got {len(log)}"

    def writes_without_blocking(payload: bytes) -> bool:
        read_fd, write_fd = os.pipe()
        os.set_blocking(write_fd, False)
        stream = io.BufferedWriter(io.FileIO(write_fd, "wb", closefd=False))
        try:
            stream.write(payload)
            stream.flush()
            return True
        except BlockingIOError:
            return False
        finally:
            os.close(read_fd)
            with contextlib.suppress(OSError):
                os.close(write_fd)

    assert not writes_without_blocking(log.encode()), (
        "the gate's own volume must still fail this write, or the test is not exercising #2117"
    )
    assert writes_without_blocking(summary.encode()), (
        f"the summary is {len(summary)} bytes and still does not fit a non-blocking pipe"
    )

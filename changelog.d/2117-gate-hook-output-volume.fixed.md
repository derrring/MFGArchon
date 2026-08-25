- **The pre-push gate no longer reports a PASSING run as `Failed`** (Issue #2117). pre-commit writes
  a hook's whole captured stdout in one `output_stream.write(s)`. `scripts/local_ci.sh` emits
  **804,934 bytes** — the suite, the durations table, four ratchets, the capability matrix, the
  discrimination and assertion-strength reports — against a non-blocking pipe that holds **65,536**
  on this platform. Measured: an `io.BufferedWriter` over such a pipe completes at 1 KB and raises
  `BlockingIOError: [Errno 35]` at 805 KB, which is the traceback in `~/.cache/pre-commit/pre-commit.log`.

  The exception propagates out of `_run_single_hook` and the hook is recorded as failed. **The gate
  had passed.** That is the whole defect: a green gate reported as red, with the one line that
  distinguishes them — `GATE GREEN` — inside the output that could not be written. Three pushes hit
  it in one session; the natural response is to hunt a broken test that does not exist, and the
  second-nature response is `--no-verify`, which is what a pre-push gate exists to prevent.

  `scripts/gate_hook.sh` now stands between them: it runs the gate, writes the full stream to a log
  (`$MFGARCHON_GATE_LOG`, default `$TMPDIR/mfgarchon-gate.log`), and prints a bounded summary —
  identity lines, per-check verdicts, any `FAIL`/`WARN`/`FAILED`, and the gate's own verdict. 6,353
  lines become 20; 805 KB becomes 1.1 KB. **`scripts/local_ci.sh` is unchanged** and still the
  authoritative gate; run it directly for the full stream.

  Two defects were written into the adapter before its tests existed, and both are now pinned by
  `tests/unit/test_gate_hook.py`:

  - `rc=$?` inside `if ! cmd; then` reads the status of the **negation**, so every red gate came
    back green. An adapter that always reports success is strictly worse than the defect it was
    written to fix.
  - the summary matched `FAIL ` on a stream where the gate's ANSI escape sits between `FAIL` and
    the space, dropping every per-check verdict while still matching pytest's uncoloured `FAILED`
    lines — a summary that looks populated having omitted exactly what it exists to surface.

  The verdict is printed outside the cap, so a run with 200 failing tests cannot push its own
  verdict off the end, and a gate that never reaches a verdict is named as one rather than passing
  silently.

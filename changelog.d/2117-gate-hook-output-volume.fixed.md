- **The pre-push gate can print why it went red** (Issue #2117), by not handing pre-commit 805 KB.

  pre-commit writes a hook's captured stdout in one `output_stream.write(s)`, guarded by
  `verbose or hook.verbose or retcode or files_modified` (`run.py:217`). For this hook the first
  two are False, so the write fires on a **red** gate — or on a run during which tracked files
  changed. Either way the payload is the whole gate: **804,837 bytes** against a non-blocking pipe
  that holds **65,536** on this platform. Measured with pre-commit's own write path: an
  `io.BufferedWriter` over such a pipe completes at 1 KB and raises `BlockingIOError: [Errno 35]`
  at 805 KB, the traceback in `~/.cache/pre-commit/pre-commit.log`.

  pre-commit then dies with *"An unexpected error has occurred"* and exit 120 **instead of printing
  which check failed**. That reads as a pre-commit bug rather than a test failure, and the
  second-nature response is `--no-verify` — past a gate that was genuinely red.

  A **green** gate can also be reported as `Failed`, but only through the `files_modified`
  disjunct. The one fully-logged instance here was exactly that: `GATE GREEN` on line 6376 and
  `- files were modified by this hook` on line 6. The modification was not the gate's — a full run
  on a clean tree leaves `git status` empty, measured — it was a concurrent edit while the hook ran.

  **The volume is cut at source in the same change.** 95.7% of those 805 KB is pytest's warnings
  summary: 6,030 of 6,354 lines, 770,223 of 804,837 bytes. `--disable-warnings` on the gate's
  pytest invocation takes the whole run to **324 lines / 34,614 bytes — 0.53× the pipe instead of
  12.3×**. It suppresses the listing, not the warnings: the tail still reads `N passed, M warnings`
  (`-p no:warnings` would drop the count, which is why it is not used). What that listing was
  carrying is a backlog, not noise — 456 of its lines are this repository's own tests calling its
  own deprecated `MFGProblem(geometry=, ...)`, across 103 files. Filed as #2119.

  **The adapter stays, as defence in depth rather than the fix.** Review measured three failing
  tests adding 5,022 bytes — about 1,674 per failure — against the 30,896 bytes of headroom that
  now remain below the pipe. **Roughly 18 failing tests puts the payload back over the buffer**,
  which is a reachable number on a bad refactor and is exactly when the reader most needs
  pre-commit to survive long enough to print which check failed.

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

#!/usr/bin/env bash
# pre-push adapter for `scripts/local_ci.sh`. NOT a second gate -- it runs that one and reports it.
#
# WHY THIS EXISTS (#2117). pre-commit captures a hook's stdout and writes it in ONE call:
# `output.write_line_b(out.strip(), ...)` -> `output_stream.write(s)`. The gate emits ~805 KB
# (6,353 lines: the suite, the durations table, four ratchets, the capability matrix, the
# discrimination and assertion-strength reports). A non-blocking pipe holds 64 KB on this platform
# -- measured -- so that write is 12.3x the buffer and raises `BlockingIOError: [Errno 35]`. The
# exception propagates out of `_run_single_hook`, and pre-commit records the hook as **Failed**.
#
# The gate had PASSED. That is the whole problem: the failure mode is a green gate reported as red,
# and the one line that distinguishes the two -- `GATE GREEN` -- is inside the output that could not
# be written. Three pushes hit it in one session; the natural response is to go looking for a broken
# test that does not exist, and the second-nature response is `--no-verify`, which is what a
# pre-push gate exists to prevent.
#
# So: full output to a file, a bounded summary to stdout, exit code passed through unchanged.
# Run `scripts/local_ci.sh` directly for the full stream -- that path is untouched.
set -uo pipefail

LOG="${MFGARCHON_GATE_LOG:-${TMPDIR:-/tmp}/mfgarchon-gate.log}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# `rc=$?` on the next line and NOT inside an `if ! ...; then` -- there, `$?` is the status of the
# NEGATION, so every non-zero gate came back as 0 and this adapter reported a red gate as green.
# Caught by stubbing local_ci.sh at exits 0/1/2 before this shipped; it would have disabled the
# gate silently, which is strictly worse than the #2117 it was written to fix.
"$HERE/local_ci.sh" "$@" >"$LOG" 2>&1
rc=$?

# Bounded by construction: one line per verdict, plus pytest's own failure lines, capped. A gate
# with every check red prints ~10 verdicts, so the cap is slack, not a truncation anyone will meet.
# ANSI first, then match. The gate colours its verdicts, so the escape sits between `FAIL` and the
# space that follows it, and a naive `grep 'FAIL '` silently drops every per-check verdict while
# still matching pytest's uncoloured `FAILED` lines -- a summary that looks populated having
# omitted exactly the lines it exists to surface. Measured on a synthetic log before this was
# written, which is the only reason it is not still wrong.
plain() { sed $'s/\033\\[[0-9;]*m//g' "$LOG"; }

# The verdict is printed unconditionally and OUTSIDE the cap. Putting it inside would let a run
# with 40+ failing tests push its own verdict off the end -- the one line whose absence started
# #2117 in the first place.
# Deduped: the gate prints its identity block at head AND tail on purpose (a mismatched run and a
# matched one otherwise have byte-identical tails), but in a summary that is just the same three
# lines twice.
plain | grep -aE '^gate [a-z]+ +:' | awk '!seen[$0]++' || true
plain | grep -aE '^(FAILED|ERROR) |^(FAIL|WARN) ' | head -40
plain | grep -aE '^(PASS|SKIPPED) ' | head -20
plain | grep -aE '^GATE (GREEN|RED|CANNOT RUN)' || printf 'GATE VERDICT MISSING -- the gate did not reach its own conclusion\n'
printf 'gate log (%s lines, %s bytes): %s\n' "$(wc -l <"$LOG" | tr -d ' ')" "$(wc -c <"$LOG" | tr -d ' ')" "$LOG"
exit "$rc"

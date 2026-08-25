#!/usr/bin/env bash
# pre-push adapter for `scripts/local_ci.sh`. NOT a second gate -- it runs that one and reports it.
#
# WHY THIS EXISTS (#2117). pre-commit captures a hook's stdout and writes it in ONE call --
# `output.write_line_b(out.strip(), ...)` -> `output_stream.write(s)` -- but only when
# `verbose or hook.verbose or retcode or files_modified` (`run.py:217`). For this hook the first
# two are False (`hook_impl.py` hardcodes `verbose=False`; the config sets no `verbose:` key), so
# the write happens on a RED gate, or on a run during which tracked files changed.
#
# Either way the payload is the whole gate: ~805 KB against a non-blocking pipe that holds 64 KB
# -- measured. `io.BufferedWriter` over such a pipe completes at 1 KB and raises
# `BlockingIOError: [Errno 35]` at 805 KB, which is the traceback in
# `~/.cache/pre-commit/pre-commit.log`. pre-commit then dies with "An unexpected error has
# occurred" and exit 120 INSTEAD OF PRINTING WHY THE GATE WENT RED. The common case is not a
# green gate misreported -- it is a red gate that cannot say what failed, which reads as a
# pre-commit bug and invites `--no-verify` past a genuinely red gate.
#
# A green gate CAN be reported as Failed, but it needs the `files_modified` disjunct. The one
# fully-logged instance in this repository was exactly that: `GATE GREEN` on line 6376 and
# `- files were modified by this hook` on line 6. The modification was not the gate's -- a full
# run on a clean tree leaves `git status` empty, measured -- it was a concurrent edit while the
# hook ran.
#
# #2118 also cut the volume at source: 95.7% of those 805 KB is pytest's warnings summary, and
# `--disable-warnings` in `local_ci.sh` takes the gate to ~35 KB, which is 0.53x the pipe rather
# than 12.3x. THIS ADAPTER IS THEREFORE DEFENCE IN DEPTH, not the primary fix: a red gate's
# pytest output can grow past 64 KB again on its own, and that is exactly when the reader most
# needs pre-commit to survive long enough to print it.
#
# Full output to a file, a bounded summary to stdout, exit code passed through unchanged. Run
# `scripts/local_ci.sh` directly for the full stream -- that path is untouched.
set -uo pipefail

# Per-PID by default. A fixed path lets two concurrent gates truncate each other's log while both
# hold open fds, and the summary then reports the OTHER run's interpreter identity with nothing
# marking it -- measured. The identity block exists precisely because a version-mismatched run and
# a matched one have byte-identical tails, so cross-attributing it is the one corruption that
# cannot be caught downstream. Nothing has to hunt for the file: the last line prints its path.
LOG="${MFGARCHON_GATE_LOG:-${TMPDIR:-/tmp}/mfgarchon-gate.$$.log}"
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
# LC_ALL=C: BSD sed in a UTF-8 locale ABORTS at the first invalid byte and discards everything
# after it, so one stray byte from a test would drop the FAIL lines and the verdict while leaving
# the summary looking populated -- the same shape as the ANSI bug above. Every grep here already
# carries -a for the same reason; sed needed the locale instead.
plain() { LC_ALL=C sed $'s/\033\\[[0-9;]*m//g' "$LOG"; }

# The verdict is printed unconditionally and OUTSIDE the cap. Putting it inside would let a run
# with 40+ failing tests push its own verdict off the end -- the one line whose absence started
# #2117 in the first place.
# Deduped: the gate prints its identity block at head AND tail on purpose (a mismatched run and a
# matched one otherwise have byte-identical tails), but in a summary that is just the same three
# lines twice.
plain | grep -aE '^gate [a-z]+ +:' | awk '!seen[$0]++' || true
plain | grep -aE '^(FAILED|ERROR) |^(FAIL|WARN) ' | head -40
plain | grep -aE '^(PASS|SKIPPED) ' | head -20
# CANNOT RUN prints its FULL body, not a matched line. That verdict is a paragraph -- it says
# whether anything was measured at all, and how to fix the environment -- and the volume problem
# does not exist in this case: the whole log is under 1 KB when the gate cannot start. Summarising
# it to one clause was a strict regression against printing nothing at all.
# Read the verdict ONCE into a variable rather than `plain | grep -q`. Under `pipefail`, `grep -q`
# exits at the first match and `sed` takes SIGPIPE, so the pipeline reports 141 and the `if` takes
# the WRONG branch whenever enough output follows the verdict. Not reachable today only because
# `cannot_run()` ends in `exit 2` so nothing follows it -- which couples this file's correctness to
# that property of local_ci.sh. Reproduced: `seq 1 500000 | grep -q '^1$'` is 141, `'^499999$'` is 0.
verdict=$(plain | grep -aE '^GATE (GREEN|RED|CANNOT RUN)' || true)
case "$verdict" in
  *"GATE CANNOT RUN"*)
    # The full paragraph, not the matched line: it says whether anything was measured at all and
    # how to fix the environment, and the log is under 1 KB in this case -- no volume problem.
    plain | sed -n '/^GATE CANNOT RUN/,$p'
    ;;
  "") printf 'GATE VERDICT MISSING -- the gate did not reach its own conclusion\n' ;;
  *)  printf '%s\n' "$verdict" ;;
esac
printf 'gate log (%s lines, %s bytes): %s\n' "$(wc -l <"$LOG" | tr -d ' ')" "$(wc -c <"$LOG" | tr -d ' ')" "$LOG"
exit "$rc"

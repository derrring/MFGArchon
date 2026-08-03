#!/usr/bin/env bash
# The authoritative pre-merge gate. Runs what GitHub's PR checks no longer do.
#
# Why local: the heavy suite takes ~2.5 min here and >25 min on a GitHub runner
# (measured: coverage costs 1.5x, the runner itself ~7x). Online execution of the
# full suite bought latency, not signal. GitHub keeps the fast tier; nightly keeps
# the full run incl. @slow as the backstop.
#
# Usage:  ./scripts/local_ci.sh            # full gate
#         ./scripts/local_ci.sh --fast     # skip the test suite (lint/format/ratchet only)
set -uo pipefail

FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1
cd "$(dirname "$0")/.."

# Resolve the interpreter and the linter EXPLICITLY, because this script's callers do not agree
# on the environment. Interactively it is run from an activated conda env, where bare `python`
# and `ruff` resolve. The pre-push hook is not: pre-commit invokes it with its own PATH, conda
# is not activated, and every single check then failed with `command not found` while the script
# printed a per-check FAIL and `GATE RED -- do not push`. That is indistinguishable at a glance
# from a real red gate on content, so the hook has never once been able to pass, and the habit it
# produced was `--no-verify`.
#
# The probe is the load-bearing part, not the path search: any python on PATH satisfies a path
# search and would then run the "authoritative gate" against an interpreter that cannot run the
# suite -- a green-looking run that measured nothing. See `resolved_python` for why the obvious
# form of that probe does not work.
cannot_run() {  # environment failure: say that nothing was measured, and exit distinguishably
  printf '\n\033[31mGATE CANNOT RUN\033[0m -- %s\n' "$1"
  printf 'This is an ENVIRONMENT failure, not a code failure: nothing was measured, so it says\n'
  printf 'nothing about whether the change is sound. Activate the env (conda activate mfg_env)\n'
  printf 'or set MFG_PYTHON to an interpreter with that tooling installed.\n'
  exit 2
}

# The probe runs from a scratch directory OUTSIDE the source tree, and that placement is the
# whole of it. Run from here it proves nothing: line 15 already put cwd at the repo root, which
# CONTAINS `mfgarchon/`, and `python -c` puts cwd on sys.path -- so any interpreter with the
# third-party dependencies imports the source tree in place and passes. Measured:
# `/usr/bin/python3 -c 'import mfgarchon'` reaches base_solver.py from the repo root and fails
# only at `import numpy`, and an empty `mfgarchon/__init__.py` in any directory satisfies it.
#
# A scratch dir rather than `/`, which would be the obvious choice: importing this package has a
# filesystem side effect -- `utils/performance/monitoring.py:413` builds a module-level
# PerformanceMonitor whose __init__ runs a cwd-relative `Path("performance_data").mkdir()` -- so
# `cd / && python -c 'import mfgarchon'` dies with `Errno 30 Read-only file system` on a perfectly
# good interpreter (#1674; delete this paragraph when that is fixed).
#
# Demanding the token back on stdout is the second half, and it is not paranoia: `MFG_PYTHON=echo`
# makes `echo -c 'import mfgarchon'` exit 0, so an exit-status-only probe accepts /bin/echo as the
# interpreter and every subsequent check passes trivially -- GATE GREEN, exit 0, nothing run. It
# is not adversary-proof: a purpose-built script that prints the token defeats it. The bar it
# raises is from "defeated by /bin/echo by accident" to "requires a deliberate forgery".
#
# The candidate is ABSOLUTIZED before the probe runs, and `resolved_python` echoes that absolute
# path back for the caller to use. Resolving at one cwd and executing at another silently refused
# every relative interpreter -- `.venv/bin/python`, the commonest project-local layout -- while
# claiming it could not import the package. It could; only the probe's `cd` could not find it.
#
# The probe tests TOOLING, never the package under test. That distinction is the point, and
# getting it wrong inverts this script's whole thesis: under the editable install this repo uses,
# `import mfgarchon` eagerly loads 268 submodules -- the very tree being reviewed -- so a branch
# whose `__init__` is mid-refactor made the probe fail, and the gate then announced "ENVIRONMENT
# failure, not a code failure: nothing was measured" over what was purely a code failure, with an
# exit code telling the reader not to look at the code. `main` ran the ruff and ratchet checks and
# reported it correctly.
#
# So `mfgarchon` is a PREFERENCE for choosing between interpreters (pass 1 below), never a reason
# to refuse. If the package does not import, the checks that need it say so, with the traceback,
# under a GATE RED -- which is a verdict about content, and is the true one.
#
# What must import is what will actually run. `--fast` (ruff plus three stdlib-only AST scanners,
# per CLAUDE.md the iterate-while-working mode) needs neither the package nor the test tooling.
# `yaml` is listed because the workflow-integrity step needs it and it is NOT a declared
# dependency -- it arrives transitively via omegaconf, so an environment can look complete and
# still fail that step with a bare ModuleNotFoundError under a GATE RED.
probe_modules() {
  if [[ $FAST -eq 1 ]]; then printf 'yaml'; else printf 'yaml, numpy, pytest, xdist'; fi
}

# Usage: resolved_python <candidate> [with_package]
# Echoes the absolute interpreter path on success. `probe_err` carries the interpreter's own
# stderr, because discarding it leaves "no module named xdist" indistinguishable from a traceback
# out of the package -- which is exactly the distinction the message above has to get right.
# The file, not a variable: `PY=$(resolved_python ...)` runs the function in a SUBSHELL, so any
# variable it sets is discarded before the caller can read it. Writing to a path the parent knows
# is what makes the interpreter's own stderr survive the capture.
# ONE scratch dir for the whole script, not one per probe. A per-probe `mktemp -d` cleaned up
# inside the function leaks on signal: bash propagates the parent's traps into a command
# substitution, so `PY=$(resolved_python ...)` runs `exit 130` in the subshell and the function's
# own `rm -rf` never executes. Owning both paths at script level puts them under the one trap
# that does run.
PROBE_ERR_FILE=$(mktemp 2>/dev/null) || PROBE_ERR_FILE=""
PROBE_DIR=$(mktemp -d 2>/dev/null) || PROBE_DIR=""
# Two traps, not one. A handler installed for INT that only cleans up does NOT end the script:
# bash runs it and RESUMES at the next command. Measured with that single trap in place, Ctrl-C
# during --fast let the run continue and render `GATE RED -- do not push` over an interrupted
# run -- an operator event wearing a content verdict, which is the failure class this script
# exists to remove. Worse, a SIGINT inside the 4.2 s pass-1 probe killed that probe and let the
# search fall through to the next candidate, silently changing which interpreter the gate used.
# `exit 130` is what restores the untrapped behaviour that `main` had for free.
trap 'rm -f "$PROBE_ERR_FILE"; rm -rf "$PROBE_DIR"' EXIT
trap 'rm -f "$PROBE_ERR_FILE"; rm -rf "$PROBE_DIR"; exit 130' INT TERM
# tail -5, not -3: a ModuleNotFoundError is exactly 3 lines, but a SyntaxError spends them on the
# source echo and the caret, dropping the `File ...` line that names the culprit.
probe_err() { [[ -n "$PROBE_ERR_FILE" ]] && tail -5 "$PROBE_ERR_FILE" 2>/dev/null; }

resolved_python() {
  local candidate=$1 want_package=${2:-} out probe_dir modules
  candidate=$(command -v "$candidate" 2>/dev/null) || return 1
  [[ -n "$candidate" ]] || return 1
  [[ "$candidate" == /* ]] || candidate="$PWD/$candidate"
  modules=$(probe_modules)
  [[ -n "$want_package" ]] && modules="mfgarchon, $modules"
  out=$(cd "$PROBE_DIR" && "$candidate" -P -c "import $modules, sys; sys.stdout.write('MFGARCHON_OK')" 2>"${PROBE_ERR_FILE:-/dev/null}")
  [[ "$out" == "MFGARCHON_OK" ]] || return 1
  # ruff is 2 of the 6 --fast checks, so it belongs in the predicate that SELECTS an interpreter.
  # Checked after resolution instead, the search committed to the first candidate satisfying an
  # incomplete predicate and then refused outright -- while a working interpreter was still
  # sitting further down CANDIDATES. Measured: `GATE CANNOT RUN` on a machine where forcing
  # candidate 3 runs the whole gate.
  (cd "$PROBE_DIR" && "$candidate" -P -m ruff --version) >/dev/null 2>>"${PROBE_ERR_FILE:-/dev/null}" || return 1
  printf '%s' "$candidate"
  return 0
}

# Fail where the failure IS. Letting an unmade scratch dir make every candidate "fail" produced a
# verdict composed entirely of false claims about the operator's interpreter -- that it could not
# import the tooling, when it was never executed -- and told them to fix the thing they had
# already done. The scratch directory is this script's own resource, not evidence about theirs.
[[ -n "$PROBE_DIR" ]] || cannot_run "could not create a scratch directory (mktemp -d failed).
No interpreter was probed, so this says nothing about any of them."

# An explicitly set MFG_PYTHON is an operator statement, not a hint. `main` honoured it
# unconditionally (`PY="${MFG_PYTHON:-python}"`), so a wrong value failed loudly and attributably.
# Searching past it would silently run the authoritative gate against an interpreter the operator
# did not name -- a silent fallback, in the script whose own third check ratchets against those.
# `+x`, not `:-`: an explicitly EMPTY MFG_PYTHON is still an operator statement, and it is what
# `MFG_PYTHON="$SOME_UNSET_VAR"` produces. Under `-n` it read as unset and silently searched --
# the one surviving fall-through beneath an absolute "it is used or nothing is" claim.
CANDIDATES=(python python3 /opt/homebrew/Caskroom/miniforge/base/envs/mfg_env/bin/python)

if [[ -n "${MFG_PYTHON+x}" ]]; then
  PY=$(resolved_python "$MFG_PYTHON") \
    || cannot_run "MFG_PYTHON=${MFG_PYTHON:-<empty>} is unusable: it must exist and import
$(probe_modules) plus ruff, probed from a scratch directory outside the source tree.
$(probe_err)
It is set explicitly, so it is used or nothing is: this does NOT fall back to another interpreter."
else
  # Pass 1 prefers an interpreter that ALSO imports the package, so that on a normal machine the
  # gate picks the project env rather than whichever python happens to carry pytest.
  PY=""
  for candidate in "${CANDIDATES[@]}"; do
    if PY=$(resolved_python "$candidate" with_package); then break; fi
    PY=""
  done
  # Pass 2 drops that preference. Refusing here would turn a broken package into an "environment
  # failure" verdict; instead the gate runs, and the checks that need the package report why.
  if [[ -z "$PY" ]]; then
    for candidate in "${CANDIDATES[@]}"; do
      if PY=$(resolved_python "$candidate"); then break; fi
      PY=""
    done
  fi
  [[ -n "$PY" ]] || cannot_run "no interpreter found with $(probe_modules) plus ruff.
$(probe_err)"
fi

# `-m ruff`, not a path next to $PY: `dirname` is wrong for any shimmed or symlinked interpreter
# (pyenv/asdf/uv shims, /usr/local/bin/python), where `command -v` deliberately does not resolve
# the link and the sibling `ruff` does not exist beside the shim. Presence is already part of the
# selection predicate above; this only binds the invocation.
#
# `-P` is load-bearing, not hygiene. `-m` puts cwd at the FRONT of sys.path, and line 15 makes cwd
# the repo root -- so a `ruff/` package committed to the root shadows the real linter, and the gate
# reports GATE GREEN with 435 files unlinted while `gate ruff : ruff 0.16.0` sits in the pasted
# tail as forged evidence. `main` invoked ruff as a PATH executable, which no file in the tree can
# shadow; switching to `-m` is what opened this, so closing it belongs to that change. The probe
# above uses `-P` for the same reason.
RUFF=("$PY" -P -m ruff)

# Print what was measured, at the head for a live run and again beside the verdict, because the
# PR template asks a human to paste the LAST lines and a head-only line never reaches them.
# This line is the only tell for a forged interpreter, so it has to be in the pasted evidence.
printf 'gate interpreter : %s (%s)\n' "$PY" "$("$PY" -V 2>&1)"
printf 'gate ruff        : %s\n' "$("${RUFF[@]}" --version 2>&1)"

fail=0
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
check() {
  if [[ $1 -eq 0 ]]; then printf '\033[32mPASS\033[0m %s\n' "$2"
  else printf '\033[31mFAIL\033[0m %s\n' "$2"; fail=1; fi
}

# This script calls itself the authoritative gate, so it must not run whatever ruff happens to
# be on PATH. pyproject/environment.yml specify `ruff>=0.6.0` -- a floor, not a pin -- so a
# contributor who installs today gets a different formatter from the one CI and pre-commit use,
# and goes red on files they never touched. Warn rather than fail: an unexpected version is a
# real signal, but blocking the whole gate on it would be worse than running it.
# `$PWD`, not `$(dirname "$0")/..`: line 15 already put cwd at the repo root, so the old path
# resolved to the repo's PARENT whenever the script was invoked as `./local_ci.sh` from scripts/.
# RUFF_PIN came back empty and the version WARN silently never fired -- which matters now that
# the ruff version is printed in the tail as merge evidence.
RUFF_PIN=$(grep -A1 'astral-sh/ruff-pre-commit' "$PWD/.pre-commit-config.yaml" 2>/dev/null \
  | grep -oE 'rev: v[0-9]+\.[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || RUFF_PIN="")
RUFF_HAVE=$("${RUFF[@]}" --version 2>/dev/null | awk '{print $2}')
if [[ -n "$RUFF_PIN" && -n "$RUFF_HAVE" && "$RUFF_PIN" != "$RUFF_HAVE" ]]; then
  printf '\033[33mWARN\033[0m ruff %s in the gate interpreter, but .pre-commit-config.yaml pins %s -- formatting may disagree with CI\n' \
    "$RUFF_HAVE" "$RUFF_PIN"
fi

step "Ruff format"
"${RUFF[@]}" format --check mfgarchon/; check $? "ruff format --check mfgarchon/"

step "Ruff lint (full ruleset, includes tests/ which CI does not)"
"${RUFF[@]}" check mfgarchon/ tests/; check $? "ruff check mfgarchon/ tests/"

step "Workflow integrity"
# Parsing is NOT sufficient and this check knows it: a workflow gutted down to one job,
# or one whose `needs:` points at a job that was deleted, parses perfectly. GitHub rejects
# a dangling `needs:` at load time and then NO job in that file runs on any event -- a
# whole workflow silently switched off. Both failure modes were shipped during this
# repo's CI cleanup, one after the other.
"$PY" -c "
import sys, yaml, pathlib
bad = []
for f in sorted(pathlib.Path('.github/workflows').glob('*.y*ml')):
    try:
        d = yaml.safe_load(open(f)) or {}
    except Exception as e:
        bad.append(f'{f}: does not parse: {e}')
        continue
    jobs = d.get('jobs') or {}
    if not jobs:
        bad.append(f'{f}: declares no jobs')
        continue
    for name, job in jobs.items():
        needs = job.get('needs')
        if needs:
            needs = [needs] if isinstance(needs, str) else needs
            for n in needs:
                if n not in jobs:
                    bad.append(f'{f}: job {name!r} needs {n!r}, which is not defined -- GitHub will reject the whole file')
        if not (job.get('steps') or job.get('uses')):
            bad.append(f'{f}: job {name!r} has no steps')
for b in bad:
    print(b)
sys.exit(1 if bad else 0)
"
check $? "workflows parse, declare jobs, and have no dangling needs"

step "Fail-fast ratchet"
"$PY" scripts/check_fail_fast.py --path mfgarchon --check-baseline scripts/fail_fast_baseline.json
check $? "no new silent fallbacks vs baseline"

# Docs are the one artefact nothing else runs: this suite never imports a doc example, so a
# rename leaves every tutorial that used the old name teaching a NameError (Issue #1759).
# Pure AST, no imports -- importing would make the count depend on which optional
# dependencies are installed, and would drift with the environment rather than with the docs.
step "Doc-API ratchet"
"$PY" scripts/check_doc_api.py --path . --check-baseline scripts/doc_api_baseline.json
check $? "docs teach no more missing API than the baseline records"

step "Frozen prototype areas"
# alg/neural and alg/reinforcement are prototypes, not under development (CLAUDE.md). The
# counter-intuitive half of that freeze is that ADDING TESTS is also out: coverage reads as a
# promise the behaviour is intended, and on a placeholder that promise is false. Prose alone does
# not hold this line -- hasattr is banned by the same conventions and was written into a test on
# 2026-07-30 because the fail-fast ratchet does not scan tests/ (#1780).
"$PY" scripts/check_frozen_areas.py --check-baseline scripts/frozen_areas_baseline.json
check $? "no new tests against a frozen prototype paradigm"

if [[ $FAST -eq 0 ]]; then
  # ~40 s. Not in --fast: every cell is a real coupled solve, so this is the one check
  # here that measures the product rather than the source. Counts of tests, issues or
  # fail-fast violations all move for reasons unrelated to whether anything solves;
  # this is the quantity that does not. Bidirectional -- a recovered cell fails until
  # the baseline records it, so a fix cannot land without saying so.
  step "Capability matrix (public solve surface vs external oracles)"
  "$PY" scripts/capability_matrix.py --check-baseline scripts/capability_baseline.json
  check $? "no capability change vs baseline"

  step "Test suite (CI marker set, xdist parallel, no coverage)"
  "$PY" -m pytest tests/ -n auto \
    -m "not slow and not benchmark and not experimental and not optional_torch and not environment" \
    -q --durations=10
  check $? "full suite"
else
  printf '\n\033[33mSKIPPED\033[0m test suite (--fast)\n'
fi

printf '\ngate interpreter : %s (%s)\n' "$PY" "$("$PY" -V 2>&1)"
printf 'gate ruff        : %s\n' "$("${RUFF[@]}" --version 2>&1)"
# Reprinted here, not only at the head: a version-mismatched run and a matched run otherwise
# produce byte-identical tails, so the comparison this WARN performs is not recoverable from the
# pasted evidence. Same rule that put the interpreter line here.
if [[ -n "$RUFF_PIN" && -n "$RUFF_HAVE" && "$RUFF_PIN" != "$RUFF_HAVE" ]]; then
  printf '\033[33mWARN\033[0m ruff %s ran, but .pre-commit-config.yaml pins %s\n' "$RUFF_HAVE" "$RUFF_PIN"
fi
if [[ $fail -eq 0 ]]; then
  printf '\033[32mGATE GREEN\033[0m -- safe to push.\n'
else
  printf '\033[31mGATE RED\033[0m -- do not push.\n'
fi
exit $fail

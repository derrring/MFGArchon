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
# suite -- a green-looking run that measured nothing. See `usable_python` for why the obvious
# form of that probe does not work.
cannot_run() {  # environment failure: say that nothing was measured, and exit distinguishably
  printf '\n\033[31mGATE CANNOT RUN\033[0m -- %s\n' "$1"
  printf 'This is an ENVIRONMENT failure, not a code failure: nothing was measured, so it says\n'
  printf 'nothing about whether the change is sound. Activate the env (conda activate mfg_env)\n'
  printf 'or set MFG_PYTHON to an interpreter with the package and the test tooling installed.\n'
  exit 2
}

# The probe runs from `/`, and that is the whole of it. Run from here it proves nothing: line 15
# already put cwd at the repo root, which CONTAINS `mfgarchon/`, and `python -c` puts cwd on
# sys.path -- so any interpreter with the third-party dependencies imports the source tree in
# place and passes. Measured: `/usr/bin/python3 -c 'import mfgarchon'` reaches base_solver.py from
# the repo root and fails only at `import numpy`, while from `/` it is a clean ModuleNotFoundError.
# An empty `mfgarchon/__init__.py` in any scratch directory satisfies the from-here version.
#
# Demanding the token back on stdout is the second half, and it is not paranoia: `MFG_PYTHON=echo`
# makes `echo -c 'import mfgarchon'` exit 0, so an exit-status-only probe accepts /bin/echo as the
# interpreter and every subsequent check passes trivially -- GATE GREEN, exit 0, nothing run.
# Probed from a WRITABLE scratch directory rather than `/`, which would be the obvious choice:
# importing this package has a filesystem side effect -- `utils/performance/monitoring.py:413`
# builds a module-level PerformanceMonitor whose __init__ runs `Path("performance_data").mkdir()`,
# cwd-relative -- so `cd / && python -c 'import mfgarchon'` dies with `Errno 30 Read-only file
# system` on a perfectly good interpreter. The scratch dir keeps the property that matters (the
# source tree is not in cwd) without depending on cwd being writable.
usable_python() {
  local candidate=$1 out probe_dir
  command -v "$candidate" >/dev/null 2>&1 || return 1
  probe_dir=$(mktemp -d) || return 1
  out=$(cd "$probe_dir" && "$candidate" -c 'import mfgarchon, pytest, xdist, sys; sys.stdout.write("MFGARCHON_OK")' 2>/dev/null)
  rm -rf "$probe_dir"
  [[ "$out" == "MFGARCHON_OK" ]] || return 1
  return 0
}

# An explicitly set MFG_PYTHON is an operator statement, not a hint. `main` honoured it
# unconditionally (`PY="${MFG_PYTHON:-python}"`), so a wrong value failed loudly and attributably.
# Searching past it would silently run the authoritative gate against an interpreter the operator
# did not name -- a silent fallback, in the script whose own third check ratchets against those.
if [[ -n "${MFG_PYTHON:-}" ]]; then
  usable_python "$MFG_PYTHON" \
    || cannot_run "MFG_PYTHON=$MFG_PYTHON cannot import mfgarchon + pytest + xdist (probed from /).
It is set explicitly, so it is used or nothing is: this does NOT fall back to another interpreter."
  PY=$(command -v "$MFG_PYTHON")
else
  PY=""
  for candidate in python python3 /opt/homebrew/Caskroom/miniforge/base/envs/mfg_env/bin/python; do
    if usable_python "$candidate"; then PY=$(command -v "$candidate"); break; fi
  done
  [[ -n "$PY" ]] || cannot_run "no interpreter on PATH can import mfgarchon + pytest + xdist."
fi

# `-m ruff`, not a path next to $PY: `dirname` is wrong for any shimmed or symlinked interpreter
# (pyenv/asdf/uv shims, /usr/local/bin/python), where `command -v` deliberately does not resolve
# the link and the sibling `ruff` does not exist beside the shim.
"$PY" -m ruff --version >/dev/null 2>&1 || cannot_run "ruff is not installed in $PY."
RUFF=("$PY" -m ruff)

# Print what was measured. The tail of this run is the merge evidence the PR template asks for,
# and a verdict that does not name the interpreter it ran is not evidence.
printf 'gate interpreter : %s (%s)\n' "$PY" "$("$PY" -V 2>&1)"
printf 'gate ruff        : %s\n' "$("$PY" -m ruff --version 2>&1)"

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
RUFF_PIN=$(grep -A1 'astral-sh/ruff-pre-commit' "$(dirname "$0")/../.pre-commit-config.yaml" 2>/dev/null \
  | grep -oE 'rev: v[0-9]+\.[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || RUFF_PIN="")
RUFF_HAVE=$("${RUFF[@]}" --version 2>/dev/null | awk '{print $2}')
if [[ -n "$RUFF_PIN" && -n "$RUFF_HAVE" && "$RUFF_PIN" != "$RUFF_HAVE" ]]; then
  printf '\033[33mWARN\033[0m ruff %s on PATH, but .pre-commit-config.yaml pins %s -- formatting may disagree with CI\n' \
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

printf '\n'
if [[ $fail -eq 0 ]]; then
  printf '\033[32mGATE GREEN\033[0m -- safe to push.\n'
else
  printf '\033[31mGATE RED\033[0m -- do not push.\n'
fi
exit $fail

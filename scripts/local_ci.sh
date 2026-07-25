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

PY="${MFG_PYTHON:-python}"
FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1
cd "$(dirname "$0")/.."

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
RUFF_HAVE=$(ruff --version 2>/dev/null | awk '{print $2}')
if [[ -n "$RUFF_PIN" && -n "$RUFF_HAVE" && "$RUFF_PIN" != "$RUFF_HAVE" ]]; then
  printf '\033[33mWARN\033[0m ruff %s on PATH, but .pre-commit-config.yaml pins %s -- formatting may disagree with CI\n' \
    "$RUFF_HAVE" "$RUFF_PIN"
fi

step "Ruff format"
ruff format --check mfgarchon/; check $? "ruff format --check mfgarchon/"

step "Ruff lint (full ruleset, includes tests/ which CI does not)"
ruff check mfgarchon/ tests/; check $? "ruff check mfgarchon/ tests/"

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

if [[ $FAST -eq 0 ]]; then
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

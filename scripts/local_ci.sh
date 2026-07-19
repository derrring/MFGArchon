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

step "Ruff format"
ruff format --check mfgarchon/; check $? "ruff format --check mfgarchon/"

step "Ruff lint (full ruleset, includes tests/ which CI does not)"
ruff check mfgarchon/ tests/; check $? "ruff check mfgarchon/ tests/"

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

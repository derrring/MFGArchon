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
cannot_run() {  # environment failure: say what was measured, and exit distinguishably
  printf '\n\033[31mGATE CANNOT RUN\033[0m -- %s\n' "$1"
  # Every cannot_run used to sit before the first verdict, so "nothing was measured" was simply
  # true. The mypy step can fire one after a PASS has printed, and a blanket claim there would be
  # a false statement about the run in the same breath as a true one about the check.
  if [[ ${verdicts:-0} -eq 0 ]]; then
    printf 'This is an ENVIRONMENT failure, not a code failure: nothing was measured, so it says\n'
    printf 'nothing about whether the change is sound. Activate the env (conda activate mfg_env)\n'
  else
    printf 'This is an ENVIRONMENT failure, not a code failure. The %d verdict(s) above did run and\n' "$verdicts"
    printf 'stand; everything from this check onward was NOT measured. Activate the env\n'
    printf '(conda activate mfg_env)\n'
  fi
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
# mypy joins ruff in that predicate. Note what that changed: `--fast` used to need neither the
# package nor the test tooling, and the type gate runs in BOTH tiers, so `--fast` now requires the
# `dev` extra. The old sentence was left standing above its own correction for one revision -- a
# reader who reached it and stopped read something false -- so it is rewritten here rather than
# annotated.
#
# What must import is what will actually run. `--fast` (ruff, mypy, and three stdlib-only AST
# scanners, per CLAUDE.md the iterate-while-working mode) does not need pytest. It DOES need the
# package: `scripts/check_internal_deprecation.py` does `import mfgarchon`, and on a package-less
# interpreter that step exits with ModuleNotFoundError. The sentence claimed otherwise, and so did
# its first correction, which fixed the tooling clause and preserved the package one -- the same
# shape as the defect it was written to fix.
# `yaml` is listed because the workflow-integrity step needs it. It used to arrive transitively
# and was declared nowhere -- from omegaconf, and from jupyterlab's dependency chain -- so an
# environment could look complete and still fail that step with a bare ModuleNotFoundError under a
# GATE RED. #1687 removed both suppliers and declared `pyyaml` directly, which is what keeps this
# probe honest: it now names a dependency the project actually asks for.
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
# The mypy control's probe must live INSIDE the checked scope to see a per-module override, so it
# cannot go in PROBE_DIR. It goes under the same trap instead, which is the half that matters.
MYPY_PROBE="mfgarchon/config/_gate_probe.py"
# Two traps, not one. A handler installed for INT that only cleans up does NOT end the script:
# bash runs it and RESUMES at the next command. Measured with that single trap in place, Ctrl-C
# during --fast let the run continue and render `GATE RED -- do not push` over an interrupted
# run -- an operator event wearing a content verdict, which is the failure class this script
# exists to remove. Worse, a SIGINT inside the 4.2 s pass-1 probe killed that probe and let the
# search fall through to the next candidate, silently changing which interpreter the gate used.
# `exit 130` is what restores the untrapped behaviour that `main` had for free.
trap 'rm -f "$PROBE_ERR_FILE" "$MYPY_PROBE"; rm -rf "$PROBE_DIR"' EXIT
trap 'rm -f "$PROBE_ERR_FILE" "$MYPY_PROBE"; rm -rf "$PROBE_DIR"; exit 130' INT TERM
# tail -5, not -3: a ModuleNotFoundError is exactly 3 lines, but a SyntaxError spends them on the
# source echo and the caret, dropping the `File ...` line that names the culprit.
probe_err() { [[ -n "$PROBE_ERR_FILE" ]] && tail -5 "$PROBE_ERR_FILE" 2>/dev/null; }

resolved_python() {
  local candidate=$1 want_package=${2:-} out modules
  candidate=$(command -v "$candidate" 2>/dev/null) || return 1
  [[ -n "$candidate" ]] || return 1
  [[ "$candidate" == /* ]] || candidate="$PWD/$candidate"
  modules=$(probe_modules)
  [[ -n "$want_package" ]] && modules="mfgarchon, $modules"
  out=$(cd "$PROBE_DIR" && "$candidate" -P -c "import $modules, sys; sys.stdout.write('MFGARCHON_OK')" 2>"${PROBE_ERR_FILE:-/dev/null}")
  [[ "$out" == "MFGARCHON_OK" ]] || return 1
  # ruff is 2 of the 8 --fast checks, so it belongs in the predicate that SELECTS an interpreter.
  # Checked after resolution instead, the search committed to the first candidate satisfying an
  # incomplete predicate and then refused outright -- while a working interpreter was still
  # sitting further down CANDIDATES. Measured: `GATE CANNOT RUN` on a machine where forcing
  # candidate 3 runs the whole gate.
  (cd "$PROBE_DIR" && "$candidate" -P -m ruff --version) >/dev/null 2>>"${PROBE_ERR_FILE:-/dev/null}" || return 1
  # mypy for the same reason, and it was added here rather than beside its own step for exactly
  # the reason the paragraph above records: a hard refusal placed AFTER resolution skips every
  # remaining candidate. Measured on this repo -- an interpreter carrying pytest, xdist and ruff
  # but not mypy aborted the gate in 0.62 s having run 2 of 11 steps, while candidate 3 would
  # have run all of them.
  (cd "$PROBE_DIR" && "$candidate" -P -m mypy --version) >/dev/null 2>>"${PROBE_ERR_FILE:-/dev/null}" || return 1
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
$(probe_modules) plus ruff and mypy, probed from a scratch directory outside the source tree.
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
  [[ -n "$PY" ]] || cannot_run "no interpreter found with $(probe_modules) plus ruff and mypy.
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
printf 'gate mypy        : %s\n' "$("$PY" -P -m mypy --version 2>/dev/null || echo unknown)"

# A discrimination sweep mutates production source in place and restores it in a `finally`.
# That survives an exception and SIGINT; it does not survive SIGKILL or a harness timeout, and
# what a killed sweep leaves behind is a silently wrong solver -- `bc_type_to_geometric_operation`
# answering "clamp" for no_flux, or `hjb_residual_norm` with its load-bearing sqrt(dx) deleted.
# Both observed (#1849, and again 2026-08-13 in the MAIN checkout).
#
# The script's own `_assert_clean_tree()` runs at ITS startup, so it protects the next SWEEP and
# nothing else. This is the guard at the point of CONSUMPTION: whatever killed the sweep, the gate
# refuses to report on a mutated tree. Verified complete rather than assumed -- all 24 mutations
# in scripts/test_discrimination.py carry the marker (`0 without a MUTATED marker`), so grepping
# for it cannot miss one.
#
# GATE CANNOT RUN, not FAIL: nothing was measured about the code you meant to test, and a red
# gate here would read as a defect in the working tree's content rather than in its state.
if MUTATED_LEFTOVER=$(grep -rn '# MUTATED' mfgarchon/ 2>/dev/null) && [[ -n "$MUTATED_LEFTOVER" ]]; then
  cannot_run "a mutation marker is still in the source tree -- a discrimination sweep was killed
before it could restore. Recover with \`git checkout -- mfgarchon/\` and re-run.

$MUTATED_LEFTOVER"
fi

fail=0
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
check() {
  verdicts=$((${verdicts:-0} + 1))
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

# The one gate that lived ONLY on GitHub. ci.yml runs this exact command as a blocking step named
# "MyPy type gate (config subpackage, blocking)" and nothing here mirrored it -- measured before
# this change, `grep -c mypy scripts/local_ci.sh` returned 0 against 23 for ruff. So a type error
# in `mfgarchon/config` could not be seen before pushing, which is the mirror image of this file's
# own reason to exist: CLAUDE.md warns that a GitHub-green PR has not had its tests run, and this
# was the check nobody could run locally at all (Issue #2101, option 4).
#
# Cost, measured, not estimated: ~22 s the first time (a cold `.mypy_cache`, which is what a fresh
# clone or a mypy upgrade gets), on a gate whose full run is ~350 s. The warm figure that stood
# here (~0.6 s) was measured for the real check ALONE and no longer describes the step: there are
# five mypy invocations per run now and the control never hits its own cache. Withdrawn rather than
# replaced -- every attempt to re-measure it has been on a loaded machine. The
# cache is ~84 MB, gitignored. The first number is the one that lands on a new contributor.
#
# Scope and flags are copied from ci.yml verbatim, not chosen here. Raw mypy over the package
# pulls 1800+ transitive errors from un-annotated dependencies; `--follow-imports=silent` plus the
# `mfgarchon/config` scope is what makes the count meaningful, and a second opinion about scope
# would make this gate and CI disagree -- which is worse than either scope. Note what the copy
# inherits: `pyproject.toml` gives `mfgarchon.factory.*` the identical strict treatment, and
# `mypy mfgarchon/factory --follow-imports=silent` reports 3 errors that neither CI nor this gate
# sees. Inherited, not introduced. Widening the LOCAL scope is the one change that would create
# the gate/CI divergence this step exists to prevent, so the fix belongs on ci.yml and is filed as
# #2107; widening it here would close the gap by breaking the property.
#
# An interpreter without mypy is rejected during interpreter SELECTION (see `resolved_python`),
# not here: a refusal at this point skips every remaining candidate, which is the defect the ruff
# paragraph up there records having already paid for once.
step "MyPy type gate (config subpackage -- mirrors the blocking job in ci.yml)"
# `-P`, like every other -m in this file: without it a `mypy/` package sitting in the repo root
# shadows the real one, and this step reports PASS having checked nothing. Verified both ways. The
# version line is NOT printed here: printed inside the step it reaches neither the head nor the
# pasted tail, and because `cannot_run` exits at this step it would appear exactly when the gate
# ABORTS and never on a green run -- which is the run a reviewer is asked to paste.
# A POSITIVE CONTROL, because this file's own rule is "the instruments, before their numbers" and
# this step was the one instrument in the gate without one. A clean PASS is indistinguishable from
# a mypy that has been silenced. So make it fire on an error it must catch.
#
# THE PROBE LIVES INSIDE THE CHECKED SCOPE, and that is not tidiness. Written to a scratch dir its
# module name is `probe`, so it shares only the GLOBAL config with the real check -- and review
# measured three silencers that pass the real check while the probe still errors: a
# `[[tool.mypy.overrides]] ignore_errors` on `mfgarchon.config.*`, a file-level
# `# mypy: ignore-errors`, and an inline `# type: ignore`. In the scope, the per-module override is
# caught: the probe passes, which is the signal.
#
# WHAT IT STILL CANNOT CATCH, said plainly because the previous comment claimed otherwise: per-file
# and per-line silencing in the files under test. No probe can -- that is a property of those files,
# not of the configuration a probe shares. The earlier text named the `# type: ignore` sweep as
# covered and it was not.
#
# The control is version-independent on purpose: assigning a `str` to an `int`-annotated local and
# returning it fires under every mypy in the supported range. The one published discrimination
# example for this step -- reverting the `keys[: i + 1 : 1]` slice -- errors under 2.3.0 and passes
# under 1.20.2, both of which satisfy `mypy>=1.5`, so it cannot serve as the control.
# `$MYPY_PROBE` is declared and trapped beside PROBE_DIR at the top, not with a local `mktemp`:
# an unguarded `_probe=$(mktemp -d)` leaves `$_probe` EMPTY when mktemp fails, so the probe file is
# never written, mypy errors on a nonexistent path, and the control silently does not fire while
# the gate proceeds as though the instrument were verified -- a silent-instrument failure inside
# the instrument-verification code. It was also outside both traps and its `rm` sat after the `fi`,
# so the `cannot_run` path leaked it. The file records that lesson 200 lines up and this broke it.
printf '# MUTATED -- local_ci.sh mypy control. A leftover means a gate run was killed; DELETE THIS\n# FILE (rm -f mfgarchon/config/_gate_probe.py). The guard above prescribes `git checkout --`,\n# which cannot touch this one: it is untracked AND gitignored, so `git clean -n` will not even\n# list it, and following that instruction loops.\ndef _gate_probe() -> int:\n    x: int = "not an int"\n    return x\n' > "$MYPY_PROBE"
# The write must be CONFIRMED, not assumed. Two concurrent runs on one checkout -- the pre-push
# hook and a manual invocation is the realistic pair -- share this fixed path, and the loser can
# have its probe deleted between the write and the check: the `if` is then false, no `cannot_run`
# fires, and the gate proceeds as though the instrument were verified. Measured. A `$$` in the
# name would fix the race and reopen the leftover problem, which the .gitignore entry closes only
# because the path is fixed.
[[ -f "$MYPY_PROBE" ]] || cannot_run "the mypy control's probe could not be written to $MYPY_PROBE,
or was removed before it could be read -- a concurrent gate run on this checkout is the usual
cause. Nothing about the type gate was verified, so its result is not reported."
# GREP FOR THE PLANTED ERROR CODE, do not test mypy's exit status. Exit status answers "did mypy
# report something", and the control needs "did mypy report THE thing". A probe truncated mid-
# statement -- a partial write, a full disk -- exits 2 with `[syntax]`, which an exit-status test
# reads as INSTRUMENT VERIFIED while proving nothing about whether assignment errors are reported
# in this scope. `[[ -f ]]` above does not close it either: the marker is line 1, so a write that
# stops there leaves a syntactically valid, error-free, NON-EMPTY module, which `[[ -s ]]` would
# also wave through. Measured on four write outcomes; this grep is strictly stronger on every one.
# Capture, then match. `set -o pipefail` is on, and mypy exits 1 when it finds the error the
# control WANTS -- so `mypy | grep -q` returns mypy's 1 even on a match, and the control fires on
# every healthy run. Measured: the one-liner passes standalone and inverts inside this script.
_probe_out=$("$PY" -P -m mypy "$MYPY_PROBE" --follow-imports=silent 2>&1)
if ! grep -q '\[assignment\]' <<<"$_probe_out"; then
  rm -f "$MYPY_PROBE"
  cannot_run "the mypy control did not report the assignment error planted in $MYPY_PROBE. Either
mypy is silenced in mfgarchon.config -- check for an ignore_errors override or a disabled error
code -- or the probe was not written whole. Nothing about the type gate was verified."
fi
rm -f "$MYPY_PROBE"

_mypy_out=$("$PY" -P -m mypy mfgarchon/config --follow-imports=silent 2>&1)
_mypy_rc=$?
printf '%s\n' "$_mypy_out"
# CLASSIFY ON THE OUTPUT, NOT THE EXIT CODE. The first version branched on `rc >= 2` and was wrong
# in both directions, measured on this tree:
#
#   - `omegaconf` absent -- a DECLARED runtime dependency -- exits 1, not 2, so the environment
#     failure this branch exists for was never caught. The comment claimed it was.
#   - renaming `mfgarchon/config` away, an ordinary refactor on a perfect environment, exits 2:
#     the operator was told "ENVIRONMENT failure, install the dev extra", and the gate ABORTED the
#     remaining checks. Before this step existed the same rename gave FAIL -> GATE RED with correct
#     attribution and the gate carried on.
#
# mypy's exit code says whether it finished, not why. Its OUTPUT carries the distinction, so match
# that -- and default everything else, exit 2 included, to `check`, which is the pre-existing
# behaviour and the safe side of a wrong guess.
_mypy_env_fault=""
grep -q "Error importing plugin" <<<"$_mypy_out" && _mypy_env_fault="a mypy plugin failed to load"
# `import-not-found` alone is NOT enough: a typo'd import is a code error and belongs in GATE RED.
# It is an environment fault only when the module pyproject.toml DECLARES is the one missing, which
# is decidable -- so decide it rather than guessing from the message.
while read -r _m; do
  [[ -n "$_m" ]] || continue
  if grep -qE "^\s*[\"']${_m}([\"'~=<>!\[]|\s|$)" pyproject.toml; then
    _mypy_env_fault="the declared dependency '${_m}' is not installed"
    break
  fi
# FLATTEN FIRST. `pretty = true` wraps mypy's output, and the phrase this greps for is split by
# the wrap at some widths and not others -- measured on an unmodified tree, COLUMNS=100 gave 0
# matches and a misattributed GATE RED where the default width gave the correct exit 2. Worse,
# within ONE run at 80 columns two import sites wrapped differently and only one was seen. The
# predicate is decidable; extracting the module name from wrapped text is not.
done < <(tr '\n' ' ' <<<"$_mypy_out" | grep -oE 'library stub for module named "[^"]+"' | sed 's/.*"\(.*\)"/\1/' | cut -d. -f1)

if [[ -n "$_mypy_env_fault" ]]; then
  cannot_run "$_mypy_env_fault, so mypy could not analyse the subject -- this is not a type error in
the code. See its output above. Install the dev extra into $PY and re-run."
fi
check $_mypy_rc "mfgarchon/config type-checks clean (the gate that otherwise only runs on GitHub)"

step "Workflow integrity"
# Parsing is NOT sufficient and this check knows it: a workflow gutted down to one job,
# or one whose `needs:` points at a job that was deleted, parses perfectly. GitHub rejects
# a dangling `needs:` at load time and then NO job in that file runs on any event -- a
# whole workflow silently switched off. Both failure modes were shipped during this
# repo's CI cleanup, one after the other.
"$PY" -P -c "
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

# A ratchet whose measurement has gone blind reports a stable or FALLING count and reads exactly
# like success. Every ratchet below therefore carries a positive control, and the controls are run
# here rather than existing unrun: check_doc_api and capability_matrix have had one since they were
# written and this gate never invoked either. ~6.8 s for all five (three runs: 7.1 / 6.7 / 6.7),
# of which check_internal_deprecation is 5.0 s and check_citations adds 0.4 s.
# check_doc_api also self-tests inside --check-baseline, so it runs twice. Kept in this loop on
# purpose: this is the ONE visible place asserting that every instrument is controlled, and if that
# internal call is ever dropped the coverage would vanish with nothing here to say so.
step "Ratchet self-tests (the instruments, before their numbers)"
for _selftest in check_fail_fast check_doc_api check_assertion_strength check_internal_deprecation check_citations; do
  "$PY" "scripts/${_selftest}.py" --self-test || { check 1 "ratchet self-tests: ${_selftest} cannot see what it counts"; }
done
check 0 "every fast ratchet still detects what it claims to detect"

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

# CLAUDE.md names three quantities that must have exactly one owner (diffusion_from_volatility,
# fp_drift_coefficient, hjb_residual_norm) and nothing measured whether the restatements were
# growing. Bidirectional, like the capability matrix: a consolidation fails until the baseline
# records it. Exit 2 is distinct from exit 1 on purpose -- a search pattern that stops matching
# returns 0 hits, which reads exactly like clean code, so the checker refuses to report a verdict
# when its own sentinels do not fire.
step "Single-source ratchet"
"$PY" scripts/check_single_source.py --baseline scripts/single_source_baseline.json
check $? "no new site restating a single-owner quantity"

# Over 200 `path.py:NNN` citations sit in tracked prose and nothing checked them; 19 of the 39 that
# can be judged -- 49% -- point at a line their named symbol has moved away from (#2102). Checking
# the number is in range is nearly useless: exactly one of 226 points past EOF. So a citation counts
# as judged only when ITS OWN LINE names a symbol; nothing is borrowed from a neighbour, and the
# other 154 are recorded unadjudicable rather than passing. Two numbers are pinned, not one:
# `drifted` bidirectionally, and `adjudicable` against SHRINKING, because deleting the symbol name
# from the prose otherwise lowers `drifted` and reads as an improvement. The baseline records WHICH
# claims are drifted, not only how many: counts alone are satisfied by a compensating pair, and
# review shipped exactly that -- two rows hidden, two fresh ones added, gate green. 0.4 s.
#
# EXPECT THIS TO GO RED ON A VERSION BUMP. AGENTS.md step 2 collates `changelog.d/` into the exempt
# `CHANGELOG.md`; step 3 is re-recording the baseline. Also measured before shipping: replaying the
# last 40 commits on main, 5 would have gone red, and 3 of those on prose the author never opened.
# That is the cost this buys the coverage with, and it is why the failure names the citations and
# prints the command rather than only moving a number.
step "Citation ratchet"
"$PY" scripts/check_citations.py --check-baseline scripts/citation_baseline.json
check $? "no citation newly entered the review queue, and none left it unrecorded"

if [[ $FAST -eq 0 ]]; then
  # ~40 s. Not in --fast: every cell is a real coupled solve, so this is the one check
  # here that measures the product rather than the source. Counts of tests, issues or
  # fail-fast violations all move for reasons unrelated to whether anything solves;
  # this is the quantity that does not. Bidirectional -- a recovered cell fails until
  # the baseline records it, so a fix cannot land without saying so.
  step "Capability matrix (public solve surface vs external oracles)"
  # 92s, so it sits in the slow tier beside the matrix it guards rather than with the fast four.
  "$PY" scripts/capability_matrix.py --self-test
  check $? "the capability cells still go red under injected drift"

  "$PY" scripts/capability_matrix.py --check-baseline scripts/capability_baseline.json
  check $? "no capability change vs baseline"

  step "Test suite (CI marker set, xdist parallel, no coverage)"
  # PYTHONSAFEPATH as well as -P: `-P` is per-process and xdist's execnet workers do not inherit
  # it, so a repo-root `pytest/` package still reaches them. Measured on a probe that must fail:
  # `-P` alone under `-n` crashes every worker (no tests ran); with PYTHONSAFEPATH=1 the real
  # pytest runs and correctly reports it. Anything that forks needs the env var, not just the flag.
  # `--disable-warnings` suppresses the warnings SUMMARY, not the warnings: the count still lands
  # in the terminal tail ("N passed, M warnings"), and `-W error` promotion still works. Measured
  # on a green run: that block is 6,030 of 6,354 lines and 770,223 of 804,837 bytes -- 95.7% of
  # everything this gate prints. Without it the whole run is 324 lines / 34,614 bytes, which is
  # 0.53x a 64 KB pipe instead of 12.3x, and #2117's failure mode has nothing to grow into.
  #
  # It is a REPORT of a backlog, not a signal: 456 of those lines are this repository's own tests
  # calling its own deprecated `MFGProblem(geometry=, components=, ...)`. Printing the list every
  # run for a year has not retired one of them; #2119 is where they get counted instead.
  PYTHONSAFEPATH=1 "$PY" -P -m pytest tests/ -n auto \
    -m "$(cat "$(dirname "$0")/ci_markers.txt")" \
    -q --durations=10 --disable-warnings
  check $? "full suite"
else
  printf '\n\033[33mSKIPPED\033[0m test suite (--fast)\n'
fi

# Printed beside the suite result on purpose. "N passed" has never been the quantity worth
# growing, and printing it alone invites growing it; this is the one that says whether green
# means anything. Reports, does not gate -- measuring it costs a full suite run per mutation,
# so the gating lives in the weekly `test_discrimination.py --check-baseline` tier. (#1901)
if [[ $FAST -eq 0 ]]; then
  "$PY" scripts/report_discrimination.py || true
  "$PY" scripts/check_assertion_strength.py || true
fi

printf '\ngate interpreter : %s (%s)\n' "$PY" "$("$PY" -V 2>&1)"
printf 'gate ruff        : %s\n' "$("${RUFF[@]}" --version 2>&1)"
printf 'gate mypy        : %s\n' "$("$PY" -P -m mypy --version 2>/dev/null || echo unknown)"
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

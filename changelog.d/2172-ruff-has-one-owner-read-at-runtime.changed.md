- **ruff left `[dependency-groups].dev`** (Issue #2172). Its version has one owner,
  `.pre-commit-config.yaml`, one writer, `scripts/update_ruff_version.py`, and consumers that READ
  it at runtime — `ci.yml`'s quick-checks job already does exactly that:
  `pip install "ruff==$(python scripts/update_ruff_version.py --print-current)"`. A floor in the dev
  group was a second statement of the same value, and a resolver takes the newest release satisfying
  it: measured on `main`, a real `uv lock` resolved **0.16.5** against the pinned **0.16.0**, so a
  tracked lock would have handed every contributor a ruff the gate warns about. #2123 removed the
  last second pin for this reason and `update_ruff_version.py` still carries the sentence — *"a
  bumper that touches more than the one owner is how the owner stops being one."*
- Pinning `ruff==<pin>` in the dev group and having the bumper write both files was considered and
  rejected against that rule. It would have made the bumper touch two owners to keep them equal,
  where the repository's answer is that consumers read the one owner.
- `scripts/setup_development.sh` installs the pinned ruff on both its uv and pip branches, guarding
  the command substitution the way `ci.yml` does rather than inlining it bare, and **both** of
  `scripts/local_ci.sh`'s refusal sites name the command — the `MFG_PYTHON` branch as well as the
  candidate search, since `AGENTS.md` tells worktree users to set `MFG_PYTHON` and that is the
  branch a repeat contributor hits. Both name `uv pip install` **and** `pip install`, because
  `uv sync` and `uv venv` install no pip and a bare `pip` either fails or silently installs into a
  different environment.
- **No CI job depends on ruff arriving from the dev group** — which is the claim, narrower than the
  one first written here. `check-ruff-updates.yml` does run `ruff format .`, and installs the
  *latest* rather than the pin: correct for a bumper, and self-contained either way. The first
  version of this entry described that file as a `--print-current` match, from a substring search
  whose one acknowledged false positive was `ruff` inside `trufflesecurity/trufflehog` — a search
  shape that reports its own unreliability and was believed anyway.
- **Three contributor-facing entry points did rely on the dev group**, and the sharpest was
  fail-silent. `scripts/quick_type_check.py` printed "Ruff not available, skipping lint check", set
  `ruff_success = True` and reported overall success — in the repository whose gate step is named
  *"no new silent fallbacks"*, reachable in a standard install for the first time because of this
  change, and named in `setup_development.sh`'s own closing banner. It is a cannot-run now, with the
  install command. `make lint` and `make format` refuse the same way instead of dying on
  `ruff: command not found`.

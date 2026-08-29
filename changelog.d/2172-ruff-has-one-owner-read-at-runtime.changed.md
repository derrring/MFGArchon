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
- `scripts/setup_development.sh` installs the pinned ruff on both its uv and pip branches, and
  `scripts/local_ci.sh`'s refusal now names the command — under this change a fresh `uv sync` has no
  ruff, and that refusal is the first thing a contributor sees. No CI job loses ruff: the only job
  that runs it already installed the exact pin, and the `ruff` matches elsewhere were
  `--print-current` (reads, does not run) and a substring of `trufflesecurity/trufflehog`.

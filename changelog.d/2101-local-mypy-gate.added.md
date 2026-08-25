- **`scripts/local_ci.sh` now runs the blocking mypy gate** (Issue #2101, option 4). `ci.yml:168`
  runs `mypy mfgarchon/config --follow-imports=silent` as a **blocking** step, and nothing here
  mirrored it — measured, `grep -c mypy scripts/local_ci.sh` returned **0** against **23** for ruff.
  That made it the one gate that could not be reproduced or pre-checked before pushing, the mirror
  image of this repository's own warning that a GitHub-green PR has not had its tests run. Scope and
  flags are copied from `ci.yml` verbatim rather than chosen again, so the two cannot disagree about
  what "clean" means. A missing mypy is an environment failure (`cannot_run`), not a pass: reporting
  clean because the instrument is absent is the silent-instrument shape tracked under #1918. ~10 s on
  a ~150 s gate. Verified to discriminate: reverting one slice step in `omegaconf_manager.py` takes
  the step from `Success` to `Found 1 error`. This does not answer #2101's other question — whether
  the now-unbounded mypy should be pinned — which is deliberately separable.

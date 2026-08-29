- **`environment.yml` was not actually deleted.** The commit titled *"Delete environment.yml"*
  touched five files and the file was not among them: splitting the deletion out of an earlier
  commit restored it to the worktree, and the later `git add -A` saw an unchanged tracked file and
  staged nothing. Every consumer had been removed by then, the changelog announced the deletion, and
  `scripts/README.md` said the file no longer existed — so the branch would have merged a state
  strictly worse than `main`: the second manifest surviving, still declaring `ruff>=0.6.0`, with the
  ratchet that compared them gone. Found by adversarial review before merge.
- The check that missed it was `git status --porcelain | wc -l` → 0. That answers "is anything
  uncommitted", not "did the deletion happen"; the two agree in every case except the one that
  mattered. `git diff --cached --name-status -- environment.yml` answers the question asked.

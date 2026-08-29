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
- **The refusal message ran two commands.** Its text said ``uv sync`` and ``uv venv`` in backticks,
  inside a double-quoted bash string — where a backtick is command substitution. So the gate's
  cannot-run path executed `uv sync` and `uv venv`, created a `.venv`, installed 72 packages, and
  substituted their (empty) stdout into the sentence, which rendered as *"is read at runtime.  and
  install no pip"*. Found by rendering the message rather than reading it; a syntax check passes on
  it, and the earlier verification of this rework had only run `bash -n`. Both sites now use single
  quotes, and the same run confirms no `.venv` appears.
- **Every job that runs the suite needs ruff, and CI is what proved it.** Sixteen test files invoke
  `scripts/local_ci.sh`, which needs ruff, so the dependency chain is
  `workflow → pytest → test → local_ci.sh → ruff` — a workflow that needs ruff while never
  mentioning it. The survey behind "no CI job loses ruff" searched the workflows *for ruff* and
  therefore could not see it; three jobs went red on the runner with `No module named ruff`. The
  local gate cannot catch this class: it runs in an environment that already has ruff.
- That revises the cost of taking ruff out of the dev group from "one extra command for a fresh
  `uv sync`" to **five call sites**. They are still not five pins: each reads
  `update_ruff_version.py --print-current`, so they cannot drift from the one owner — which is the
  property that made this preferable to restating the version in `pyproject.toml`.
- A guard now fails when a job runs pytest without installing the pinned ruff, with one exemption
  carrying its reason: the bumper installs the *candidate* version, because validating a proposed
  bump is its job. The exemption is itself asserted to name a job that still exists. Three
  mutations killed, including a bare unpinned `pip install ruff`.

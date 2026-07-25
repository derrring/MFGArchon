- **`CLAUDE.md` now states what actually gates `main` — which is nothing** (Issue #1720). Two
  passages claimed branch protection enforced the PR workflow; measured, the effective-rules
  endpoint returns `[]`, classic protection returns 404, and the sole ruleset is
  `enforcement=disabled`. The local pre-push hook does not close the gap either: `local_ci.sh`
  contains no branch logic and no `no-commit-to-branch` hook is configured, so it gates test
  quality rather than which ref you are on. Also corrects a self-contradictory comment in
  `.pre-commit-config.yaml` that claimed `pre-commit install` does not wire pre-push — the
  `default_install_hook_types` line two lines below it does exactly that.

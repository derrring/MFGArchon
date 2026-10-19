- **AGENTS.md: what to search before filing an issue and before merging** (user rulings, 2026-10-09, 2026-10-18 and
  2026-10-19), in § *GitHub issue/PR management*.
  - Before filing: search issues by `family:` label and by code site or owner, comments and closed issues included,
    up to the search API's cap of 1000. Then the first rule that applies: a new issue when the finding needs a
    separate decision or folding would lose its priority or size, even if an open issue covers it; otherwise a fold
    into an open issue whose fix, or whose approved staged plan for the family, would remove it, as a comment with
    the reproducer and a checklist line under `## Folded findings`; a reopen of a closed issue with the same cause;
    during the #2512 phase, a line in the phase ledger for a low-priority behaviour defect off the default path, a
    duplicate or a dead surface, with a pin gap going in the PR body's "Known pin gaps"; and otherwise a new issue.
  - Before merging: search open issues for the functions and files the diff changes, close each one the diff fixes,
    by reference or with the commit named, and report each issue closed beyond the PR's `Fixes` list, or that there
    was none.

- **AGENTS.md: what to search before filing an issue and before merging** (user rulings, 2026-10-09 and
  2026-10-18), in § *GitHub issue/PR management*.
  - Before filing: search open issues by family label and by code site or owner. A finding is folded into an open
    issue whose fix, or whose approved staged plan for the family, would remove it: a comment with the reproducer and
    a line under `## Folded findings`. A low-priority defect off the default path, a duplicate or a dead surface is a
    line in the phase ledger. A new issue is opened only for a different fix, a separate decision, or a priority or
    size folding would lose, and a closed issue with the same cause is reopened.
  - Before merging: search open issues for the functions and files the diff changes, and close each one the diff
    fixes, by reference or with the commit named.

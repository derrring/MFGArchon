- **AGENTS.md: a Blocker's correction is re-reviewed whatever it touches** (user ruling, 2026-10-26), in
  § *Branches & PRs*, *Review before merge*.
  - Only a must-fix correction, after a MERGE-OK, merges without a further review when it touches only tests or
    prose; a test change still needs revert-red evidence.
  - This narrows the 2026-10-13 bullet, which read "a test-only or prose-only correction" and so also covered a
    Blocker answered by tests alone, against the Blocker bullet's "Fix it, then re-review".

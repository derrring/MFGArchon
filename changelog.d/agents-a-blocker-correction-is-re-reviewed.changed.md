- **AGENTS.md: a Blocker's correction is re-reviewed whatever it touches** (user ruling, 2026-10-26), in
  § *Branches & PRs*, *Review before merge*.
  - After a MERGE-OK, its must-fix and should-fix corrections merge without a further review, whatever they touch; a
    test change among them still needs revert-red evidence.
  - This replaces the 2026-10-13 bullet, which read "a test-only or prose-only correction" and so also covered a
    Blocker answered by tests alone, against the Blocker bullet's "Fix it, then re-review".

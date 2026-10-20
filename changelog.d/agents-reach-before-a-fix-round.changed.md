- **AGENTS.md: a code change ordered between reviews owes its reach first** (user ruling, 2026-10-20), in
  § *Branches & PRs*, *Review before merge*.
  - The order, whoever gives it, states the reach: the inputs each condition the change alters admits, and every
    reader of each value it changes. The session that pushes does not push an order without one.
  - Before the next review the change is measured at the base sha and the head on one case of each input class, and
    the PR body states each case beside its command and both shas. A missing statement is a must-fix.
  - A defect present at the base sha, found once review 1 has begun, goes through the filing rule instead of a fix
    round, unless its fix is a class fix with its own class line.
- **AGENTS.md: the 2026-10-13 re-review rulings land** (#2641).
  - A narrow re-review also reads the PR body's class line and gating claims at the new head. When the only blockers
    were user-facing false statements, it reads only the corrected text.
  - The PR body states a class line where the PR fixes a defect. The review record is the one number written with no
    command. Measurements no gating claim rests on go on an issue the PR addresses.
  - A test-only or prose-only correction merges without a further review, and a test change needs revert-red evidence.

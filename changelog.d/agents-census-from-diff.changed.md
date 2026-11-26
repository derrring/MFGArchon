- **AGENTS.md: a census draws the change's callers, and a stated behaviour meets a mutant** (user ruling,
  2026-11-26), in § *Local patches are the anti-pattern*, § *Closing out a fix* and § *Review before merge*.
  - A census's candidates include every caller and reader of each symbol the change deletes, moves or alters in
    behaviour, drawn by script at plan and again from the diff at the head. The plan names what the census holds
    constant, and a member the head's draw finds that the plan's did not is a missed member.
  - Before review 1, each sentence of the PR body that says what the library does on some input names a mutant and
    the test it turns red, or goes under *Known pin gaps* with its reason.
  - A claimed fix listed under *Known pin gaps* as having no oracle yet, with the issue that records it, is not a
    Blocker.

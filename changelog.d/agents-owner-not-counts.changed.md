- **AGENTS.md: a class line names the owner and the check that holds it, not member counts** (user ruling,
  2026-11-27), in § *Local patches are the anti-pattern*, § *A PR stays at its plan*, § *Closing out a fix* and
  § *Review before merge*.
  - A class line states its fact, its owner, the check that holds it after the merge (a structural one-owner test
    where the deciding places can be found by reading the code, otherwise pins over each class of input the census
    found) and class or partial fix. The census's list goes in the plan, not the PR body; the counts at two shas,
    the census's control in the count and the table on the tracking issue are no longer asked for.
  - The PR body gives each behaviour as a row of a table: the sentence, the mutant, the test it turns red.
  - A reach case between reviews goes in an evidence comment, which the body links.

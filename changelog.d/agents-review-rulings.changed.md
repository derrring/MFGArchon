- **AGENTS.md: a re-review is narrow, and a PR body carries claims, not tallies** (user rulings, 2026-10-13).
  - § *Review before merge*:
    - Review 1 reads the whole PR.
    - Every later review reads only:
      - each finding of the review before it, with the change that answers it;
      - every file the fix round touched;
      - a regression pass over the PR's own tests and the gate's result.
    - The PR body states what changed, the class line, and the evidence for each claim that gates the merge. Each number sits beside the command that printed it and the sha it ran at.
    - Other measurements go in a comment on the PR's issue, which the body links.
  - § *Local patches are the anti-pattern*:
    - A partial fix's remaining members are listed in an open issue that the PR does not close.
    - A staged plan's record says who approved it, someone other than its author.

    These are #2636's S1 and S3.

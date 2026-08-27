- **The ruff pin is located by its `repo:` line, not by a bare occurrence of the repo URL** (Issue
  #2139). `.pre-commit-config.yaml` pins more than one repository, and the expression routing from
  the ruff repo to its `rev:` began at any occurrence of `astral-sh/ruff-pre-commit`, which a
  comment in an *earlier* block naming that URL satisfies. Traced on such a config, that was three
  failures rather than one: the reader returned `6.0.0` for a pin of `0.16.0`; `re.sub` replaces
  every match, so a bump rewrote the earlier block's `rev:` as well, sending `pre-commit-hooks` to
  a version that is not one of its tags, after which `pre-commit` can fetch no hook environment at
  all; and the post-bump check raised nothing, because the corruption had written the asked-for
  version into the block it inspected. That is #2123's damage arriving from the reader's side and
  reported as success.
- **Anchoring only the post-bump check does not fix it**, which is worth recording because it was
  the first attempt: the check then reads the correct block, finds the correct value, and passes
  while the corruption continues — behaviour byte-identical to the unfixed state. The tests assert
  on the resulting config for that reason.
- The check's single diagnostic became three. One message for "no block recognised", "no `rev:`
  line" and "the pin reads something else" asserted the second over files that plainly had a
  `rev:`, sending the reader to inspect the half that was fine.

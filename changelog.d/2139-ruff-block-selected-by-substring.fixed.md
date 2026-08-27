- **The ruff pin is located by its `repo:` line, not by a bare occurrence of the repo URL** (Issue
  #2139). `.pre-commit-config.yaml` pins more than one repository, and the expression that routes
  from the ruff repo to its `rev:` started at any occurrence of `astral-sh/ruff-pre-commit` — a
  comment in an *earlier* block naming that URL satisfied it. Traced on that config, the
  consequences were three, not one: `get_current_version` returned `6.0.0` for a pin that read
  `0.16.0`; `re.sub` replaces every match, so a bump rewrote the earlier block's `rev:` as well,
  sending `pre-commit-hooks` from `v6.0.0` to a ruff version that does not exist as one of its
  tags, after which `pre-commit` can fetch no hook environment at all; and the post-bump check did
  not fire, because the corruption had just written the asked-for version into the very block it
  was inspecting. That is #2123's damage arriving from the reader's side, reported as success. The
  expression is now anchored to a line that *is* the ruff `repo:` declaration, which a comment
  cannot satisfy, while still spanning a comment between `repo:` and `rev:` — the shape #2123 was
  about. `tests/unit/test_ruff_block_selected_by_repo_line_2139.py` asserts on the resulting config,
  not on the selector: anchoring only the check leaves the corruption in place and the check green.

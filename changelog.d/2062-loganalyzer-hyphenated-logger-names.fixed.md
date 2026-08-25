- **`LogAnalyzer` reads back every name the writer can emit** (Issue #2062). Its logger-name group was
  `([^-]+?)`, which by construction cannot match a hyphen, so any line whose logger name carried one
  was skipped with no diagnostic — `parse_log_file` simply reported a lower count. The group is now
  bounded by the ` - ` separator the writer emits. The same change closes the second hole the issue
  named: an empty or whitespace-only message under `include_location=False` leaves the line ending in
  `-` with nothing after it, which a literal trailing space could not match; the separator is now
  `\s?` — and deliberately not `\s+`, which was measured to eat the message's own leading whitespace
  and to swallow the `[location]` field into an empty message. Swept group-by-group against the
  previous regex over 540 lines (6 logger names × 5 levels × 9 message shapes × `include_location`):
  **0 regressions, 0 differing groups, 220 newly matched — 160 from the logger group, 60 from the
  `\s?`**, and 40 of the 220 fall on logger names with no hyphen at all.
- **Scope of the workflow-logger claim.** Every workflow logger *is* named `mfg_workflow.<uuid>`, so
  a uuid's hyphens would have made such a file parse to zero — but **no such file has ever been
  written**: `WorkflowManager._setup_logging` passes `log_file=None`, and `common.py`'s `FileHandler`
  sits inside an `if not logger.handlers:` branch its own comment records as dead for production
  callers. This fixes a latent defect, not an observed one. A user-chosen
  `get_logger("my-experiment")` is the live case.
- **Still unreadable, and not fixed here**: `.mfg_sweeps/parameter_sweep.log` — 11,566 lines, the
  largest log in the repository — parses to **0 entries before and after**, because its pre-#2058
  writer emitted `,%(msecs)03d` and the timestamp group this change does not touch rejects it.
  Multi-line messages likewise remain unparsed, a structural limit of line-at-a-time reading.

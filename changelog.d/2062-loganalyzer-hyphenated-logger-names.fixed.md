- **`LogAnalyzer` reads back every name the writer can emit** (Issue #2062). Its logger-name group was
  `([^-]+?)`, which by construction cannot match a hyphen, so any line whose logger name carried one
  was skipped with no diagnostic — `parse_log_file` simply reported a lower count. Every workflow
  logger is named `mfg_workflow.<uuid>` and a uuid always has hyphens, so those files parsed to
  **zero** entries; a user-chosen `get_logger("my-experiment")` produced a log nothing could read
  back. The group is now bounded by the ` - ` separator the writer emits. The same change fixes the
  second hole the issue named: an empty or whitespace-only message under `include_location=False`
  leaves the line ending in `-` with nothing after it, which a literal trailing space could not
  match; the separator is now `\s?` — and deliberately not `\s+`, which was measured to eat the
  message's own leading whitespace and to swallow the `[location]` field into an empty message.
  Swept group-by-group against the previous regex over 540 lines (6 logger names × 5 levels × 9
  message shapes × `include_location`): 0 regressions, 0 differing groups, 160 newly matched, all on
  hyphenated names. Multi-line messages remain unparsed — a structural limit of line-at-a-time
  reading, unchanged here.

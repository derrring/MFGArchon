`LogAnalyzer` could never parse a `CRITICAL` log line, and `CRITICAL` is one of the two levels its
failure reports select for (#1918). The writer formats with `%(levelname)-8s`; `CRITICAL` is exactly
8 characters, so it is the one level that arrives with no padding, leaving a single space before the
separator where the reader's `(\w+)\s+ - ` needed two — one for `\s+` and one literal. Every shorter
level is padded and matched. Measured end-to-end through the real `MFGFormatter` and the real
`LogAnalyzer`: 5 levels emitted, 4 parsed, `CRITICAL` the only miss. `get_summary_statistics` and
`find_error_patterns` both select `level in ("ERROR", "CRITICAL")`, so the highest severity was
absent from both — a gap on the error channel, which reads as "no critical events" rather than as a
parse failure.

The level field now matches `(\w+)\s+- ` with a **literal** trailing space. `\s+` there would be greedy and
eat whitespace belonging to the message — leading indentation, and on an empty message under
`include_location=True` the `[location]` field itself, which parses as `message="[solver.py:42]"`,
`location=None`. The literal space is byte-identical to the previous behaviour by construction, not
by sweep: `([^-]+?)` cannot contain a dash, so neither group can backtrack, and both forms consume
exactly one space after the dash.

This fixes one reader against one writer. Two other writers — `LoggingHook`
(`hooks/visualization.py:328`) and `setup_workflow_logging` (`workflow/common.py:107`) — install
formatters with no `datefmt`, so their `asctime` carries milliseconds and `LogAnalyzer` returns zero
entries for every line they write, at every level. That is a larger instance of the same shape and
is #2058, for which this change is a prerequisite: with `datefmt` added but this fix absent, their
unpadded levelname leaves a single space at *every* level and none of the five parse.

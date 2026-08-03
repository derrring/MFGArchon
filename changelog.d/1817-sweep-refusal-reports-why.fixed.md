- **The discrimination sweep's two refusals now print the pytest output they refused on** (Issue
  #1817). `_pytest` parsed failure names out of `proc.stdout` and discarded the rest, so a weekly
  run that aborts on a pre-existing failure named the node id and nothing else -- no assertion, no
  traceback. Diagnosing one red therefore cost a full CI round-trip, and on 2026-08-03 it bought
  nothing: the same argv on the same commit gave `5833 passed` locally, which makes the runner's
  own output the only evidence that the failure existed. `Run` now carries the output and both
  `sys.exit` paths append a `_failure_excerpt`, anchored at the `FAILURES` banner so the progress
  dots do not crowd out the reason, tail-truncated at 120 lines because `-q` puts the short summary
  last, and explicit about how many lines it dropped.

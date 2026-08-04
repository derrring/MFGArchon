- **`test_auto_selection_logging` measured global logging configuration, not backend
  auto-selection** (Issue #1821). It asserted that `capfd`'s stdout was non-empty after
  `create_backend("auto")`. The message is a `logger.info`, and `mfg_logging/logger.py` attaches a
  `StreamHandler(sys.stdout)` **once**, at the first `get_logger` for a name -- binding whatever
  `sys.stdout` was then, caching the logger, and setting `propagate = False`. So whether the record
  reaches file descriptor 1 depends on which module imported first. On CI both streams came back
  empty and the test was red there while green locally, which blocked the weekly discrimination
  sweep (#1817).
  Worse, the assertion it failed on came *first*, so **the auto-selection half was never reached on
  CI at all** -- nothing had ever checked that `create_backend("auto")` returns what the documented
  torch > jax > numpy priority says it should.
  Split into the two claims it was conflating: one test asserts the selection against
  `get_available_backends()`, the other captures the `mfgarchon.backends` logger's own records with
  a handler attached for the call (`propagate = False` is also why `caplog` cannot see it).
  Mutation-verified: ignoring the priority order fails the first, removing the log fails the second.

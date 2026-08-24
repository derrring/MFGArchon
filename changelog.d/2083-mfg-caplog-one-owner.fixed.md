- **One owner for mfgarchon log capture in tests** (Issue #2083). `MFGLogger` disables propagation
  (`logger.py:211`), and what pytest does about that differs by version: 8.4.1 attaches its capture
  handler to the root logger only and sees **no** mfgarchon record whatever its creation site, while
  9.1.1 also attaches to every non-propagating logger that already exists when `catching_logs.__enter__`
  runs. That sweep runs once per test phase, so on 9.1.1 a logger born mid-solve stays invisible — 34
  of the package's 104 `get_logger` calls are inside a function — and whether a test passes depends on
  what ran before it in the same worker: the gfdm drift test fails run alone and passes when a sibling
  solve ran first. The new `mfg_caplog` fixture (`tests/conftest.py`) attaches to the emitting logger on
  demand and depends on neither the version nor the order. A misspelt logger name still captures nothing silently: two
  guards against that were built and both removed, each having re-created the order-dependence the
  fixture exists to remove (8 of the 10 logger names this package uses are not module paths, so a
  static check cannot decide them either). The limitation is pinned by a test and the discipline that
  replaces it — pair an absence assertion with a presence assertion on the same name — is documented. It replaces six hand-rolled collectors and two `logger.warning` spies, and the four
  mass-drift tests plus `test_auto_mode_verbose_shows_selection` — whose assertion had an
  `or result is not None` branch that could not fail — now assert on the log itself.

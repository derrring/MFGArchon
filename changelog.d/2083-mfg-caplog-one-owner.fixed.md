- **One owner for mfgarchon log capture in tests** (Issue #2083). `MFGLogger` disables propagation
  (`logger.py:211`), and what pytest does about that differs by version: 8.4.1 attaches its capture
  handler to the root logger only and sees **no** mfgarchon record whatever its creation site, while
  9.1.1 also attaches to every non-propagating logger that already exists when `catching_logs.__enter__`
  runs. That sweep runs once per test phase, so on 9.1.1 a logger born mid-solve stays invisible — 34
  of the package's 104 `get_logger` calls are inside a function — and whether a test passes depends on
  what ran before it in the same worker: the gfdm drift test fails run alone and passes when a sibling
  solve ran first. The new `mfg_caplog` fixture (`tests/conftest.py`) attaches to the emitting logger on
  demand and depends on neither the version nor the order. It refuses a logger name that is neither a module in the
  package nor one the package has already handed out — a static criterion, so a misspelt name can no
  longer satisfy an absence assertion and the verdict does not move with test order. It replaces six hand-rolled collectors and two `logger.warning` spies, and the four
  mass-drift tests plus `test_auto_mode_verbose_shows_selection` — whose assertion had an
  `or result is not None` branch that could not fail — now assert on the log itself.

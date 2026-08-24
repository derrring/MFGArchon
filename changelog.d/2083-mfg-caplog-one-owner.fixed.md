- **One owner for mfgarchon log capture in tests** (Issue #2083). `MFGLogger` disables
  propagation, so whether pytest's `caplog` sees an mfgarchon record depends on the pytest
  version *and* on when the logger was created: 8.4.1 attaches its capture handler to the root
  logger only and sees nothing, while 9.1.1 also attaches to every non-propagating logger that
  **already exists** at phase start — so a module-level `get_logger` is visible and one obtained
  inside a function (34 call sites, `fp_gfdm.py:575` among them) is not. Measured: the gfdm drift
  test fails run alone under 9.1.1 and passes when a sibling solve runs first. The new
  `mfg_caplog` fixture (`tests/conftest.py`) attaches to the emitting logger on demand and
  depends on neither; it replaces six hand-rolled collectors and two `logger.warning` spies, and
  the four mass-drift tests plus `test_auto_mode_verbose_shows_selection` — whose assertion had
  an `or result is not None` branch that could not fail — now assert on the log itself.

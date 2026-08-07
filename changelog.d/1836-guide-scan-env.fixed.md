- **The nightly can read the live library end to end, and the deprecation-guide tests state the
  environment they need instead of erroring in it** (Issue #1836, refs #1830, #1713). Reading the
  runtime registry made a complete walk a precondition: `scan_all_deprecations` refuses a partial
  one, because the guide is user-facing and a guide generated from half a tree teaches half an API
  as if it were the whole one. From #1830 the `registry` fixture therefore raised
  `IncompleteScanError` at setup and three tests errored every night, while the local gate — which
  runs in a complete environment — stayed green. Measured on `b88b4781` in a fresh
  `.[dev,numerical]` environment, the walk cannot import **20** modules: 19 under the frozen
  paradigms, and `mfgarchon.backends.numba_backend`, which raises a deliberate `ImportError`
  without numba. The twentieth is live library, so the only automatic full-suite tier could not read
  the package end to end — the nightly now installs `numba`, for the same measured reason
  `deprecation-check.yml` already carried it, and still not torch. `discrimination.yml` — the weekly
  sweep, and a second automatic tier that runs `pytest tests` in full — had the identical hole and
  gets the same one-word fix. The fixture skips with the reason
  and the remedy, and only when every unreadable module is inside a frozen paradigm: a live module
  that will not import still raises, since turning a real breakage into a skip is how a suite goes
  quiet about what it was built to catch. The scope rule is read from the ratchet that owns it
  (`check_internal_deprecation.FROZEN`) rather than restated, and both branches of the guard are
  pinned against the nightly's own 20-module failure.
  Installing numba is not inert: `USE_NUMBA` flips to True and three `@njit` kernels become
  compiled. Measured rather than assumed — collection is identical in all four shards with and
  without it, and no test in the nightly's marker selection branches on numba availability, so
  the change moves that tier toward the local gate, whose environment already has it.


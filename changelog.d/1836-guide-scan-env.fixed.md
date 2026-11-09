- **The nightly can import the live library end to end** (Issue #1836, refs #1830). Measured on `b88b4781` in a
  fresh `.[dev,numerical]` environment, a walk over the package cannot import **20** modules: 19 under the frozen
  paradigms, and `mfgarchon.backends.numba_backend`, which raises a deliberate `ImportError` without numba. The
  twentieth is live library, so the only automatic full-suite tier could not read the package end to end — the
  nightly now installs `numba`, and still not torch. `discrimination.yml` — the weekly sweep, and a second automatic
  tier that runs `pytest tests` in full — had the identical hole and gets the same one-word fix.
  Installing numba is not inert: `USE_NUMBA` flips to True and the `@njit` kernels become
  compiled. Measured rather than assumed — collection is identical in all four shards with and
  without it, and no test in the nightly's marker selection branches on numba availability, so
  the change moves that tier toward the local gate, whose environment already has it.

- **`mfgarchon.backends` no longer promises a tier list the implementation abandoned** (Issue #1922).
  `create_backend(None)` has returned `NumPyBackend` unconditionally since #1921, while
  "Tiered auto-selection priority: torch > jax > numpy" survived in **four** live places — the
  module docstring, `create_backend`'s summary line, its `Args` block, and the `>>>` Example a
  reader would copy — plus twice more in `docs/user/guides/backend_usage.md`. All six corrected.
  Marked as corrections rather than deleted: this was public API documentation, believed for long
  enough to matter.

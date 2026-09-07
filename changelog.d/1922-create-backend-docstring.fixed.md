- **`mfgarchon.backends`'s own docstrings no longer promise a tier list the implementation
  abandoned** (Issue #1922). `create_backend(None)` and `create_backend("auto")` have returned
  `NumPyBackend` unconditionally since #1921, while "Tiered auto-selection priority:
  torch > jax > numpy" survived in four live places in `backends/__init__.py` — the module
  docstring, `create_backend`'s summary, its `Args` block, and the `>>>` Example a reader copies.
  All four corrected, marked rather than deleted, because this was public API documentation
  believed long enough to matter.

  **Scoped to the source docstrings on purpose.** `docs/user/guides/backend_usage.md` makes the
  same promise many more times and mostly without the phrase — as `create_backend("auto")` beside
  "optimal", "prefers CUDA", "usually selects torch_mps", and in output examples that print a
  backend name the call cannot return. Three attempts to patch it inside this PR each missed sites
  and one left a heading contradicting its own section, so the guide is deliberately untouched here
  and filed separately. A lexical search cannot enumerate that population; the query that can is
  the call itself.

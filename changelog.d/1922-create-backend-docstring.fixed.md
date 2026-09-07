- **`create_backend`'s docstring no longer promises a tier list the implementation abandoned**
  (Issue #1922). It advertised "Tiered auto-selection priority: torch > jax > numpy" in two places
  while `create_backend(None)` has returned `NumPyBackend` since #1921 — public API documentation
  telling readers that auto-selection would reach for an accelerator.

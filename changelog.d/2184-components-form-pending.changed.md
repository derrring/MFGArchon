- **The components= form warns that it is planned to be replaced, not that it is deprecated** (Issue #2184).
  `MFGProblem(geometry=, components=, ...)` and `MFGProblem(spatial_bounds=, components=, ...)` now emit a
  `PendingDeprecationWarning`: "The components= form of MFGProblem (geometry= or spatial_bounds=, components=, ...)
  is planned to be replaced by the model/domain/conditions form, MFGProblem(model=, domain=, conditions=). Nothing
  needs to change now." The model/domain/conditions form is not fixed yet, so the warning names no version, does not
  tell a caller to migrate, and the form has no entry in the deprecation registry (user ruling, 2026-10-12). The old
  text also named `geometry=` to a caller who had passed `spatial_bounds=`. `docs/user/CONVENTIONS.md`,
  `docs/user/LEGACY_PARAMETERS.md` and the test comments that called the form deprecated now say the same.

  **Who is affected.** `PendingDeprecationWarning` is not a subclass of `DeprecationWarning`, and Python's default
  filters ignore it everywhere.
  - A script that calls the form at top level printed the old `DeprecationWarning` (the default filters show one
    attributed to `__main__`), and now prints nothing. `-W default::PendingDeprecationWarning` shows the new one.
  - Code that catches the call's warning as a `DeprecationWarning` -- `pytest.warns(DeprecationWarning)`, a
    category check under `warnings.catch_warnings` -- no longer matches it: the constructor emits no
    `DeprecationWarning` now.
  - A call from inside an imported module was already silent under the default filters, and still is.

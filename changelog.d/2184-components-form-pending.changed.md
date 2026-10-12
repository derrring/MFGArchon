- **The components= form warns that it is planned to be replaced, not that it is deprecated** (Issue #2184).
  Every `MFGProblem` construction that passes none of `model=`, `domain=` and `conditions=` -- by `geometry=`,
  `spatial_bounds=` or `network=`, with or without `components=` -- now emits a `PendingDeprecationWarning`: "The
  components= form of MFGProblem is planned to be replaced by the model/domain/conditions form, MFGProblem(model=, domain=,
  conditions=). Nothing needs to change now." The model/domain/conditions form is not fixed yet, so the warning
  names no version, does not tell a caller to migrate, and the form has no entry in the deprecation registry (user
  ruling, 2026-10-12). It names the form and none of its routes: the old text named `geometry=` to a caller who had
  passed `spatial_bounds=`. `docs/user/CONVENTIONS.md`, `docs/user/LEGACY_PARAMETERS.md`,
  `docs/user/GEOMETRY_FIRST_API_GUIDE.md` and the test comments that called the form deprecated now say the same.

  **Who is affected.** `PendingDeprecationWarning` is not a subclass of `DeprecationWarning`. Python's default
  filters ignore it; pytest shows it.
  - A call made from code in `__main__` -- a script's top level or a function the script defines -- printed the
    old `DeprecationWarning` under Python's default filters (they show one attributed to `__main__`), and now prints
    nothing. `-W default::PendingDeprecationWarning` shows the new one. `-W error::DeprecationWarning:__main__`
    raised on that call and no longer does.
  - Code that catches the call's warning as a `DeprecationWarning` -- `pytest.warns(DeprecationWarning)`, a
    category check under `warnings.catch_warnings` -- no longer matches it: the constructor emits no
    `DeprecationWarning` now.
  - A downstream test suite run by pytest with no warning filter of its own still sees the call's warning in its
    summary, as a `PendingDeprecationWarning`.
  - A call from inside an imported module was already silent under Python's default filters, and still is. That
    includes the library's own subclasses that use the form, `StochasticMFGProblem`, `MultiPopulationMFGProblem` and
    `NetworkMFGProblem`, whose warning is attributed to the library.

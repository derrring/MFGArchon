- **BC geometric-operation mapping is now tested in a tier that runs** (Issue #1736) —
  the assertions lived in `bc_utils.py`'s `__main__` block, which could not execute under
  either invocation (`factories` has never existed at that path) and had been dead since
  at least the package rename. Promoted to `tests/unit/test_geometry/`, extended to every
  branch including the absent-BC default, and the dead block deleted.

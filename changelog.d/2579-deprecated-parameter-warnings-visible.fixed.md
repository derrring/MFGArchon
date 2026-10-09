- **A call that still passes a deprecated parameter now warns where it can be seen** (#2579). Two
  filters hid it from the test suite and its warning census:
  - `pytest.ini` ignored every `Parameter … is deprecated` warning. The filter is gone.
  - Under two stacked `@deprecated_parameter` decorators, the inner warning was attributed to
    `mfgarchon/utils/deprecation.py`, the outer wrapper's frame, so pytest.ini's
    `ignore::DeprecationWarning:mfgarchon.*` filtered it. `deprecated_parameter` now skips every frame of
    its own module (`deprecation.WRAPPER_FRAMES`), and the warning names the caller's file and line.
    That holds for users too. Five functions stack them: `TensorProductGrid.__init__`,
    `WeakFormFPSolver.solve_fp_system`, `FPNetworkSolver.solve_fp_system`, `BlockIterator.__init__` and
    `UnstructuredMesh.visualize_mesh`.

  The test call sites this surfaced are migrated: `damping_factor` → `relaxation` (16), the redundant
  `TensorProductGrid(dimension=)` dropped (8), and `drift_field` → `potential_field` (2). The three tests
  of `dimension`'s own validation now assert its warning with `pytest.warns`.

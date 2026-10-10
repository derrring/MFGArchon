- **A call that still passes a deprecated parameter now warns where it can be seen** (#2579). Two
  filters hid it from the test suite and its warning census:
  - `pytest.ini` ignored every `Parameter … is deprecated` warning. The filter is gone.
  - Under any other wrapper from `mfgarchon/utils/deprecation.py`, the warning was attributed to that
    wrapper's frame. The suite's two module filters then dropped it: pytest.ini's
    `ignore::DeprecationWarning:mfgarchon.*`, and the same filter in `tests/conftest.py`'s autouse
    `suppress_warnings` fixture. Both stay. The other
    wrapper is either a second `@deprecated_parameter` or `retired_parameters`, which
    `retired_volatility_keywords` uses.
    - Measured at the base: 10 (function, parameter) pairs in 7 functions were hidden this way.
    - The 7 functions: `TensorProductGrid.__init__`, `WeakFormFPSolver.solve_fp_system`,
      `FPNetworkSolver.solve_fp_system`, `BlockIterator.__init__`, `UnstructuredMesh.visualize_mesh`,
      `FPFDMSolver.solve_fp_system` and `FPSLSolver.solve_fp_system`.
    - In three of them (`BlockIterator`, `FPNetworkSolver` and `WeakFormFPSolver`), the outer warning was
      hidden too.

    `deprecated_parameter` now skips every frame of its own module (`deprecation.WRAPPER_FRAMES`). The
    warning names the file and line of the call that passed the deprecated keyword. For a user that holds
    when they call the function directly. A keyword they pass through another library function's
    `**kwargs` is still attributed to the library line that forwards it; for example,
    `BlockGaussSeidelIterator(..., damping_factor_M=...)` names `block_iterators.py`. In a user's own
    pytest run that warning is shown at the library line. In this repository's suite the two module
    filters hide it.

  The test call sites this surfaced are migrated: `damping_factor` → `relaxation` (16), the redundant
  `TensorProductGrid(dimension=)` dropped (8), `drift_field` → `potential_field` (2), and one slow-tier
  `TensorProductGrid(num_points=)` → `Nx_points`. The three tests
  of `dimension`'s own validation now assert its warning with `pytest.warns`.

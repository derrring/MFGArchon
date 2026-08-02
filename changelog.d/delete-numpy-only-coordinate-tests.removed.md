- **Deleted `tests/unit/test_visualization/test_coordinate_transforms.py`** — 18 tests, 375 lines,
  none of which could be reddened by any change to this repository. The file imports only `pytest`
  and `numpy`; every call in every test is a NumPy or Python builtin. It asserts that
  `np.meshgrid` returns the shapes NumPy documents, that `(x - min) / (max - min)` lands in
  $[0, 1]$, and that `np.argmin(|x - target|)` finds the nearest index.

  There is no production counterpart: `mfgarchon/visualization/` contains `convergence_plots.py`,
  `extract.py` and `vtk_export.py`, and no coordinate-transform function at all. The tests were not
  covering an untested module — there was never a subject.

  The `grid_2d_time` fixture is kept: `density_2d_gaussian` and `value_function_2d` depend on it.

- **Deleted `tests/unit/test_utils/test_gradient_notation_standard.py`** — 11 tests, 380 lines,
  imports only `pytest` and `numpy`. This one is worse than the file above, because it claims to
  be a regression test for a real silent-failure bug and pins none of it.

  Its stated subject is that "all HJB solvers use tuple-indexed derivative dictionaries", guarding
  Bug #13 (user code sent `{"dx": ..., "dy": ...}` while the Hamiltonian read `{"x": ..., "y": ...}`,
  so the control term was silently zero). What it actually asserts:

  - `test_bug13_scenario_prevented` defines its own `hamiltonian_string_keys` inside the test body,
    feeds it the wrong keys, and asserts `dict.get()` returned the default.
  - `test_hjb_gfdm_compliance` hand-writes a dict literal; its own docstring says "This simulates
    what hjb_gfdm.py:1544-1556 does" and "Full integration test should verify tuple notation in
    actual solver runs".
  - `test_no_silent_failures_with_typo` asserts `d.get(missing, 0.0) == 0.0` and that `d[missing]`
    raises `KeyError` — Python's dict contract.
  - `test_tuple_keys_are_hashable_and_immutable` asserts that tuples are immutable.

  If `hjb_gfdm.py` switched to string keys tomorrow, all 11 stay green. The file made Bug #13 look
  pinned while pinning nothing, which is worse than an absent test.

  The convention **is** pinned, elsewhere and for real — but not where this fragment first said.
  Mutating `gfdm_strategies.py:931` to emit string keys and running the five files originally cited
  here gives **134 passed**. They cannot fail: `test_hjb_gfdm_solver.py:424` is `if (1,) in derivs:`
  with the assertion at :425 *inside* the guard, so string keys make the body vacuous rather than
  raising, and `test_collocation_gfdm_hjb.py:164` carries the same shape. The file that actually
  kills that mutation is `tests/unit/test_alg/test_hjb_gfdm_bc_newton_residual.py` — 9 failed under
  it, 9 passed after `git checkout --`, which is the counterfactual this claim owes.

  The five guarded assertions are their own defect and are filed as #1799: a convention check
  written as `if <convention holds>: assert <consequence>` is self-satisfying, the #1714/#1715
  family with a different mechanism.

  Known false negative, left for #1800: `tests/unit/test_alg/test_bug15_sigma_fix.py` is the same
  construct as the file deleted above — it re-implements the production dispatch inline and asserts
  on its own copy, and two mutations of the real `_get_sigma_value` leave all four of its tests
  green. It survived the read-through because its docstring contains the literal string
  `mfgarchon/alg/numerical/hjb_solvers/hjb_gfdm.py:1573-1583`, so a substring grep matches it where
  an import-parse does not. Recorded rather than swept in, since the Bug #15 convention is genuinely
  pinned by `test_hjb_gfdm_bug15.py`.

  Net for this change: **755 test lines deleted, −717 for the branch**; **6205 → 6176 collected**
  (the two files carry 29 tests between them).

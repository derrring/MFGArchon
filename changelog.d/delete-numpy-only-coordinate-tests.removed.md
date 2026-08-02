- **Deleted `tests/unit/test_visualization/test_coordinate_transforms.py`** — 18 tests, 375 lines,
  none of which could be reddened by any change to this repository. The file imports only `pytest`
  and `numpy`; every call in every test is a NumPy or Python builtin. It asserts that
  `np.meshgrid` returns the shapes NumPy documents, that `(x - min) / (max - min)` lands in
  $[0, 1]$, and that `np.argmin(|x - target|)` finds the nearest index.

  There is no production counterpart: `mfgarchon/visualization/` contains `convergence_plots.py`,
  `extract.py` and `vtk_export.py`, and no coordinate-transform function at all. The tests were not
  covering an untested module — there was never a subject.

  The `grid_2d_time` fixture is kept: `density_2d_gaussian` and `value_function_2d` depend on it.

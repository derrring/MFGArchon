- **`HJBGFDMSolver(use_wind_dependent_bc=True)` constructs when the boundary is detected** (#2604). It printed
  `len(boundary_indices)`, the constructor's parameter, which is None by default, so the documented option raised
  `TypeError: object of type 'NoneType' has no len()` unless `boundary_indices` was passed. It now logs the detected
  boundary's size through the module logger, not stderr.

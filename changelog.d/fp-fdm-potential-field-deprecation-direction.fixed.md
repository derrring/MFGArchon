- **FP-FDM `potential_field` is no longer deprecated.** v0.18.6 (#919) swapped two parameter
  names — `velocity_field` became `drift_field` (velocity), and the old `drift_field` (which
  held the value function `U`) became `potential_field`. A `@deprecated_parameter` was attached
  to the *destination* of that rename, pointing users back at `drift_field`, whose meaning had
  just changed. The advice was actively wrong: on `FPFDMSolver`, `drift_field` is the velocity
  α\*, so obeying the warning and passing `U` there silently solves a different problem. It was
  also un-completable — `resolve_fp_drift_kwargs`, the single owner of this routing, emits
  `potential_field=U` on the default smooth-separable path used by both Picard and Newton, so
  every such solve raised a `DeprecationWarning` from library-internal code and the scheduled
  v0.25.0 removal would have broken the library's own primary path.

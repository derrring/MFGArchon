- **AdaptiveTrainingStrategy reads the canonical training_mode, not the raw deprecated booleans**
  (Issue #1572). The strategy gated curriculum/refinement on the raw `self.config.enable_*` booleans,
  which default to `None`; `__post_init__` maps them to `training_mode` but never back-fills them. So a
  canonical `training_mode=CURRICULUM` (and the default `FULL_ADAPTIVE`) silently skipped the features —
  the inverse of the #616 trap (the canonical API was the silent no-op). The consumer now reads the
  `uses_curriculum` / `uses_refinement` properties. The deprecated boolean redirect is unchanged.

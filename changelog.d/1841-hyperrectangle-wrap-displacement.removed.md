- **`Hyperrectangle.wrap_displacement` is removed** (Issue #1841). The minimum-image displacement
  wrap now lives in `mfgarchon.geometry.boundary.periodic.wrap_displacement`, taking the `periods`
  dict that `SupportsPeriodic.get_periods()` already promises, so every geometry satisfying the
  protocol can use it rather than only the one class that implemented the method. The method had a
  single caller in the tree and none in `mfg-research`; it is deleted rather than deprecated because
  it is a pre-1.0 internal with no external users, and leaving it would have made the change an added
  layer instead of a consolidation.

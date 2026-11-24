- **One damping policy, set per field, on every coupling route** (#2997). `DampingPolicy` (`mfgarchon.config`) holds a
  `FieldDamping` for U and one for M, each with its theta, schedule (`none`, `constant`, `harmonic`, `sqrt`,
  `polynomial`, `exponential` or a callable), floor, adaptation and first-sweep rule, and Anderson acceleration on U
  (`AndersonOptions`). `FixedPointIterator`, the Block iterators, `FictitiousPlayIterator` and `NewtonMFGSolver` take it
  as `damping=`, and `PicardConfig.damping` holds it, written to and read from YAML as a mapping. Presets:
  `DampingPolicy.constant(theta, theta_M=None)` and `DampingPolicy.fictitious_play(alpha0, schedule, floor=,
  damp_value_function=)`.

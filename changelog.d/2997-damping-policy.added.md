- **One damping policy, set per field, on four coupling routes** (#2997, stage 1a). `DampingPolicy` (`mfgarchon.config`) holds a
  `FieldDamping` for U and one for M, each with its theta, schedule (`none`, `constant`, `harmonic`, `sqrt`,
  `polynomial`, `exponential` or a callable), floor, adaptation and first-sweep rule, and Anderson acceleration on U
  (`AndersonOptions`). `FixedPointIterator`, the Block iterators, `FictitiousPlayIterator` and `NewtonMFGSolver` take it
  as `damping=`. `PicardConfig.damping` holds one for a solve run from a config: set, it decides; left None, the
  default, the route keeps its own; given both there and to the constructor, the solve refuses. A policy whose schedules
  are named is written to and read from YAML as a mapping; one with a callable schedule is not, and `to_yaml` raises.
  Presets: `DampingPolicy.constant(theta, theta_M=None)` and `DampingPolicy.fictitious_play(alpha0, schedule, floor=,
  damp_value_function=)`. `GraphMFGSolver`, `RegimeSwitchingIterator` and `MultiPopulationIterator` keep their own
  damping until stage 1b.

- **One damping policy, set per field, on every coupling route** (#2997, stages 1a and 1b). `DampingPolicy` (`mfgarchon.config`) holds a
  `FieldDamping` for U and one for M, each with its theta, schedule (`none`, `constant`, `harmonic`, `sqrt`,
  `polynomial`, `exponential` or a callable), floor, adaptation and first-sweep rule, and Anderson acceleration on U
  (`AndersonOptions`). `FixedPointIterator`, the Block iterators, `FictitiousPlayIterator`, `NewtonMFGSolver`,
  `GraphMFGSolver`, `RegimeSwitchingIterator` and `MultiPopulationIterator` take it as `damping=`; the last three apply
  it to each component, node, regime or population, whose fields adapt on that component's own history. `PicardConfig.damping` holds one for a solve run from a config, on `FixedPointIterator` and
  `FictitiousPlayIterator`, whose solves take one: set, it decides; left None, the default, the route keeps its own;
  given both there and to the constructor, the solve refuses. The other routes take no config, so their damping is the
  constructor's alone. A policy whose schedules
  are named is written to and read from YAML as a mapping; one with a callable schedule is not, and `to_yaml` raises.
  Presets: `DampingPolicy.constant(theta, theta_M=None)` and `DampingPolicy.fictitious_play(alpha0, schedule, floor=,
  damp_value_function=)`.

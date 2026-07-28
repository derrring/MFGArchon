- **HybridMazeGenerator works again** (Issue #1712) — all four algorithm branches were
  broken: three imported modules that never existed under those names (the `maze_` prefix
  was added by a rename), and the fourth read `get_cell` off `generate()`'s return value
  instead of the `Grid` it wraps. Function-local imports meant none of this failed at
  import time. Fixed, with `tests/unit/test_geometry/test_maze_hybrid_dispatch.py`
  exercising every branch end to end. The byte-identical maze-generator copies under
  `alg/reinforcement/environments/` are deleted; `mfgarchon.geometry.graph` owns them and
  the RL package re-exports from there.

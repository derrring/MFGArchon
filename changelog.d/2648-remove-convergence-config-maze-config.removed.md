- **Two configuration surfaces that nothing consumed are removed** (#2648, rulings 5 and 6, user, 2026-10-14). Breaking
  (pre-1.0; deprecation clause 4 waived, as for stage 1).
  - `ConvergenceConfig`, `MFGConvergenceChecker`, `HJBConvergenceChecker`, `FPConvergenceChecker`, the
    `ConvergenceChecker` protocol and `create_convergence_checker`, exported from `mfgarchon.utils` and
    `mfgarchon.utils.convergence`.
    - No solver took a `ConvergenceConfig`, and seven of its fourteen fields had no reader.
    - The checkers read the rest, and no solver used them.
    - Each solver takes its own `tolerance` and `max_iterations`. The change a coupling sweep made is `sweep_change`'s.
    - #2657 records one convergence configuration across solvers, built on those.
    - `compute_norm` stays (#2566).
  - `mfgarchon.geometry.graph.maze_config`: `MazeConfig`, exported from `mfgarchon.geometry` and
    `mfgarchon.geometry.graph`, with its `maze_Config` alias. Also removed: `PlacementStrategy`, `MazeTopology`,
    `PhysicalDimensions`, and the three `create_*_config` factories.
    - No maze generator consumed a `MazeConfig`.
    - Build a maze with `MazeGeometry(rows=..., cols=..., algorithm=..., seed=...)`, which generates on construction.

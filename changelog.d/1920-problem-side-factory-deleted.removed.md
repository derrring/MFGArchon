- **The problem-side factory is deleted, and the problem factories refuse construction** (#1920, #1989; recorded by
  #2654). Breaking (pre-1.0).
  - `GeneralMFGFactory` and `get_general_factory` are removed, with `mfgarchon.factory.general_mfg_factory`. At
    v0.21.0 both were exported from `mfgarchon` and `mfgarchon.factory`.
  - Nine problem factories remain importable, and calling one raises `NotImplementedError`:
    `create_general_mfg_problem`, `create_mfg_problem`, `create_lq_problem`, `create_standard_problem`,
    `create_crowd_problem`, `create_network_problem`, `create_highdim_problem`, `create_stochastic_problem` and
    `create_variational_problem`. #1920 found that the implementation behind them silently replaced the caller's
    `BoundaryConditions` with a hardcoded no-flux wall, so it was removed rather than left to answer wrongly.
  - Build the problem directly instead: `MFGProblem(geometry=..., components=...)`.

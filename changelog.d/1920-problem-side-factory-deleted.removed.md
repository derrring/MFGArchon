- **The problem-side factory is deleted, and the problem factories refuse construction** (#1920, #1989; recorded by
  #2654). Breaking (pre-1.0).
  - `GeneralMFGFactory` and `get_general_factory` are removed, with `mfgarchon.factory.general_mfg_factory`. At
    v0.21.0 both were exported from `mfgarchon` and `mfgarchon.factory`. Their `create_from_hamiltonian` -- behind
    `create_general_mfg_problem`, `create_from_config_file` and `create_from_config_dict` -- built a legacy 1-D domain
    (`xmin`/`xmax`/`Nx`) with a hardcoded no-flux wall, and the caller's `boundary_conditions=` reached only the
    components, while solvers read the geometry's (#1920).
  - Nine problem factories remain importable, and calling one raises `NotImplementedError`:
    `create_general_mfg_problem`, `create_mfg_problem`, `create_lq_problem`, `create_standard_problem`,
    `create_crowd_problem`, `create_network_problem`, `create_highdim_problem`, `create_stochastic_problem` and
    `create_variational_problem`. `create_network_problem` already raised at v0.21.0. The other eight were removed with
    the general factory (#1989) rather than kept without it.
  - Build the problem directly instead: `MFGProblem(geometry=..., components=...)`.

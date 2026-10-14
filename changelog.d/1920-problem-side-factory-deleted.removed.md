- **The problem-side factory is deleted, and the problem factories refuse construction** (#1920, #1989; recorded by
  #2654). Breaking (pre-1.0).
  - `GeneralMFGFactory` and `get_general_factory` are removed, with `mfgarchon.factory.general_mfg_factory`. At
    v0.21.0 both were exported from `mfgarchon` and `mfgarchon.factory`. Given an `xmin`/`xmax`/`Nx` domain, which
    `create_general_mfg_problem` always built and the config loaders built for a domain in that format, their
    `create_from_hamiltonian` built the grid with a hardcoded no-flux wall and put the caller's `boundary_conditions=`
    only in the components. The HJB solvers, which read the geometry's condition, then ran on the wall, while
    `FPFDMSolver` and `FPFVMSolver`, which read the components' first, ran on the caller's boundary (#1920).
  - Nine problem factories remain importable, and calling one raises `NotImplementedError`:
    `create_general_mfg_problem`, `create_mfg_problem`, `create_lq_problem`, `create_standard_problem`,
    `create_crowd_problem`, `create_network_problem`, `create_highdim_problem`, `create_stochastic_problem` and
    `create_variational_problem`. `create_network_problem` already raised at v0.21.0. The other eight were removed with
    the general factory (#1989) rather than kept without it.
  - Build the problem directly instead: `MFGProblem(geometry=..., components=...)`.

- **Star imports no longer raise `AttributeError`** (#2651).
  - `from mfgarchon import *` and `from mfgarchon.factory import *`: both `__all__` lists named
    `create_general_mfg_problem`, which neither module bound after the problem-side factory was deleted (#1920). Both
    now bind it to the stub its sibling problem factories use. Calling it raises #1920's `NotImplementedError`, with
    its guidance, where `mfgarchon.create_general_mfg_problem` was an `AttributeError`.
  - `from mfgarchon.alg.optimization import *` when POT is not installed: that branch listed
    `create_obstacle_variational_mfg` and `create_quadratic_variational_mfg`, which the package never imports. They
    are no longer listed, and both remain in `mfgarchon.alg.optimization.variational_problem`.

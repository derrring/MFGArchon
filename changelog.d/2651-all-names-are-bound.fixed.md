- **`from mfgarchon.alg.optimization import *` no longer raises `AttributeError` when POT is not installed**
  (#2651). That branch's `__all__` listed `create_obstacle_variational_mfg` and `create_quadratic_variational_mfg`,
  which the package never imports. They are no longer listed, and both remain in
  `mfgarchon.alg.optimization.variational_problem`. `mfgarchon` and `mfgarchon.factory` bind
  `create_general_mfg_problem` again, to the stub described in the #1920 entry, so their star imports work as they
  did at v0.21.0.

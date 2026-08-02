- **README no longer advertises the frozen packages as complete** (#1789). `Reinforcement Learning -
  Complete RL framework (DDPG, TD3, SAC)` and `Neural (DGM, PINN) - Deep learning for high
  dimensions` were the strongest unreconciled maturity claims in the repo, on the front page, while
  both packages are frozen design prototypes (`CLAUDE.md`).

  They survived the #1782 marker sweep because that sweep enumerated documents with the frozen
  checker's own prefix matching — deliberately, since an ad-hoc grep counts `mfgarchon.alg.
  neural_solvers`, an unrelated package, as a hit. README describes the packages in prose with no
  dotted module path, so the rule that made the enumeration correct is the rule that made it blind.
  A marker sweep keyed on import paths cannot see a capability claim that does not import.

  The `[nn]` extra line is split rather than marked wholesale: measured, `torch` is imported by
  four non-frozen modules (three under `backends/`, one under `utils/acceleration/`) while
  `gymnasium` and `stable-baselines3` have zero importers outside `alg/reinforcement/` and
  `tensorboard` has none anywhere. (`utils/dependencies.py` names `torch` in a string for
  `is_available` and does not import it, so it is not a fifth.)

  Review found the first pass had marked the bullet list and missed the two strongest claims in the
  same file: the one-line project description (`...GPU acceleration, and reinforcement learning`)
  and a feature-parity bullet listing `Neural` as interchangeable with FDM/FEM/WENO. The same
  sentence lives in `CITATION.cff`, which is DOI-attached and rendered by Zenodo, and `pyproject.toml`
  still attributed `torch` to DGM/PINN — a divergence this PR's own README edit created. All four
  corrected, and the per-dependency attribution is now recorded beside the requirement it explains.

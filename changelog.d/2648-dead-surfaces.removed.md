- **Declared surfaces that nothing reads are removed** (#2648, stage 1). Breaking (pre-1.0; deprecation clause 4 waived
  by the user, ruling of 2026-10-14, as for #2343 batch 1, ruling 16 of 2026-09-28 and #2422).
  - `mfgarchon.core.component_configs` (`StandardMFGConfig`, `NetworkMFGConfig`, `VariationalMFGConfig`,
    `StochasticMFGConfig`, `NeuralMFGConfig`, `RLMFGConfig`, `ImplicitGeometryConfig`, `AMRConfig`,
    `TimeDependentDomainConfig`, `MultiPopulationMFGConfig`) and `mfgarchon.core.mfg_components_nested` (its own
    `MFGComponents`). `component_configs` was imported only by `mfg_components_nested`, which nothing imported. This
    answers #2558's first row.
  - `ErgodicConditions`, exported from `mfgarchon` and `mfgarchon.core` at v0.21.0. It declared a stationary problem
    that no `MFGProblem` accepted. It comes back together with an ergodic mode, should one be designed.
  - `mfgarchon.factory.solver_factory.SolverFactoryConfig` and
    `mfgarchon.utils.functional_calculus.FunctionalDerivativeConfig`, two dataclasses that nothing instantiated
    (#1663's `SolverFactoryConfig` row).
  - Everything above except `ErgodicConditions` was reachable at v0.21.0 only by its module path.

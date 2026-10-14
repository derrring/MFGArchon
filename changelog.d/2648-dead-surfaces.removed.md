- **Declared surfaces that nothing reads are removed** (#2648, stage 1). Breaking (pre-1.0; deprecation clause 4 waived
  by the user, ruling of 2026-10-14, as for #2343 batch 1, ruling 16 of 2026-09-28 and #2422).
  - `mfgarchon.core.component_configs` (`StandardMFGConfig`, `NetworkMFGConfig`, `VariationalMFGConfig`,
    `StochasticMFGConfig`, `NeuralMFGConfig`, `RLMFGConfig`, `ImplicitGeometryConfig`, `AMRConfig`,
    `TimeDependentDomainConfig`, `MultiPopulationMFGConfig`) and `mfgarchon.core.mfg_components_nested` (its own
    `MFGComponents`). Each was imported only by the other, and nothing imported the pair. This closes #2558's first row.
  - `ErgodicConditions`, which was exported at the top level. It declared a stationary problem that no `MFGProblem`
    accepted. It comes back together with an ergodic mode, should one be designed.
  - `mfgarchon.factory.solver_factory.SolverFactoryConfig` and
    `mfgarchon.utils.functional_calculus.FunctionalDerivativeConfig`, two dataclasses that nothing instantiated
    (#1663's `SolverFactoryConfig` row).
  - `scripts/docstring_kwargs_baseline.json` is re-recorded. Its four findings in `mfg_components_nested` went with the
    module, which held the scan's one block and 9 keyword arguments of the difference.

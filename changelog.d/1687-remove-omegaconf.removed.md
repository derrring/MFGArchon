- **OmegaConf is removed; the config stack is Pydantic plus `config/io.py`** (Issue #1687). A
  numerical library should not own a config framework — it should accept typed objects, and loading
  config files belongs to the application layer. Gone: `OmegaConfManager` and its module functions,
  `bridge_to_pydantic` (with one config system there is nothing to bridge — it collapses to
  `model_validate`), `save_effective_config` / `load_effective_config`,
  `create_parameter_sweep_configs`, the four shipped `mfgarchon/config/configs/*.yaml`, and the
  `OMEGACONF_AVAILABLE` flag. YAML interpolation, composition and merging go with them; they were
  OmegaConf features, not library features. `load_solver_config` / `save_solver_config` /
  `validate_yaml_config` are unchanged and remain the flat-YAML path.

  Unknown-field rejection (#1766) is unaffected: it is `extra="forbid"` on the models in
  `config/core.py`, not bridge behaviour. Two tests that exercised it *through* the bridge are
  replaced by one that exercises it directly.

- **`jupyter`, `jupyterlab` and `seaborn` are no longer dependencies.** Measured: zero imports in
  the package, the tests, the examples and the benchmarks. `nbformat` stays, because the package
  writes notebooks; the application you read them in is not a library dependency.

- **`pyyaml` is now declared.** `config/io.py` has always imported it at module level while nothing
  declared it — it arrived transitively from omegaconf and from jupyterlab's dependency chain, both
  removed above. A fresh install would otherwise have lost `load_solver_config` and the local gate's
  workflow-integrity step.

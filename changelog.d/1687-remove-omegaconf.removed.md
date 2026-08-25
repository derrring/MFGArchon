- **OmegaConf is removed; the config stack is Pydantic plus `config/io.py`** (Issue #1687). A
  numerical library should not own a config framework — it should accept typed objects, and loading
  config files belongs to the application layer. Gone: `OmegaConfManager` with `create_omega_manager`,
  `load_beach_config`, `load_experiment_config` and `create_parameter_sweep_configs`;
  `bridge_to_pydantic`; `save_effective_config` / `load_effective_config`; the four
  `mfgarchon/config/configs/*.yaml`; and the `OMEGACONF_AVAILABLE` flag.

  **`bridge_to_pydantic` does not collapse to a plain `model_validate`.** It carried `strict=True`,
  so `{'picard': {'max_iterations': '100'}}` raised there and is silently coerced to `int` without
  it. A caller relying on that rejection must pass `strict=True` — **but only on a dict built in
  Python, not on YAML**; see the scalar note below. The four YAMLs were never in any wheel either —
  `package-data` is commented out — they were written into the installed package directory at
  runtime by `_create_default_configs`, so this also removes a write-into-site-packages pattern.

  YAML interpolation, composition and merging go with them; they were
  OmegaConf features, not library features. `load_solver_config` / `save_solver_config` /
  `validate_yaml_config` are unchanged and remain the flat-YAML path.

- **Behaviour change: YAML scientific notation.** OmegaConf resolved scalars itself and handed
  Pydantic a `float`. PyYAML implements YAML 1.1, whose float rule requires **both a decimal point
  and a signed exponent**, so `1e-8`, `1e8`, `-1e-8` and even `1.0e8` (unsigned exponent) now arrive
  as `str`, while `1.0e-8` and `1.0E+8` arrive as `float`. For **float** fields
  `load_solver_config` is unaffected — it validates non-strictly and coerces the string back — so
  `tolerance: 1e-8` still loads. For **int** fields it is not: a string will not coerce to `int`
  even non-strictly, so `picard.max_iterations: 1e3` raises out of `load_solver_config` itself,
  where OmegaConf accepted it by resolving to `1000.0` first. Four fields are exposed
  (`hjb.accuracy_order`, `hjb.newton.max_iterations`, `picard.anderson_memory`,
  `picard.max_iterations`). Nothing shipped here was affected; the removed default configs all used
  the `1.0e-6` spelling. `tests/unit/test_config/test_yaml_scalar_typing_1687.py` pins the table,
  both field-type outcomes, and the four int fields by name, because a documented behaviour is a
  claim that rots on the next PyYAML release.

  Unknown-field rejection (#1766) is unaffected: it is `extra="forbid"` on the models in
  `config/core.py`, not bridge behaviour. In `test_unknown_fields_are_rejected_1766.py`, two tests
  that exercised it *through* the bridge are replaced by one that exercises it directly. Verified by
  mutation: reverting to `extra="ignore"` fails 3 of 4 tests on the branch against 4 of 5 before, so
  the guard is not weakened.

  Separately, `tests/unit/test_config/test_bridge.py` (14 tests) is deleted. Thirteen were
  bridge-scoped. The fourteenth, `test_the_alias_map_has_one_owner`, imports only `PicardConfig` and
  is **carried over** to `test_picard_relaxation_alias.py`: it checks every entry of
  `LEGACY_FIELD_ALIASES` by iterating the map, where the rest of that file uses a hardcoded
  parametrize list — a second copy. Without it, adding an alias whose canonical target does not
  exist leaves the config suite green at 173 passed.

- **`jupyter`, `jupyterlab` and `seaborn` are no longer dependencies.** Measured: zero imports in
  the package, the tests, the examples and the benchmarks. `nbformat` stays, because the package
  writes notebooks; the application you read them in is not a library dependency.

- **`pyyaml` is now declared.** `config/io.py` has always imported it at module level while nothing
  declared it — it arrived transitively from omegaconf and from jupyterlab's dependency chain, both
  removed above. A fresh install would otherwise have lost `load_solver_config` and the local gate's
  workflow-integrity step. `utils/cli.py`'s `try/except ImportError` around the same import, and its
  two unreachable `raise` branches, go with it: this change is what made them unreachable. The test
  that patched the removed `YAML_AVAILABLE` flag is replaced by one asserting the declaration is
  present, which is the invariant the unconditional imports actually need.

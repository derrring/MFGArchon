- **The config translator threads what the user SET, not what differs from a default**
  (Issue #2266). `config/translator.py` passed a field to a solver constructor only when its value
  differed from the Pydantic default — sound if and only if `Pydantic default == solver constructor
  default` for every threaded field. Where the two disagreed, a user who wrote the config's own
  documented default silently got the solver's different value. The rule is now
  `field in model_fields_set`: an explicitly-set field is threaded whatever its value. 47 sites, at
  both levels — the per-field checks and the sub-config guards above them, which compared a whole
  sub-config by value and would otherwise have skipped the block before any per-field check ran.

  **`NewtonConfig.max_iterations = 10` now means 10.** It is the Pydantic default, so it was dropped,
  and `HJBFDMSolver` resolved `None` to `DEFAULT_NEWTON_MAX_ITERATIONS = 30` — a 3x budget nobody
  asked for, on the axis along which #1873's error is non-monotone.

  **`SLConfig.interpolation_method` now reaches the solver, and the issue's own matrix was wrong
  about what that fixes.** #2266 reports "SL_CUBIC runs linear interpolation. Always." Measured
  through the shipped path with a spy on the solver constructor, it does not:
  `factory/scheme_factory.py` re-supplies the value with
  `hjb_config.setdefault("interpolation_method", "linear"|"cubic")`, so before this change
  `SL_CUBIC` + `'cubic'` ran **cubic** and only `SL_LINEAR` + `'cubic'` silently ran linear. One row
  of four, not four. That accidental masking is the argument for fixing the rule rather than the
  fields: the same dropped field is visible under one scheme and invisible under another, so which
  instances get found is a matter of luck.

  **Ordering.** This lands after #2250 and could not have landed before it. Threading
  explicitly-set fields means `config.backend.type = "numpy"` — the documented default — starts
  arriving at `FixedPointIterator`, where before #2250 it raised
  `AttributeError: 'str' object has no attribute 'zeros'`. Fixing this rule first would have turned a
  crash on non-default backends into a crash on the default one.

  An untouched config still threads nothing, so callers relying on solver defaults are unaffected —
  that was the whole point of the original rule and it is preserved.

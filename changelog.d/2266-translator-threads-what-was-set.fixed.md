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

  **Two questions, two tests — refined after adversarial review.** Threading asks about INTENT
  (`field in model_fields_set`); refusing a field as *unmapped* asks about EFFECT and still compares
  values. Writing a field's own default into a YAML is ordinary practice — this package's documented
  example does it — and asks for nothing the library is not already doing, so refusing it would be
  noise. A scheme *mismatch* is refused on intent, because configuring a solver you are not using is
  an error at any value.

- **`model_fields_set` now means what it says** (Issue #2266). Two things made it lie, both found by
  adversarial review and neither covered by any test:

  A `model_validator(mode="after")` that *assigns* is recorded by Pydantic exactly as a caller's
  value would be. A **fresh** `FEMConfig()` reported `{'quadrature_order'}`, a fresh `HJBConfig()`
  reported `{'fdm'}`, and a fresh `FPConfig()` reported `{'particle'}` — so the translator refused
  configurations nobody had configured, and **FEM was unreachable through Safe and Auto mode**. New
  single owner `BaseConfig._forget_derived`, called in each of the three validators at the point of
  assignment; every branch is guarded by `is None`, so it can never erase a caller's value.

  And `save_solver_config` / `model_dump_yaml` dumped with `exclude_none`, not `exclude_unset`, so
  every field was written out and `from_yaml` returned a config in which *everything* read as
  explicitly set — a round trip raised `NotImplementedError` on all four translator entry points.
  Both now use `exclude_unset`, so a save/load round trip preserves provenance.

- **The coupling loop resolves a backend name and refuses one it cannot write into** (Issue #2250,
  revised after review). A backend *object* is passed through and a *name* is resolved via
  `create_backend` — a `NumPyBackend` object solved end to end before this change, so refusing it
  would have been a capability regression, and the repository's own
  `examples/basic/solvers/acceleration_comparison.py` assigns one. Whether the allocated array can
  be **written** is checked at the allocation, in `allocate_state_arrays`, not in the constructor:
  `self.backend` is a public attribute that callers assign *after* construction, so a constructor
  guard cannot see that path and — with the allocation branch present — would silently ignore it.
  jax and torch are refused there with a message naming #1922 instead of failing deep in the solve.

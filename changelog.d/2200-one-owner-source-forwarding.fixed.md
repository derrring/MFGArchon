- **One owner decides whether a solver can be handed a `source_term`, at all four coupling call
  sites** (Issue #2200). `volatility_field` and `drift_field` each had a `resolve_*` owning that
  decision; `source_term` had none, so both Picard sides and both Newton sides restated it inline
  with four different refusal messages. The observable cost was three disagreeing counts of one
  property — #1991 reported seven of ten HJB solvers dropping a source, #2020 five of ten accepting
  it, and the behavioural census five of eight. `resolve_source_kwarg` now owns whether the solver
  can take a source at all (a `**kwargs` signature counts as *cannot*), the callable convention
  `source_term(t, x) -> (N,)` at the solver's own evaluation points, and the refusal message. It
  deliberately does not own the consumption: the HJB half subtracts from a Newton residual and the
  FP half adds to a timestep, which are different equations and belong to the schemes.
- **The refusal message stopped telling users to "use an FDM solver"** (Issue #2200). That was true
  when written and false by the time it was read: six HJB solvers and seven FP solvers accept a
  source after #1991, #2198 and #2020. It now states the rule and hands the reader the predicate
  this check itself uses — `'source_term' in inspect.signature(solver.solve_hjb_system).parameters`
  — rather than a list, which goes stale, or a path under `tests/`, which `pyproject.toml` excludes
  from the wheel and so does not exist for anyone who installed the package.

- **The coupling kwarg gate no longer drops `volatility_field` when a solver declares `**kwargs`**
  (#1783). `_build_hjb_kwargs` / `_build_fp_kwargs` asked `inspect.signature` whether the solver
  named the parameter — which answers "does this callable name it", not "can this solver consume
  it". `MeshlessGalerkinHJBSolver` delegates through `(*args, use_newton=None, **kwargs)`, so the
  field was dropped on the HJB side while the paired FP solver consumed it: measured with
  `problem.sigma = 0.3` and a field of mean 0.7, HJB ran at `D = 0.045` against FP's `D = 0.245`, a
  5.4x mismatch with no warning and a converged density for a problem nobody posed. The gate now
  refuses, matching the `source_term` branch three lines below it (#1424).

  Review found the same silent drop live on two further sites the issue had not named:
  `MFGResidual.compute_hjb_output` and `.compute_fp_output`, the Newton coupling path, where the
  identical 5.4x mismatch was fully reachable through `NewtonMFGSolver`. The four sites each
  restated the same three-way decision — is this field a hazard, can the solver take it, do we
  forward it — and the restatement is what let them disagree. All four now call one
  `resolve_volatility_kwarg`; its guard test scans the whole coupling package for a surviving
  inline membership test rather than one class, which is why the first version of that guard was
  green while the two Newton copies were live.

  Two corrections to the first version of this fix. It put the forward inside the hazard branch,
  so a scalar equal to `problem.sigma` stopped being forwarded at all — silent-wrong in the same
  way, because `problem.volatility_field` is not always `problem.sigma`: construct with an array
  sigma and the field is the array while `sigma` is its mean, so a solver falling back through
  `get_diffusion_coefficient_field(None)` picked up the array instead of the constant asked for.
  Forwarding is now unconditional whenever the solver names the parameter, and the exemption
  decides only whether to refuse. The scalar test is `numbers.Real` rather than `(int, float)`, so
  `np.float32(0.3)` no longer draws a refusal for a solve byte-identical to one that is accepted.

  Known limitation, not addressed here: `MeshlessGalerkinHJBSolver.solve_hjb_system` is a pure
  delegation to `WeakFormHJBSolver.solve_hjb_system`, which does name `volatility_field` and does
  consume it (`D = 0.045` without, `D = 0.245` with). The refusal therefore forecloses a capability
  that solver has; declaring the parameter on the wrapper, or resolving the signature through the
  delegate, is the better fix. Left to the weak-form work in flight rather than edited underneath
  it, and tracked separately. Until then the refusal is still the right behaviour — the prior
  alternative was a silently wrong answer, not a working solve.

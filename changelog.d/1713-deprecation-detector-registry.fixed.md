- **The only deprecation artifact wired into CI could not fail** (Issue #1713).
  `check_internal_deprecation.py` discovered symbols by walking the AST for `@deprecated` and
  required the decorator to literally pass `removal_blockers=` — an optional kwarg **no call site
  in the package passes**. It therefore found nothing by construction: `Total deprecated symbols: 0`,
  exit 0, green workflow, while `audit_all_deprecations` reported **72 live** in the same tree.
  Same class as `check_fail_fast`'s `broad_except` reading 11 against a true 115 for 32 days (#1706):
  a discovery predicate that silently excludes the whole population.
  It now asks the **runtime registry**, which cannot under-count the way a syntax guess can, and
  pins the counts in `scripts/deprecation_baseline.json` — failing in **both** directions, so a
  retirement must be recorded and a baseline cannot be lowered without doing the work. Verified by
  constructing both: injecting a `@deprecated` symbol takes it to 73 and exits 1; claiming one more
  in the baseline than exists also exits 1.
  Stated boundary: this covers deprecations **declared through the decorators**. Something retired
  without one is invisible here too.
  The internal-usage check is kept and now runs over a real symbol list, but is vacuous today by
  construction — **0 of 72** have `internal_usage` cleared — which the count now makes visible
  instead of hiding behind a green tick.

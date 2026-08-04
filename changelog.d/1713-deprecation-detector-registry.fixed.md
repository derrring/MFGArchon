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
  construction — **0 of 63** have `internal_usage` cleared — which the count now makes visible
  instead of hiding behind a green tick.
- **One unimportable subpackage decided what the deprecation count was** (Issue #1713, #1774).
  `scan_deprecated` passed no `onerror` to `pkgutil.walk_packages`, which re-raises anything that is
  not `ImportError`, and the enclosing `except Exception: logger.debug(...)` swallowed the abort.
  Without torch, `alg/reinforcement/multi_population` raises `AttributeError` (`nn = None` used as a
  base class, #1773), so the walk stopped at **160 of 428 modules**: `geometry/`, `utils/` and
  `operators/` were never reached and the count came back **41** against a true **72**. The first
  version of the registry ratchet compared exactly that number to a committed baseline and went red
  on a tree nobody had touched. 29 of the 31 missing were truncation, not torch — only 2 live in a
  module that needs it.
  The walk now continues past a broken subpackage, records every module it could not import, and
  `scan_deprecated` **refuses** by default rather than returning the smaller number
  (`IncompleteScanError`; `allow_incomplete=True` opts into best-effort). `generate_deprecation_guide.py`
  refuses on the same grounds instead of writing a partial guide as if it were the whole API.
  The ratchet's census is scoped to the **live** library — `alg/neural` and `alg/reinforcement` are
  frozen prototypes and out of scope for repo-wide campaigns (CLAUDE.md), and measured, they are
  also exactly where every torch-dependent module lives. That makes the count **63 with torch and 63
  without**, so the baseline means something on any runner; and any module *in scope* that fails to
  import exits 2 with the module named, because a smaller number and a number measured over less
  tree are indistinguishable from outside.
  Mutation-verified: restoring the missing `onerror` loses `old_c` in a synthetic three-subpackage
  tree whose middle package raises at import — the symbol after the break, which is the one the
  truncating version cannot reach.
  `audit_all_deprecations` items now carry `modules` — every module a symbol was found in, not just
  the one that survived the dedup — because a scope filter reading the survivor judges walk order:
  found in review, a live deprecation added under `mfgarchon/utils/` took the count 63 → 64 and went
  red correctly, and one `from … import` of it appended to `alg/neural/__init__.py` put it back to
  63 and green. A symbol is out of scope only when the frozen paradigms are the *only* place it
  appears, and containment is by package, not by string prefix. Both halves of that rule are pinned
  and mutation-verified: reverting the scope filter to the reported bug reddens
  `tests/unit/test_check_internal_deprecation.py`, and so does removing the filter altogether, which
  is the control that keeps "stop excluding" from passing as a fix.

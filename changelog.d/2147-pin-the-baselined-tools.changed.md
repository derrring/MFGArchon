- **pytest is pinned exactly; mypy and everything else keep their floors** (Issue #2147, trial
  policy). The criterion is *does a tracked baseline change when this tool's version changes, and is
  that dependence deliberate?* For pytest both halves hold: `scripts/warning_baseline.json` keys each
  identity on `(origin file, class, message)` **by design** — one deprecated API called from 153 test
  files is 153 identities, and that number falling is the migration the ratchet tracks — and pytest
  computes the origin attribution, so a major upgrade moves some of them. Measured: 8 → 9 moved 6 of
  224, and the only line in the gate naming a cause was about ruff.
- **A version-dependent red is usually a defect in the check, and the alternatives were measured
  before the pin, not after.** Scoping the census to `mfgarchon`-origin identities would delete the
  194 that come from our own test files. Excluding `site-packages` would delete three real findings
  plus one of our own warnings whose frame sits inside pytest. Re-keying without the origin file
  collapses 224 identities to 44 and destroys the call-site count that is the whole point. What
  remains is a 2.7% coupling in a check whose value is elsewhere.
- **mypy is deliberately not pinned.** The criterion admits it and the second half does not: the
  baseline it guards is one subpackage type-checking clean, and pinning a type checker suppresses the
  new checks that find real defects. The cost of floating is a red gate on an unrelated PR; the cost
  of pinning is not finding things, which nobody reports.
- **ruff is not pinned a second time.** It already has an owner in `.pre-commit-config.yaml`, which
  `ci.yml` reads and `check-ruff-updates.yml` bumps monthly. Forty-nine releases in twelve months is
  the reason not to pin it twice.
- **The gate warns on every `==` pin in the dev extra, not just ruff.** It had one comparison,
  against the axis that did not break. Running the gate under the `uv.lock` toolchain now prints
  three `WARN`s where it printed one.
- Unresolved, and stated rather than dropped: the conventional home for an exact dev pin is a lock
  file, not published package metadata. `uv.lock` is that home, is five months stale, and
  `uv lock --check` exits 1 — it does not currently describe a resolvable environment. That is the
  other half of #2147.

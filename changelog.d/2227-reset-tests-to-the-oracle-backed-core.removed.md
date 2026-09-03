**The deletion removed 243 test files and 58,785 lines** (`bf44de93`). Review restored six, so the
change against `main` is **238 files**. Line totals are deliberately NOT restated here: they moved
three times during review and again afterwards, and a figure tracking the branch tip is one that will
be wrong by the time you read it. `git diff --stat main...HEAD -- tests/` owns them.

Git history is the archive, the same disposition #1710 gave `archive/` and the one that removed
`alg/neural/` and `alg/reinforcement/`.

**Why:** a suite you cannot audit is not an asset. 5,643 test functions, and reading them file by
file was estimated at roughly six days of full-time review — an estimate, with no measured basis.

## What replaced "audit the tests" as the question

The Joplin `[Principle] Conventions Index` names **35** load-bearing conventions;
`scripts/discrimination_baseline.json` carries **24** falsifiable mutations, each with an `owner`
field. That makes the reviewable unit **tens, not thousands** — 5,643 / 35 is a factor of about 161,
not the "three orders of magnitude" an earlier draft claimed.

**The two sets are nearly disjoint, which an earlier draft got badly wrong.** It said the 24 were a
falsifiable form *of* the 35 and that ~11 conventions were uncovered. Of the 19 issue numbers the
mutation owners cite, **2** appear in the index; roughly 6–9 of the 24 land on index rows. So the
uncovered figure is closer to **29 of 35**, and the mutations defend a dozen conventions the index
does not name. Note also that the index is a **private Joplin note**: no reader of this changelog can
resolve it.

Measured before the deletion, **every one of the 24 mutations is defended**: `kill_count` 2 … 168,
median **21.5**, no zeros, no ones. The three at the floor of 2:

| convention | killers |
|---|---|
| absent BC defaults to clamp/absorbing (#1698) | 2 |
| `SolverResult.mass_conservation_error` is drift from the initial mass, not deviation from 1 | 2 |
| Picard convergence requires **both** the relative and the absolute criterion | 2 |

`kill_count` counts tests that *notice* a mutation, so it tracks how widely the mutated code is
exercised rather than defensive depth — 168 is a constant everything routes through, not a
convention 84× better defended. What makes these three thin is a fact the count does not show: **each
has both killers inside a single file**, so one file-level loss takes each to zero in one step. All
three are already recorded by name in #2148.

## The keep-set: one arm measured, three judged, and the three judged arms all failed

Union of: files containing a test that **kills a mutation** (`killmatrix.json`, 125 files —
measured); **external-oracle** files; **defect pins**; tests **guarding the instruments** in
`scripts/`; the ratchet self-test; `conftest.py`/`__init__.py`.

Arm 1 was applied by measurement. Arms 2–4 were applied by keyword, and independent review found each
had dropped something it should have kept. Restored:

- `test_check_assertion_strength.py` — the **only** test of `scripts/check_assertion_strength.py`, an
  instrument the gate itself runs.
- `test_fp_particle_anisotropic_sigma_1256.py` and `test_issue_1079_anisotropic_sigma.py` — an Itô-isometry
  oracle on a **non-symmetric** Σ, with an explicit anti-tautology control. Without it the surviving
  pin tests only symmetric Σ, where `ΣΣᵀ ≡ ΣᵀΣ`, and the transpose-order convention becomes
  **inexpressible** in the tree.
- `test_issue1285_source_term_time_slice.py`, `test_hjb_1071_control_cost_lambda.py`,
  `test_mfg_caplog.py` — the last being the contract test for the fixture `AGENTS.md` mandates, which
  otherwise had six consumers and no test of its own.

Earlier, the **positive control** had already caught a fourth: a first keep-set would have deleted
`test_capability_warnings.py`, carrying the `== 12` defect pin for #1878.

## Acceptance: the deletion is cleaner than the first claim

| | |
|---|---|
| killer node IDs resolving after the deletion | **611 / 614** |
| killer node IDs **lost to this deletion** | **0** |
| killer files deleted | 0 (and 0 before, so this figure discriminates nothing) |
| conventions reaching `kill_count` 0 | 0 |

The three unresolved node IDs were **already** unresolvable on `main` — a class rename in
`test_fp_particle_solver.py`, a file this change does not touch, from #2181/#2185. That is the
pre-existing killmatrix staleness #2176 was filed for, and this PR's node-ID check is the first
instrument that surfaces it. Its true effect is `particle_mass_counting_measure` 13 → **11** and
`optimal_control_sign` 40 → **39**, neither caused here.

An earlier draft said "613 of 614 survive" and attributed one loss to this change. Both halves were
wrong, in opposite directions, because the check used a `def <name>` regex that a class rename
satisfies while the node ID does not resolve.

## Mechanical consequences

- `citation_baseline.json` re-recorded. **The deletion moved `missing` from 40 to 42** — an earlier
  draft read the post-state, 42, as the number of citations pointing at deleted files; the number
  caused here is **2**. The standing backlog moved 13 → **12** (13 → 11 at the deletion, back to 12
  once review restored six files), and that figure is hand-written in at least **seven** durable
  sites, and **this fragment has now asserted a total three times and been wrong three times** — six,
  then seven, then eight, each found by the next search rather than by the last count. So it no
  longer asserts one. Two of the sites are structurally invisible to the checker: `scripts/local_ci.sh`
  (it scans `.md` and `.py` only) and a second `#2102` changelog fragment. **A checker cannot audit
  the predicate defining its own population.** The live figures now have one owner —
  `check_citations.py` — `local_ci.sh` points at it instead of restating, and the #2102 fragments are
  marked as measurements taken when written rather than current.
- `warning_baseline.json` re-recorded: **336 → 220** identities (212 at the deletion, back to 220
  once review restored six files). The checker refused to write without
  a census — *"Nothing was measured, so this says nothing about whether warnings changed"* — which is
  the correct refusal.
- The full discrimination sweep was **not** re-run: ~26 minutes measured (seven full-suite runs), and a run killed by a timeout leaves
  a mutant in the production tree (#1849, #2229 — it happened twice during this work). The node-ID
  check answers the same question in seconds and mutates nothing.
- The CI smoke tier named `tests/unit/test_config`, which this change emptied; pytest given a
  non-existent path collects **nothing** and reports "no tests ran". Invisible locally, because a
  stale `__pycache__` left the directory present in a working tree.

## What this does not claim

**That the deleted volume was worthless** — only that its value could not be established at a price
worth paying. **Rebuild cost has never been measured**, for this suite or any other. And the
discrimination *fraction* will rise sharply because the denominator shrank while the killers were
kept: **that is arithmetic, not improvement**, the same misreading the record already documents from
when the mutation table grew 6 → 24. The gate prints its own staleness warning saying so. (#2227)

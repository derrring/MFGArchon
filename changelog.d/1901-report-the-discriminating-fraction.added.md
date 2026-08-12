- **The gate now prints the discriminating fraction beside the suite result, with its denominator.**
  `N passed` has never been the quantity worth growing, and printing it alone invites growing it.
  `scripts/report_discrimination.py` prints the one that says whether green means anything:

  ```
  discrimination : 212 of 5872 tests notice at least one of 6 conventions = 3.61%  (measured at db3496f9)
                   the suite has since moved 5872 -> 6142 (+4.6%): the fraction above is STALE, and
                   adding tests lowers it unless they discriminate
  ```

  Reports, does not gate — measuring it costs one full suite run per mutation, so the gating stays
  in the weekly `test_discrimination.py --check-baseline` tier. Skipped under `--fast`.

  The staleness line is the point. The recorded fraction ages silently as the suite grows, which is
  #1901's class 2 (a verdict without its denominator) applied to the instrument that measures class
  2. Its own parser is pinned against **both** pytest summary forms, because the first version read
  `line.split()[0]`, returned `None` on the `N/M tests collected (K deselected)` form the CI marker
  set actually prints, and reported "current suite size unknown" while the number was on screen.

- **`CLAUDE.md`'s discrimination figures corrected, and the citation fixed.** They read `192 of
  5,683 = 3.4%`; the committed artifacts give **212 of 5,872 = 3.6%**, so **96.4% notice nothing** —
  the stronger and more actionable direction. The 60%-inert claim was cited to #1715, whose *body*
  says the prevalence "is not established"; it comes from a **comment**, now linked directly and
  verified to contain the figure.

- **"Inert" is not "worthless", and acting on that distinction was the whole of #1901's item 3.**
  All five tests #1715 names were read: `test_granular_primitives_byte_identical_to_inline` and
  `test_h_eval_helpers_delegate_byte_identical` are delegation pins (`evaluate_H` is literally
  `np.asarray(self(...), dtype=float)`, so the assertion is `f(x) == f(x)` while delegation holds —
  and fires the moment a shim is reimplemented divergently);
  `test_coupled_equals_drift_byte_identical` compares two FP advection entry points and carries its
  own vacuity control; `test_nonlcr_weights_byte_identical_adaptive_off` compares builder-path
  against operator-path GFDM weights; `test_newton_matches_picard_small` pins two independent
  solvers on one fixed point. **None is deletable.** Each is inert on six conventions precisely
  because those conventions are not what it pins.

  So the deletable set is the **structurally tautological** one — found by reading — not the inert
  one, found by counting. Recorded rather than acted on, which is what
  `feedback_net_negative_test_mass` already prescribed: the correct form of net-negative is "add
  less", not "hunt for deletions", and a kill count cannot prove a test is removable, only fail to
  falsify it.

- **A second quality number beside it, and the deletion PR it was meant to justify does not exist.**
  `scripts/check_assertion_strength.py` counts tests whose assertions a well-formed **wrong** answer
  would satisfy — every assertion being `is not None` / `isfinite` / `.shape` / `len` / `isinstance`,
  or none at all:

  ```
  assertion strength : 1036 of 5369 collected tests assert only what a well-formed WRONG answer
                       satisfies = 19.3%
  ```

  A **structural** selector, and that is the point. "Inert under the six convention mutations" is
  not: it selects for *tests something else*, which is why all five tests #1715 named that way are
  genuine cross-path pins. An assertion that only checks well-formedness cannot separate right from
  wrong for **any** input.

- **It is a review queue, not a delete list.** Of the 71 assertion-free tests: **32 are negative
  controls for fail-loud guards** (`test_x_accepts_supported` beside `test_x_fails_loud_on_unsupported`
  — without it the guard could reject everything and its `pytest.raises` siblings would still pass),
  **24 are capability cells** ("can this configuration run at all", a close-out `CLAUDE.md` allows),
  and **15 have no stated purpose**. The first two are assertion-free *by nature*. A promised
  115-deletion PR was withdrawn on that evidence.

  ~~37 capability cells / 15 negative controls / 10 dependency probes, of 115~~ **[CORRECTED]** —
  that reading was taken over the *superseded* population, before the classifier defects below were
  fixed, and it inverted the two largest buckets. Re-derived above over the committed scan.

- **The scanner needed the discipline it enforces, twice.** Three defects found by reading its own
  output: it counted `def test_helper()` **nested inside** a test, which pytest never collects; it
  omitted `pytest.raises`/`pytest.warns`, so **every fail-loud guard in the tree** was flagged; and
  the first figure (21.9%) was taken over that inflated population. Two more found by review:

  - **The frozen-paradigm filter excluded nothing.** `FROZEN = ("alg/neural", "alg/reinforcement")`
    names the *source* layout and matches **zero** files under `tests/`, where those live as
    `test_dgm_*`, `test_pinn_*`, `test_rl_*`. 131 frozen test functions sat in the denominator at a
    47% flag rate. Worse, the test "verifying" it asserted `"alg/neural" in cas.FROZEN` — the
    constant containing itself, which is the tautological shape this very script exists to count.
    It now asserts the **behaviour**: a frozen-named file must be absent from the scan.
  - **The separation assertion was called the weakest class when it is the strongest.**
    `assert not allclose(a, b)` says two things must *differ* — this repo's own doctrine, "assert on
    disagreement, not validity; byte-identity is the defect, not the pass". Every `not` was treated
    as weak, inverting it on **70** tests including `test_coupling_affects_solution` and
    `test_fp_velocity_consumes_cross_density_1071`. Only a bare `assert not x` is weak now.

  Net effect of both: **20.6% → 19.3%**.

- **The staleness line was itself measured over the wrong denominator.** The baseline records
  `"excluded": "tests/unit/test_discrimination_ratchet.py"` and the sweep ignores it; the reporter
  did not, comparing 5872 against 6168 and printing **+5.0%** where the like-for-like figure is
  **+4.6%**. #1901 class 2, inside the instrument built to report class 2. The exclusion is now
  threaded from the field that was already being read.

- **Mutation table, which the first revision shipped without.** Review found 18 of 23 mutations
  surviving 26 tests, because neither test file called `main()` — so nothing about either *printed
  line* was pinned. Now:

  | mutation | reddens |
  |:--|--:|
  | halve the percentage (`100 *` → `50 *`) | 1 |
  | drop `of {then} tests` — the denominator disappears | 2 |
  | `now != then` → `now > then` (a shrinking suite never called stale) | 1 |
  | drop the `--ignore` exclusion from the collect | 1 |
  | invert the fraction (`len(weak)/total` → its complement) | 1 |
  | drop `of {total} collected tests` | 1 |

- **`CLAUDE.md`**: the cited comment date was **2026-07-30**; `created_at` is **2026-07-27**. Fixed.
  The link resolves and does contain `39 (60%)` and `96.8% notice nothing`, verified against the API.

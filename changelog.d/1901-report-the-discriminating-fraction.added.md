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

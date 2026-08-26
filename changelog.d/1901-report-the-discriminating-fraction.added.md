- **The gate prints the discriminating fraction beside the suite result, with its denominator.**
  `N passed` has never been the quantity worth growing, and printing it alone invites growing it.
  `report_discrimination.py` prints the one that says whether green means anything — how many tests
  notice at least one load-bearing convention being broken — together with a staleness line, because
  a recorded fraction ages silently as the suite grows and adding tests lowers it unless they
  discriminate.

  It **reports and does not gate**: measuring it costs a full suite run per mutation, so gating stays
  in the weekly `test_discrimination.py --check-baseline` tier. Skipped under `--fast`.

  A second number joins it: `check_assertion_strength.py` counts tests whose assertions a well-formed
  **wrong** answer would satisfy — `is not None`, `isfinite`, `.shape`, `len`, `isinstance`, or none
  at all. It is a review queue, not a delete list; capability cells and fail-loud negative controls
  are assertion-free by nature.

  **"Inert" is not "worthless".** A test can notice none of the tracked conventions and still pin
  something real — a delegation shim, two entry points that must agree, two independent solvers on
  one fixed point. The deletable set is the *structurally tautological* one, found by reading, not
  the inert one, found by counting.

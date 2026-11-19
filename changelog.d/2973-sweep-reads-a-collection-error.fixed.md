- **The discrimination sweep reads a module the mutation stopped from collecting as that module's tests killed**
  (#2973).
  - `optimal_control_sign` read INEFFECTIVE in every new recording since c6a920ef (#2901 stage 1c-i), and the weekly
    sweep failed on it. #2901's consistency check refuses a sign-flipped control while a problem is built, and
    `test_multi_population_density_2335.py` builds problems at import, so under the mutation that module fails to
    collect. A run short of 99% of the baseline was INEFFECTIVE, though thousands of tests failed in it.
  - `scripts/test_discrimination.py` now collects the baseline's node IDs once. A short run is read for the modules
    that failed to collect. When their baseline tests, together with the tests that errored outside their body,
    number exactly the tests the run is short of, the mutation is `ok`, and those baseline tests are its kills in
    place of the module's path. Any other short run, and any run where pytest exits other than 0 or 1, stays
    INEFFECTIVE. Measured at d8af309a under the sign flip: short by 191, which is the module's 80 baseline tests and
    111 setup errors.
  - The fix is in the instrument, not the module: 67 test modules make 450 calls into `mfgarchon` at import
    (d8af309a), and each fails to collect the same way under a mutation that makes one of those calls raise.
  - The kill parser read pytest's live-log lines at ERROR level as kills, so a kill matrix could name a file that
    does not exist. It now reads only node IDs and collection errors' paths, which start with `tests/`.

- **The discrimination sweep reads baseline tests a mutation left with no outcome as killed, when they account for
  the whole shortfall** (#2973).
  - `optimal_control_sign` read INEFFECTIVE in a new recording since c6a920ef (#2901 stage 1c-i), and the weekly
    sweep failed on it. #2901's consistency check refuses a sign-flipped control while a problem is built. Under the
    mutation, `test_multi_population_density_2335.py`, which builds problems at import, fails to collect (80 tests),
    and 111 tests in 11 modules, whose fixtures build one, error at setup. Neither kind has an outcome, so the run
    was short of the baseline by 191, and a run short by more than 1% was INEFFECTIVE, though thousands of tests
    failed in it.
  - `scripts/test_discrimination.py` now collects the baseline's node IDs once. A short run is read for the modules
    that failed to collect and the tests that errored at setup. When the modules' baseline tests and those errored
    tests number exactly the tests the run is short of, the mutation is `ok`, with the modules' baseline tests among
    its kills in place of their paths, and the log prints the account. A test that has an outcome and then errors at
    teardown is not missing from the run: the account holds only when pytest's setup-error headers, which the sweep's
    own `--tb=short` prints, number the errored tests. Any other run short by more than 1%, and any run where pytest
    exits other than 0 or 1, stays INEFFECTIVE.
  - A module that stops building at import moves its tests from failing to collect to erroring at setup, so the fix
    reads both kinds in the classification, not in that module.
  - The kill parser read ERROR-level records from a failing test's captured log as kills, so a kill matrix could name
    a file that does not exist. It now reads only node IDs and collection errors' paths, which start with `tests/`.

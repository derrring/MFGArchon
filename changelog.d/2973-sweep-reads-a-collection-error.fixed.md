- **The discrimination sweep reads a module the mutation stopped from collecting as that module's tests killed**
  (#2973).
  - `optimal_control_sign` read INEFFECTIVE in a new recording since c6a920ef (#2901 stage 1c-i), and the weekly
    sweep failed on it. #2901's consistency check refuses a sign-flipped control while a problem is built. Under the
    mutation `test_multi_population_density_2335.py`, which builds problems at import, fails to collect, and 111 tests
    whose fixtures build one error at setup; neither has an outcome. A run short of the baseline by 1% or more was
    INEFFECTIVE, though thousands of tests failed in it.
  - `scripts/test_discrimination.py` now collects the baseline's node IDs once. A short run is read for the modules
    that failed to collect. When their baseline tests, together with the tests that errored outside their body,
    number exactly the tests the run is short of, the mutation is `ok`, and those baseline tests are its kills in
    place of the module's path. Any other run short by 1% or more, and any run where pytest exits other than 0 or 1,
    stays INEFFECTIVE. Measured at d8af309a under the sign flip: short by 191, which is the module's 80 baseline tests
    and 111 setup errors.
  - The fix is in the instrument, not the module: 67 test modules make 450 calls into `mfgarchon` at import
    (d8af309a), and a mutation that makes one of those calls raise stops that module from collecting, unless the
    module catches the exception at import.
  - The kill parser read ERROR-level records from a failing test's captured log as kills, so a kill matrix could name
    a file that does not exist. It now reads only node IDs and collection errors' paths, which start with `tests/`.

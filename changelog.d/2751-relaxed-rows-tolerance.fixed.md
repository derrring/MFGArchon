`test_opted_in_relaxed_rows_are_mains`, #1689's oracle for `socp_allow_relaxed_rows=True`, compared the relaxed
rows' L, D and eps_M with main's captured output at `atol=1e-12`, calling that a floor for cross-platform last bits
(#2751). It is not one. Moving the cloud by one ULP moves the rows by up to 3.4e-9 in L, 2.7e-8 in D and 1.2e-11 in
eps_M, and the Linux nightly differed from the macOS capture by 2.3e-10 in L, with the row set and the neighbour
indices exact. The tolerances are now 1e-5 for L and D and 1e-7 for eps_M. A 1% change in either slack penalty moves
the rows by at least 3.8e-2, 8.4e-3 and 2.0e-4, and still fails the test.

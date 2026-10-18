`test_opted_in_relaxed_rows_are_mains`, #1689's oracle for `socp_allow_relaxed_rows=True`, compared the relaxed
rows' L, D and eps_M with main's captured output at `atol=1e-12`, calling that a floor for cross-platform last bits
(#2751). It is not one. On Linux the first comparison to fail was row 73's L, off by 2.3e-10. CLARABEL stopping one
iteration earlier or later in a relaxed solve moves the rows by up to 1.6e-6 in L, 1.3e-5 in D and 6.8e-9 in eps_M,
and a platform's last bits can move such a stop. The tolerances are now 2e-4 for L, 3e-4 for D and 1e-6 for eps_M:
at least 22 times that band, while a 1% change in either slack penalty, at least 3.8e-2, 8.4e-3 and 2.0e-4, still
fails the test. A CLARABEL stop 100 times looser passes, since it moves the rows by the band's own mechanism.

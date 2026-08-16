Three shipped declarations said the FP-FDM path supports Robin. It does not: `_BOUNDARY_HANDLERS`
is keyed on the advection scheme, all four entries are `add_boundary_no_flux_entries_*`, and a
ROBIN segment assembles byte-identically to no-flux (measured at alpha=3.2 and alpha=999). Corrected
`solve_fp_system_1d`'s deprecation docstring, the legacy-BC diagnostic that recommended `robin_bc`
on a path that cannot assemble it, and `robin_bc`'s own docstring — which now states which solvers
do honour a Robin wall (`FPFEMSolver` / `HJBFEMSolver` in weak form, `HJBGFDMSolver` for the
adjoint-consistent case) and which refuse it. (#1975)

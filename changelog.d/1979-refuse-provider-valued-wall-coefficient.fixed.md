A provider-valued wall coefficient no longer disappears (#1979). `BCValueProvider` exists so a wall
coefficient can be recomputed each Picard iterate — that is what `AdjointConsistentProvider` and
#1970's `NormalDriftProvider` are for, and both live on `alpha`, not on `value`. Two consumers took
one and did the wrong thing with it.

**FDM**: `_BOUNDARY_HANDLERS` is keyed on the advection scheme and its handlers take no
`boundary_conditions` argument at all, so nothing read `alpha` and the segment assembled
byte-identically to a plain no-flux wall, with no diagnostic — the wall a user wired a coefficient
to *avoid*, returned as though it were their request. `solve_fp_nd_full_system` now refuses at entry,
naming the offending `segment.field`, the segment types involved, and `with_resolved_providers` as
the remedy.

**FEM**: `assemble_robin_terms` coerced with `float()`, giving a bare builtin `TypeError` — unnamed,
ungreppable, silent about the remedy — three lines above a `NotImplementedError` guard that already
did this correctly for `g`. `alpha` and `beta` now get the same guard, with the same wording.

**Correction to the issue's premise, which changes what the fix is for.** #1979 was filed against a
ROBIN segment, and that route is not reachable: every grid FP solver refuses `ROBIN` at construction
(`_validate_bc_support`, #1456, stated in `conditions.py:1149`), measured — `FPFDMSolver` raises
before any assembly. The reachable case is a provider on the `alpha` of a **NO_FLUX** or **NEUMANN**
segment, which passes the capability gate. That is precisely what `NormalDriftProvider` produces,
since an impermeable wall *is* Robin in `m` (`alpha*m - D*d_n m = 0`) and its coefficient lives
there. So the defect is not narrower than filed; it sits on the path the provider layer was built
for.

Refusing is the fix, not reading. The conservative grid schemes already impose `J·n = 0`
structurally (#1975), so teaching those handlers to add `(alpha, beta, g)` would count the drift
twice — measured there at −79.5% mass against −7.4e-15.

Pinned in `tests/unit/test_geometry/test_provider_robin_coefficient_refused_1979.py`, both paths,
each `raises` paired with a float-coefficient control so that a blanket refusal cannot pass. Both
guards verified mutation-red: removing the FDM one fails 2 of 7, removing the FEM one fails 3 of 7.

Unblocks #1970, which is in draft pending this.

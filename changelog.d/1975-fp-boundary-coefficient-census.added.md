A census pin over every solver's `_SUPPORTED_BC_TYPES`, plus two assertions on the gap it
records (#1975). The Fokker-Planck reflecting wall is Robin in `m`; no FP solver accepts
`BCType.ROBIN`, the only solver that does is on the HJB side where the condition is Neumann, and
`FPResolver` emits exactly the type every consumer refuses. Measured: a mixed `ROBIN(alpha=999)`
wall assembles byte-identically to no-flux, so `(alpha, beta, g)` is read by nothing on the FP
side and the gate refusing ROBIN is honest rather than an oversight. Both gap assertions pin
neither side, so they go green when the gap closes from either end.

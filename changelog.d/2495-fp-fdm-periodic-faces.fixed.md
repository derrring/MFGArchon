FP-FDM reads periodicity from a boundary condition's faces, not from how it is spelt. One PERIODIC
segment per face, or an empty segment list over a periodic `default_bc`, solved as no-flux on every
face, bit for bit and with no error: the periodic test was `bc.is_uniform and bc.type ==
"periodic"`, and every wall node of a non-uniform BC went to the no-flux handler. Both spellings now
solve bit-identically to `periodic_bc`; on the issue's coupled 1-D `FDM_UPWIND` problem the gap was
5.2e-02 in U and 3.2 in M on main. The one predicate is `conditions.periodic_on_every_face`, read
by `periodic_axis_span`, the assembly's wall routing and the linearised operator.

A BC periodic on some faces only -- a channel, or an axis periodic on one face -- was solved the same
way, as no-flux everywhere. FP-FDM wraps every axis or none, so it now refuses such a BC, at
construction and where the wall handlers are dispatched; per-axis FP-FDM is #2505. HJB-FDM, which
reads its ghosts per face, already solved both periodic spellings bit-identically to `periodic_bc`.

`scripts/capability_census.py` and its ratchet: what each class **declares**, and the wall-ratio
**sequence** each FP path produces.

Lane 1 discovers its population with `walk_packages` + `issubclass` + `inspect.isabstract`, keyed
on class identity -- a predicate independent of the declaration it audits -- so "declares nothing"
is a recorded row rather than an absence, and own declarations are separated from inherited ones by
walking the MRO.

**Lane 2 renders no verdict.** An earlier version classified each path from the ratio's trend
across three resolutions; independent review showed the rule cannot do that. `FPFVMSolver` reads
0.392 / 0.649 / 1.106 and keeps going to 12.699 at nx=1281, so a "within tolerance of 1" clause
fires on a value the sequence merely transits, and `FPSLSolver` reads 0.998 at nx=201 then 1.926
and 3.500. Three resolutions cannot separate approach from transit, and neither can six. The
sequence is reported and the reading is left to a person.

Mass drift is reported beside it as a **form property**: neither sufficient (streamline diffusion
conserves to 1e-12 while the ratio collapses) nor necessary (`FPSLJacobianSolver` is the Lagrangian
form, non-conservative by construction, deprecated for adjoint inconsistency rather than for mass).

Also folded: the empty deprecated subclass `FPSLAdjointSolver(FPSLSolver)`, via its own
`_deprecation_meta["alias_for"]` -- class-keying collapses `X = Y` but not `class X(Y): pass`, and
two identical rows read as two independent confirmations. The clip exemption is keyed on the
declared `kde_boundary_smoothing` flag rather than a substring of the class name. (#1975, #1977)

`scripts/capability_census.py` and its ratchet: what each class **declares** about the boundary
conditions it accepts.

The 2026-08-13 design census had four lanes and every one looks for reality falling *short* of a
claim; it found 77 over-claims. Nobody counted the other direction, and that is what made #1975
wrong -- `FPFEMSolver` implements a general Robin wall and declares nothing, so a census keyed on
`_SUPPORTED_BC_TYPES` reported the capability absent.

The population comes from `walk_packages` + `issubclass`, a predicate independent of the
declaration audited, so "declares nothing" is a recorded row rather than an absence; own
declarations are separated from inherited ones by walking the MRO. `honors_inhomogeneous_neumann`
defaults to `True` on `BaseMFGSolver`, so 20 solvers claim to honour an inhomogeneous flux by a
default nobody chose. Three classes apply BC segments without being subclasses of any root and are
named rather than discovered; that list is stated to be incomplete.

A second lane measuring which wall each FP path imposes was removed. Its findings are recorded in
#1975. It needed a discrimination rule for the LIMIT behaviour of a numerical scheme -- four
attempts failed to state one, and 41% of a 32-mutation sweep survived the ratchet built over it,
including the pins for three of the defects that ratchet claimed to have fixed. A ratchet over an
unstated rule pins nothing.

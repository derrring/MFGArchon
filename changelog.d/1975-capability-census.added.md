`scripts/capability_census.py` and its ratchet: what each class **declares** about the boundary
conditions it accepts.

The 2026-08-13 design census had four lanes and every one looks for reality falling *short* of a
claim; it found 77 over-claims. Nobody counted the other direction, and that is what made #1975
wrong -- `FPFEMSolver` implements a general Robin wall and declares nothing, so a census keyed on
`_SUPPORTED_BC_TYPES` reported the capability absent.

The population comes from `walk_packages` + `issubclass`, a predicate independent of the
declaration audited, so "declares nothing" is a recorded row rather than an absence; own
declarations are separated from inherited ones by walking the MRO, and a class bound twice in its
own module is one row. **18 solvers claim to honour an inhomogeneous Neumann flux by inheriting
`BaseMFGSolver`'s `True`, a default nobody chose.** A nineteenth inherits the same field from a
sibling and inherits `False` -- a deliberate refusal, the opposite case, and pinned separately.
Three classes do a root's job without subclassing one; they are named rather than discovered and
that list is stated to be incomplete. ~~"apply BC segments"~~ [CORRECTED] -- `HJBHowardSolver`
reads `seg.bc_type` and `ParticleApplicator` interprets `BCType` throughout, but
`ImplicitHeatSolver` only forwards `bc` to `get_laplacian_operator` and prints a segment in a
`__repr__`. The script's own comment already had the accurate wording.

A second lane measuring which wall each FP path imposes was removed. Its findings are in #1975. It
needed a discrimination rule for the LIMIT behaviour of a numerical scheme -- four attempts failed
to state one, and 41% of a 32-mutation sweep survived the ratchet built over it, including the pins
for three of the defects that ratchet claimed to have fixed.

~~After the removal the same sweep leaves 19% surviving.~~ [CORRECTED 2026-08-17] -- it cannot be
the same sweep. The surviving test set is a strict subset of the one that scored 41%, and deleting
tests turns kills into survivals, never the reverse; 13 survivors cannot become 6 against a smaller
suite over an identical mutation set. The second sweep is a different 32 mutations, retargeted at
lane 1 because lane 2's code no longer exists to mutate, so the two percentages are not comparable
and neither is auditable from this branch -- the harness for THESE 32 mutations is not
committed. (`scripts/test_discrimination.py` is a committed mutation harness, for a different set.) What IS reproducible
is stated per-mutation in the PR body, each with its liveness check.

The script emits JSON only. An earlier revision also printed a prose report; nothing invoked it
(zero references outside this changelog), it restated what the test file asserts with nothing
keeping the restatement honest, and a review round went to correcting its labels -- it had called
an inherited `False` and a `property` object "permissive defaults". 53 lines removed rather than
relabelled: a mechanism with no invocations is fairly judged by that, and a second surface saying
what the assertions already say is the failure this census exists to find, committed by the census.

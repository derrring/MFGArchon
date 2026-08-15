Applicators now declare which `BCType` members they handle, and refuse the rest.

`_SUPPORTED_BC_TYPES` and `_validate_bc_support` mirror the mechanism the solver layer has used
since #1456 rather than introducing a second one — `dispatch.py` already calls that gate "the
authoritative source", and it fires. The applicator layer had a namesake, `supports_bc_type`, whose
body checks the *dimension* rather than the type and which has no callers anywhere; it is left in
place as public surface and superseded.

Measured over the `(applicator × BCType)` product before declaring: an unhandled type produced four
different outcomes across the family — a silent fall-through, a bare `pass`, `else: 0.0`, and a
raise. Only the raise is right. `ImplicitApplicator` with `PERIODIC` returned the field untouched
and now refuses it.

The conformance table derives its cell count from the enums, so adding a `BCType` member makes it
fail until someone decides what each applicator does with the new member. Declaring a type asserts
that a branch exists, not that it is correct; correctness is a separate axis tracked per cell.

Both silent cells are closed. `ImplicitApplicator` no longer declares `PERIODIC`, and
`MeshfreeApplicator` no longer declares `NO_FLUX`: its branch was a bare `pass` whose own comment
said boundary values "should match nearby interior" and then kept them unchanged, which makes no
claim at all. It now raises for the same reason `NEUMANN` already did — a meshfree method has no
ghost layer, so a normal derivative needs the solver's own operators. Measured before withdrawing
it: across `tests/unit/test_geometry` and `tests/unit/test_alg` (2857 tests) `MeshfreeApplicator.apply`
is reached 8 times, all from the conformance table, and never with `NO_FLUX`.

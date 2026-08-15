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

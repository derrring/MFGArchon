A test file recording which solvers declare their BC support and that the FDM boundary assembly
reads none of a ROBIN segment's coefficients (#1975, #1977, #1979).

Assertions only. Four revisions put the reasoning, the history and the physics in the file as
prose, and four independent reviews found a false statement in that prose each time -- five wrong
numbers, one wrong sign, and a correction of a reviewer that was itself wrong. Nothing is now
stated that the file does not compute; the narrative lives in #1975.

The population is discovered by `walk_packages` + `issubclass` + `inspect.isabstract`, keyed on
class identity, with the root package imported explicitly and cross-module re-exports collapsed by
the defining-module filter. Verified against injected escapes: a solver in `alg/__init__.py`, one
named `Baseline*`, a factory-built class, an alias sorting alphabetically earlier, and the deletion
of a dead alias.

The gate reads `supported_bc_types`; this file reads `_SUPPORTED_BC_TYPES`. Those coincide only
while every property forwards, so that premise is now itself asserted: all 11 declaring solvers
have a property whose body is exactly `return self._SUPPORTED_BC_TYPES`, and the 11 that declare
nothing have no property at all. Widening the live gate through the property while leaving the
private attribute alone previously left the file green.

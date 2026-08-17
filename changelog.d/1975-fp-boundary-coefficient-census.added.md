A test file recording which solvers declare their BC support and that the FDM boundary assembly
reads none of a ROBIN segment's coefficients (#1975, #1977, #1979).

Assertions only. Five revisions put the reasoning, the history and the physics in the file as
prose, and five independent reviews found a false statement in that prose each time -- six wrong
numbers, one wrong sign, a correction of a reviewer that was itself wrong, and an argument for
keeping a filter that measurement showed did nothing. No NUMBER is now stated that the file does
not compute; the traced source references it carries instead each name a file and a line.

The population is discovered by `walk_packages` + `issubclass` + `inspect.isabstract`, keyed on
class identity, with the root package imported explicitly and the canonical name pinned to
`cls.__name__` so an alphabetically earlier alias cannot displace it. There is no defining-module
filter: it gained nothing (21 classes with it, 21 without) and hid one real cross-module alias,
`HJBNetworkSolver` at `network_solvers/__init__.py:19`, from the very test written to catch that
case. Verified against injected escapes: a solver in `alg/__init__.py`, one named `Baseline*`, an
alias sorting alphabetically earlier, a cross-module alias, and the deletion of a dead alias.

A factory-built class whose `__module__` points at another scanned module is NOT caught -- that
escape is real as a mechanism and has no instance in this tree (measured: 0 classes defined in a
scanned module and never bound there).

The gate reads `supported_bc_types`; this file reads `_SUPPORTED_BC_TYPES`. Those coincide only
while every property forwards, and that premise is NOT asserted. A revision asserted it by
comparing `inspect.getsource(...).splitlines()[-1]`; review measured that open in three shapes and
falsely closed on a fourth, so it is deleted rather than repaired and the hole is stated in the
file: a solver can widen its live support through the property alone and every assertion stays
green. Of 21 concrete solvers, 11 declare and 10 do not.

Two disclosed scope limits. The population predicate is `BaseNumericalSolver`, while
`_validate_bc_support` and `honors_inhomogeneous_neumann` live on `BaseMFGSolver`: 4 concrete
classes (`PrimalDualMFGSolver`, `SinkhornMFGSolver`, `VariationalMFGSolver`,
`WassersteinMFGSolver`) are in the gate's reach and outside the census. None mentions a boundary
condition anywhere, so none hides a live defect. And the behavioural oracle is 1D with
`interface_velocity=None`, so the conservative wall's multi-axis branch and its interface-velocity
branch are exercised by nothing here.

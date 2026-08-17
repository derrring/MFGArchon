A test file recording which solvers declare their BC support and that the FDM boundary assembly
reads none of a ROBIN segment's coefficients (#1975, #1977, #1979).

Assertions, with the reasoning in #1975. Six revisions put the reasoning in the file as prose and
six independent reviews found a false statement in it each time -- including in the sentences that
announced the file's own discipline, which were false in the same commit that added them. Those
sentences are gone; what the file can compute it asserts, and what it cannot it cites at a line.

The population is discovered by `walk_packages` + `issubclass` + `inspect.isabstract`, keyed on
class identity, with the root package imported explicitly and the canonical name pinned to
`cls.__name__` so an alphabetically earlier alias cannot displace it. There is no defining-module
filter. Removing it gained no classes and admitted one name -- the cross-module alias
`HJBNetworkSolver` at `network_solvers/__init__.py:19`, which the test written for that case could
not see -- and it also closed the factory escape, since a `type()`-built class over an `ABCMeta`
base gets `__module__ == 'abc'` and was rejected in every scanned module while the filter stood.
Verified against injected escapes: a solver in `alg/__init__.py`, one named `Baseline*`, an alias
sorting alphabetically earlier, a cross-module alias, a factory-built class with a retargeted
`__module__`, and the deletion of a dead alias.

The gate reads `supported_bc_types`; this file reads `_SUPPORTED_BC_TYPES`. Those coincide only
while every property forwards, and that premise is NOT asserted. A revision asserted it by
comparing `inspect.getsource(...).splitlines()[-1]`; review measured that open in three shapes and
falsely closed on a fourth, so it is deleted rather than repaired and the hole is stated in the
file: a solver can widen its live support through the property alone and every assertion stays
green.

Two scope limits. The population predicate is `BaseNumericalSolver` while `_validate_bc_support`
and `honors_inhomogeneous_neumann` live on `BaseMFGSolver`, so four concrete classes are in the
gate's reach and outside the census; that gap is now computed by a test rather than described,
because the prose version of it shipped a false negative claim about their contents. And the
behavioural oracle is 1D with `interface_velocity=None`, so the conservative wall's multi-axis and
interface-velocity branches are exercised by nothing here.

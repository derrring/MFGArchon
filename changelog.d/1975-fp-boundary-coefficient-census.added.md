A test file recording which solvers sit outside the BC capability gate, and that the FDM boundary
assembly reads none of a ROBIN segment's coefficients (#1975, #1977, #1979).

The population is discovered by `walk_packages` + `issubclass` + `inspect.isabstract`, keyed on
class identity: a predicate independent of the declaration it audits. Two same-module alias pairs
(`NetworkFPSolver = FPNetworkSolver`, `NetworkHJBSolver = HJBNetworkSolver`) are collapsed and
pinned separately, the root package is imported explicitly because `walk_packages` never yields it,
and the `startswith("Base")` name heuristic is gone -- it excluded any concrete solver whose name
began with "Base" and bought nothing `isabstract` did not already cover. Verified against four
injected escapes: a solver in `alg/__init__.py`, one named `Baseline*`, a factory-built class, and
the deletion of a dead alias.

No count is restated in prose. Three independent attempts at this file produced three different
counts, each written down as a fact; the frozen sets are the measurement.

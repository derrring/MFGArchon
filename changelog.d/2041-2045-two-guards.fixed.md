Two guards, each closing a case where the answer was right for the wrong reason.

`_solved` in `scripts/capability_matrix.py` keyed its trigger on a hardcoded pair, so
`fvm_picard_converged` could **veto** a verdict but never cause one to be evaluated: an artifact
whose only convergence evidence said the solve did *not* converge fell through to the
"absent means not applicable" branch and gated **green** (#2041). Every `*picard_converged` key now
both triggers and contributes, so a future prefix is covered by construction.

`NetworkPolicyIterationHJBSolver` sets `_honors_node_bc` and therefore skips its base class's
deliberate node-BC refusal, falling through to an unguarded `problem.num_nodes` — an
`AttributeError` from inside construction, indistinguishable from a defect in the solver, where its
sibling raises `NotImplementedError` naming the reason (#2045). Both now refuse loudly. The guard is
a `try/except AttributeError` around all three network reads rather than `hasattr` on one: the
fail-fast policy names try-except as the replacement (a `hasattr` moved that ratchet 107 → 108), and
a single-attribute guard would wave through a problem carrying `num_nodes` but no adjacency accessor.

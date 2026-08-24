`inner_solver='howard'` no longer hard-requires SOCP-precomputed stencils. It runs on whatever
derivative operator the provider carries, and **warns** rather than refusing when those stencils are
not monotone (#2066).

Howard needs three things: `D_lap`, `D_grad`, and an interior/boundary split. **None is
SOCP-specific.** `get_derivative_weights` on the live `TaylorOperator` / `LocalRBFOperator` returns
the same dict keys the SOCP object does — `neighbor_indices`, `grad_weights`, `lap_weights`,
`center_idx_in_neighbors` — which is the entire contract `_build_dlap_from_socp` and
`_build_dgrad_central` consume.

**Monotonicity is a real hypothesis, but the gate was not enforcing it.** Policy iteration's global
convergence assumes a monotone consistent stable scheme (Bokanowski–Maroso–Zidani 2009, quoted in
this module's own docstring). That is a statement about **convergence**, not about whether the solve
can run — and this module already ships `discretisation="central"`, documented as *"Does NOT preserve
monotonicity under advection-dominant regime; included for comparison only"*. Refusing at the door
while offering that was inconsistent.

Worse, the gate did not deliver the property it appeared to guard: a `joint_socp` run logs
SOCP-infeasible interior nodes falling *"through to bare Wendland-Taylor LSQ (NON-MONOTONE)"* and the
check passed anyway, because it tested for the **object** rather than for monotonicity.

**The interior/boundary split was a by-product of SOCP feasibility.** `interior_mask` was built from
the keys of `socp_data`, so a point the SOCP could not solve silently became a *boundary* point and
received BC treatment instead of interior treatment. Where the provider carries a real classification
(`boundary_indices`), that is now used.

**The SOCP path is unchanged.** Where `_joint_socp_stencils` exists, both the operator source and the
split behave exactly as before; nothing about an existing run moves. The refusal is narrowed, not
removed: no SOCP **and** no operator is still a hard error, because then there is no source for the
operators at all.

Measured: a provider built without `joint_socp` now constructs (with the non-monotone warning) and
solves — finite, correct shape, terminal preserved bit-for-bit, non-constant field. Against `main`
the same construction raises `RuntimeError: has no _joint_socp_stencils`.

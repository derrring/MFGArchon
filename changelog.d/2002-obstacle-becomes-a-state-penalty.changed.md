**BREAKING.** `problem.obstacle` is retired and raises (#2002). It was documented as the
variational inequality `v >= Psi(x)` and implemented as `max(0, Psi(x))` — a term with no `v`,
byte-identical at a node satisfying the constraint and one violating it. It resolved as a **soft
wall**, which is what it always computed: a cost that is `alpha`-free and `u`-free is a POTENTIAL,
so it becomes `problem.state_penalty` (cost-signed, positive where expensive, with
`state_penalty_scale`) and is composed into the Hamiltonian's `V` at problem construction rather
than into `source_term`. The composition SUBTRACTS, because `potential` is reward-signed
(G-001) — measured, not assumed: a potential amplitude of `-5` raises `u(0, mid)` to `+0.555`
while `+5` lowers it to `-1.419`. The obstacle branch is deleted from `compose_hjb_source` in the
same change. The **variational inequality** remains a reserved, deliberately unfinished slot:
`HJBFDMSolver(constraint=ObstacleConstraint(...))` (#591, #2036, #2046). Note `obstacles` (plural,
geometric regions excluded from the domain) is a different field and is unaffected.

The composition COPIES the Hamiltonian rather than rebuilding it (an earlier draft rebuilt from
five of its six constructor parameters and silently dropped `population_index`, which a
multi-population problem needs), returns exactly what the base potential returns (so a vectorised
base stays vectorised), and updates `_lagrangian_class` as well — `MFGComponents` snapshots the
potential there at construction, and `HJBSemiLagrangianSolver` reads that copy whenever the control
cost is non-smooth, so composing into the Hamiltonian alone left a solver silently unwalled.

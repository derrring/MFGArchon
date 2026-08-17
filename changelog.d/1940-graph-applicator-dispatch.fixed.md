`GraphApplicator.apply` silently returned the field unmodified for any `field_type` other than
`"value"` or `"density"`, including `"VALUE"`. `Literal` is a type annotation, not a runtime check,
and every branch tested for one of the two names, so a typo applied no boundary condition and raised
nothing. It now raises.

Its two boundary-node detectors used `adj.sum(axis=1)` — the weighted degree, i.e. the node
*strength* — where they meant the combinatorial degree. On a path graph `0-1-2-3` with edge weight
0.5 they returned nodes 1 and 2, the two non-leaves, so the boundary condition was applied to the
interior. Both now share one helper using `(adj != 0).sum(axis=1)`, the quantity a leaf detector needs. There
is no single owner to route through: of the three `network_backend` implementations of
`node_degrees`, igraph and networkit return the combinatorial degree and networkx returns the
weighted one, which is a pre-existing single-source violation filed separately. The weighted sum is
left untouched at the six sites that feed a graph Laplacian or a CFL bound, where it is correct.

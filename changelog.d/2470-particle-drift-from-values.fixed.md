`FPParticleSolver` now takes the drift at a wall node from U's own slope rather than from the finite-difference ghost (#2470). This is the change #2467 made for the semi-Lagrangian pair, and both now share one owner, `mfgarchon.operators.differential.gradient.value_gradient`.

- On a non-periodic axis, the wall nodes take U's second-order one-sided difference.
- Interior nodes and periodic axes are unchanged.
- The solver's own `boundary_conditions` is still the BC the gradient is taken with (#1255).

**Before.** The cell-centred no-flux ghost gave the wall half U's slope. For U = -0.4 x the gradient at both walls was -0.2 (a drift of +0.2); it is now -0.4 (a drift of +0.4).

**What it prevents.** The node-centred ghost #1935 needs would have made the wall drift 0, whatever U is. Particles leaving a wall then stall: the mean position at T was off by 6.6e-02 at 51 points, against 6.1e-03 with this change.

The non-tensor-grid path is unchanged.

**Newly refused.** The refusals of #2467 now apply here too, and both ran before:

- a geometry mixing periodic and non-periodic faces with a segment that has no `boundary`, for example a region-only exit with a periodic default;
- an axis periodic on one face only.

For the region-only exit, the two face resolvers disagree about which faces the segment covers, so its periodicity cannot be read. Two shapes are accepted:

- a segment named by its face and restricted to part of it, such as a door;
- a uniform BC, whatever its unused `default_bc`.

**Also narrowed.** The #2467 refusal itself is narrowed to segments with no `boundary` in a BC that is not uniform. It had also refused two shapes that both resolvers read the same way:

- face-named segments carrying a region;
- one periodic segment beside a non-periodic default, the shape `mixed_bc` and `resolution.to_boundary_conditions` produce.

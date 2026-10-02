`FPParticleSolver` now takes the drift at a wall node from U's own slope rather than from the finite-difference ghost (#2470). This is the change #2467 made for the semi-Lagrangian pair, and both now share one owner, `mfgarchon.operators.differential.gradient.value_gradient`.

- On a non-periodic axis, the wall nodes take U's second-order one-sided difference.
- Interior nodes and periodic axes are unchanged.
- The solver's own `boundary_conditions` is still the BC the gradient is taken with (#1255).

**Before.** The cell-centred no-flux ghost gave the wall half U's slope. For U = -0.4 x the drift at both walls was -0.2; it is now -0.4.

**What it prevents.** The node-centred ghost #1935 needs would have made the wall drift 0, whatever U is. Particles leaving a wall then stall: the mean position at T was off by 6.6e-02 at 51 points, against 6.1e-03 with this change.

The non-tensor-grid path is unchanged. The new refusals of #2467 apply here too: a geometry mixing periodic and non-periodic faces through a segment not named by a recognised face, and an axis periodic on one face only.

- **`HJBResolver` refuses a `BCType` it cannot map, as `FPResolver` already did** (Issue #2293).
  It logged a WARNING and returned `ResolvedBC(NEUMANN, g=0)` — a homogeneous reflecting wall
  indistinguishable at the call site from a real resolution, on a condition nobody wrote. Both
  resolvers now raise, which is the "total-or-fail-loud" contract #1471 wrote down for the layer.

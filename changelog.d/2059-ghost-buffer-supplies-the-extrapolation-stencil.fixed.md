**BREAKING for direct callers of the extrapolation calculators.** `GhostBuffer` served a *different
boundary condition* than `pad_array_with_ghosts` for `EXTRAPOLATION_LINEAR` and
`EXTRAPOLATION_QUADRATIC`, and both calculators now refuse rather than degrade (#2059).

`GhostBuffer.update()` never supplied `second_interior_value`, so `LinearExtrapolationCalculator`
took its `return interior_value` fallback — **the zero-gradient ghost**. Not a rare degradation: on
that path it was the *only* branch that ever ran, so every `EXTRAPOLATION_LINEAR` request served
through `GhostBuffer` got a Neumann-0 wall, silently.

Measured on `u = [2.5, 3.1, 4.0, 5.2, 6.6]`, `ghost_depth=1`:

| | `pad_array_with_ghosts` | `GhostBuffer` before | after |
|---|---|---|---|
| LINEAR, low ghost | 1.9 | **2.5** — literally `u[0]` | 1.9 |
| LINEAR, high ghost | 8.0 | — | 8.0 |
| QUADRATIC | 2.2 / 8.2 | — | 2.2 / 8.2 |

**The repo had already decided this question**, in the path that works. `pad_array_with_ghosts` reads
the stencil inward from the wall (`buf[g + j]` low, `buf[-g - 1 - j]` high) and **refuses** when the
grid cannot carry it, with the reason written in the code: *"Refuse rather than silently dropping to
a lower order."* `GhostBuffer` simply had not followed. It now uses the same indices, so the two
public paths compute the same ghost from the same cells.

The quadratic calculator's degradation chain was worse than a single fallback: `quadratic → linear →
edge extension`, all silent, so a caller asking for `EXTRAPOLATION_QUADRATIC` could receive any of
**three different boundary conditions** depending on how many arguments happened to arrive. Both
calculators now raise, naming the formula and what the substitution would have imposed instead.

The test's oracle is **agreement between the two public paths**, not a formula restated in the test —
with #1958's own measured figures pinned alongside, so a change breaking both paths identically is
still caught.

Howard's decomposition guard had two coverage holes that made it **accept what it exists to refuse**
(#2072). Both are pre-existing on `main`, independent of #2069.

**The gradient bound had the wrong dimensional content, and that is what made the probe miss.**
`_gT` was `spread / hmin` — a *global* range divided by a *local* spacing — so it **diverges under
refinement** while the gradient it stands for converges:

| `u_T = cos(2πx)` | nx=11 | nx=21 | nx=41 | nx=201 |
|---|---|---|---|---|
| `spread/hmin` | 10 | 20 | 40 | 200 |
| discrete Lipschitz | 3.090 | 3.090 | 3.129 | 3.141 |
| true `max\|du/dx\|` | 3.142 | 3.142 | 3.142 | 3.142 |

On a linear ramp it gives 200 where the true gradient is 1. This is why the *previous* widening
failed: told the probe missed `|p| = 6.18`, it widened the ladder to multiples of a quantity 6.5×
too large, moving the new rungs **further from** the hole. `_gT` is now
`max |u_i − u_j| / |x_i − x_j|` — a genuine upper bound that converges to the truth, over arrays
already materialised at that point.

**The ladder is denser, not hole-free**, and an earlier revision of this note claimed otherwise.
Adjacent rungs of a 12-point geomspace sit at a ratio of ~1.59, so a bump narrower than that in
relative width still fits between two. What closes the operating-range hole is the corrected bound
putting the rungs where the solve lives. The old ladder's **exact** rungs at `0.5·_gT` and `_gT` are
kept in the union rather than dropped — a geomspace grid lands on neither — and the low end is
`min(0.5, _gT/4)` rather than a hard `0.5`, which stopped scaling down when `_gT < 1`.

Measured, all with **no alpha-free part** so the alpha-free measurement cannot be what catches them:

| fixture | old bound + old ladder | corrected |
|---|---|---|
| bump on \|p\| ∈ (5.1, 7.9), straddling the operating magnitude | **ACCEPTED** | **REFUSED** |
| bump on \|p\| ∈ (3, 10) | **ACCEPTED** | **REFUSED** |
| `H = \|p\|²/2` exactly (control) | ACCEPTED | ACCEPTED |

**A single NaN defeated the guard, and the comment claimed the opposite:**

```
np.maximum(0.0, nan) = nan     nan > tol = False   -> ACCEPT
max(0.0, nan)        = 0.0     0.0 > tol = False   -> ACCEPT
```

Identical. Measured on a quartic returning NaN at one probe magnitude: **accepted**, all-finite
output, **153% wrong** against Newton. The guard now refuses on a non-finite `_af`, `_ke` or
`_scale` before comparing anything to a tolerance.

The NaN fixture places its NaN at `|p| = 1`, which is in the unconditional `{0.5, 1, 2}` union. An
earlier revision used `|p| = 80` — a magnitude that existed only because `spread/hmin` was 40 on
that fixture, so the test was coupled to the very quantity this change replaces and would have gone
green-by-vacuity the moment the bound was corrected.

**False-refusal check.** No legitimate configuration is newly refused: the review that found the
ladder defect swept 52 (declared `control_cost` with λ ∈ [0.1, 1000], potentials and couplings
across 9 decades, four grid sizes, four terminal amplitudes) with a validated positive control and
found **0**. My own confirmation covered 8 — 24 of my 32 intended configurations never reached the
guard, failing at construction on the `potential` signature, which an accounting line caught before
the result was reported as 32.

**Not changed here**, and now its own PR: `hjb_howard.py:506` returned the previous timestep with a
`logger.warning` on a non-finite policy iterate.

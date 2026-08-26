- **`_compat.py`'s Neumann ghost branches delegate to `ghost_cell_neumann` instead of restating a
  retired convention** (Issue #2067). `_compute_ghost_pair` took one `g` and handed it to two
  branches four lines apart that read it as **different quantities**: the Neumann branch applied one
  signed value with opposite signs at the two walls, which is `du/dx` by construction, while
  `ghost_cell_robin` beside it reads `du/dn`. Whichever convention a caller held, one of those two
  branches was wrong at the low wall.

  Measured on `u = 3x`, `dx = 0.1`, cell-centred, exact ghosts `−0.1500` / `+3.1500`:

  | branch | fed `g = du/dx = +3` | fed `g = du/dn` |
  |:-------|:---------------------|:----------------|
  | NEUMANN | −0.1500 / +3.1500 | low wall **+1.0500** |
  | ROBIN(α=0, β=1) | low wall **+0.4500** | −0.1500 / +3.1500 |

  **It was not only the convention, and this is the half the issue did not have.** The branch also
  used the `2*dx` separation measured off the **second** interior cell, the form #1972 retired from
  `ghost_cells.py` for being self-consistent with a wrong wall position. So it disagreed with the
  owner at `g = 0` as well — that is, on a plain no-flux wall, which is the default:
  **0.300** on `u = 3x` and **0.588** on `u = sin(2πx)`. A constant field was the only one that
  agreed, which is why nothing caught it.

  Both Neumann branches (`_compute_ghost_pair` and `_compute_single_ghost`) now call
  `ghost_cell_neumann(u_int, g, dx)`. The `side` split in the second one existed only to undo the
  `du/dx` sign and is gone with it. Implementation count of the inline form in `mfgarchon/`:
  **3 → 0**.

  **The pin is an external oracle, not path-A-vs-path-B**, because agreement with
  `ghost_cell_neumann` is tautological the moment the branch calls it — and a characterization test
  restating the branch's own arithmetic is exactly what carried the retired convention through two
  reviews. `test_a_neumann_wall_reproduces_a_linear_field_exactly` feeds `u = slope*x` at three
  slopes and asserts the ghost continues it exactly. Verified against two mutations — reading `g`
  as `du/dx`, and reverting to the `2*dx` separation — each of which turns the characterization
  case and the `slope = 3.0` and `slope = −1.7` cases red while **`slope = 0.0` survives both**,
  which is the non-discriminating input named in the test's own docstring.

  Two characterization expectations move with the fix and say why in place:
  `(2.75, 5.55) → (2.675, 6.775)` in the #1961 file, and `field[1]`/`field[-2]` → `field[0]`/
  `field[-1]` in `test_hjb_fdm_solver.py`.

  `get_ghost_values_nd` stays deprecated with its declared v0.25.0 removal (#1955); this makes it
  satisfy the deprecation policy's first clause — the old API calls the new one internally — which
  it did not. **#2068 is the same divergence one function over**, in `ghost_cell_fp_no_flux`'s
  vertex-centred branch, and is not touched here.

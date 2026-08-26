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
  `ghost_cells.py`. So it disagreed with the owner at `g = 0` as well — on a plain no-flux wall,
  which is the default. Measured on a **cell-centred** grid, `dx = 0.1`: **0.300** on `u = 3x` and
  **0.500** on `u = sin(2πx)`. (An earlier draft printed `0.588` for the sine inside a cell-centred
  discussion; `0.587785 = sin(2π·0.1)` needs nodes at `j·dx`, the vertex layout. `u = 3x` gives
  0.300 on both, which is why the pair looked consistent.) A constant field is the only one that
  agrees on either layout.

  **The two forms are order mirror images, and this says so rather than implying the retired one was
  simply wrong.** Measured on `u = sin x + 0.3x²` with `du/dn` prescribed at the min wall:

  | centring | `u_int + dx·g` (delegated) | `u_next ∓ 2dx·(du/dx)` (retired) |
  |:---------|:---------------------------|:---------------------------------|
  | **cell** | rate 3.00 → `O(h³)` | rate 1.92, 1.96, 1.98 → `O(h²)` |
  | **vertex** | rate 2.04, 2.02, 2.01 → `O(h²)` | rate 3.00 → `O(h³)` |

  At a vertex wall the retired form's difference is centred **on the wall node** — the classical
  no-flux mirror — and is a full order better. Three things make delegating right anyway: no caller
  passes a non-default `GhostCellConfig` to `get_ghost_values_nd`; `u_int + dx·g` is what
  `ghost_cell_neumann`, `ghost_cell_robin` and `NeumannCalculator` all produce at **both**
  centrings, so delegating makes `_compat` agree with the live path where keeping the old form
  would make it disagree; and the order question therefore belongs to `ghost_cell_neumann` and
  #1972, not here. **#2129 records that #1972's stated evidence — "verified exact on 12
  combinations … against `u = a·x`" — cannot discriminate the two**, because a linear field makes
  them identical to machine precision at both centrings and both walls.

  Both Neumann branches (`_compute_ghost_pair` and `_compute_single_ghost`) now call
  `ghost_cell_neumann(u_int, g, dx)`. The `side` parameter of the second existed only to undo the
  `du/dx` sign; nothing else in that function read it, so it is deleted along with both call sites'
  arguments. Counting the inline form in `mfgarchon/`, and saying which count, because
  `ghost_cells.py:311` warns that a tally over a hand-chosen literal cannot audit the predicate that
  chose it: **executable lines 3 → 0**, expressions 4 → 0, dispatch sites 2 → 0.

  **The pin is an external oracle, not path-A-vs-path-B**, because agreement with
  `ghost_cell_neumann` is tautological the moment the branch calls it — and a characterization test
  restating the branch's own arithmetic is exactly what carried the retired convention through two
  reviews. `test_a_neumann_wall_reproduces_a_linear_field_exactly` feeds `u = slope*x` at three
  slopes and asserts the ghost continues it exactly.

  Verified at full-suite scope, and the counts below are against the test set this PR **ships**, not
  the one it started from — an earlier draft quoted 3 and 4, measured before the mixed-face test
  below existed. Reading `g` as `du/dx` in both branches turns **5** red; reverting the paired
  branch to the verbatim pre-#2067 arithmetic turns **4**, one of them
  `test_hjb_fdm_solver.py::test_get_ghost_values_nd_neumann` in another file. `slope = 0.0` survives
  both, which is the non-discriminating input the test's docstring names, and it does not survive
  everything — replacing the branch body outright kills it too.

  **What the linear oracle cannot separate, stated rather than left to be found:** the `2·dx`
  **vertex mirror** `u_neighbor + 2·dx·g` reproduces a linear field exactly as well, so it reddens
  neither oracle test. Two characterization rows separate it — the `neumann` row here and
  `test_get_ghost_values_nd_neumann` in `test_hjb_fdm_solver.py`, which does it at `g = 0` where
  `u_next ≠ u_int` — and that is measured, not assumed: the vertex-mirror mutation reddens exactly
  those two.

  **`_compute_single_ghost`'s Neumann branch had zero discrimination and now has a pin.** Replacing
  its whole body with a constant turned nothing red in 6610 tests: the only test executing that line
  asserts the refusal beside it, not the value.
  `test_a_neumann_face_on_a_MIXED_boundary_reproduces_a_linear_field` routes a non-uniform
  BoundaryConditions through it and applies the same linear oracle; the constant-body mutation now
  turns exactly its two cases red.

  Two characterization expectations move with the fix and say why in place:
  `(2.75, 5.55) → (2.675, 6.775)` in the #1961 file, and `field[1]`/`field[-2]` → `field[0]`/
  `field[-1]` in `test_hjb_fdm_solver.py`.

  `get_ghost_values_nd` stays deprecated with its declared v0.25.0 removal (#1955); this makes it
  satisfy the deprecation policy's first clause — the old API calls the new one internally — which
  it did not. **Clause 2, the `old == new` equivalence test, is discharged by construction rather
  than owed**: from this change `_compat` calls `ghost_cell_neumann` directly and
  `pad_array_with_ghosts` reaches the same function through `NeumannCalculator`, so any such test
  compares an owner to itself and nothing can redden it. That is the objection this file's own
  header already raises for the Robin path. **#2068 is the same divergence one function over**, in `ghost_cell_fp_no_flux`'s
  vertex-centred branch, and is not touched here.

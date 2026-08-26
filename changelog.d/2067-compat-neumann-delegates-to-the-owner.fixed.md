- **`_compat.py`'s Neumann ghost branches call `ghost_cell_neumann` instead of restating it**
  (Issue #2067). `_compute_ghost_pair` took one `g` and handed it to two branches four lines apart
  that read it as different quantities: the Neumann branch applied one signed value with opposite
  signs at the two walls, which is `du/dx`, while `ghost_cell_robin` beside it reads `du/dn`.
  Whichever convention a caller held, one of them was wrong at the low wall.

  It also measured across `2·dx` from the *second* interior cell, so it disagreed with the owner at
  `g = 0` as well — on a plain no-flux wall, which is the default. Only a constant field agreed.

  **This is a behaviour change on a deprecated path.** `get_ghost_values_nd` keeps its declared
  v0.25.0 removal (#1955); the change is what makes it satisfy the deprecation policy's first
  clause, that the old API calls the new one internally. Two characterization expectations moved
  with it and say so in place.

  **The two forms are order mirror images**, which is why nothing symmetric in the two centrings
  could have chosen between them: at cell centring the shipped rule is `O(h³)` and the retired one
  `O(h²)`; at vertex centring it is the reverse. Delegating is right because the whole live path
  uses this convention, not because it is uniformly more accurate — and the vertex half of that
  convention is what **#1904** and **#1935** are open on.

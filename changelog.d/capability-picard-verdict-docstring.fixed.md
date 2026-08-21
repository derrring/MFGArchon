Corrected `_picard_verdict`'s docstring in `scripts/capability_matrix.py`, which still opened with
"Deliberately NOT part of any cell's `ok`". That has been false since #1893 routed all four verdict
lines through `_solved`, which gates on exactly this field — the record and the gate sit one line
apart in every cell. Struck in place with a pointer to the four line pairs; the measurement below
it is kept, since it is what justified turning three greens red rather than raising the iteration
budgets (#1871).

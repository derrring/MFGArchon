- **`levy_integro_diff.py`'s "Tests MUST include non-uniform grid adjoint consistency" is enforced in
  CI again** (Issue #2268). The test satisfying it was deleted by `18d8cc80` (#2227). Restored as the
  binding half only — the discrete duality `<J[v], m>_W == <v, J*[m]>_W` on a non-uniform grid —
  rather than the whole deleted file, whose remaining cases were happy-path assertions the #2227 bar
  removed on purpose. Note the operator's own `if __name__ == "__main__"` block already asserted this
  at `1e-10`; nothing runs it, so the gap was in CI rather than in the repository.

  Admitted under **class 1, it kills a mutation** — three distinct `apply_adjoint` mutations (drop the
  `W^{-1}...W` conjugation, drop either half alone), each failing 3 of the 5 tests, which is the
  ceiling since only 3 call `apply_adjoint`. Named in the file and deliberately not added to
  `discrimination_baseline.json`; AGENTS.md admits a test on a convention in neither registry "under
  class 1 or 2 on its own merits". Not an external oracle, which the first draft claimed. `apply_adjoint` reaches `J^T`
  through `as_sparse()`, which is assembled from `_matvec`, so the identity holds for any linear `J`
  and positive `W`: scaling the operator by 3, dropping the `-v(x)` term, or dropping the Lévy
  density all leave five tests passing. A *non-linear* wrong `J` is caught (an affine return fails 3
  of 5), so the scope is "cannot see a wrong linear `J`". What it does pin is `apply_adjoint` against
  `_matvec`, where dropping the `W^{-1}...W` conjugation or either half of it fails 3 of 5 — the
  defect this issue names.

  Two things measured while writing it, both contradicting the constraint's obvious reading. A
  `linspace` grid does **not** give `W ∝ I` — trapezoidal weights halve both endpoints, so the W-free
  transpose fails there too, by order 1 — and restricting the pairing to interior indices does not
  rescue it, because `(W^{-1} J^T W m)_i` sums over every `j`. What the non-uniform grid uniquely
  buys is **interior** weight variation (ratio 2.19 against 1.00 for `linspace`).

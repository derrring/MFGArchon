- **The 1-D FDM Jacobian's advection block is derived from the same gradient the residual applies,
  not restated by hand** (Issue #1896). The hand-written stencil was right in the interior and wrong
  at both walls under every BC — and the missing central entry has the *opposite* sign under no-flux
  and Dirichlet, so no single correction to it could have covered both. The bands now come from the
  two operators that are linear and already have owners, `forward = central + (dx/2)·laplacian` and
  `backward = central − (dx/2)·laplacian`, with the branch read off the gradient rather than
  re-derived from a second rule.

  **Nothing caught it because the default configuration is the one cell where it is invisible.**
  `use_upwind=True` with no-flux gives an empty true row *and* forces `p = 0` at the wall, so
  `dH/dp` multiplies the spurious diagonal away. Under Dirichlet or periodic nothing masks it.

  **Upwind has no Jacobian at a switching node** — the map is nondifferentiable there and only a
  Clarke generalised Jacobian exists. Probing it does not lose accuracy, it disagrees with itself,
  which is why the bands are assembled from linear operators rather than extracted.

  **Unchanged and still wrong: the default path.** The analytic block is reached only under
  `HJBFDMSolver(analytic_jacobian=True)`; the default per-point finite-difference fallback remains
  wrong at periodic row 0 and on upwind interior rows. Pre-existing, measured, not addressed here.

  Also corrected: `_extract_bands`' docstring claimed its O(Nx²) tier was reached in practice. It is
  reached, but by `Nx = 6` — `tests/conftest.py`'s `tiny_problem` — for a stated reason, not by the
  obstacle masks and Robin BCs the docstring named.

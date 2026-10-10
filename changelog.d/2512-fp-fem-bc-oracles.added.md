- **FP-FEM's no-flux wall and shared Dirichlet exit are checked against exact densities** in 1-D and 2-D
  (#2512 (a)).
  - **Fixtures.** Each density is `m = w(t) exp(-phi/D) g(x)`, so the total flux is
    `J = -D w exp(-phi/D) grad g`. The drift `-grad(phi)` crosses every wall.
    - At a no-flux wall, `dg/dn = 0` makes `J.n = 0` while `dm/dn ≠ 0`. The two readings of "no flux"
      therefore disagree there by `(v.n) m`, instead of coinciding as they would on a symmetric density.
    - At the exit, the shared `DIRICHLET(0.7)` is absorbing with its value dropped (row B3), so
      `m = 0` there.
  - **Order.** Each cell converges at second order (P1). Measured max-error ratios are 3.98–4.00 in 1-D
    and 3.63–3.80 in 2-D, on a non-square domain with a different drift per axis.
  - **Wall properties.** Each cell checks the property it is named for.
    - **No flux.** The boundary norm of `J_h.n` falls at first order (ratios 1.98–1.99 in 1-D, 1.94–1.98
      in 2-D). The same solve with the wall written as `dm/dn = 0` misses the density by a constant
      (0.65 in 1-D, 1.155 in 2-D) at every resolution.
    - **Exit.** The exit nodes hold 0 at every time. A rival that keeps the shared value misses.
  - **Mass identity.** It is pinned beside the oracle, not as the oracle: the loss equals the source
    minus the integrated exit flux, to 0.02 relative.
  - **Two production mutants each redden their cells:**
    - the advection without its by-parts form reddens all four cells;
    - the exit left unimposed reddens exactly the exit cells.
  - **Neumann cells.** The shared-NEUMANN cells, which solve as no-flux (row B3), are now met through
    the no-flux oracle.

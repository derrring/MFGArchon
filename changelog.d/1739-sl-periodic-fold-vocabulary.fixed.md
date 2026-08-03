- **Periodic boundary conditions are honoured by the semi-Lagrangian departure fold** (Issue #1739).
  Three sites in `hjb_semi_lagrangian.py` dispatched on `bc_op == "wrap"`, a spelling
  `bc_type_to_geometric_operation` has never emitted -- its alphabet is
  `{'reflect', 'periodic', 'clamp'}`. All three branches were unreachable, so a periodic foot fell
  through to a clamp or to an extrapolation with no exception and no warning, and the solve returned
  a value function for boundary conditions the problem did not declare. Measured on a unit domain, a
  foot at `-0.15` came back as `0.0` where its periodic image is `0.85`. The fold now has one owner,
  `fold_into_domain`, dispatching on the mapping's own vocabulary and **raising** on anything else,
  so the next drift stops the solve instead of quietly choosing a boundary condition.
- Pinned by the seam `|u(t, x_min) - u(t, x_max)|`, which is zero for any true periodic solution:
  it fell from 8.67e-01 to 2.45e-16 under `diffusion_method='canonical_cs'` and from 1.82e+00 to
  8.96e-04 under `'stochastic'`. The seam assertion carries a positive control -- it counts the feet
  that actually left the domain and refuses to report a seam unless the fold was exercised and the
  value function moved -- because a solver returning its terminal data untouched also has a seam of
  exactly zero.
- **Periodic solves move on the default `'adi'` method too**, wherever CFL substepping reaches the
  batch advection site. The `Nx=41, sigma=0.05` fixture in `test_sl_one_solve_one_interpolant.py`
  goes from `-75.511615372759422` to `-75.584015936808527`.
- Consequence for the out-of-bounds policy: the fold now runs unconditionally at every site, and
  every branch of it lands inside the domain -- `clamp` included, since that is `np.clip` against
  the same bounds `x_grid` is built from. So **no `bc_op` reaches the batch interpolant out of
  bounds**, and extrapolate-vs-clamp there is unobservable: swapping the extrapolating `interp1d`
  for a clamping `np.interp` moves that fixture by 1 ULP where it moved it by 1.9e-3 before. The
  extrapolation is kept because it is what the site has always done, not because a route to it
  exists.

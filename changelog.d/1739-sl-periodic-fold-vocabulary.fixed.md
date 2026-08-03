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
  8.96e-04 under `'stochastic'`.
- **Periodic solves move on the default `'adi'` method too**, wherever CFL substepping reaches the
  batch advection site: the `Nx=41, sigma=0.05` fixture in `test_sl_one_solve_one_interpolant.py`
  goes from `-75.511615372759422` to `-75.584015936808527`. A `Nx=21, sigma=0.3` configuration does
  not reach that site and is unchanged, so "the default path is unaffected" is false in general.
- Consequence for the out-of-bounds policy: the solver supports only
  `{NO_FLUX, NEUMANN, PERIODIC}`, which map to `{reflect, periodic}`, so once periodic feet fold
  **no supported BC reaches the batch interpolant out of bounds**. Swapping its extrapolating
  `interp1d` for a clamping `np.interp` now moves that fixture by 1 ULP where it moved it by 1.9e-3
  before. The extrapolation is kept -- the fold above is what makes it unobservable, and a
  `clamp`-mapped BC or a post-construction BC swap (#1699) restores it.

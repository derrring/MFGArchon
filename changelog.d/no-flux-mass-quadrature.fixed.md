- **The no-flux mass tests measured the rectangle rule, which is not the mass on this grid.**
  `TensorProductGrid` is endpoint-inclusive, so the two end nodes carry half a cell each and
  `sum(m)*dx` over-counts them by `dx*(m[0]+m[-1])/2` — 3.5% on the fixture, before any evolution.
  Both zero-drift mass tests now use `mass_drift`, the trapezoid owner the periodic test beside
  them already used, and both are `xfail(strict=True)`: under the correct quadrature the shipped
  no-flux solve loses **5.8e-3** of its mass, O(h). The rectangle-sum invariant they used to assert
  is kept under a name that says what it measures — it is a real guard on the wall assembly and it
  still catches the #1250 absorbing-wall regression.

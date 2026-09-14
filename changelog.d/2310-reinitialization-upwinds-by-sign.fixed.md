- **`reinitialize(method="pde")` upwinds by the sign of the initial level set** (Issue #2310). It used
  the `phi0 > 0` Godunov form everywhere, which is downwind where `phi0 < 0`. At 6c0610d2 an exact
  signed distance drifted without bound (2.99e+01 after 100 iterations on `x - 0.5`, 101 points), and
  a `phi0 < 0` minimum moved by 3.73e-02 on the default call. It now takes `gradient_upwind` of `phi`
  where `phi0 > 0` and of `-phi` where it is negative, per axis. A signed distance plane along either
  axis of a 2-D grid, and a 1-D one under a matching Dirichlet wall, stay fixed to rounding. The
  default `method="fmm"` does not change.

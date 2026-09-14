- **`reinitialize(method="pde")` upwinds by the sign of the initial level set** (Issue #2310). It used
  the `phi0 > 0` Godunov form everywhere, which is downwind where `phi0 < 0`. An exact signed distance
  drifted without bound (1.1e+01 after 100 iterations on `x - 0.5`), and #2308 made `phi0 < 0` minima
  worse. It now takes `gradient_upwind` of `phi` where `phi0 > 0` and of `-phi` where it is negative;
  a signed distance is a fixed point to rounding. The default `method="fmm"` does not change.

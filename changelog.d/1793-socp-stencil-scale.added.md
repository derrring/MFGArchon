- **The joint-SOCP stencil scale `h_i` is pinned** (Issue #1793). `h_i` non-dimensionalises the
  monotonicity cone AND appears in the per-edge ratio reported about it, so inflating it tightens
  the constraint, shrinks the optimum's gradient weights by the same factor, and leaves `kappa`
  back inside the bound looking healthy. A 25-axis mutation sweep found the whole suite blind to
  it: 5770 passed with `median` replaced by `max`. Two tests, doing different jobs — an external
  oracle asserting a uniform 2D cross returns the analytic five-point weights (which says the
  construction is right and provably *cannot* see the scale, since `median == max` there), and a
  pin on the reported `kappa` at a stencil whose neighbour distances spread 3x. Measured: the
  mutation moves `kappa` 5.03% and the Laplacian weights 0.018%, so `kappa` is the discriminator
  and the weights are not — a third test records that, so the file is not later "strengthened" by
  pinning the wrong quantity. Verified against `median -> max` and against `median -> mean`, an
  unintended change neither test was written for.

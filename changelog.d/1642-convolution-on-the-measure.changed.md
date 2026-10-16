- **The interaction convolution integrates on the problem's measure** (#1642, stage 0 item 4).
  - `ConvolutionCouplingOperator` on a regular grid weights each node by its trapezoid weight, the grid's measure
    (`quadrature_weights_nd` of its endpoint-inclusive axes), on the FFT path and the dense path alike. It weighted
    every node by `prod(spacings)`, the rectangle rule, which gives each end node a full cell past the domain: an
    O(dx) error. For a Gaussian kernel (amplitude 1, length scale 0.1) and m = 1 on [0, 1], the error against the
    exact box integral at 321 nodes is 4.9e-6, where it was 1.6e-3.
  - **Behaviour change:** its output changes on every regular-grid use, at the nodes the end weights reach. A
    per-node `cell_volume` is accepted on a regular grid, and copied, and a scalar one is refused there. Without a
    `cell_volume`, a one-node axis is refused, since it carries no measure; it was weighted by its spacing.
  - The FFT path's adjoint is `w * (K * g)`. It was the forward map, which is the adjoint only for uniform weights.
  - `ConvolutionCouplingOperator.from_problem(kernel, problem)` reads the problem's grid, its measure
    (`spatial_measure().weights`) and its topology. A periodic axis is refused by name: the convolution is Euclidean
    and does not wrap. A geometry that is not a `TensorProductGrid` is refused.
  - `create_nonlocal_source(W, weights)` takes the measure's per-node weights and computes `W @ (w * m)`, for a
    dense, sparse or matrix-free `W` (anything with a shape that applies itself with `@`, a wrapper of one included).
    Its second parameter, `grid_spacing`, a scalar cell volume, is renamed `weights`; a scalar there is refused with
    the remedy `problem.spatial_measure().weights`. `W` is the unweighted kernel. The library's weighted operators
    passed as built, a `ConvolutionCouplingOperator` or a `GraphonCouplingOperator`, are refused, since their `@`
    carries the weights already; the remedies are `create_lions_source(QuadraticInteractionEnergy(op))` or
    `op.kernel_matrix()`, and the graphon kernel's own matrix. A form derived from one, such as `op.T`, `op.H`,
    `2 * op`, `op.as_dense()` or a wrapper of one, cannot be told from an unweighted kernel and is weighted twice.

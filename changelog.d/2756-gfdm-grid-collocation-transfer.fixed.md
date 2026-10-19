- **HJB-GFDM moves a grid density to its collocation points by position, on the grid's own nodes** (#2756). Two
  silent wrong answers of one owner, the solver's grid-to-collocation transfer:
  - A grid density whose node count equalled the number of collocation points was read as already at the points, so
    grid node i's value went to collocation point i, and the terminal value with it. With the 21 nodes of a grid
    reversed as the collocation set, U moved by 4.592e-01. Such a density is now the grid's, mapped to the points,
    unless the points are the grid's nodes in its order, where the two readings agree and nothing is interpolated.
  - `GridCollocationMapper` rebuilt the grid's axes from its bounds, `np.linspace(lo, hi, n)`, so a graded
    `TensorProductGrid` was interpolated as if uniform in both directions: m = x mapped to 31 points was 2.500e-01
    off. It now works on the grid's axes, read off the geometry's nodes by `tensor_grid_axes`, and accepts them as
    `grid_axes=`. A uniform grid's axes are its `linspace`, so uniform-grid solves are unchanged bit for bit.
  - A geometry whose nodes are not a tensor grid, such as an implicit domain's sampled points, has no grid to map from:
    a density that is not at the collocation points is refused there, where it was read on invented uniform axes.
  - "These points are the grid's nodes in its order" has one owner, `points_are_grid_nodes` in
    `geometry.grids.tensor_grid`, which the coupling cost's node check (#1642) now calls too.

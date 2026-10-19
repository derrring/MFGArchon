- **HJB-GFDM moves a grid density to its collocation points by position, on the grid's own nodes** (#2756). Two
  silent wrong answers of one owner, the solver's grid-to-collocation transfer:
  - A grid density whose node count equalled the number of collocation points was read as already at the points, so
    grid node i's value went to collocation point i, and the terminal value with it. With the 21 nodes of a grid
    reversed as the collocation set, U moved by 4.592e-01. Such a density is now the grid's, mapped to the points,
    unless the points are the grid's nodes in its order, where the two readings agree and nothing is interpolated.
  - `GridCollocationMapper` rebuilt the grid's axes from its bounds, `np.linspace(lo, hi, n)`, so a graded
    `TensorProductGrid` was interpolated as if uniform in both directions: m = x mapped to 31 points was 2.500e-01
    off. It now works on the grid's axes and accepts them as `grid_axes=`.
- **A grid's node axes have one owner, `geometry_grid_axes` in `geometry.grids.tensor_grid`**, which reads them off
  the geometry's nodes. The other consumers that rebuilt them from the bounds now read them there too:
  `compute_fp_velocity_field`, FP-FVM's face velocity, the 1-D grids of HJB-SL and FP-SL-adjoint, and `FPParticleSolver`'s
  grid-to-particle interpolation and density estimate. Those solvers stop on a graded grid in `get_grid_spacing()`
  (#2567), so for them this removes a second definition rather than a wrong answer. A uniform grid's axes are its
  `linspace`, so uniform-grid results are unchanged bit for bit.
- "These points are the grid's nodes in its order" has one owner, `points_are_grid_nodes`, which the coupling cost's
  node check (#1642) now calls too.

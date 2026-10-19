- **HJB-GFDM moves a grid density to its collocation points by position, on the grid's own nodes** (#2756). Two
  silent wrong answers of one owner, the solver's grid-to-collocation transfer:
  - A grid density whose node count equalled the number of collocation points was read as already at the points, so
    grid node i's value went to collocation point i, and the terminal value's with it. With the 21 nodes of a grid
    reversed as the collocation set, in a run with no coupling term, U came back 4.592e-01 off in point order, and
    4.000e-01 off with that order undone, from the terminal value read by index. Such a density is now the grid's, mapped to the points, unless the points are the grid's nodes
    in its order, where the two readings agree and nothing is interpolated.
  - `GridCollocationMapper` rebuilt the grid's axes from its bounds, `np.linspace(lo, hi, n)`, so a graded
    `TensorProductGrid` was interpolated as if uniform in both directions: m = x mapped to 31 points was 2.500e-01
    off. It now works on the grid's axes and accepts them as `grid_axes=`.
- **A grid's node axes have one owner, `geometry_grid_axes` in `geometry.grids.tensor_grid`**, which reads them off
  the geometry's nodes. The other consumers that rebuilt them now read them there too: `compute_fp_velocity_field`,
  FP-FVM's face velocity, the 1-D grids of HJB-SL and FP-SL-adjoint, and `FPParticleSolver`'s grid-to-particle
  interpolation. On a uniform grid their axes are unchanged bit for bit, with one exception that was a wrong answer:
  - `FPParticleSolver`'s 1-D path read its drift on `np.linspace(xmin, xmin + Lx, Nx)`. Over bounds such as
    (0.2, 0.9), `xmin + Lx` is 0.8999999999999999, so a particle sampled on the last node lay past the axis and got
    the interpolation's fill value, zero drift. Over (0.2, 0.9) with 15 nodes and 3000 particles, the 201 that start
    on the last node did not move at the first step, where the drift moves every particle by -0.025. They now get the
    grid's drift.
  - On a graded grid those solvers stop in `get_grid_spacing()` (#2567), so there the change removes a second
    definition of the axes rather than a wrong answer.
- "These points are the grid's nodes in its order" has one owner, `points_are_grid_nodes`, which the coupling cost's
  node check (#1642) now calls too.

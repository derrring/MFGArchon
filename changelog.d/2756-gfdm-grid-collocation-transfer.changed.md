- **HJB-GFDM reads a density whose shape fits both representations as the grid's** (#2756). When a grid has as many
  nodes as there are collocation points, `solve_hjb_system` maps `M_density` from the grid to the points, which is
  what the coupling iterators pass; it uses M as it stands only where the points are the grid's nodes in their order.
  A density at the collocation points whose count happens to equal the grid's is therefore read as grid values. The
  rule is stated in `solve_hjb_system`'s docstring.
- **HJB-GFDM refuses by name a density whose shape fits neither the collocation points nor the grid** (#2756). Such
  a density used to go to the grid mapper, which reshaped it: some shapes solved silently, such as `(Nt, 21, 1)` on
  a 21-node grid, or a transposed `(Nt, 4, 5)` on a 5 × 4 grid. On a geometry with no tensor grid, such as an implicit
  domain's sampled points, a density not at the collocation points is refused the same way, where it was read on
  uniform axes made up from the bounds.
- **`compute_fp_velocity_field` refuses by name a geometry with no tensor grid, or a U whose spatial shape is not the
  grid's** (#2756). It used to build its axes over U's shape. FP-FVM's face velocity and the 1-D HJB-SL and
  FP-SL-adjoint grids refuse a geometry with no tensor grid too, though on the library's own geometries an earlier
  check stops those solvers first.
- `interpolate_grid_to_particles` and `interpolate_particles_to_grid` take the grid's axes as `grid_axes=`; without
  them the grid is uniform over `grid_bounds`, as before.

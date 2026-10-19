- **HJB-GFDM reads a density whose shape fits both representations as the grid's** (#2756). When a grid has as many
  nodes as there are collocation points, `solve_hjb_system` maps `M_density` from the grid to the points, which is
  what the coupling iterators pass; it uses M as it stands only where the points are the grid's nodes in their order.
  A density at the collocation points whose count happens to equal the grid's is therefore read as grid values. The
  rule is stated in `solve_hjb_system`'s docstring.
- **A grid field on a geometry with no tensor grid is refused, not read on uniform axes made up from its bounds**
  (#2756). HJB-GFDM refuses a density, a terminal value or a volatility field that is not at its collocation points
  when the problem's geometry, such as an implicit domain's sampled points, has no tensor grid to map it from.
  `compute_fp_velocity_field`, FP-FVM's face velocity and the 1-D HJB-SL and FP-SL-adjoint solvers refuse such a
  geometry by name too.
- `interpolate_grid_to_particles` and `interpolate_particles_to_grid` take the grid's axes as `grid_axes=`; without
  them the grid is uniform over `grid_bounds`, as before.

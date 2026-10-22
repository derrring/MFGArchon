- **`FPParticleSolver` ends every step at its first contacts with the particles' boundary, reflecting specularly**
  (#2733 PR 2; fixes #2707 and #2679). One exact search, `ImplicitDomain.segment_intervals`, finds where the step's
  path first leaves the domain, over the obstacles and the box walls together (`reflect_at_first_contacts`).
  - **The action.** A contact on the grid's box reads the boundary condition there. Every other contact, on an
    obstacle's surface or a curved boundary, reflects. A reflection is in the outward normal at the contact, from the
    surface's own form, and the rest of the step is re-traced.
  - **On the box alone** this is the fold it replaces, bitwise.
  - **#2707's table:** the peak count of particles on an obstacle's far side is 0 in all four cases. Before this change
    it was 500, 500, 0 and 9: a step through a thin obstacle was placed past it.
  - **A slab thinner than 1/32 of a step** (#2733's thin-feature row, the folded #2710): of width 1e-3 or 1e-4, against
    a step of 0.075, it holds all 500 particles behind it. Before this change all 500 ended past it.
  - **A 3-D room with a box obstacle and an exit:** 1250 to 1289 of 2000 particles absorbed over 12 seeds, against the
    noise-free shadow's 1260. Before this change it was 1944 to 1961: the projection slid about 700 particles past the
    obstacle.
  - **Ties.** Two contacts at one point reflect in each face in turn. Two faces with opposite normals bound a
    zero-width corridor, such as one an obstacle flush with a wall leaves: the rest of the step loses its normal part
    and runs along the corridor. Past the corridor's end that is a convention, and the particle stays on its line.
  - **A step left with no direction along a corridor is refused, naming the point.** In 1-D, a grid node on a box
    face flush with an obstacle is an isolated point of the domain. Before this change the step raised in
    `project_to_interior`. The remedy is no initial mass on that node.
  - **The cap.** A step reflected more than 20 times returns to its start, and a solve warns once when any step did.
- **Refused at construction, where a particle domain is set** (an `ImplicitDomain` geometry or `implicit_domain=`):
  - **An absorbing condition the boundary owner can place off the box**: one placed by `region`, `sdf_region`,
    `normal_direction`, `region_name`, `boundary="all"` or nothing, and an absorbing `default_bc`.
    - An exit on a disc's arc absorbed 400 of 400 particles drifting toward (1.2, 0.5), read where the arc touches the
      box, and 0 of 400 toward (0.95, 1.05), with 252 particles on the arc inside the exit. It is refused whatever the
      drift.
    - A `dirichlet_bc(...)` or `default_bc=DIRICHLET` beside an obstacle is this case. It absorbed at the box and
      reflected at the obstacle, 305 of 400. The remedy is one exit per box face with `boundary=` and a reflecting
      `default_bc`.
    - Reading the condition on an obstacle's surface or a curved boundary is #2789.
  - **An absorbing segment on box-face cells no particle can cross.** A `boundary="x_max"` exit on a disc in its
    bounding box absorbed 400 of 400 on axis, at the tangent point, and absorbs none under the contact loop. So does an
    exit on the corridor an obstacle flush with a wall leaves.
    - A cell counts where the domain's trace on the face, read 1e-12 of the box's diagonal inside, has a piece at least
      2e-4 of the diagonal long (`ImplicitDomain.face_trace`). This is exact in 1-D and 2-D, and a midpoint-line rule
      in 3-D and above.
    - Two cases are misjudged:
      - a ball of radius 5000 diagonals or more, tangent to a face with the domain inside it, is admitted although no
        particle crosses there;
      - a crossable piece shorter than 2e-4 of the diagonal is refused.
- **The boundary condition is read on the solver's box.** A BC whose `domain_bounds` differ from the solver's box is
  read where particles cross the solver's box. An exit placed on a face of the BC's own box that is not the solver's
  is refused, naming both boxes.
- **Initial particles are held to the box as they are to an obstacle domain.**
  - A sampled jitter at a boundary node that falls outside the box is redrawn within its cell (#2760's solver case).
    Before, 9% of a uniform 2-D sample started outside the box and was folded in at the first step, and 16.5% of a
    bump's sample on a [0, 1] x [0, 0.5] box. The t = 0 law is unchanged: for
    a uniform jitter, folding its outside half onto its inside half and redrawing it within the cell give the same law
    on the inside half. Seeded random streams change.
  - `initial_particles` outside the box are refused by name, where they used to be folded silently.
- **`exit_positions_history`** records where a particle met the exit, not where its raw step ended past it.
- **`obstacle_placement_history`** records each step's contact counts, on every solve: searched, contacts,
  reflections, corridors, offsets, fallbacks and absorbed.

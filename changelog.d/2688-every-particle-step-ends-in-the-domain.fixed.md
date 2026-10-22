- **Every forward step of `FPParticleSolver` with an `implicit_domain=` records its particles inside that domain,
  or refuses** (#2688, #2679 stage 2; the second half of #1561).
  - The problem's geometry, when it is an `ImplicitDomain` other than a `Hyperrectangle` on the grid's box, is read
    as that domain too (#2708).
  - **One end of step.** The 1-D grid-drift, n-D grid-drift and callable-drift paths end each step in one place,
    `_finish_step`.
    - The outer boundary condition runs on the raw step first: absorption at Dirichlet segments, then reflection.
    - Then the survivors that ended outside the domain are placed, with their positions before the step.
    - Before, the 1-D grid-drift path never placed them. The n-D paths placed them before the wall reflection,
      which put particles back inside an obstacle that meets the wall.
  - **The face of entry.** A particle that ends a step outside the domain is placed by the boundary its step met, not
    by the projection alone. #2733 replaced this stage's face-of-entry placement in the same release: its crossing scan
    of 32 points per leg could step over a thin feature or a clipped corner (#2710), and each step now ends at its
    exact first contact, reflected specularly (#2733's fragment). The walk toward the bounding-box centre is deleted.
  - **Initial particles are drawn navigable.**
    - `sample_from_density(..., navigable=)` redraws, within its cell, a jitter that lands where `navigable` is
      False.
    - Cells carrying mass with no navigable point on a 5-per-axis probe lattice hold mass outside the domain, inside
      an obstacle or beyond its outer boundary, and no redraw can place it. Their share of the mass, over every cell
      that carries any, is decided by `mass_fabrication_gate.leave_out_or_raise`, against the threshold that bounds a
      non-negativity clip (1e-8):
      - above it the sample is refused, naming the share;
      - at or below it those cells are left out, silently, as a clip of that size is, and the rest renormalised.
    - The 1-D grid-drift path samples the nodes themselves and decides its nodes outside the domain the same way.
    - `initial_particles` outside the domain are refused.
  - **Refused rather than delivered wrong:**
    - an `implicit_domain` with any periodic face on the outer boundary, since a wrapped step is not the segment
      from where it started (#2704);
    - an `implicit_domain` whose bounding box is not the grid's box. The boundary condition reflects at the grid's
      box and the placement acts inside the domain's: where the grid's box reaches further, a particle between the
      two is placed by neither; where the domain's does, a particle can be placed beyond the reflecting wall;
    - an `implicit_domain` on the GPU path, whose particles never met the placement.
  - **Known limitation,** which predates this change: a step that passes through an obstacle and ends past an
    absorbing exit is absorbed before any placement (#2707).

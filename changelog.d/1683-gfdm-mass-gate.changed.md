GFDM FP solves no longer renormalise the density to the initial mass after clipping
negatives. The path routes through `clip_nonnegative_or_raise` (#1683) and stops when a clip
would fabricate mass -- measured at 61% of the present mass on one configuration that
previously returned a final mass of exactly 1.0000. Removing the renormalisation exposed
#1752: the unstabilised central flux divergence (`upwind_scheme="none"`) diverges under
refinement, reaching a final mass of 2.79 where the initial mass was 1.0, and refining the
timestep makes it worse rather than better. That drift is now reported at WARNING with the
timestep it peaked at, rather than being scaled away.

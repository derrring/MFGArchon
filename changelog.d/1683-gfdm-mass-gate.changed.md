GFDM FP solves no longer renormalise the density to the initial mass after clipping
negatives. The path now routes through `clip_nonnegative_or_raise` (#1683) and stops when a
clip would fabricate mass -- measured at 61% of the present mass on one configuration that
previously returned a final mass of exactly 1.0000. Removing the renormalisation also
exposed #1752: this scheme is not conservative, drifting 179% on a configuration that clips
nothing, and that drift is now reported at WARNING rather than being scaled away.

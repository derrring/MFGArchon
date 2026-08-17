One owner for `BCType -> particle action`. The mapping was written out twice --
`MeshfreeApplicator.apply_particles` and `ParticleApplicator.apply` -- and had drifted on
**four of `BCType`'s eight members**. Measured on the same five particles and the same domain:

| BCType | MeshfreeApplicator | ParticleApplicator |
|---|---|---|
| `REFLECTING` | **ValueError** | reflecting |
| `ROBIN` (default `alpha=1, beta=0`) | **absorbing** | **reflecting** |
| `EXTRAPOLATION_LINEAR` / `_QUADRATIC` | ValueError | **reflecting** |

The `ROBIN` row is the defect that matters: the *same* `BoundaryConditions` object made one
path build an absorbing wall and the other an impermeable one -- opposite physics, no error.

`particle_action_for_bc_type` is now the single owner and both dispatches are deleted. Each
divergence is resolved toward whichever path was right, so this is not either previous
behaviour:

- `REFLECTING` reflects. It is `BCType`'s own documented particle spelling of an impermeable
  wall, and refusing it was the defect.
- `ROBIN` dispatches on its coefficients. `beta = 0` makes the condition `alpha*u = g`, which
  is Dirichlet, so the particle is absorbed; `alpha = 0` is a flux condition, so it reflects;
  a genuinely mixed Robin reflects, conserving mass.
- `EXTRAPOLATION_*` raises, naming the type. These describe how to continue a *field* past a
  truncated domain and carry no boundary datum; the segment-aware path's `else` branch used to
  reflect them silently.

The coefficients now come off the **matched segment** on the segment-aware path, so a mixed BC
with a different Robin coefficient per wall is read per wall. The uniform path's
`_robin_alpha_beta` cannot do that -- it takes the first Robin segment in the whole
specification whichever face is being processed -- and that remains open.

**Behaviour changes**, all of them repairs: `MeshfreeApplicator` accepts `REFLECTING` instead
of raising; `ParticleApplicator` absorbs at a `beta = 0` Robin wall instead of reflecting, and
raises on `EXTRAPOLATION_*` instead of silently reflecting. Of 22 probed cells, **15 are
byte-identical** across the change and 2 differ only in an exception message.

The pin does not compare the two paths against each other -- once both route through the owner
that is tautological and would pass over a broken owner. It pins the table against the physics
and each repaired cell against the behaviour captured before the merge. Three mutations were
measured to redden it: dropping `REFLECTING` from the impermeable arm (2 failures), restoring
the always-reflect Robin (4), and restoring the silent `EXTRAPOLATION_*` fallback (4).

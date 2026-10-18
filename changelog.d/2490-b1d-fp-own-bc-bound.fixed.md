- **A BC handed to an FP solver is bound to its problem's geometry** (#2490, B1d's binding). Only a grid bound a BC,
  so the shared BC was bound and a BC passed as `boundary_conditions=` was not, and a reader that asks the owner
  without a geometry could not read a `region_name` the grid marks. `BaseFPSolver._fp_own_bc`, the one route every
  FP solver's own BC takes, binds it; `bind_geometry` returns a copy, so the caller's BC is not written into.
  - `FPParticleSolver` with an explicit BC whose exit is a `region_name` marked on the grid raised "This reader has no
    geometry" at the first boundary particle; it now absorbs there, as the same BC set on the grid does. Measured
    over every FP solver that takes an explicit BC, that is the only answer the binding changes.
  - **Behaviour change:** a BC of another dimension than the problem's geometry is refused, naming the dimension to
    build it in. `FPParticleSolver`, `FPFDMSolver` and `MeshlessGalerkinFPSolver` accepted a 1-D BC on a 2-D grid.

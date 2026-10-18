- **A BC handed to an FP solver is bound to its problem's geometry** (#2490, B1d's binding). Only a grid bound a BC,
  so the shared BC was bound and a BC passed as `boundary_conditions=` was not, and a reader that asks the owner
  without a geometry could not read a `region_name` the grid marks. `BaseFPSolver._fp_own_bc`, the one route every
  FP solver's own BC takes, binds it. `bind_geometry` and `bind_dimension` return a copy with a segment list of its
  own, so the caller's BC is not written into.
  - `FPParticleSolver` with an explicit BC whose exit is a `region_name` marked on the grid raised "This reader has no
    geometry" at the first boundary particle; it now absorbs there, as the same BC set on the grid does.
  - A `region_name` that is also a face label (`"x_max"`), on a grid that marks a region of that name over part of
    the face, was read by the explicit route as the whole face. It is now read by the mark, as the grid's own BC is:
    `FPParticleSolver` absorbs only on the marked part of the face and reflects elsewhere on it, and
    `MeshlessGalerkinFPSolver`, which solved, now raises #2717's refusal, as each does with the BC set on the grid.
  - Over the five FP solvers and four exits the binding was measured on, those are the only answers that change.
  - **Behaviour change:** a BC of another dimension than the problem's geometry is refused, naming the dimension to
    build it in. `FPParticleSolver`, `FPFDMSolver`, `FPFVMSolver`, `FPSLSolver`, `FPFEMSolver` and
    `MeshlessGalerkinFPSolver` accepted a 1-D BC on a 2-D grid; for a uniform BC its reading was the 2-D BC's.

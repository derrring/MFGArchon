- **A coupling loop refuses an HJB or FP solver built for a geometry whose points are not those of the problem it
  solves** (#2775). Each solver discretizes on its own problem's geometry. Before, a solver built for another geometry
  with the same number of points was coupled with nothing raised, and the solve was wrong:
  - a mesh with its interior moved, 3.7e-02 off;
  - a 1-D grid over another interval, 3.9e-01 off.
  - The only check was #1489's DOF count, in `FixedPointIterator` alone.
  - One owner, `assert_paired_solver_geometry`, now compares each solver's geometry with the solved problem's by
    `get_spatial_grid()`, exactly. It also holds #1489's count.
  - Every loop that pairs an HJB and an FP solver reads it, at construction:
    - `FixedPointIterator`, the `BlockIterator` family, `FictitiousPlayIterator` and `NewtonMFGSolver`;
    - `MFGResidual`;
    - `MultiPopulationIterator`, `GraphMFGSolver` and `RegimeSwitchingIterator`, once per pair.
  - Two separately built equal grids or meshes are still one geometry. An implicit domain's `get_spatial_grid` is a
    random sample, so two implicit domains compare equal only when they are one object.
  - A solver with no `problem` attribute is refused by name. Every built-in HJB and FP solver has one, and a solver
    written to the protocol sets `self.problem` to the problem it was built for.
  - FEM P2 beside P1 on one mesh is refused by name in every loop. Before, only `FixedPointIterator` refused it, and
    the other loops raised an unnamed broadcast error.

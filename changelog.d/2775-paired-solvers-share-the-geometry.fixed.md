- **A coupling loop refuses an HJB / FP pair that does not hold its values at the points of the problem it solves**
  (#2775). Before, a pair whose halves held their values at other points was coupled with nothing raised whenever the
  number of points agreed, and the solve was wrong:
  - a mesh with its interior moved, 3.7e-02 off;
  - a 1-D grid over another interval, 3.9e-01 off;
  - a meshless Galerkin pair on two clouds of 21 points, 7.7e-02 off.
  - The only check was #1489's DOF count, in `FixedPointIterator` alone and only on a mesh.
  - One owner, `assert_paired_solver_geometry`, now compares each solver's geometry with the solved problem's by
    `get_spatial_grid()`, exactly. A weak-form solver (HJB-FEM, FP-FEM, the meshless Galerkin pair) declares the points
    its DOFs sit at as `dof_points`, and the two halves' points are compared too: `dof_points` where declared, the
    geometry's points otherwise. The owner also holds #1489's count, for a pair that reports `n_dof` without
    `dof_points`, on every geometry.
  - Every loop that pairs an HJB and an FP solver reads it, at construction:
    - `FixedPointIterator`, the `BlockIterator` family, `FictitiousPlayIterator` and `NewtonMFGSolver`;
    - `MFGResidual`;
    - `MultiPopulationIterator`, `GraphMFGSolver` and `RegimeSwitchingIterator`, once per pair.
  - Two separately built grids or meshes whose points are bit-identical are still one geometry. An implicit domain's
    `get_spatial_grid` is a fresh random sample on every call (#2778), so two implicit domains are one geometry only as
    one object, and an equal one built separately is now refused by name.
  - A solver with no `problem` attribute is refused by name. Every built-in HJB and FP solver has one, and a solver
    written to the protocol sets `self.problem` to the problem it was built for.
  - FEM P2 beside P1 on one mesh is refused by name in every loop. Before, `FixedPointIterator` refused it by #1489's
    count, `FictitiousPlayIterator` refused every mesh, and the five other loops measured raised an unnamed broadcast
    error.

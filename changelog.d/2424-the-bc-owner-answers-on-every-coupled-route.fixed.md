- **Every coupled route reads the problem's boundary conditions from their one owner** (#3005), so the owner's refusals answer whatever BC the route's solvers carry: two declarations that differ (#2530), and a FLUX on the shared BC (#2581). `IterationState`, which every coupled route builds per component, reads `problem.get_boundary_conditions()` when it is built and at each `start`, which every route calls at each solve.
  - **Main solves, this change refuses** (the user's ruling of 2026-11-26, that two declarations that differ are refused on every route):
    - `GraphMFGSolver`, both of whose solvers carry their own BC, with two declarations that differ or a FLUX on the shared BC. Graph never asked the owner.
    - A BC changed on the problem between two solves of one iterator, so that its declarations differ, is refused at the next solve. At main FixedPoint refused it, reading the owner every sweep. FictitiousPlay, both Block iterators, Newton, Graph, RegimeSwitching and MultiPopulation solved it.
    - A BC changed so that its declarations differ after the iterator is built and before its first solve is refused at that solve. At main FixedPoint and `SolverFactory.create_solver` refused it. FictitiousPlay, `BlockIterator` and its two subclasses, Newton, MultiPopulation, RegimeSwitching and Graph solved it.
    - A warm-start accessor of `FixedPointIterator` (`set_warm_start_data`, `get_warm_start_data`, `clear_warm_start_data`, `has_warm_start_data`) called while the declarations differ refuses there, since it builds the iterator's state. At main it returned, and a solve after the declarations agreed again solved.
  - **Main refuses, this change solves** (the approver's ruling of 2026-11-26): `FixedPointIterator`, with the problem's BC changed during one solve from an iteration callback, on two set-ups:
    - both solvers carrying their own BC;
    - HJB-GFDM carrying its own BC, with FP-FDM on the problem's. FP-FDM reads the owner once, when it is built.

    At main FixedPoint's per-sweep read refused both at the next sweep. The state now reads once per solve, and the solve completes on the BC each solver holds. With HJB-GFDM on the problem's BC, HJB-GFDM refuses the change at the next sweep, at main and here. `FictitiousPlayIterator` solves both set-ups at main and here. Stage 2a's driver is to read the owner at every sweep on every route (#3002).
  - **Not read, at main or here:** a BC changed after `MFGResidual` is built, when it is used on its own. It evaluates on the state it built (#3005).
  - **The callers of the provider check this release removes (#2424), and of FixedPoint's per-sweep resolution,** each read the owner, and the reads in `IterationState` keep their refusals, except the change during one solve above. They are FixedPoint, `problem.solve()` and `SolverFactory.create_solver` with solvers of your own, FictitiousPlay, `BlockIterator` and its two subclasses, `MFGResidual` used on its own and through Newton, RegimeSwitching and MultiPopulation.
  - **When they refuse:**
    - `MFGResidual` and `NewtonMFGSolver` when built, as at main;
    - FixedPoint and `problem.solve()` at the solve, as at main;
    - `FictitiousPlayIterator`, the Block iterators, `RegimeSwitchingIterator` and `MultiPopulationIterator` at the solve, where at main they refused when built;
    - Graph at the solve;
    - FixedPoint, FictitiousPlay and the Block iterators also at a warm-start accessor called before the solve.
  - **Measured:** on a 1-D grid with HJB-GFDM and FP-FDM each given their own BC, on every entry point above; the change during one solve also with only HJB-GFDM given its own, and with neither. A network problem, whose owner answers None or a `GraphBCConfig`, is refused by name, as this release's #2923 entry says.

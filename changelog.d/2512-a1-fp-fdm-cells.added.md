- **FP-FDM has exact-density oracles on every BC it declares, in 1-D and 2-D; FP-FEM's prescribed-density pair is
  counted** (#2512 (a)). The manifest's oracle line goes from 37/67 to 49/67.
  - New, in `tests/unit/test_alg/test_fp_fdm_bc_oracles_2512.py`, on the default scheme (`divergence_upwind`) and the
    default drift arm (`potential_field`):
    - a drifted no-flux wall, non-stationary, run with the drift's sign both ways on every axis, so each wall row's
      upwind selection takes both branches;
    - a periodic seam crossed by the drift, with a density that is not even about the seam;
    - a 2-D exit whose shared value is nonzero.
  - The no-flux and periodic cells assert a rate, the error at every level, and at every level the error on each
    face's interior and on the corners, or on the seam, each against its recorded value. The exit asserts m = 0 at
    the exit at every step, a rate, and the error at each level. The rivals each fixture separates are measured and
    recorded there: 13 mutants of the no-flux wall row, the torus one cell too long (#1822), the seam's U difference
    dropped, no wrap (with the repeated endpoint's constraint row), and the exit read literally or made a wall.
  - The no-flux and periodic cells converge at first order. That is the upwind advection's: the 1-D no-flux fixture
    gives order 2 drift-free and with `divergence_centered`.
  - Counted from existing tests, each asserting its wall nodes:
    - FP-FDM's 1-D exit (#2525), which already did;
    - FP-FDM's prescribed density in 1-D and 2-D (#2535), and FP-FEM's (#2538's level test), which gain that
      assertion here.
  - FP-FDM's NEUMANN cells are met through the NO_FLUX oracle they solve as (#2512 comment 6079979873).

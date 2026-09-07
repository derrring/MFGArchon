- **Two test comments that asserted coverage which does not exist** (Issue #2257). Both misled in the
  confident direction: a reader checking them concluded a guard was pinned when it is not.
  `test_particle_gpu_pipeline.py` said the #1910 GPU absorbing-BC refusal "is asserted in
  test_gpu_particle_refuses_absorbing_bc_1910.py" — a file `18d8cc80` deleted. It now names the
  refusal by symbol (`FPParticleSolver._solve_fp_system_gpu`) and states plainly that it currently
  has no test. `test_newton_mfg_solver.py` said `local_ci.sh` filters without `not manual`, so a
  manual-only test still runs in the gate; that was true at `ef6c4bb2` and is false now that
  `scripts/ci_markers.txt` is the single owner and carries `not manual`.

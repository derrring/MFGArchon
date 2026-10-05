- **The FP-FDM gradient advection schemes `gradient_upwind` and `gradient_centered` are removed, with their legacy aliases `"centered"` and `"upwind"`** (Issue #2007, maintainer ruling 2026-10-04).
  - Asking for one raises `ValueError`, naming `divergence_upwind` or `divergence_centered` instead. `"flux"` (-> `divergence_upwind`) still resolves.
  - The removal is not the HJB-FDM option of the same name: `HJBFDMSolver(advection_scheme="gradient_upwind")` is unchanged.
- **Why removal and not a fix.** The schemes carried two separate defects:
  - The gradient form discretizes v·∇m, which drops m∇·v from ∇·(vm). With a non-constant drift it solves a different equation. On a source-free instance, repointing its wall at the conservative routine left it non-convergent: error 5.81e-1 -> 8.02e-1, EOC about 0.1.
  - Its wall imposed ∂m/∂n = 0 instead of J·n = 0, giving EOC 0.00 at a drifting wall.

  No library route selected these schemes: `FDM_UPWIND` and `FDM_CENTERED` route to the divergence form. Explicit selection was the only way to reach them, and a test recorded a decision to keep that selection. That decision is retired with the test.
- **Removed with them:**
  - their two assembly modules;
  - the non-conservation warning;
  - `fp_fdm_operators.add_interior_entries`, a compatibility wrapper whose only callee was the gradient interior;
  - the tests that pinned their defects or compared them with the divergence form.

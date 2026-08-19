`TestDualityConvergence::test_mesh_refinement_improves_accuracy` is removed rather than repaired
(#1998). It is the whole of "Nightly full test suite is failing": the validation shard reports
`1 failed, 27 passed, 4 xfailed` — that one test — and every other shard is green, on both red runs
the issue covers.

It recorded `result.max_error` — the final **Picard residual** — and compared two of them across
`Nx = 20 / 40` under the name "h-convergence". Measured, with the test's own `tolerance=1e-8` and
`max_iterations=30`, **both legs run to the cap**: `converged=False, iterations=30/30`. Raising the
cap does not help — at 300 neither converges: the residual peaks at **1.26e+03 around iteration 4**,
then settles onto a `1e-4`…`2e-3` floor it never leaves, ending far from its own best
(`1.78e-06` at iteration 194). `max_error` here is a sample from a cycle, so no assertion over two samples
can measure a rate. The precedent is `TestConvergenceRate::test_upwind_first_order_convergence`,
removed for this exact reason and named by #1728; the other removal in this file,
`test_centered_fdm_higher_order`, is a different defect (one grid, tautological assertion).

**The window is not clean, and saying so was wrong.** A real numerical change landed inside it:
`c98a9c5f` (#1902) deletes the hand-rolled `u[0] = u[1] - g*dx` no-flux enforcement — the boundary
treatment this test runs through — and moves these numbers ~5× on darwin:

| ref | `Nx=20` | `Nx=40` | verdict |
|---|---|---|---|
| `179e55a7`, head of the last green nightly | 1.185395e-04 | 5.550650e-04 | **fails** on darwin |
| `c98a9c5f` and after | 5.935364e-04 | 4.237936e-04 | passes on darwin |
| linux (CI), `4a50e27b` and after | 8.814973e-05 | 9.901234e-04 | **fails**, 11× |

The verdict flipped in *opposite directions* on the two platforms across the same change, which is
what a limit-cycling residual does and is stronger grounds for deletion than a clean-window story.
No platform conditional exists in the window and CI pins the BLAS thread counts to 1. Anyone tracing
an FDM_UPWIND no-flux discrepancy to this window should start at #1902 and #1904.

**What the deletion costs, and the first version of this understated it.** `e < 1000` was **not**
vacuous: the residual crosses it by 26% in this very solve. So it was a crude but **live** guard on
the transient decay rate — anything costing ~eight Picard iterations of convergence speed pushes
iteration 30 back over 1000 and fires it. No domination claim survives either: `max_error` here is
entirely the **U** increment (at `Nx=40`, iteration 29: `err_U` 4.238e-04 against `err_M` 4.807e-08),
while `test_fdm_upwind_stable`'s mass-conservation and `M.min() > 0` assertions constrain M's
physics. The two sets are incomparable, not ordered, and that test stops at iteration 20.

The rest is smaller than first stated. The `Nx=40` leg duplicates `test_fdm_upwind_stable` on every
listed parameter, differing only in `max_iterations` (30 vs 20) and tolerance. And the coarse leg is
not uncovered in kind: `tests/integration/test_fvm_hjb_coupling.py`'s `fdm_1d` fixture runs a coupled
1D no-flux `FDM_UPWIND` solve at `Nx=25, Nt=12, sigma=0.4` asserting convergence, a 10× error drop,
mass and positivity — strictly stronger, at comparable coarseness. What has no counterpart is the
**low-σ / high-Péclet** coarse regime, and the transient guard.

Deleting is still right — a threshold on an unconverged residual is a poor instrument even when live,
and the *rate* assertion beside it is indefensible at any threshold. But the transient goes unguarded
and that is recorded, not claimed away.

Coupled EOC through the production `FixedPointIterator` already exists in
`tests/integration/test_coupled_mfg_mms.py`. What remains uncovered is coupled EOC **at a no-flux
wall** — that suite is periodic — which the deleted test never measured either. #2006's standalone FP
order study covers the wall but not the coupling.

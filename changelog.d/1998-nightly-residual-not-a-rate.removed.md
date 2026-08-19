`TestDualityConvergence::test_mesh_refinement_improves_accuracy` is removed rather than repaired
(#1998). It is the whole of "Nightly full test suite is failing": the validation shard reports
`1 failed, 27 passed, 4 xfailed` — that one test — and every other shard is green, on both red runs
the issue covers.

It recorded `result.max_error` — the final **Picard residual** — and compared two of them across
`Nx = 20 / 40` under the name "h-convergence". Measured, with the test's own `tolerance=1e-8` and
`max_iterations=30`, **both legs run to the cap**: `converged=False, iterations=30/30`. Raising the
cap does not help — at 300 neither converges: the residual peaks at **1.26e+03 around iteration 4**,
then settles onto a noise floor — over iterations 26–300 it ranges `1.78e-06`…`3.44e-03` (`Nx=20`)
and `7.12e-06`…`3.40e-03` (`Nx=40`) — ending far from its own best. `max_error` here is a sample from a cycle, so no assertion over two samples
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
vacuous: the residual crosses it by 26% in this very solve, so it was a **live** guard on the
transient decay rate — but a very coarse one. Measured: the assertion samples the *final* residual,
so tripping it needs the whole decay curve delayed by **24–26 of the 30 iterations**, or the
per-iteration decay factor to fall from its actual **~1.85 to ≤ 1.01**. A solve that merely slowed
down stays green; only one that has stopped converging fires it. No domination claim survives either: `max_error` here is
entirely the **U** increment (at `Nx=40`, iteration 29: `err_U` 4.238e-04 against `err_M` 4.807e-08),
while `test_fdm_upwind_stable`'s mass-conservation and `M.min() > 0` assertions constrain M's
physics. The two sets are incomparable, not ordered, and that test stops at iteration 20.

The rest is smaller than first stated: the `Nx=40` leg duplicates `test_fdm_upwind_stable` on every
listed parameter, differing only in `max_iterations` (30 vs 20) and tolerance.

**The coarse leg has no counterpart anywhere, and an earlier version of this fragment said
otherwise.** It claimed `tests/integration/test_fvm_hjb_coupling.py`'s `fdm_1d` fixture asserts
"convergence, a 10× error drop, mass and positivity — strictly stronger". Those four assertions are
made on the **`fvm_1d`** fixture, which runs `FVM_MUSCL`. `fdm_1d` runs `FDM_UPWIND` and has exactly
one consumer, `test_1d_fvm_fdm_agreement`, which asserts cross-scheme agreement (`rel_m < 0.07`) and
nothing about convergence, mass or positivity. Verified: `grep -n "fdm_1d\|fvm_1d"` over that file
returns 9 lines, 3 of them the `fvm_1d`-only tests, 1 the shared agreement test.

So a raise or a NaN from the coarse `FDM_UPWIND` solve is now caught by nothing — which is what this
fragment originally said before the false citation replaced it. The **low-σ / high-Péclet** coarse
regime is uncovered, and so is the coarse regime generally.

**The case for repairing instead** is stronger than "fix the two-grid rate test", and worth weighing:
`test_fdm_upwind_stable` already runs the identical `Nx=41, Nt=20, sigma=0.1, FDM_UPWIND` solve and
already holds `result`, so a transient bound there is one line at zero marginal runtime — measured
margin for `final < 1e-2 * peak` at its own 20-iteration budget: **73.5×** at `Nx=20`, **31.9×** at
`Nx=40`.

**And the transient is not left unguarded** — an earlier draft of this fragment said it was.
Measured on `test_dual_fdm_pair_converges`' own configuration (`Nx_points=[41]`, `Nt=20`,
`sigma=0.1`, `FDM_UPWIND`, `max_iterations=50`), its `:94` assertion `final_error <= initial_max*2.0`
reads threshold `1.003133e+03` against final `1.823556e+02` — a **5.50× margin** — and it is *more*
sensitive than the deleted one: `:94` fires at a uniform post-peak decay factor below **1.0440**,
`e < 1000` at below **1.0088**, so `:94`'s firing region strictly contains it, 20 iterations sooner.
What the deletion actually costs is the `sigma=0.1` **coarse** leg (`Nx_points=[21]`), which has no
such neighbour.

Deleting is still right, and not because the guard was worthless. Whatever the guard becomes belongs
on the **surviving** test, so this one goes either way. It also wants a **decay** bound rather than a
magnitude one — `final < 1e-2 * peak` is scale-free, margin **31.9×** at `test_fdm_upwind_stable`'s
own budget. Writing a threshold here to fit what was observed is how this file got its numbers:
`c1a3696b`, the day the file was created, is *"Relax convergence test to allow oscillatory behavior
… final error ≤ 2× initial max error"* — and that is `:94`, the assertion now doing the real work.
(`e < 1000` arrived with the file at `fc07ae31`, 2026-01-17; `nightly.yml` did not exist until
`8fa80818`, 2026-04-02, so no nightly pressure was involved.) Tracked as #2014.

Coupled EOC through the production `FixedPointIterator` already exists in
`tests/integration/test_coupled_mfg_mms.py`. What remains uncovered is coupled EOC **at a no-flux
wall** — that suite is periodic — which the deleted test never measured either. #2006's standalone FP
order study covers the wall but not the coupling.

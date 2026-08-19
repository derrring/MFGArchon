`TestDualityConvergence::test_mesh_refinement_improves_accuracy` is removed rather than repaired
(#1998). It is the whole of "Nightly full test suite is failing": the validation shard reports
`1 failed, 27 passed`, that one test, and every other shard is green.

It recorded `result.max_error` — the final **Picard residual** — and compared two of them across
`Nx = 20 / 40` under the name "h-convergence". Measured, with the test's own `tolerance=1e-8` and
`max_iterations=30`, **both legs run to the cap**: `converged=False, iterations=30/30` on either
grid. So the two numbers are residuals of unconverged iterations and their ratio is a fact about how
far each got in thirty steps, not about mesh resolution. #1728 names this exact defect in a
different test — "it compared `result.max_error`, the final Picard residual — so removing it lost
nothing" — and a sibling in this same file, `test_centered_fdm_higher_order`, was removed on
2026-07-25 for the same reason, with the account left in place above where this one sat.

That also explains why bisecting the green-to-red window found nothing. Same code, same nightly
invocation (`-n auto`, the nightly marker set, which does not exclude `slow`):

| platform | `Nx=20` | `Nx=40` | assertion |
|---|---|---|---|
| darwin/arm64 | 5.935364e-04 | 4.237936e-04 | holds |
| linux (CI) | 8.814973e-05 | 9.901234e-04 | fails, 11x |

Both deterministic on their platform — CI produced identical digits on two different commits — so
it is one BLAS path through a non-converged iteration against another, not a regression from any
commit in the window.

Nothing is lost. The `Nx=40` leg's configuration is identical to
`TestNumericalStability::test_fdm_upwind_stable` in the same file — `Nx_points=[41]`, `Nt=20`,
`T=1.0`, `sigma=0.1`, the same components, `FDM_UPWIND` — and that test asserts `U` and `M` finite
and the density positive, strictly stronger than this one's `e < 1000` on a residual. Real
h-convergence has a home as of #2006: `tests/unit/test_alg/test_fp_mms_wall_order_1728.py` measures
observed order against a source-free exact solution, with error-level bounds verified mutation-red.

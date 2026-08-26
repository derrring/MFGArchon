`TestDualityConvergence::test_mesh_refinement_improves_accuracy` is removed rather than repaired
(#1998). It was the whole of "Nightly full test suite is failing" — that one test, every other shard
green.

It recorded `result.max_error`, the final **Picard residual**, and compared two of them across
`Nx = 20 / 40` under the name "h-convergence". Both legs run to the iteration cap without
converging, and raising the cap does not help: the residual settles onto a noise floor and ends far
from its own best. A sample from a noise floor cannot measure a rate, whichever direction the
comparison happens to come out.

The precedent is `TestConvergenceRate::test_upwind_first_order_convergence`, removed for the same
reason and named by #1728. The other removal in this file, `test_centered_fdm_higher_order`, is a
different defect — one grid, tautological assertion.

A real numerical change landed inside the window this test was blamed for (#1902's deletion of the
hand-rolled no-flux enforcement), so "the nightly went red on its own" was not true either.

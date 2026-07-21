`converged` on the three PINN solvers reported whether an epoch had run, not whether training
converged. `len(self.training_history.get("total_loss", [])) > 0` is `True` whenever a single epoch
completed -- and the guard four lines above each site already raises `RuntimeError` when that list
is empty, so the expression could only ever be `True`. A run whose final loss was `1e30` reported
convergence.

The training loop has always carried the real criterion: it breaks when
`total_loss < convergence_tolerance`. The defect was that the reporting layer ignored the
computation it reports on and substituted its own test.

`PINNBase.converged` is now the one owner, defined as `best_loss < convergence_tolerance` -- the
same comparison the loop breaks on, so the reported flag and the loop's own decision cannot
disagree. All three solvers read it rather than recomputing. Exits via `early_stopping_patience` or
`max_epochs` leave `best_loss` at or above tolerance and now report `False`; an untrained solver
keeps `best_loss = inf` and reports `False`.

This is the first of the seven decoupled `converged` flags on #1684; the other four reporting
defects and the two that can stop a solver early are not addressed here.

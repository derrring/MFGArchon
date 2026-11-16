- **A custom network Hamiltonian is no longer refused at construction for what it returns at a grid probe**
  (#2901, stage 1c-iii). Construction's checks read a `NetworkHamiltonian`'s `__call__`, `dm` and `dp` at a grid probe
  point: x a node's coordinates cast to a node, p of the coordinate dimension and m one number, a form no network solver
  hands. Each of these was refused, though it returned a number where the solvers read it, and now builds:
  - a `hamiltonian_func` that raised `TypeError`, returned `None` or returned NaN there;
  - a `hamiltonian_dm_func` that raised `TypeError` or returned `None` there;
  - a `NetworkHamiltonian` subclass handed to `MFGProblem(model=Model(hamiltonian=...))` on a network geometry, whose
    `dp` raised `TypeError` there. No network solver accepts that problem.

  Five of those six refusals said "wrong signature". No grid probe reads a network Hamiltonian, and its sources are
  compared nowhere: with the built-in control `__call__`, the rates and `source_term` come from one formula, and with a
  custom `hamiltonian_func` both readers of the rates, `FPNetworkSolver` and `NetworkPolicyIterationHJBSolver`, refuse
  it (#1545). Those two refusals are now pinned.

Corrected four false claims about the `obstacle` field and pinned the defect they concealed
(#2002). `mfg_problem.py` declares `obstacle` as the variational inequality `v >= Psi(x)`, but
both paths acting on it compute `max(0, Psi)` — no `v` — so the term is identical at a node
satisfying the constraint and one violating it: it penalises position, not violation. The two
also disagree by 1e10 (`(1/eps) * max(0, psi)` with `eps = 1e6` against
`penalty_parameter * max(0, psi)` with `1e4`), while `source_composition`'s docstring asserted
they "match rather than silently diverge" and pointed to `PenaltyHJBSolver` as "proper handling"
— that wrapper carries the same stub, and its own docstring described the intended
`(1/eps) * max(0, Psi - v)` rather than the implemented term. All four sentences are withdrawn
at their sites. The constraint itself is implemented correctly and reachably elsewhere —
`ObstacleConstraint.project` (#591), applied by `HJBFDMSolver(constraint=...)` — so this is a
single-source-of-truth defect, not a missing capability. Behaviour is unchanged; which owner
survives is #2002's open question.

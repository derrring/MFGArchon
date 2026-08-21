Corrected false claims about the `obstacle` field and pinned the defect they concealed (#2002).
`mfg_problem.py` declares `obstacle` as the variational inequality `v >= Psi(x)`, but both paths
acting on it compute `max(0, Psi)` — no `v` — so the term is identical at a node satisfying the
constraint and one violating it: it penalises position, not violation. The two also disagree by
1e10 (`(1/eps) * max(0, psi)` with `eps = 1e6` against `penalty_parameter * max(0, psi)` with
`1e4`), and they are additive rather than alternative — `PenaltyHJBSolver` adds its term on top of
whatever `source_term` it receives, which for a problem with `obstacle` set is the other one.
Withdrawn at their sites: the `PenaltyHJBSolver` pointer as "proper handling" (that wrapper carries
the same stub), in both copies of that claim; and `hjb_penalty`'s docstring, which described the
intended `(1/eps) * max(0, Psi - v)` rather than the term it computes. Behaviour is unchanged;
which owner survives is #2002's open question. Two adjacent defects found while establishing the
above are filed separately: #2036 (the `constraint=` path clips post-hoc in 1D and returns an
infeasible terminal slice in nD) and #2037 (`PenaltyHJBSolver` has never run compatibility
validation).

Added a **measured** account of which solvers a manufactured source can reach
(`tests/unit/test_alg/test_source_term_reachability_2020.py`, #2020). Both prior counts were
signature audits, and a signature is a coarser predicate than the hazard — `HJBWENOSolver` named
`source_term` while threading it through the 1D path only. Each solver is now solved twice, with a
zero source and a strong one, and classified by whether the answer moved: 5 thread it, 6 refuse
loudly, 10 are out of this fixture's reach and are listed by name. The class the test exists for is
the third one — accepting a source and returning a byte-identical answer, which would let an MMS
measure the order of an equation without its source and report a clean, wrong number. The verdict is
read against a per-solver noise floor from a same-source repeat, because `FPParticleSolver` differs
by 1.3e-01 on byte-identical inputs and "the answer moved" is not evidence for a stochastic solver.

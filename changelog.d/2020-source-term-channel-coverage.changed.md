`test_source_term_channel_2020.py` now covers **14 of the 21** concrete solvers instead of 6, and
holds its parametrisation against discovery so the gap cannot reopen silently (#2020). The eight
added rows are `HJBGFDMSolver`, `HJBWENOSolver` and `FPFVMSolver` (honour a source) and
`HJBSemiLagrangianSolver`, `FPGFDMSolver`, `FPParticleSolver`, `FPSLSolver`, `FPSLJacobianSolver`
(refuse it — a correct outcome, recorded per row so that one which starts honouring it fails and
gets its row updated deliberately). The remaining seven are named in `_UNCOVERED` with the reason,
rather than being absent. The file's status line said "every solver here now honours a source",
which was true of its six rows and not of the channel: five of the eight added solvers refuse.

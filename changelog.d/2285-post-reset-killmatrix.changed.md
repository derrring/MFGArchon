- **The discrimination killmatrix and baseline are re-recorded post-reset** (Issue #2285). Both were
  measured at `cb4f2adf`, three days *before* the #2227 test reset removed 2,927 collected tests, so
  every figure read off them described a suite that no longer existed. They now carry a run at
  `2c923694`: 665 of 3822 tests kill at least one of 24 conventions, no convention undefended, and
  no kill count lower than the pre-reset record. `AGENTS.md` gains the worktree isolation recipe the
  sweep requires and a sweep cost derived from the matrix instead of estimated.

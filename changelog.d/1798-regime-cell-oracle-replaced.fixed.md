The `regime_switching/non_negativity` capability cell no longer gates a proposition that is false of
a correct solver. Its verdict required each regime's mass to stay **constant** to 1e-6; in a
Markov-switching MFG the generator `Q` transfers mass between regimes by construction, so per-regime
mass follows `M(t) = M(0) expm(Q t)` and only the **sum** is conserved. Integrating the FP system
over a no-flux domain gives `dM/dt = M Q` directly. The cell was red on something no implementation
can satisfy — worse than no check, because it teaches a reader to ignore the report.

**Replaced, not relaxed.** #1767 named tolerance-fitting as the route the matrix exists to catch, and
the previous `intended` note explicitly refused it. The quantity changed instead. The 8.88e-02 that
note called drift is almost entirely the physical transfer: at `t = 1` the measured masses are
0.594648 / 0.499156 against the oracle's 0.593323 / 0.498957.

**What is gated now:** `max_rel_vs_expm_oracle` — the external oracle, computed without reference to
the scheme — at 1e-2. The tolerance is bracketed on both sides by measurement rather than fitted to
one: it admits a correct solve at this resolution (2.43e-03, first order in `dt` at `Nt = 10`) and
refuses the harness's own 10% density mutation (1.11e-01).

A mass term had to remain in the gate. This cell is a member of `MASS_ORACLE_CELLS`, and that listing
is a claim its verdict reads mass — measured, non-negativity alone does **not** discriminate, the
same mutation moving `min_density` only from 1.087e-04 to 1.141e-04. Total-mass drift
(1.40e-03 → 1.02e-01 under mutation) is recorded rather than gated, because a leak in the sum shows
up as a departure from the oracle too.

**The cell stays FAIL, and no status moved** (FAIL=7, PASS=2, UNSUPPORTED=2, unchanged). It is now red
on `picard_converged: false` at its 3-sweep budget — the same axis as its FDM and SL siblings — rather
than on the per-regime mass question. `min_density` is 1.087e-04, strictly positive, so #1681 remains
fixed. Whether it converges at a real budget is unmeasured.

Regenerating the baseline dropped this cell's `intended` note, which is the carry-forward mechanism
working: a changed artifact is meant to make the cell read as unexplained so someone looks at it. The
other ten notes are byte-identical, verified by hash, and the unexplained count is back to 0. (#1798)

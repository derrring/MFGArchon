- **The weekly discrimination sweep fails only on a loss, and lists a gain** (#1715, user ruling of 2026-10-15).
  - `scripts/test_discrimination.py --check-baseline` exits 1 on any of these:
    - a kill count that drops;
    - a test that stops killing a mutation;
    - a mutation that disappears or stops applying;
    - a baseline mutation the kill matrix lacks.
  - A count that rises, a new killer and a new mutation exit 0. They are listed under their own heading and in one
    `::warning` annotation.
  - Until now a gain failed the run too. Four pending fragments state that two-way rule, and each is now marked
    "one-way since #1715": `test-discrimination-ratchet.added.md` (#1737), `1817-discrimination-baseline.fixed.md`,
    `1901-discrimination-ratchet-killer-sets.changed.md` and `2342-preset-threading-mutation.changed.md`.
  - The baseline is re-recorded monthly by a full `--write-baseline` sweep in its own PR. The sweep stamps
    `_measured_at.swept_on` on the baseline and the kill matrix, and that date, not the baseline's last commit, is
    when it was last re-recorded.
  - The baseline and kill matrix are re-recorded at `ac8eae9f`: 27 of 27 mutations are live.
    `bc_uniform_dispatch_reads_as_mixed` read INEFFECTIVE because its live check expected a ghost value that had
    since moved. The check is now structural, and the mutation is live again.
  - `test_sl_pair_symmetry_1880::test_the_pair_stays_stable_when_its_hjb_half_does_not_substep` is deleted. Its only
    perturbation was round-off, and seeded at 1e-6 the same pair grows instead of converging (measured at
    `f7aeead2`, #2465 comment 6185002012).

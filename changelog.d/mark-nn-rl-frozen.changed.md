- **`alg/neural/` and `alg/reinforcement/` are marked FROZEN — design prototypes, not under
  development.** Three layers, because prose alone does not hold a line here: a section in
  `CLAUDE.md` (what agents read), a banner in each package docstring (what a human reading the
  code sees), and `scripts/check_frozen_areas.py` wired into `local_ci.sh` (what fires without
  being read). The counter-intuitive half of the freeze is that adding TESTS is also out —
  coverage reads as a promise that behaviour is intended and load-bearing, and on a placeholder
  that promise is false, so later readers preserve decisions nobody made. Keeping the packages
  importable, one-line build fixes, and filing issues remain allowed. The checker counts test
  files importing either package (12 and 2 today) and fails when the count grows; it ships a
  `--self-test` and was verified against a real added test before being wired.
  Review hardened the checker: it now walks the AST and reads string literals, because the regex
  form missed `importlib.import_module("mfgarchon.alg.neural.nn")` — the verbatim idiom of one of
  the two files in its own baseline, which was counted only via an unrelated static import
  elsewhere in that file — plus `pytest.importorskip`, `patch("...")`, and
  `from mfgarchon.alg import neural`. It compares file SETS rather than counts, so delete-one-add-one
  no longer nets to zero, and it runs its own `--self-test` inside `--check-baseline` rather than
  leaving verification to an opt-in flag nothing invokes. Two docs that advertised the frozen RL
  package as "Production-Ready ✅" are marked, `CLAUDE.md`'s Scope line no longer lists DGM/PINN/
  Actor-Critic under ✅ two lines above the ⛔, and four frozen-area issues are labelled
  `status: blocked` instead of reading as available work.
  A second review found the two branches added to close the first review's blockers had no
  positive control: the self-test covered exactly the three shapes the original regex already
  handled, so deleting either new branch left the gate green while reopening the exact blocker.
  Neither is load-bearing for any of the 14 baselined files -- all 14 carry a static import -- so
  a baseline regeneration, the full suite and CI would all have passed over it. There is now one
  fixture per detector branch, each labelled with the branch it exercises, and deleting a branch
  fails naming it. Four documents teach a frozen package, enumerated with the checker's own prefix
  matching rather than a grep (which counted `mfgarchon.alg.neural_solvers` as a hit); all four are
  marked, and every trailing `**Status**` line in them -- one still read "Production-Ready" 435
  lines under a freeze banner -- now agrees with the header.
  A third review found the same gap one level up: `--self-test` covers `_references`, the code that
  turns a source file into a detection, and stops there. The two set differences in `main()` that
  turn a detection into a failure had no control, and three single-line mutations left
  `--check-baseline` at exit 0 while `--self-test` printed PASSED -- including one that killed the
  drop-detection this ratchet is advertised on. `tests/unit/test_check_frozen_areas.py` now pins
  both directions against a synthetic tree, as `test_check_fail_fast.py` does for its sibling.

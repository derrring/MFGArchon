- **`alg/neural/` and `alg/reinforcement/` are deleted** — 60 files, 23,118 lines, 11.6% of the
  package. They were frozen design prototypes for months; this ends them rather than freezing them
  further.

  The deciding measurement: **`boundary_conditions` appears zero times in `alg/neural/`'s 10,610
  lines.** Not hardcoded, as #1555 has it — absent. Both paradigms were designed before the
  geometry and boundary-condition layer this library is built on, so the gap is architectural
  rather than a set of defects to patch: there is no seam to make them BC-aware without rewriting
  each solver around a layer that did not exist when it was written.

  Severing them touched **one import** — `alg/__init__.py` — plus documentation, examples and the
  guards that existed only to police the freeze. Nothing in the numerical core depended on them.
  Recorded defects that go with them: #1555 (BC ignored), #1570 (shares none of the single-sourced
  conventions), #1342 (DGM is one abstract base with 22 unimplemented methods and one concrete
  subclass), #1684 item 1 (`converged` is a tautology), #1789 (the README advertised them as
  complete).

  **Maze generation is not affected.** It lives in `geometry/graph/` and the RL package only
  re-exported it; the example and the guide that used the re-export now import from the owner.

  Also removed: `gymnasium`, `stable-baselines3` and `tensorboard` from `[nn]` and `[all]` — the
  first two served only `alg/reinforcement`, the third had no importer anywhere;
  `scripts/check_frozen_areas.py` and its baseline, a ratchet whose entire purpose was to stop
  tests being added to the deleted packages; 15 test files, 12 examples, 3 user guides and a
  benchmark that exercised them.

  **This does not fix #1776.** `import mfgarchon` still costs 5.3s and still pulls torch: the cause
  is `utils/acceleration/__init__.py`, reached from `utils/numerical/nonlinear_solvers.py` for a
  `HAS_JAX` constant, and it is unrelated to the frozen paradigms. Measured before and after.

  Two ratchets recorded the gain rather than the loss: fail-fast violations 327 → 311, doc-API
  findings 137 → 126. Both baselines tightened in this change.

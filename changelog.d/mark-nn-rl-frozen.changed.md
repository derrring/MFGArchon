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

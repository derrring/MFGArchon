- **The capability harness no longer silences the library it measures** (Issue #1879).
  `scripts/capability_matrix.py` carried a bare, uncommented `warnings.filterwarnings("ignore")` at
  import — process-wide, so anything importing it inherited the deafness. While the matrix decided
  whether a configuration can solve at all, the library could tell it nothing. What that hid,
  measured the moment it was removed: `fdm_upwind/mass_conservation`, PASS on every run, emits **39**
  warnings reading *"the value function returned for this timestep is not a root of the discrete
  HJB, and the outer iteration will consume it as if it were"*, and `regime_switching` emits **42**
  (Issue #1878). Each cell now runs under `catch_warnings` and what the library said lands in its
  artifact as `library_said`, folded by category and message with digits collapsed so near-identical
  warnings count rather than repeat. Import and deprecation warnings are excluded: they differ
  between machines and would fake a baseline diff, and they are not statements about whether the
  configuration solves. **Recorded, not gated** — `--check-baseline` still compares status only, and
  no cell changes status; the field's purpose is that the next regeneration shows this in a diff
  instead of requiring a five-hour investigation to find.

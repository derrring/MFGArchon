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
- **The exclusion is by message as well as by category, after adversarial review.** The first
  version excluded import and deprecation warnings only, on the reasoning that what remained —
  including the JAX-autodiff fallback, which fires only where JAX is importable — was "capability
  information rather than noise". Measured, it is not: forcing `_JAX_AVAILABLE = False` leaves the
  numeric half of all three affected 2-D cells byte-identical, so the entry says what is installed,
  not whether the configuration solves. Recorded, it made this committed baseline machine-dependent,
  and via `_note_still_applies` a regeneration on a JAX-free machine **silently dropped three
  `intended` notes**, including the ~1500-character #1865 investigation record — which
  `--check-baseline` cannot catch, because it compares status only.
- **Numbers are collapsed, hyphens are not.** The normalising regex `[-+0-9][0-9.eE+-]*` also ate the
  hyphen inside ordinary words, folding `non-negativity` to `nonNnegativity` and
  `stable-baselines3` to `stableNbaselinesN` — merging warnings that differ in their text rather
  than in their numbers.
- **Per-cell attribution is now tested.** Every original test monkeypatched `CELLS` to a *single*
  stub, so the field's actual claim — that a cell carries its own warnings and no one else's — was
  unverified by construction: hoisting `catch_warnings` out of the per-cell loop, which credits each
  cell with its predecessors' output, passed all five. Verified on a real two-cell run to credit
  `gfdm_rbf/construction` with two non-convergence warnings emitted by `fdm_centered`. A two-cell
  test now kills it, and kills nothing else.
- **The import-time guard's structural half was scoped wrong in both directions.** It tested
  `category is Warning and module is None`, which missed `simplefilter("ignore")` (whose `module` is
  `""`) and `filterwarnings("ignore", category=UserWarning)` — the latter reopens the process-wide
  trap for one category and left every test green. Widening it to any `Warning` subclass then
  flagged CPython's own always-installed defaults. It now asks the only question that matters:
  would this filter swallow a category the harness records.
  Five mutations, each asserted to apply before it ran and each killed by exactly the test written
  for it; both files restored and SHA-256-verified afterwards.

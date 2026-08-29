- **A moved warning identity set now says what else moved** (Issue #2158). The baseline's keys were
  `_comment`, `occurrences_when_written`, `identities` — nothing recorded what produced it, so the
  report could say *that* the set moved and never *why*. The one time that mattered, six identities
  were attributed to pytest when five came from elsewhere, and the only line in the gate naming a
  cause was about ruff. `tests/conftest.py` now records `{python, pytest, numpy, scipy, scikit-fem,
  torch, cvxpy, osqp}` in every census, and `check_warnings.py` reports the delta beside a moved
  identity set.
- **Each name carries its own warrant, and they are not the same strength.** An earlier draft of
  this entry claimed all of them had been *measured* moving an identity; review found that false for
  three. Measured in that comparison: cvxpy and torch (absent, so their tests skipped or were not
  collected — three of the six), numpy (2.2.6 against 2.4.6), pytest (9 added the check). Owning
  identities outright in the baseline, so a bump moves their text with no inference: scikit-fem
  (3 of 224) and osqp — the `OSQP failed: …` text is the solver's own status string, and
  `OSQP_AVAILABLE` selects a different warning entirely, so its absence moves identities
  structurally. Co-varying only: scipy, named in that comparison inside a disjunction with numpy
  and owning no identity, and python.
- **A name recorded on one side only is a schema difference, never a package moving.** Comparing the
  union of both key sets reported `cvxpy <absent> -> 1.9.2` and `scikit-fem <absent> -> 12.0.1`
  against a five-field baseline, with both packages installed and unchanged throughout — and the
  reverse, `cvxpy 1.9.2 -> <absent>`, for a census written by an older `conftest.py`. That is worse
  than the standing guess it replaced, because it wears the authority of a measurement. Only names
  present on both sides can have moved; one-sided names are reported as what they are.
- **The "went absent" note is printed only when something went absent.** It is the one line telling
  the reader what to do, and under a package that *appeared* it said the opposite of what happened.
- **`importlib.metadata.version` returns the first match and is silent about the rest.** A stale
  `.dist-info` beside a current one records a version no code in the run used. `osqp` carries
  1.1.1 and 1.1.3 in this environment and `ruff` carries 0.15.17 and 0.16.0; which one wins is
  `os.listdir` order, unsorted on this volume, so "correct today" is luck. No count is stated,
  because the population is not stable — an editable reinstall adds and removes a second
  `mfgarchon` record underneath you, and a review and a re-measurement forty minutes apart
  disagreed on the total for exactly that reason. The reader now takes the **distinct versions**
  across the records — two records at the same version dedupe to one — and returns
  `ambiguous:1.1.1|1.1.3` rather than picking. A wrong version is worse than none: it invents a
  package move, or hides a real one behind the stale pair.
- **The census writer had no test at all** — four mutations survived, including deleting the
  `toolchain` key from the payload outright, which makes the attribution cease to exist while every
  `check_warnings.py --self-test` case stays green. Those cases build their baselines and censuses
  by hand, so none of them touches the code that writes one in a real run. The reader was lifted out
  of the hook (a closure cannot be tested, and the duplicate-`.dist-info` case cannot be reached
  from an end-to-end run at all) and
  `tests/unit/test_warning_census_records_the_toolchain_2158.py` covers both.
- The payload key-set assertion is written out rather than read from `TOOLCHAIN_NAMES`. Keyed on the
  constant, it followed the constant: deleting `osqp` from the list survived. A test cannot be keyed
  on the thing it is testing.
- **A side that is wholly absent is not a one-sided name.** Introduced while fixing the union bug
  above and caught reading the production path before push: a baseline predating
  `toolchain_when_written` would have listed all eight recorded names as "recorded on one side only"
  the moment the identity set moved — eight lines of noise, the exact class this block exists to
  remove. Covered, and three mutations of the guard are killed.
- **`osqp` is deliberately not yet in the committed `toolchain_when_written`.** This machine has two
  `osqp` records (1.1.1 and 1.1.3), so a baseline regenerated here would carry
  `ambiguous:1.1.1|1.1.3` and every contributor with a clean install would see a spurious drift line
  against it. Until it is regenerated somewhere clean, the report says `Recorded on one side only:
  osqp`, which is exactly true. The ambiguity marker doing its job on a real pair is itself the
  first evidence that the reader works.

- **The comparability band was withdrawn from this change.** It is being redesigned around cause
  rather than magnitude — see #2165. Three defects decided it: the standing
  "usually an optional backend is absent" guess still fired in the commonest case (the toolchain
  unchanged, the cause in the tree); `--write-baseline` sat *before* the guard, so the refusal's own
  second option destroyed exactly what the guard existed to protect, printing only a count of what
  it dropped; and `0.005` was calibrated against the largest single-commit *growth*, which a
  one-sided shortfall band can never see. Measured independently on `9f84c22c`, marker set held
  fixed on both sides: 6195 → 6002 collected, a legitimate −193, which is 5.7× the band.
- **`--write-baseline` refuses a census carrying an unresolved marker.** The guard for this was
  written in the *test* and not in the code, while the path that reaches the committed artifact had
  none — and it is the path the report itself prints as the remedy. Measured on this machine's real
  census: `--write-baseline` would have committed `osqp: ambiguous:1.1.1|1.1.3`, after which every
  clean machine reads `osqp ambiguous:1.1.1|1.1.3 -> 1.1.3`, a permanent fabricated attribution line
  on every future red gate. Both halves asserted: poisoned → exit 2 and the artifact untouched at 224
  identities, clean → exit 0 and written at 223.
- **A schema difference no longer prints under a heading that claims a move.** `The toolchain also
  moved since the baseline was written:` followed three lines later by *"says nothing about whether
  those packages moved"*. That is the configuration this change ships — a seven-name baseline against
  an eight-name census — so it fired on the first red gate after merge. Two blocks, two headings, and
  the `(no drift, one-sided names)` shape now has a case: without it, reverting to the single
  `if drift or unrecorded:` heading survives.
- **A record that will not say its version is not an absent package.** `importlib.metadata` swallows
  `PermissionError` and `FileNotFoundError` inside `read_text` and returns an empty message, so four
  of five corruption modes — including `chmod 000`, the one `unreadable` is named for — recorded
  `None`, which means *not installed*. The report then prints `numpy 2.4.6 -> <absent>` for an
  installed, working numpy, under the note saying its tests did not run. Six modes now answer
  `unreadable`, with a control that a genuinely absent name still answers `None`.
- **A malformed `toolchain` is a schema error, not a warnings regression.** A non-empty non-dict
  reached the comparison as a raw `TypeError`, which `local_ci.sh` reports as the warnings check
  failing. It gets a diagnostic and exit 2 now, like every other malformed-census class. An *empty*
  list still folds into the same "nothing recorded" state as `None`, which is right.
- **The floor case proved only that the floor is positive.** `tests_run: 0` against `MIN_TESTS * 2`
  is scale-invariant: measured, it survived `MIN_TESTS` at 1, 2, 50 and 6000, and at 6000 a
  legitimate 10% suite shrink is refused with a red no warning fix can clear. Three boundary points
  now pin its location and a literal pins its value — six mutations killed.
- **The census writer's stub had one outcome class, so `tests_run` was not really covered.**
  `{"passed": [None] * 7}` left the other three at `[]` either way, and dropping `skipped` from the
  sum survived — a mutation this change's own table listed as killed. Four classes with distinct
  counts now.
- **The identity-extraction block ran in no test.** The stub had no `warnings` key, so the regex, the
  path normalisation, the digit normalisation, the 40-character truncation, the kind field and the
  occurrence count were all dead — seven mutations survived. A `WarningReport`-shaped fixture covers
  them, and the truncation is pinned to exactly 40 with a message longer than the cut: a `<= 40`
  bound passes for any wider truncation when every fixture message is already shorter.
- **No count of duplicate `.dist-info` pairs is stated anywhere.** A review and a re-measurement forty
  minutes apart disagreed on the total, because an editable reinstall adds and removes a second
  `mfgarchon` record underneath you. The mechanism and the two stable examples are stated; the
  population is not.
- The baseline's `_comment` — the artifact's own self-description — now documents
  `toolchain_when_written` and the values that cannot be written into it. Verified byte-identical
  against what `--write-baseline` produces.
- **The corruption handling emitted a warning from the census writer itself, and the ratchet caught
  it.** `Message.__getitem__` returns `None` for a missing header and warns
  `Implicit None on return values is deprecated` on Python 3.12+, so reading a record that has no
  `Version` — the case this function exists to handle — added a new identity originating in
  `tests/conftest.py`. `GATE RED`, one NEW identity, named exactly. `.get("Version")` is the
  non-deprecated form. Worth stating plainly: this fix was found by the guard this change is about,
  in the shape that guard exists for.

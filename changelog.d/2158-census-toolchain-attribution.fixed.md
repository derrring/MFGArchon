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
  `.dist-info` beside a current one records a version no code in the run used. Two distributions
  carry two records each in this environment today (`osqp` 1.1.1/1.1.3, `ruff` 0.15.17/0.16.0), and
  which one wins is `os.listdir` order — unsorted on this volume, so "correct today" is luck. The
  effect was reproduced on a constructed pair, since neither real pair is one of the recorded names
  yet — `osqp` becomes one in this change. The reader now counts the records and returns
  `ambiguous:2.2.6|2.4.6` rather than picking one. A wrong version is worse than none: it invents a
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

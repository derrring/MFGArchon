- **A non-PASS capability cell now says whether it is intended.** `scripts/capability_baseline.json`
  carries an `intended` note on each non-PASS cell; cells without one are unexplained and are the
  actual backlog, which `--check-baseline` now reports. Three defects found in review are fixed
  with it: the `#1745` note cited residual `2.42e-01` as if the number were still current and described it as a step norm; it was a real residual, from a full-step Newton diverging geometrically, which `#1763` fixed by defaulting `line_search=True`.17e-05`, 11.7x the `1e-6` tolerance, so the
  cell is not diverging and the diagnosis is the convergence rate, not a broken Jacobian;
  `regime_switching/non_negativity` was labelled INTENDED while `#1681` is open and labelled
  `type: bug`, which dropped an open bug out of the very backlog metric this change introduces;
  and the written `_comment` told readers to "See `--explain`", a flag that does not exist.
- **Carry-forward no longer preserves a note across a changed failure.** Requiring only "still
  non-PASS" let an annotation survive an arbitrary change of exception while the run still reported
  zero unexplained cells — and because the note is an unchanged line, it never appeared in the
  reviewer's diff. It is now dropped unless the artifact's exception is unchanged.
- **`--json` emits JSON.** The cells run real coupled solves that wrote to stdout from three
  independent places (library INFO logging, Rich progress bars, plain prints), so the flag never
  produced parseable output. The evaluation is redirected to stderr rather than each source being
  silenced, so the next thing added cannot break it again.
